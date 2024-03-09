# Adapted from https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_marlin.py

import numpy as np
import torch
import torch.nn as nn

try:
    import autogptq_marlin_cuda
except ImportError as e:
    marlin_import_exception = e

    def error_raiser_marlin(*args, **kwargs):
        raise ValueError(
            f"Trying to use the marlin backend, but could not import the C++/CUDA dependencies with the following error: {marlin_import_exception}"
        )

    autogptq_marlin_cuda = error_raiser_marlin


def mul(A, B, C, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    """Marlin FP16xINT4 multiply; can be used within `torch.compile`.
    @A: `torch.half` input matrix of shape `(m, k)` in standard row-major layout
    @B: `torch.int` weight matrix of original shape `(k, n)` in Marlin format; see `Layer.pack()`
    @C: `torch.half` out matrix of shape `(m, n)` in standard row-major layout
    @s: `torch.half` scales of shape `(m / group_size, n)`
    @workspace: `torch.int` tensor with at least `n / 128 * max_par` entries that are all zero
    @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
    @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    """
    autogptq_marlin_cuda.mul(A, B, C, s, workspace, thread_k, thread_n, sms, max_par)


# Precompute permutations for Marlin weight and scale shuffling


def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single

# _perm, _scale_perm, _scale_perm_single = _get_perms()

# def unpack_qzeros(qzeros):
#     unpacked_zeros = torch.zeros(
#         (qzeros.shape[0], qzeros.shape[1] * 8),
#         dtype=torch.int8,
#         device=qzeros.device,
#         requires_grad=False,
#     )

#     for col in range(unpacked_zeros.shape[1]):
#         i = col % 8
#         unpacked_zeros[:, col] = (qzeros[:, col // 8] >> (4 * i)) & 0xF

#     return unpacked_zeros + 1

def pack(x, nbits=4):
    pack_size = 32 // nbits
    out = torch.zeros((x.shape[0]//pack_size, x.shape[1]), dtype=x.dtype, device=x.device)
    bitmask = 2**nbits - 1
    for i in range(pack_size):
        out |= (x[i::pack_size] & bitmask) << (nbits*i)
    return out

def unpack(x, nbits=4, axis=0):
    assert nbits == 4
    bitmask = 2**nbits - 1
    pack_size = 32 // nbits
    dim0_size = x.shape[0] * pack_size if axis == 0 else x.shape[0]
    dim1_size = x.shape[1] * pack_size if axis == 1 else x.shape[1]
    output = torch.empty((dim0_size, dim1_size), dtype=x.dtype, layout=x.layout, device=x.device)

    if axis == 0:
        for i in range(pack_size):
            output[i::pack_size, :] = (x >> (i*nbits)) & bitmask
    elif axis == 1:
        for i in range(pack_size):
            output[:, i::pack_size] = (x >> (i*nbits)) & bitmask
    else:
        assert False, "invalid unpack axis"
    return output


class MarlinQuantLinear(nn.Module):
    QUANT_TYPE = "marlin"

    def __init__(self, qweight, qzeros, scales, g_idx, bias, bits, group_size):
        super().__init__()
        
        pack_size = 32 // bits
        infeatures = qweight.shape[0] * pack_size
        outfeatures = qweight.shape[1]
        
        device_capability = torch.cuda.get_device_capability()
        if not device_capability[0] >= 8:
            raise ValueError(f'Can not use Marlin int4*fp16 kernel with a device of compute capability {device_capability}.')
        if infeatures % 128 != 0 or outfeatures % 256 != 0:
            raise ValueError("`infeatures` must be divisible by 128 and `outfeatures` by 256.")
        if bits not in [4]:
            raise NotImplementedError("Only 4 bits are supported.")
        if group_size not in [-1, 128] and group_size != infeatures:
            raise ValueError("Only group_size -1 and 128 are supported.")
        if infeatures % group_size != 0:
            raise ValueError("`infeatures` must be divisible by `group_size`.")
        
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.group_size = group_size if group_size != -1 else infeatures
        
        self.desc_act = not ( g_idx is None 
                              or torch.equal(g_idx, torch.arange(infeatures, device=qweight.device) // group_size) )
        
        if self.desc_act:
            # shuffle weight rows
            self.perm = torch.argsort(g_idx)
            # unpack --> shuffle --> pack
            qweight = pack(unpack(qweight)[self.perm])

        # Repack into marlin format
        self.B = autogptq_marlin_cuda.gptq_repack(qweight)
        
        # # Check symmetric quantization, very slow, skipping for now
        # dequantized_qzeros = unpack_qzeros(qzeros)
        # if not torch.all(dequantized_qzeros == 8):
        #     raise ValueError(
        #         "Marlin kernel is compatible only with checkpoints using symetric quantization. "
        #         "Found non-symmetric quantization for the weight {name}."
        #     )
        
        # Process scales
        _, _scale_perm, _scale_perm_single = _get_perms()
        s = scales.data.clone()
        if group_size != infeatures:
            s = s.reshape((1, -1))
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s = s.reshape((-1, outfeatures)).contiguous()
        self.s = s
        
        # TODO: Can the workspace be shared among all marlin invocations?
        self.workspace = torch.zeros(self.outfeatures // 128 * 16, dtype=torch.int, device=qweight.device)
        self.bias = bias if bias is not None else None

    def post_init(self):
        pass

    def forward(self, A):
        A = A.half()
        #Support activation reordering
        if self.desc_act:
            A = A[:, self.perm]
        C = torch.empty(A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device)
        mul(
            A.view((-1, A.shape[-1])),
            self.B,
            C.view((-1, C.shape[-1])),
            self.s,
            self.workspace,
        )
        C = C + self.bias if self.bias is not None else C
        return C
