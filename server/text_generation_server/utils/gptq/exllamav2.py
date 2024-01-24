# Adapted from turboderp exllama: https://github.com/turboderp/exllamav2
# Adapted AutoGPTQ: https://github.com/PanQiWei/AutoGPTQ

import torch
from torch import nn

from exllamav2_kernels import make_q_matrix, gemm_half_q_half
from text_generation_server.utils import print_rank_n

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


def ext_gemm_half_q_half(x, q_handle, q4_width, force_cuda):
    """Matrix multiplication, returns x @ q4"""
    output_shape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.half, device=x.device)
    gemm_half_q_half(x, q_handle, output, force_cuda)
    return output.view(output_shape)


def ext_make_q_matrix(w: dict, temp_dq, key: str = None):
    """
    Create Q matrix 
    """

    if w["scales"].dtype == torch.float:
        w["scales"] = w["scales"].half()

    # GPTQ with g_idx (act_order)
    g_idx = w.get("g_idx")
    if g_idx is not None and g_idx.any():
        w["q_perm"] = torch.empty((w["qweight"].shape[0] * 8,), dtype=torch.short, device=w["qweight"].device)
        w["q_invperm"] = torch.empty_like(w["q_perm"])
        # make_q4 segfaults if g_idx is not on cpu in the act-order case. In the non act-order case, None needs to be passed for g_idx.
        return make_q_matrix(
            w["qweight"],
            w["q_perm"],
            w["q_invperm"],
            none_tensor,
            none_tensor,
            none_tensor,
            w["qzeros"],
            w["scales"],
            g_idx.cpu(),
            temp_dq,
        )
    # GPTQ without g_idx
    else:
        return make_q_matrix(
            w["qweight"],
            none_tensor,
            none_tensor,
            none_tensor,
            none_tensor,
            none_tensor,
            w["qzeros"],
            w["scales"],
            none_tensor,
            temp_dq,
        )


def temp_dq_size(inout_product):
    return inout_product * 2 + 128


def _elements(size_bytes):
    size_bytes = (size_bytes + 127) & -128  # round up to nearest multiple of 128
    return size_bytes // 2


class ExLlamaV2DeviceTensor:
    def __init__(self, device, scratch_bytes):
        self.device = device
        print_rank_n(f"Allocating {scratch_bytes // (1024 * 1024)} MiB for exllama v2 scratch space")
        self.scratch = torch.empty(
            _elements(scratch_bytes), dtype=torch.half, device=self.device
        )

    def get_scratch_slice(self, size_bytes):
        size_half = _elements(size_bytes)
        return self.scratch[:size_half]


# DEVICE_TENSOR is a cuda buffer used by cublas gemm when M > 50
DEVICE_TENSOR = None
DEVICE = None
# Max of (infeatures * outfeatures), used by temp_dq_size calculation
MAX_INOUT_PRODUCT = 1


def set_device(device):
    global DEVICE, DEVICE_TENSOR, MAX_INOUT_PRODUCT
    DEVICE = device
    DEVICE_TENSOR = ExLlamaV2DeviceTensor(DEVICE, temp_dq_size(MAX_INOUT_PRODUCT))


class Ex4bitLinearV2(nn.Module):
    """Linear layer implementation with per-group 4-bit quantization of the weights"""
    def __init__(self, qweight, qzeros, scales, g_idx, bias, bits, groupsize):
        global MAX_INOUT_PRODUCT
        super().__init__()
        assert bits == 4

        self.device = qweight.device
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.g_idx = g_idx.cpu() if g_idx is not None else None
        self.bias = bias if bias is not None else None
        
        self.height = qweight.shape[0] * 8
        self.width = qweight.shape[1]

        assert self.device.type == "cuda"
        assert self.height % 32 == 0
        assert self.width  % 32 == 0

        # Update max outfeatures & inout_product so far for later call to set_device
        MAX_INOUT_PRODUCT = max(MAX_INOUT_PRODUCT, self.width * self.height)
    
    def post_init(self):
        global DEVICE_TENSOR
        assert self.qweight.device.type == "cuda"
        self.q_tensors = {
            "qweight": self.qweight,
            "qzeros": self.qzeros,
            "scales": self.scales,
            "g_idx": self.g_idx
        }

        self.q_handle = ext_make_q_matrix(
            self.q_tensors,
            DEVICE_TENSOR.get_scratch_slice(temp_dq_size(self.height * self.width)),  # temp_dq
        )

    def forward(self, x, force_cuda=False):
        out = ext_gemm_half_q_half(x, self.q_handle, self.width, force_cuda)

        if self.bias is not None:
            out.add_(self.bias)
        return out

