import torch

from loguru import logger

if not torch.cuda.is_available():
    raise ImportError("CUDA is not available")

major, minor = torch.cuda.get_device_capability()
is_sm75 = major == 7 and minor == 5
is_sm8x = major == 8 and minor >= 0
is_sm90 = major == 9 and minor == 0

HAS_FLASH_ATTN = False
HAS_FLASH_ATTN_V2 = False
try:
    if not (is_sm8x or is_sm90):
        raise ImportError(
            f"GPU with CUDA capability {major} {minor} is not supported for "
            "Flash Attention V2"
        )

    try:
        import flash_attn_2_cuda
    except ImportError as e1:
        raise ImportError("Flash Attention V2 is not installed") from e1

    HAS_FLASH_ATTN_V2 = True
except ImportError as e2:
    if not (is_sm75 or is_sm8x or is_sm90):
        raise ImportError(
            f"FLASH_ATTENTION by GPU with CUDA capability {major} {minor}"
        ) from e2

    try:
        import flash_attn_cuda
    except ImportError as e:
        raise ImportError("Flash Attention is not installed") from e

    logger.warning(f"Unable to use Flash Attention V2: {e2}")
    HAS_FLASH_ATTN = True


def attention(
    q,
    k,
    v,
    out,
    cu_seqlens,
    max_s,
    softmax_scale,
    cu_seqlens_q=None,
    max_s_q=None,
    causal=True,
):
    if cu_seqlens_q is None:
        cu_seqlens_q = cu_seqlens
        max_s_q = max_s

    if HAS_FLASH_ATTN_V2:
        return flash_attn_2_cuda.varlen_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            cu_seqlens,
            None,
            max_s_q,
            max_s,
            0.0,
            softmax_scale,
            False,
            causal,
            -1,
            -1,
            False,
            None,
        )

    if HAS_FLASH_ATTN:
        # Flash attention v1 requires q, k and v to have the same number of heads
        if k.shape[1] != q.shape[1]:
            # MQA expand
            if k.shape[1] == 1:
                k = k.expand(-1, q.shape[1], -1)
            # Grouped attention reshape
            else:
                original_shape = k.shape
                k = (
                    k.unsqueeze(2)
                    .expand(-1, -1, q.shape[1] // k.shape[1], -1)
                    .reshape(original_shape[0], -1, original_shape[2])
                )
        if v.shape[1] != q.shape[1]:
            # MQA expand
            if v.shape[1] == 1:
                v = v.expand(-1, q.shape[1], -1)
            # Grouped attention reshape
            else:
                original_shape = v.shape
                v = (
                    v.unsqueeze(2)
                    .expand(-1, -1, q.shape[1] // v.shape[1], -1)
                    .reshape(original_shape[0], -1, original_shape[2])
                )

        return flash_attn_cuda.fwd(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            cu_seqlens,
            max_s_q,
            max_s,
            0.0,
            softmax_scale,
            False,
            causal,
            False,
            0,
            None,
        )

    raise NotImplementedError("flash attention is not installed")
