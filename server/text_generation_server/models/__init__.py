import os

import torch

from text_generation_server.models.model import Model
from transformers.models.auto import modeling_auto

from text_generation_server.models.causal_lm import CausalLM
from text_generation_server.models.seq2seq_lm import Seq2SeqLM
from text_generation_server.utils.dist import get_torch_dtype, print_rank_n
from text_generation_server.utils.hub import get_model_path, TRUST_REMOTE_CODE
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM

FLASH_ATTENTION = os.getenv("FLASH_ATTENTION", "false").lower() == "true"

__all__ = ["Model", "CausalLM", "Seq2SeqLM", "get_model"]

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Disable gradients
torch.set_grad_enabled(False)


def get_model(model_name: str, revision: str, deployment_framework: str, dtype_str: str) -> Model:
    dtype = get_torch_dtype(dtype_str)
    model_path = get_model_path(model_name, revision)
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=TRUST_REMOTE_CODE)
    model_type = model_config.model_type

    supports_causal_lm = model_type in modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES \
        or type(model_config) in AutoModelForCausalLM._model_mapping \
        or (hasattr(model_config, "auto_map") and "AutoModelForCausalLM" in model_config.auto_map)
    supports_seq2seq_lm = model_type in modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES \
        or type(model_config) in AutoModelForSeq2SeqLM._model_mapping \
        or (hasattr(model_config, "auto_map") and "AutoModelForSeq2SeqLM" in model_config.auto_map)

    print_rank_n(
        f"supports_causal_lm = {supports_causal_lm}, supports_seq2seq_lm = {supports_seq2seq_lm}", rank=0,
    )

    if FLASH_ATTENTION:
        from text_generation_server.models.flash_causal_lm import FlashCausalLM

        if model_type == "gpt_neox":
            from text_generation_server.models.custom_modeling.flash_neox_modeling import FlashGPTNeoXForCausalLM

            model_class = FlashGPTNeoXForCausalLM
        elif model_type in ["RefinedWeb", "RefinedWebModel"]:
            from text_generation_server.models.custom_modeling.flash_rw_modeling import FlashRWForCausalLM, RWConfig

            RWConfig.model_type = model_type
            model_config = RWConfig.from_pretrained(model_path, trust_remote_code=TRUST_REMOTE_CODE)
            model_class = FlashRWForCausalLM
        elif model_type == "gpt_bigcode" or model_name.startswith("bigcode/"):
            from text_generation_server.models.flash_santacoder import FlashSantacoder
            return FlashSantacoder(model_name, revision, dtype, model_config)
        elif model_type == "llama":
            from text_generation_server.models.flash_llama import FlashLlama, FlashLlamaForCausalLM
            # Hack for now
            if deployment_framework != "hf_custom_tp":
                return FlashLlama(model_name, revision, dtype, model_config)
            model_class = FlashLlamaForCausalLM
        else:
            raise NotImplementedError(f"FlashAttention not supported for model type {model_type}")

        return FlashCausalLM(model_name, revision, deployment_framework, dtype, model_config, model_class)

    # For now special-casing bart, will improve this soon
    if supports_seq2seq_lm and model_type == "bart":
        supports_causal_lm = False

    if supports_causal_lm:
        return CausalLM(model_name, revision, deployment_framework, dtype, model_config)

    if supports_seq2seq_lm:
        return Seq2SeqLM(model_name, revision, deployment_framework, dtype, model_config)

    raise NotImplementedError(f"Unsupported model type {model_type}")
