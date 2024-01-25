import os
from typing import Optional

import torch

from text_generation_server.models.model import Model, PT2_COMPILE
from transformers.models.auto import modeling_auto

from text_generation_server.models.causal_lm import CausalLM
from text_generation_server.models.seq2seq_lm import Seq2SeqLM
from text_generation_server.utils.dist import get_torch_dtype, print_rank_n
from text_generation_server.utils.hub import get_model_path, TRUST_REMOTE_CODE
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM

FLASH_ATTENTION = os.getenv("FLASH_ATTENTION", "false").lower() == "true"

__all__ = ["Model", "CausalLM", "Seq2SeqLM", "get_model", "FLASH_ATTENTION", "PT2_COMPILE"]

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Disable gradients
torch.set_grad_enabled(False)


def get_model(
    model_name: str,
    revision: str,
    deployment_framework: str,
    dtype_str: str,
    quantize: Optional[str],
    max_sequence_length: Optional[int],
) -> Model:
    dtype = get_torch_dtype(dtype_str)
    model_path = get_model_path(model_name, revision)
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=TRUST_REMOTE_CODE)
    model_type = model_config.model_type

    if FLASH_ATTENTION:
        # This will raise an exception if flash attention is not supported by the device
        import text_generation_server.utils.flash_attn as flash_attn
        print(f"Using Flash Attention V2: {flash_attn.HAS_FLASH_ATTN_V2}")

        if deployment_framework != "hf_custom_tp":
            print_rank_n(
                f"WARNING: Using deployment engine hf_custom_tp rather than {deployment_framework} "
                "because FLASH_ATTENTION is enabled"
            )
            deployment_framework = "hf_custom_tp"

        if model_type in ["RefinedWeb", "RefinedWebModel", "falcon"]:
            # Custom config type for RW models
            from text_generation_server.models.custom_modeling.flash_rw_modeling import RWConfig
            RWConfig.model_type = model_type
            model_config = RWConfig.from_pretrained(model_path)

        elif model_type == "llama":
            # Custom config type for LLaMA models
            from text_generation_server.models.custom_modeling.flash_llama_modeling import LlamaConfig
            model_config = LlamaConfig.from_pretrained(model_path)

        from text_generation_server.models.flash_causal_lm import FlashCausalLM
        return FlashCausalLM(
            model_name,
            revision,
            deployment_framework,
            dtype, quantize,
            model_config,
            max_sequence_length=max_sequence_length,
        )

    elif deployment_framework == "hf_transformers" and int(os.getenv("WORLD_SIZE", "1")) > 1:
        print_rank_n(
            f"WARNING: Using deployment engine hf_custom_tp rather than {deployment_framework} "
            "because more than one shard is configured"
        )
        deployment_framework = "hf_custom_tp"

    supports_causal_lm = model_type in modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES \
        or type(model_config) in AutoModelForCausalLM._model_mapping \
        or (hasattr(model_config, "auto_map") and "AutoModelForCausalLM" in model_config.auto_map)
    supports_seq2seq_lm = model_type in modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES \
        or type(model_config) in AutoModelForSeq2SeqLM._model_mapping \
        or (hasattr(model_config, "auto_map") and "AutoModelForSeq2SeqLM" in model_config.auto_map)

    print_rank_n(
        f"supports_causal_lm = {supports_causal_lm}, supports_seq2seq_lm = {supports_seq2seq_lm}", rank=0,
    )

    # For now special-casing bart, will improve this soon
    if supports_seq2seq_lm and model_type == "bart":
        supports_causal_lm = False

    if deployment_framework != "hf_custom_tp" and (model_type == "bloom" or model_type == "t5"):
        print_rank_n(
            "WARNING: It's recommended to use the hf_custom_tp engine with safetensors weights for T5 and BLOOM models"
        )

    if supports_causal_lm:
        return CausalLM(model_name, revision, deployment_framework, dtype, quantize, model_config, max_sequence_length)

    if supports_seq2seq_lm:
        return Seq2SeqLM(model_name, revision, deployment_framework, dtype, quantize, model_config, max_sequence_length)

    raise NotImplementedError(f"Unsupported model type {model_type}")
