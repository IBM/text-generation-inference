import os
import torch
from loguru import logger
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from text_generation_server.inference_engine.engine import BaseInferenceEngine
from text_generation_server.utils.hub import TRUST_REMOTE_CODE
from typing import Any, Optional


class InferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        model_path: str,
        model_class: type[_BaseAutoModelClass],
        dtype: torch.dtype,
        quantize: Optional[str],
        model_config: Optional[Any],
        max_sequence_length: Optional[int] = None,
    ) -> None:
        super().__init__(model_path, model_config)

        kwargs = {
            "pretrained_model_name_or_path": model_path,
            "local_files_only": True,
            "trust_remote_code": TRUST_REMOTE_CODE,
        }

        # TODO: consider if Flash Attention should be enabled based on FLASH_ATTENTION=True
        if attn_impl := os.getenv("TRANSFORMERS_ATTN_IMPL"):
            kwargs["attn_implementation"] = attn_impl

        if model_config.model_type == "mpt":
            model_config.init_device = str(self.device)
            kwargs["config"] = model_config

        if quantize is None and hasattr(model_config, "quantization_config"):
            quantize = model_config.quantization_config.get("quant_method")

        if quantize == "bitsandbytes":
            # using LLM.int8()
            kwargs["load_in_8bit"] = True

        elif quantize == "gptq" and model_config.quantization_config.get("bits", 4) == 4:
            from transformers import GPTQConfig

            logger.info("Using AutoGPTQ to load 4-bit GPTQ model")
            kwargs["device_map"] = "auto"
            quantization_config = GPTQConfig(bits=4, max_input_length=max_sequence_length)
            disable_exllama = os.getenv("DISABLE_EXLLAMA", "False").lower() == "true"
            if disable_exllama:
                logger.info("Exllama kernels disabled")
                quantization_config.use_exllama = False
            else:
                exllama_version = int(os.getenv("EXLLAMA_VERSION", "2"))  # Use v2 as default
                logger.info(f"Using exllama version {exllama_version}")
                quantization_config.exllama_config = {"version": exllama_version}
            kwargs["quantization_config"] = quantization_config

        elif quantize is not None:
            raise ValueError(f"{quantize} quantization not supported by hf_transformers engine")
        else:
            kwargs["torch_dtype"] = dtype

        slow_but_exact = os.getenv('BLOOM_SLOW_BUT_EXACT', 'false').lower() == 'true'
        if slow_but_exact:
            kwargs["slow_but_exact"] = True

        with self.device:
            self.model = model_class.from_pretrained(**kwargs).requires_grad_(False).eval()
            # This seems to be necessary even with with self.device
            self.model.to(self.device)
