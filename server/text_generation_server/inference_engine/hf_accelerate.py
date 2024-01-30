import os
import torch
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
        max_sequence_length: Optional[int],
    ) -> None:
        super().__init__(model_path, model_config)

        kwargs = {
            "pretrained_model_name_or_path": model_path,
            "device_map": None,
            "local_files_only": True,
            "trust_remote_code": TRUST_REMOTE_CODE,
        }

        if self.device.type == "cuda":
            kwargs["device_map"] = "balanced_low_0" if self.world_size > 1 else "auto"

        if quantize == "bitsandbytes":
            # using LLM.int8()
            kwargs["load_in_8bit"] = True
        elif quantize is not None:
            raise ValueError(f"{quantize} quantization not supported by hf_accelerate engine")
        else:
            kwargs["torch_dtype"] = dtype

        slow_but_exact = os.getenv('BLOOM_SLOW_BUT_EXACT', 'false').lower() == 'true'
        if slow_but_exact:
            kwargs["slow_but_exact"] = True

        self.model = model_class.from_pretrained(**kwargs).requires_grad_(False).eval()
