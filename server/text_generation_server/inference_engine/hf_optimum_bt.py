import os
import torch
from text_generation_server.inference_engine.engine import BaseInferenceEngine
from text_generation_server.utils.hub import TRUST_REMOTE_CODE
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from typing import Union, Any, Optional

from optimum.bettertransformer import BetterTransformer


class InferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        model_path: str,
        model_class: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM],
        dtype: torch.dtype,
        quantize: Optional[str],
        model_config: Optional[Any],
        max_sequence_length: Optional[int],
    ) -> None:
        super().__init__(model_path, model_config)

        kwargs = {
            "pretrained_model_name_or_path": model_path,
            "local_files_only": True,
            "trust_remote_code": TRUST_REMOTE_CODE,
        }

        if self.device.type == "cuda":
            kwargs["device_map"] = "balanced_low_0" if self.world_size > 1 else "auto"

        # if dtype == torch.int8:
        #     # using LLM.int8()
        #     kwargs["load_in_8bit"] = True
        # else:
        #     kwargs["torch_dtype"] = dtype

        slow_but_exact = os.getenv("BLOOM_SLOW_BUT_EXACT", "false").lower() == "true"
        if slow_but_exact:
            kwargs["slow_but_exact"] = True

        self.model = model_class.from_pretrained(**kwargs).requires_grad_(False).eval()

        # Convert using BetterTransformer
        self.model = BetterTransformer.transform(self.model)
