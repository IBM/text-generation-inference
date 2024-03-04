import os
from typing import Any

import torch
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from text_generation_server.inference_engine.engine import BaseInferenceEngine
from text_generation_server.utils.hub import TRUST_REMOTE_CODE


class InferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        model_path: str,
        model_class: AutoModelForCausalLM | AutoModelForSeq2SeqLM,
        dtype: torch.dtype,
        quantize: str | None,
        model_config: Any | None,
        max_sequence_length: int | None,
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
