from typing import Any

import torch
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from text_generation_server.inference_engine.hf_transformers import (
    InferenceEngine as HFTransformersInferenceEngine,
)


class InferenceEngine(HFTransformersInferenceEngine):
    def __init__(
        self,
        model_path: str,
        model_class: type[_BaseAutoModelClass],
        dtype: torch.dtype,
        quantize: str | None,
        model_config: Any | None,
        max_sequence_length: int | None,
    ) -> None:
        super().__init__(
            model_path,
            model_class,
            dtype,
            quantize,
            model_config,
            max_sequence_length,
            _use_accelerate=True,
        )
