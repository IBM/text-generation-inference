import os
from abc import ABC
from typing import Tuple, Union, Any, Optional, List

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from text_generation_server.utils.hub import TRUST_REMOTE_CODE


class BaseInferenceEngine(ABC):
    def __init__(self, model_path: str, model_config: Optional[Any]) -> None:
        self._config = AutoConfig.from_pretrained(model_path, trust_remote_code=TRUST_REMOTE_CODE) \
            if model_config is None else model_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left", truncation_side="left", trust_remote_code=TRUST_REMOTE_CODE
        )
        self.model = None

        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            assert (
                self.world_size <= gpu_count,
                f"{self.world_size} shards configured but only {gpu_count} GPUs detected"
            )
            device_index = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(device_index)
            self.device = torch.device("cuda", device_index)
        else:
            self.device = torch.device("cpu")

    def get_components(self) -> Tuple[AutoConfig, AutoTokenizer, Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]]:
        return self.model.config, self.tokenizer, self.model

    def get_device(self) -> torch.device:
        return self.device
