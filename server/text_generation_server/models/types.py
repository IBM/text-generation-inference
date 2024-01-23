from functools import total_ordering

import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from text_generation_server.pb import generate_pb2
from text_generation_server.prompt_cache import PrefixCache


@dataclass
class GenerateError:
    request_id: int
    message: str

    def to_pb(self) -> generate_pb2.GenerateError:
        return generate_pb2.GenerateError(
            request_id=self.request_id,
            message=self.message,
        )


class Batch(ABC):
    @abstractmethod
    def get_id(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_pb(
            cls,
            pb: generate_pb2.Batch,
            tokenizer: PreTrainedTokenizerBase,
            dtype: torch.dtype,
            device: torch.device,
            embeddings_lookup: Optional,
            prefix_cache: Optional[PrefixCache],
            use_position_ids: bool = False,
    ) -> Tuple["Batch", List[GenerateError]]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concatenate(cls, batches: List["Batch"]) -> "Batch":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def prune(cls, batch: "Batch", completed_ids: List[int]) -> Optional["Batch"]:
        raise NotImplementedError

    def compact(self):
        # Optional method
        pass
