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


@dataclass(eq=True)
@total_ordering
class TopToken:
    token_id: int
    logprob: float = 0.0

    def __gt__(self, other):
        # We tiebreak equal logprobs with the _lower_ token_id to align with
        # greedy ordering (torch.argmax)
        return self.logprob > other.logprob \
            or (self.logprob == other.logprob and self.token_id < other.token_id)

    def to_pb(self) -> generate_pb2.TopToken:
        return generate_pb2.TopToken(
            token_id=self.token_id,
            logprob=self.logprob,
        )


@dataclass
class TokenInfo:
    token_id: int
    request_id: int = 0  # This won't be set for input tokens
    logprob: float = 0.0
    rank: int = 0

    top_tokens: Optional[List[TopToken]] = None

    def to_pb(self) -> generate_pb2.Token:
        return generate_pb2.Token(
            request_id=self.request_id,
            token_id=self.token_id,
            logprob=self.logprob,
            rank=self.rank,
            top_tokens=None if self.top_tokens is None
            else [tt.to_pb() for tt in self.top_tokens]
        )


@dataclass
class InputTokens:
    request_id: int
    tokens: List[TokenInfo]

    def to_pb(self) -> generate_pb2.InputTokens:
        return generate_pb2.InputTokens(
            request_id=self.request_id,
            tokens=[token.to_pb() for token in self.tokens],
        )
