from dataclasses import dataclass
from functools import total_ordering

from text_generation_server.pb import generate_pb2


@dataclass(eq=True)
@total_ordering
class TopToken:
    token_id: int
    logprob: float = 0.0

    def __gt__(self, other):
        # We tiebreak equal logprobs with the _lower_ token_id to align with
        # greedy ordering (torch.argmax)
        return self.logprob > other.logprob or (
            self.logprob == other.logprob and self.token_id < other.token_id
        )

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

    top_tokens: list[TopToken] | None = None

    def to_pb(self) -> generate_pb2.Token:
        return generate_pb2.Token(
            request_id=self.request_id,
            token_id=self.token_id,
            logprob=self.logprob,
            rank=self.rank,
            top_tokens=None
            if self.top_tokens is None
            else [tt.to_pb() for tt in self.top_tokens],
        )


@dataclass
class InputTokens:
    request_id: int
    tokens: list[TokenInfo]

    def to_pb(self) -> generate_pb2.InputTokens:
        return generate_pb2.InputTokens(
            request_id=self.request_id,
            tokens=[token.to_pb() for token in self.tokens],
        )
