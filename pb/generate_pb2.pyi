from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HealthRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class ServiceDiscoveryRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class ServiceDiscoveryResponse(_message.Message):
    __slots__ = ["urls"]
    URLS_FIELD_NUMBER: _ClassVar[int]
    urls: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, urls: _Optional[_Iterable[str]] = ...) -> None: ...

class ClearCacheRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class ClearCacheResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class ModelInfoRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class ModelInfoResponse(_message.Message):
    __slots__ = ["model_type", "eos_token", "batch_padding"]
    class ModelType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        CAUSAL_LM: _ClassVar[ModelInfoResponse.ModelType]
        SEQ2SEQ_LM: _ClassVar[ModelInfoResponse.ModelType]
    CAUSAL_LM: ModelInfoResponse.ModelType
    SEQ2SEQ_LM: ModelInfoResponse.ModelType
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    EOS_TOKEN_FIELD_NUMBER: _ClassVar[int]
    BATCH_PADDING_FIELD_NUMBER: _ClassVar[int]
    model_type: ModelInfoResponse.ModelType
    eos_token: int
    batch_padding: bool
    def __init__(self, model_type: _Optional[_Union[ModelInfoResponse.ModelType, str]] = ..., eos_token: _Optional[int] = ..., batch_padding: bool = ...) -> None: ...

class NextTokenChooserParameters(_message.Message):
    __slots__ = ["temperature", "top_k", "top_p", "typical_p", "min_new_tokens", "seed", "repetition_penalty", "length_penalty"]
    class LengthPenalty(_message.Message):
        __slots__ = ["start_index", "decay_factor"]
        START_INDEX_FIELD_NUMBER: _ClassVar[int]
        DECAY_FACTOR_FIELD_NUMBER: _ClassVar[int]
        start_index: int
        decay_factor: float
        def __init__(self, start_index: _Optional[int] = ..., decay_factor: _Optional[float] = ...) -> None: ...
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TYPICAL_P_FIELD_NUMBER: _ClassVar[int]
    MIN_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    REPETITION_PENALTY_FIELD_NUMBER: _ClassVar[int]
    LENGTH_PENALTY_FIELD_NUMBER: _ClassVar[int]
    temperature: float
    top_k: int
    top_p: float
    typical_p: float
    min_new_tokens: int
    seed: int
    repetition_penalty: float
    length_penalty: NextTokenChooserParameters.LengthPenalty
    def __init__(self, temperature: _Optional[float] = ..., top_k: _Optional[int] = ..., top_p: _Optional[float] = ..., typical_p: _Optional[float] = ..., min_new_tokens: _Optional[int] = ..., seed: _Optional[int] = ..., repetition_penalty: _Optional[float] = ..., length_penalty: _Optional[_Union[NextTokenChooserParameters.LengthPenalty, _Mapping]] = ...) -> None: ...

class RequestedDetails(_message.Message):
    __slots__ = ["input_toks", "logprobs", "ranks", "top_n_toks"]
    INPUT_TOKS_FIELD_NUMBER: _ClassVar[int]
    LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    RANKS_FIELD_NUMBER: _ClassVar[int]
    TOP_N_TOKS_FIELD_NUMBER: _ClassVar[int]
    input_toks: bool
    logprobs: bool
    ranks: bool
    top_n_toks: int
    def __init__(self, input_toks: bool = ..., logprobs: bool = ..., ranks: bool = ..., top_n_toks: _Optional[int] = ...) -> None: ...

class Request(_message.Message):
    __slots__ = ["id", "prefix_id", "inputs", "input_length", "truncate", "max_output_length", "parameters", "stream_response", "details"]
    ID_FIELD_NUMBER: _ClassVar[int]
    PREFIX_ID_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    INPUT_LENGTH_FIELD_NUMBER: _ClassVar[int]
    TRUNCATE_FIELD_NUMBER: _ClassVar[int]
    MAX_OUTPUT_LENGTH_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    STREAM_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    id: int
    prefix_id: str
    inputs: str
    input_length: int
    truncate: bool
    max_output_length: int
    parameters: NextTokenChooserParameters
    stream_response: bool
    details: RequestedDetails
    def __init__(self, id: _Optional[int] = ..., prefix_id: _Optional[str] = ..., inputs: _Optional[str] = ..., input_length: _Optional[int] = ..., truncate: bool = ..., max_output_length: _Optional[int] = ..., parameters: _Optional[_Union[NextTokenChooserParameters, _Mapping]] = ..., stream_response: bool = ..., details: _Optional[_Union[RequestedDetails, _Mapping]] = ...) -> None: ...

class StopSequence(_message.Message):
    __slots__ = ["tokens"]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    tokens: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, tokens: _Optional[_Iterable[int]] = ...) -> None: ...

class Batch(_message.Message):
    __slots__ = ["id", "requests", "total_tokens"]
    ID_FIELD_NUMBER: _ClassVar[int]
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    id: int
    requests: _containers.RepeatedCompositeFieldContainer[Request]
    total_tokens: int
    def __init__(self, id: _Optional[int] = ..., requests: _Optional[_Iterable[_Union[Request, _Mapping]]] = ..., total_tokens: _Optional[int] = ...) -> None: ...

class TopToken(_message.Message):
    __slots__ = ["token_id", "logprob"]
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    LOGPROB_FIELD_NUMBER: _ClassVar[int]
    token_id: int
    logprob: float
    def __init__(self, token_id: _Optional[int] = ..., logprob: _Optional[float] = ...) -> None: ...

class Token(_message.Message):
    __slots__ = ["request_id", "token_id", "logprob", "rank", "top_tokens"]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    LOGPROB_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    TOP_TOKENS_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    token_id: int
    logprob: float
    rank: int
    top_tokens: _containers.RepeatedCompositeFieldContainer[TopToken]
    def __init__(self, request_id: _Optional[int] = ..., token_id: _Optional[int] = ..., logprob: _Optional[float] = ..., rank: _Optional[int] = ..., top_tokens: _Optional[_Iterable[_Union[TopToken, _Mapping]]] = ...) -> None: ...

class GenerateError(_message.Message):
    __slots__ = ["request_id", "message"]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    message: str
    def __init__(self, request_id: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class InputTokens(_message.Message):
    __slots__ = ["request_id", "tokens"]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    tokens: _containers.RepeatedCompositeFieldContainer[Token]
    def __init__(self, request_id: _Optional[int] = ..., tokens: _Optional[_Iterable[_Union[Token, _Mapping]]] = ...) -> None: ...

class PrefillRequest(_message.Message):
    __slots__ = ["batch", "to_prune"]
    BATCH_FIELD_NUMBER: _ClassVar[int]
    TO_PRUNE_FIELD_NUMBER: _ClassVar[int]
    batch: Batch
    to_prune: _containers.RepeatedCompositeFieldContainer[CachedBatch]
    def __init__(self, batch: _Optional[_Union[Batch, _Mapping]] = ..., to_prune: _Optional[_Iterable[_Union[CachedBatch, _Mapping]]] = ...) -> None: ...

class GenerateResult(_message.Message):
    __slots__ = ["output_tokens", "errors", "batch_id", "forward_time_ns"]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    FORWARD_TIME_NS_FIELD_NUMBER: _ClassVar[int]
    output_tokens: _containers.RepeatedCompositeFieldContainer[Token]
    errors: _containers.RepeatedCompositeFieldContainer[GenerateError]
    batch_id: int
    forward_time_ns: int
    def __init__(self, output_tokens: _Optional[_Iterable[_Union[Token, _Mapping]]] = ..., errors: _Optional[_Iterable[_Union[GenerateError, _Mapping]]] = ..., batch_id: _Optional[int] = ..., forward_time_ns: _Optional[int] = ...) -> None: ...

class PrefillResponse(_message.Message):
    __slots__ = ["result", "input_tokens"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    result: GenerateResult
    input_tokens: _containers.RepeatedCompositeFieldContainer[InputTokens]
    def __init__(self, result: _Optional[_Union[GenerateResult, _Mapping]] = ..., input_tokens: _Optional[_Iterable[_Union[InputTokens, _Mapping]]] = ...) -> None: ...

class RequestsStatus(_message.Message):
    __slots__ = ["completed_ids"]
    COMPLETED_IDS_FIELD_NUMBER: _ClassVar[int]
    completed_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, completed_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class CachedBatch(_message.Message):
    __slots__ = ["batch_id", "status"]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    batch_id: int
    status: RequestsStatus
    def __init__(self, batch_id: _Optional[int] = ..., status: _Optional[_Union[RequestsStatus, _Mapping]] = ...) -> None: ...

class NextTokenRequest(_message.Message):
    __slots__ = ["batches"]
    BATCHES_FIELD_NUMBER: _ClassVar[int]
    batches: _containers.RepeatedCompositeFieldContainer[CachedBatch]
    def __init__(self, batches: _Optional[_Iterable[_Union[CachedBatch, _Mapping]]] = ...) -> None: ...

class NextTokenResponse(_message.Message):
    __slots__ = ["result"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: GenerateResult
    def __init__(self, result: _Optional[_Union[GenerateResult, _Mapping]] = ...) -> None: ...

class PruneBatchRequest(_message.Message):
    __slots__ = ["batch"]
    BATCH_FIELD_NUMBER: _ClassVar[int]
    batch: CachedBatch
    def __init__(self, batch: _Optional[_Union[CachedBatch, _Mapping]] = ...) -> None: ...

class PruneBatchResponse(_message.Message):
    __slots__ = ["batch_id"]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    batch_id: int
    def __init__(self, batch_id: _Optional[int] = ...) -> None: ...

class PrefixLookupRequest(_message.Message):
    __slots__ = ["prefix_id"]
    PREFIX_ID_FIELD_NUMBER: _ClassVar[int]
    prefix_id: str
    def __init__(self, prefix_id: _Optional[str] = ...) -> None: ...

class PrefixLookupResponse(_message.Message):
    __slots__ = ["prefix_length"]
    PREFIX_LENGTH_FIELD_NUMBER: _ClassVar[int]
    prefix_length: int
    def __init__(self, prefix_length: _Optional[int] = ...) -> None: ...
