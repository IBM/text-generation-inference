from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DecodingMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    GREEDY: _ClassVar[DecodingMethod]
    SAMPLE: _ClassVar[DecodingMethod]

class StopReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    NOT_FINISHED: _ClassVar[StopReason]
    MAX_TOKENS: _ClassVar[StopReason]
    EOS_TOKEN: _ClassVar[StopReason]
    CANCELLED: _ClassVar[StopReason]
    TIME_LIMIT: _ClassVar[StopReason]
    STOP_SEQUENCE: _ClassVar[StopReason]
    TOKEN_LIMIT: _ClassVar[StopReason]
    ERROR: _ClassVar[StopReason]
GREEDY: DecodingMethod
SAMPLE: DecodingMethod
NOT_FINISHED: StopReason
MAX_TOKENS: StopReason
EOS_TOKEN: StopReason
CANCELLED: StopReason
TIME_LIMIT: StopReason
STOP_SEQUENCE: StopReason
TOKEN_LIMIT: StopReason
ERROR: StopReason

class BatchedGenerationRequest(_message.Message):
    __slots__ = ["model_id", "prefix_id", "requests", "params"]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    PREFIX_ID_FIELD_NUMBER: _ClassVar[int]
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    prefix_id: str
    requests: _containers.RepeatedCompositeFieldContainer[GenerationRequest]
    params: Parameters
    def __init__(self, model_id: _Optional[str] = ..., prefix_id: _Optional[str] = ..., requests: _Optional[_Iterable[_Union[GenerationRequest, _Mapping]]] = ..., params: _Optional[_Union[Parameters, _Mapping]] = ...) -> None: ...

class SingleGenerationRequest(_message.Message):
    __slots__ = ["model_id", "prefix_id", "request", "params"]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    PREFIX_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    prefix_id: str
    request: GenerationRequest
    params: Parameters
    def __init__(self, model_id: _Optional[str] = ..., prefix_id: _Optional[str] = ..., request: _Optional[_Union[GenerationRequest, _Mapping]] = ..., params: _Optional[_Union[Parameters, _Mapping]] = ...) -> None: ...

class BatchedGenerationResponse(_message.Message):
    __slots__ = ["responses"]
    RESPONSES_FIELD_NUMBER: _ClassVar[int]
    responses: _containers.RepeatedCompositeFieldContainer[GenerationResponse]
    def __init__(self, responses: _Optional[_Iterable[_Union[GenerationResponse, _Mapping]]] = ...) -> None: ...

class GenerationRequest(_message.Message):
    __slots__ = ["text"]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    def __init__(self, text: _Optional[str] = ...) -> None: ...

class GenerationResponse(_message.Message):
    __slots__ = ["input_token_count", "generated_token_count", "text", "stop_reason", "stop_sequence", "seed", "tokens", "input_tokens"]
    INPUT_TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    GENERATED_TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    STOP_REASON_FIELD_NUMBER: _ClassVar[int]
    STOP_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    input_token_count: int
    generated_token_count: int
    text: str
    stop_reason: StopReason
    stop_sequence: str
    seed: int
    tokens: _containers.RepeatedCompositeFieldContainer[TokenInfo]
    input_tokens: _containers.RepeatedCompositeFieldContainer[TokenInfo]
    def __init__(self, input_token_count: _Optional[int] = ..., generated_token_count: _Optional[int] = ..., text: _Optional[str] = ..., stop_reason: _Optional[_Union[StopReason, str]] = ..., stop_sequence: _Optional[str] = ..., seed: _Optional[int] = ..., tokens: _Optional[_Iterable[_Union[TokenInfo, _Mapping]]] = ..., input_tokens: _Optional[_Iterable[_Union[TokenInfo, _Mapping]]] = ...) -> None: ...

class Parameters(_message.Message):
    __slots__ = ["method", "sampling", "stopping", "response", "decoding", "truncate_input_tokens"]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_FIELD_NUMBER: _ClassVar[int]
    STOPPING_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    DECODING_FIELD_NUMBER: _ClassVar[int]
    TRUNCATE_INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    method: DecodingMethod
    sampling: SamplingParameters
    stopping: StoppingCriteria
    response: ResponseOptions
    decoding: DecodingParameters
    truncate_input_tokens: int
    def __init__(self, method: _Optional[_Union[DecodingMethod, str]] = ..., sampling: _Optional[_Union[SamplingParameters, _Mapping]] = ..., stopping: _Optional[_Union[StoppingCriteria, _Mapping]] = ..., response: _Optional[_Union[ResponseOptions, _Mapping]] = ..., decoding: _Optional[_Union[DecodingParameters, _Mapping]] = ..., truncate_input_tokens: _Optional[int] = ...) -> None: ...

class DecodingParameters(_message.Message):
    __slots__ = ["repetition_penalty", "length_penalty"]
    class LengthPenalty(_message.Message):
        __slots__ = ["start_index", "decay_factor"]
        START_INDEX_FIELD_NUMBER: _ClassVar[int]
        DECAY_FACTOR_FIELD_NUMBER: _ClassVar[int]
        start_index: int
        decay_factor: float
        def __init__(self, start_index: _Optional[int] = ..., decay_factor: _Optional[float] = ...) -> None: ...
    REPETITION_PENALTY_FIELD_NUMBER: _ClassVar[int]
    LENGTH_PENALTY_FIELD_NUMBER: _ClassVar[int]
    repetition_penalty: float
    length_penalty: DecodingParameters.LengthPenalty
    def __init__(self, repetition_penalty: _Optional[float] = ..., length_penalty: _Optional[_Union[DecodingParameters.LengthPenalty, _Mapping]] = ...) -> None: ...

class SamplingParameters(_message.Message):
    __slots__ = ["temperature", "top_k", "top_p", "typical_p", "seed"]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TYPICAL_P_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    temperature: float
    top_k: int
    top_p: float
    typical_p: float
    seed: int
    def __init__(self, temperature: _Optional[float] = ..., top_k: _Optional[int] = ..., top_p: _Optional[float] = ..., typical_p: _Optional[float] = ..., seed: _Optional[int] = ...) -> None: ...

class StoppingCriteria(_message.Message):
    __slots__ = ["max_new_tokens", "min_new_tokens", "time_limit_millis", "stop_sequences", "include_stop_sequence"]
    MAX_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MIN_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TIME_LIMIT_MILLIS_FIELD_NUMBER: _ClassVar[int]
    STOP_SEQUENCES_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_STOP_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    max_new_tokens: int
    min_new_tokens: int
    time_limit_millis: int
    stop_sequences: _containers.RepeatedScalarFieldContainer[str]
    include_stop_sequence: bool
    def __init__(self, max_new_tokens: _Optional[int] = ..., min_new_tokens: _Optional[int] = ..., time_limit_millis: _Optional[int] = ..., stop_sequences: _Optional[_Iterable[str]] = ..., include_stop_sequence: bool = ...) -> None: ...

class ResponseOptions(_message.Message):
    __slots__ = ["input_text", "generated_tokens", "input_tokens", "token_logprobs", "token_ranks", "top_n_tokens"]
    INPUT_TEXT_FIELD_NUMBER: _ClassVar[int]
    GENERATED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_RANKS_FIELD_NUMBER: _ClassVar[int]
    TOP_N_TOKENS_FIELD_NUMBER: _ClassVar[int]
    input_text: bool
    generated_tokens: bool
    input_tokens: bool
    token_logprobs: bool
    token_ranks: bool
    top_n_tokens: int
    def __init__(self, input_text: bool = ..., generated_tokens: bool = ..., input_tokens: bool = ..., token_logprobs: bool = ..., token_ranks: bool = ..., top_n_tokens: _Optional[int] = ...) -> None: ...

class TokenInfo(_message.Message):
    __slots__ = ["text", "logprob", "rank", "top_tokens"]
    class TopToken(_message.Message):
        __slots__ = ["text", "logprob"]
        TEXT_FIELD_NUMBER: _ClassVar[int]
        LOGPROB_FIELD_NUMBER: _ClassVar[int]
        text: str
        logprob: float
        def __init__(self, text: _Optional[str] = ..., logprob: _Optional[float] = ...) -> None: ...
    TEXT_FIELD_NUMBER: _ClassVar[int]
    LOGPROB_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    TOP_TOKENS_FIELD_NUMBER: _ClassVar[int]
    text: str
    logprob: float
    rank: int
    top_tokens: _containers.RepeatedCompositeFieldContainer[TokenInfo.TopToken]
    def __init__(self, text: _Optional[str] = ..., logprob: _Optional[float] = ..., rank: _Optional[int] = ..., top_tokens: _Optional[_Iterable[_Union[TokenInfo.TopToken, _Mapping]]] = ...) -> None: ...

class BatchedTokenizeRequest(_message.Message):
    __slots__ = ["model_id", "requests", "return_tokens"]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    RETURN_TOKENS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    requests: _containers.RepeatedCompositeFieldContainer[TokenizeRequest]
    return_tokens: bool
    def __init__(self, model_id: _Optional[str] = ..., requests: _Optional[_Iterable[_Union[TokenizeRequest, _Mapping]]] = ..., return_tokens: bool = ...) -> None: ...

class BatchedTokenizeResponse(_message.Message):
    __slots__ = ["responses"]
    RESPONSES_FIELD_NUMBER: _ClassVar[int]
    responses: _containers.RepeatedCompositeFieldContainer[TokenizeResponse]
    def __init__(self, responses: _Optional[_Iterable[_Union[TokenizeResponse, _Mapping]]] = ...) -> None: ...

class TokenizeRequest(_message.Message):
    __slots__ = ["text"]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    def __init__(self, text: _Optional[str] = ...) -> None: ...

class TokenizeResponse(_message.Message):
    __slots__ = ["token_count", "tokens"]
    TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    token_count: int
    tokens: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, token_count: _Optional[int] = ..., tokens: _Optional[_Iterable[str]] = ...) -> None: ...

class ModelInfoRequest(_message.Message):
    __slots__ = ["model_id"]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class ModelInfoResponse(_message.Message):
    __slots__ = ["model_kind", "max_sequence_length", "max_new_tokens"]
    class ModelKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        DECODER_ONLY: _ClassVar[ModelInfoResponse.ModelKind]
        ENCODER_DECODER: _ClassVar[ModelInfoResponse.ModelKind]
    DECODER_ONLY: ModelInfoResponse.ModelKind
    ENCODER_DECODER: ModelInfoResponse.ModelKind
    MODEL_KIND_FIELD_NUMBER: _ClassVar[int]
    MAX_SEQUENCE_LENGTH_FIELD_NUMBER: _ClassVar[int]
    MAX_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    model_kind: ModelInfoResponse.ModelKind
    max_sequence_length: int
    max_new_tokens: int
    def __init__(self, model_kind: _Optional[_Union[ModelInfoResponse.ModelKind, str]] = ..., max_sequence_length: _Optional[int] = ..., max_new_tokens: _Optional[int] = ...) -> None: ...
