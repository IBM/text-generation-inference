import inspect
import os
import types

import torch

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, TypeVar, Type

from transformers import PreTrainedModel

from text_generation_server.models.types import Batch, GenerateError
from text_generation_server.inference_engine.engine import BaseInferenceEngine
from text_generation_server.pb import generate_pb2
from text_generation_server.prompt_cache import PrefixCache
from text_generation_server.utils.dist import print_rank_n
from text_generation_server.utils.layers import TensorParallelEmbedding
from text_generation_server.utils.token_types import TokenInfo, InputTokens

B = TypeVar("B", bound=Batch)

# TODO make configurable, possibly based on configured max seq length
MAX_PROMPT_PREFIX_LENGTH = 256

CUDA_PAD_TO_MULT_OF_8 = os.getenv("CUDA_PAD_TO_MULT_OF_8", "true").lower() != "false"
PT2_COMPILE = os.getenv("PT2_COMPILE", "false").lower() != "false"

if PT2_COMPILE:
    import torch._dynamo
    from torch._inductor.compile_fx import compile_fx
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()


class Model(ABC):
    def __init__(self, engine: BaseInferenceEngine, dtype: torch.dtype):
        self.engine = engine
        self.config, self.tokenizer, self.model = engine.get_components()
        self.device = engine.get_device()
        self.dtype = dtype

        # The tokenizer config doesn't always contain the eos token id
        if self.config.eos_token_id is not None:
            self.tokenizer.model_eos_token_id = self.config.eos_token_id

        # Check whether model supports position_ids
        self.use_position_ids = "position_ids" in inspect.signature(self.model.forward).parameters

        prompt_prefix_supported = self._setup_prompt_encoder()

        if prompt_prefix_supported:
            # Set up prefix cache
            decoder_start_token_id = self.model.config.decoder_start_token_id
            if decoder_start_token_id is None:
                decoder_start_token_id = self.tokenizer.bos_token_id

            return_zero = False
            # If the word_embeddings layer is configured not to reduce at the end of the forward() call
            # each shard will have only a partial tensor. This tensor cannot be concatenated with a
            # prefix tensor in each shard because the reduce that happens afterwards would result
            # in adding the prefix N times, where N is the world size.
            if isinstance(self.word_embeddings, TensorParallelEmbedding) and not self.word_embeddings.reduce:
                return_zero = self.word_embeddings.process_group.rank() != 0

            self.prefix_cache = PrefixCache(
                device=self.device,
                dtype=dtype,
                max_length=MAX_PROMPT_PREFIX_LENGTH,
                encoder_decoder=self.model.config.is_encoder_decoder,
                return_zero=return_zero,
                decoder_start_tok_embedding=self.word_embeddings(
                    torch.tensor([decoder_start_token_id], device=self.device, dtype=torch.long)
                ) if decoder_start_token_id is not None else None,
            )
        else:
            self.prefix_cache = None

        # For some reason, inference_mode does not work well with GLOO which we use on CPU
        self.context_manager = (
            torch.no_grad if self.device.type == "cpu" and engine.world_size > 1
            else torch.inference_mode
        )

        if not PT2_COMPILE:
            self.compiled = False
        else:

            # Perform a forward pass using a single token. This serves 2 purposes:
            # (1) work-around for PT2C issue #107721
            # (2) determine types of past_key_value output
            type_pkv_dim0, type_pkv_dim1 = self.determine_pkv_types()

            torch._dynamo.config.cache_size_limit = 512
            self.n_kernels = 0

            def count_kernels(guard):
                print("[pt2_compile] guard failed: ", guard)
                self.n_kernels += 1

            compiled_forward = torch._dynamo.optimize(
                compile_fx,
                dynamic=True,
                guard_fail_fn=count_kernels,
            )(self.model.forward)

            run_forward = torch._dynamo.run(compiled_forward)

            def parse_kwargs(kwargs):
                # after batch concatentation the past_key_value tensor is a list of lists.
                # this will lead to guard failures unless we convert them to the typical
                # types that we expect to be returned by forward.
                pkv = kwargs.get("past_key_values")
                if pkv is not None:
                    if type(pkv) != type_pkv_dim0 or type(pkv[0]) != type_pkv_dim1:
                        kwargs["past_key_values"] = type_pkv_dim0(type_pkv_dim1(t) for t in pkv)

                    for t in pkv:
                        for x in t:
                            strides = list(x.stride())
                            if strides != sorted(strides, reverse=True):
                                x.data = x.data.clone(memory_format=torch.contiguous_format)

                return kwargs

            def override_forward_with_compile(self, *args, **kwargs):
                kwargs = parse_kwargs(kwargs)
                return compiled_forward(*args, **kwargs)

            def override_forward_with_run(self, *args, **kwargs):
                kwargs = parse_kwargs(kwargs)
                return run_forward(*args, **kwargs)

            self.model.compile_forward = types.MethodType(override_forward_with_compile, self.model)
            self.model.run_forward = types.MethodType(override_forward_with_run, self.model)

            # pt2 compile still seems to have some issue with torch.inference_mode
            self.context_manager = torch.no_grad

    def stop_compile(self):
        self.model.forward = self.model.run_forward

    def start_compile(self):
        self.model.forward = self.model.compile_forward

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    @abstractmethod
    def generate_token(
        self, batch: B, first: bool = False, for_concat: bool = False,
    ) -> Tuple[List[TokenInfo], Optional[List[InputTokens]], List[GenerateError], int]:
        raise NotImplementedError

    @staticmethod
    def get_indices_to_keep(
        requests: List[generate_pb2.Request], completed_ids: List[int],
    ) -> List[int]:
        # Compile list of indices to retain
        next_batch_keep_indices = []
        completed = iter(completed_ids)
        next_id = next(completed)
        for i, r in enumerate(requests):
            while next_id is not None and r.id > next_id:
                next_id = next(completed, None)
            if r.id != next_id:
                next_batch_keep_indices.append(i)
        return next_batch_keep_indices

    def _setup_prompt_encoder(self) -> bool:
        try:
            self.word_embeddings = self.model.get_input_embeddings()
            return True
        except:
            pass

        vocab_size = getattr(self.model.config, "vocab_size", None)
        hidden_size = getattr(self.config, "hidden_size", None)

        if vocab_size is not None and hidden_size is not None and isinstance(self.model, torch.nn.Module):
            candidates = []

            for _, m in self.model.named_modules():
                if (isinstance(m, torch.nn.Embedding) or isinstance(m, torch.nn.modules.sparse.Embedding)) \
                    and m.weight.shape == (vocab_size, hidden_size):
                    candidates.append(m)
                elif isinstance(m, TensorParallelEmbedding) and m.weight.shape == (vocab_size+1, hidden_size):
                    candidates.append(m)

            if len(candidates) == 1:
                self.word_embeddings = candidates[0]
                return True

        # Prompt-tuned prefixes not currently supported for ONNX or sharded vocab cases
        print_rank_n("WARN: Could not find input embedding layer for model - prompt prefixes disabled")
        self.word_embeddings = None
        return False
