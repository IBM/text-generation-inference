import inspect
import os
import types

import torch

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, TypeVar, Type

from transformers import PreTrainedModel

from text_generation_server.models.types import Batch, TokenInfo, InputTokens, GenerateError
from text_generation_server.inference_engine.engine import BaseInferenceEngine
from text_generation_server.pb import generate_pb2
from text_generation_server.prompt_cache import PrefixCache
from text_generation_server.utils.dist import print_rank_n

B = TypeVar("B", bound=Batch)

# TODO make configurable, possibly based on configured max seq length
MAX_PROMPT_PREFIX_LENGTH = 256

CUDA_PAD_TO_MULT_OF_8 = os.getenv("CUDA_PAD_TO_MULT_OF_8", "true").lower() != "false"
PT2_COMPILE = os.getenv("PT2_COMPILE", "false").lower() != "false"

if PT2_COMPILE:
    import torch._dynamo
    from torch._inductor.compile_fx import compile_fx


class Model(ABC):
    def __init__(self, engine: BaseInferenceEngine, dtype: torch.dtype):
        self.engine = engine
        self.config, self.tokenizer, self.model = engine.get_components()
        self.device = engine.get_device()

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
            self.prefix_cache = PrefixCache(
                device=self.device,
                dtype=dtype,
                max_length=MAX_PROMPT_PREFIX_LENGTH,
                encoder_decoder=self.model.config.is_encoder_decoder,
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
            torch._dynamo.config.cache_size_limit = 512
            self.n_kernels = 0

            def count_kernels(guard):
                print("[pt2_compile] guard failed: ", guard)
                self.n_kernels += 1

            compiled_forward = torch._dynamo.optimize(
                lambda model, inputs: compile_fx(
                    model,
                    inputs,
                    config_patches={
                        "triton.cudagraphs": False,
                        "size_asserts": False,
                    },
                ),
                dynamic=True,
                guard_fail_fn=count_kernels,
            )(self.model.forward)

            run_forward = torch._dynamo.run(compiled_forward)

            def parse_kwargs(kwargs):
                if "past_key_values" in kwargs and type(kwargs["past_key_values"]) is list:
                    kwargs["past_key_values"] = tuple(tuple(t) for t in kwargs["past_key_values"])
                return kwargs

            def override_forward_with_compile(self, *args, **kwargs):
                kwargs = parse_kwargs(kwargs)
                return compiled_forward(*args, **kwargs)

            def override_forward_with_run(self, *args, **kwargs):
                kwargs = parse_kwargs(kwargs)
                return run_forward(*args, **kwargs)

            self.compiled = True
            self.model.forward = types.MethodType(override_forward_with_compile, self.model)
            self.model.run_forward = types.MethodType(override_forward_with_run, self.model)

            # pt2 compile still seems to have some issue with torch.inference_mode
            self.context_manager = torch.no_grad

    def freeze_compile(self):
        self.model.forward = self.model.run_forward

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    @abstractmethod
    def generate_token(
        self, batch: B, first: bool = False, for_concat: bool = False,
    ) -> Tuple[List[TokenInfo], Optional[List[InputTokens]], List[GenerateError]]:
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
        if hasattr(self.model, "named_children"):
            # Logic derived from https://github.com/huggingface/peft/blob/75925b1aaee47fe483a3fd0322d86df3d3eb8d22/src/peft/peft_model.py#L185
            for name, module in self.model.named_children():
                if isinstance(module, PreTrainedModel):
                    for named_param, value in list(module.named_parameters()):
                        if value.shape[0] == self.model.config.vocab_size:
                            self.word_embeddings = module.get_submodule(named_param.replace(".weight", ""))
                            return True

        # Prompt-tuned prefixes not currently supported for ONNX or sharded vocab cases
        print_rank_n("WARN: Could not find input embedding layer for model - prompt prefixes disabled")
        self.word_embeddings = None
        return False
