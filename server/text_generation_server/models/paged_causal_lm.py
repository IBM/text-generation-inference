import logging
import time
import os
from operator import itemgetter

import torch

from dataclasses import dataclass
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Union, Any

from text_generation_server.models.model import Model
from text_generation_server.models.types import Batch, GenerateError
from text_generation_server.pb import generate_pb2
from text_generation_server.prompt_cache import PrefixCache
from text_generation_server.utils import print_rank_n
from text_generation_server.utils.hub import get_model_path
from text_generation_server.utils.token_types import TokenInfo, InputTokens
from text_generation_server.utils.tokens import HeterogeneousNextTokenChooser, get_token_info, get_input_tokens_info
from text_generation_server.utils.paged import (
    prepare_inputs_without_speculation,
    prepare_inputs_with_speculation,
    process_outputs_with_speculation,
    prepare_inputs_for_prefill
)
from text_generation_server.inference_engine import get_inference_engine_class

# HF name or path to speculator model (None means no speculation will be used)
SPECULATOR_NAME = os.getenv("SPECULATOR_NAME", None)

# we will only do speculation if the batch size is <= this parameter
SPECULATOR_MAX_BATCH_SIZE = int(os.getenv("SPECULATOR_MAX_BATCH_SIZE", "16"))

# override number of KV cache manager blocks
KV_CACHE_MANAGER_NUM_GPU_BLOCKS = os.getenv("KV_CACHE_MANAGER_NUM_GPU_BLOCKS", None)


@dataclass
class PagedCausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]

    input_ids: torch.Tensor
    embeds: torch.Tensor
    sequence_ids: Optional[List[int]]

    # maximum of the input tokens across the batch
    max_seqlen: int

    # All tokens
    all_input_ids_tensor: torch.Tensor

    # Lengths of all generations present in the batch
    input_lengths: List[int]

    # Maximum possible number of tokens for each request:
    #   (truncated) original input length + prefix length + max output tokens
    # Used for sizing/preallocating kv cache and all_input_ids_tensor
    total_lengths: List[int]
    pad_token_id: int

    # Generation helpers
    next_token_chooser: HeterogeneousNextTokenChooser

    def get_id(self) -> int:
        return self.batch_id

    def __len__(self):
        return len(self.requests)

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
        embeddings_lookup: Optional,
        prefix_cache: Optional[PrefixCache],
        use_position_ids: bool = True,
    ) -> Tuple[Optional["PagedCausalLMBatch"], List[GenerateError]]:
        """Convert a text_generation.v1.Batch protobuf to a PagedCausalLMBatch"""
        errors = []
        batch_inputs = []
        requests = pb.requests

        input_lengths = []
        total_lengths = []
        max_seqlen = 0
        for r in requests:
            input_length = r.input_length
            batch_inputs.append(r.inputs)
            input_lengths.append(input_length)
            max_seqlen = max(max_seqlen, input_length)
            total_lengths.append(input_length + r.max_output_length)

        # return as lists to avoid unnecessary padding;
        # sequences will be concatenated across the batch
        batch_tokenized_inputs = tokenizer(
            batch_inputs, truncation=True, max_length=max_seqlen, return_token_type_ids=False
        )["input_ids"]

        # Process inputs to generate the needed tensors
        input_ids = []
        next_token_chooser_parameters = []
        return_logprobs = []
        # Allocate maximal all_input_ids_tensor
        all_input_ids_tensor = torch.full(
            (len(batch_inputs), max(total_lengths)),
            tokenizer.pad_token_id,
            dtype=torch.int64, device=device,
        )

        for i, (r, tokenized_input, input_length) in enumerate(zip(requests, batch_tokenized_inputs, input_lengths)):
            if r.truncate:
                tokenized_input = tokenized_input[-r.input_length:]
                # Fill in bos token in truncation case if needed
                if getattr(tokenizer, "add_bos_token", False):
                    tokenized_input[0] = tokenizer.bos_token_id
            tokenized_input = all_input_ids_tensor.new_tensor(tokenized_input)
            # Instead of adding the padding for prefix tokens to
            # tokenized_input, just embed it in all_input_ids_tensor and copy
            # the padding from it when appending to input_ids (if needed)
            all_input_ids_tensor[i, input_length - r.input_length:input_length] = tokenized_input
            input_ids.append(tokenized_input if input_length == r.input_length else all_input_ids_tensor[i, :input_length])
            next_token_chooser_parameters.append(r.parameters)
            return_logprobs.append(r.details.logprobs)

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            pb=next_token_chooser_parameters,
            model_eos_token_id=getattr(tokenizer, 'model_eos_token_id', tokenizer.eos_token_id),
            model_pad_token_id=tokenizer.pad_token_id,
            return_logprobs=return_logprobs,
            dtype=dtype,
            device=device,
        )

        return cls(
            batch_id=pb.id,
            requests=requests,
            input_ids=torch.cat(input_ids),
            max_seqlen=max_seqlen,
            input_lengths=input_lengths,
            total_lengths=total_lengths,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            pad_token_id=tokenizer.pad_token_id,
            embeds=None,
            sequence_ids=None
        ), errors

    @classmethod
    def concatenate(cls, batches: List["PagedCausalLMBatch"]) -> "PagedCausalLMBatch":
        """Concatenate multiple batches together by padding internal torch tensors"""

        # Batch attributes
        requests = []
        input_lengths = []
        total_lengths = []
        next_token_chooser_parameters = []
        ntc_current_tokens = []
        ntc_samplings = []
        ntc_return_logprobs = []
        sequence_ids = []

        #device = batches[0].input_ids.device

        # Batch tensors
        input_ids = []
        embeds = []
        max_seqlen = 0

        # Allocate the all input IDs tensor
        new_batch_size = sum(len(b) for b in batches)
        max_total_length = max(mtl for b in batches for mtl in b.total_lengths)
        all_input_ids_tensor = batches[0].all_input_ids_tensor.new_full(
            (new_batch_size, max_total_length),
            batches[0].pad_token_id,
        )

        start_index = 0
        for i, batch in enumerate(batches):
            requests.extend(batch.requests)
            input_lengths.extend(batch.input_lengths)
            total_lengths.extend(batch.total_lengths)
            sequence_ids.extend(batch.sequence_ids)

            next_token_chooser_parameters.extend(r.parameters for r in batch.requests)
            ntc_current_tokens.extend(batch.next_token_chooser.current_tokens)
            ntc_samplings.extend(batch.next_token_chooser.samplings)
            ntc_return_logprobs.extend(batch.next_token_chooser.return_logprobs)

            input_ids.append(batch.input_ids)
            embeds.append(batch.embeds)

            end_index = start_index + len(batch)
            all_input_ids_tensor[
                start_index:end_index, :batch.all_input_ids_tensor.shape[1]
            ] = batch.all_input_ids_tensor
            start_index = end_index

            max_seqlen = max(max_seqlen, batch.max_seqlen)

        first_next_token_chooser = batches[0].next_token_chooser
        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            pb=next_token_chooser_parameters,
            model_eos_token_id=first_next_token_chooser.eos_token_id,
            model_pad_token_id=first_next_token_chooser.pad_token_id,
            return_logprobs=ntc_return_logprobs,
            dtype=first_next_token_chooser.dtype,
            device=first_next_token_chooser.device,
            samplings=ntc_samplings,
            current_tokens=ntc_current_tokens,
        )

        return PagedCausalLMBatch(
            batch_id=batches[0].batch_id,
            requests=requests,
            input_ids=torch.cat(input_ids, axis=0),
            embeds=torch.cat(embeds, axis=0),
            max_seqlen=max_seqlen,
            input_lengths=input_lengths,
            total_lengths=total_lengths,
            sequence_ids=sequence_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            pad_token_id=batches[0].pad_token_id,
        )

    @classmethod
    def prune(cls, batch: "PagedCausalLMBatch", completed_ids: List[int]) -> Optional["PagedCausalLMBatch"]:
        """Prune completed entries from a batch"""

        if not completed_ids:
            # Nothing to prune
            return batch

        # this call doesn't work -> some assumption seems to break
        # keep_indices = Model.get_indices_to_keep(batch.requests, completed_ids)
        # use this code instead:
        keep_indices = [
            i for i, request in enumerate(batch.requests) if request.id not in completed_ids
        ]

        new_size = len(keep_indices)

        # If the whole batch has finished, discard it
        if new_size == 0:
            return None

        #TODO maybe a single loop for all these list slices
        slice_list = itemgetter(*keep_indices) if new_size > 1 else lambda l: (l[keep_indices[0]],)

        batch.input_lengths = list(slice_list(batch.input_lengths))
        batch.sequence_ids = list(slice_list(batch.sequence_ids))
        batch.total_lengths = slice_list(batch.total_lengths)
        batch.requests = slice_list(batch.requests)
        batch.next_token_chooser = batch.next_token_chooser.filter(keep_indices)

        batch.max_seqlen = max(batch.input_lengths)

        batch.input_ids = batch.input_ids[keep_indices]
        batch.embeds = batch.embeds[keep_indices]

        batch.all_input_ids_tensor = batch.all_input_ids_tensor[keep_indices, :max(batch.total_lengths)]

        return batch


class PagedCausalLM(Model):
    def __init__(
        self,
        model_name: str,
        revision: str,
        deployment_framework: str,
        dtype: torch.dtype,
        quantize: Optional[str],
        model_config: Union[Any] = None,
        max_sequence_length: Optional[int] = None,
    ):
        model_path = get_model_path(model_name, revision)

        inference_engine = get_inference_engine_class(deployment_framework)(
            model_path, AutoModelForCausalLM, dtype, quantize, model_config, max_sequence_length
        )

        super(PagedCausalLM, self).__init__(inference_engine, dtype)

        if self.model.config.pad_token_id is not None:
            self.tokenizer.pad_token_id = self.model.config.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            if self.model.config.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.model.config.eos_token_id
            elif self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.batch_type = PagedCausalLMBatch

        from fms_extras.utils.cache.paged import PagedKVCacheManager

        if SPECULATOR_NAME is not None:
            from fms_extras.models.hf.modeling_mlp_speculator import MLPSpeculatorPreTrainedModel
            speculator_revision = os.getenv("SPECULATOR_REVISION", None)
            speculator_model_path = get_model_path(SPECULATOR_NAME, speculator_revision)
            print_rank_n(f"Loading speculator model from: {speculator_model_path}")
            print_rank_n(f"Speculation will be enabled up to batch size {SPECULATOR_MAX_BATCH_SIZE}")
            kwargs = {
                "pretrained_model_name_or_path": speculator_model_path,
                "local_files_only": True,
                "torch_dtype": dtype,
            }
            with self.device:
                self.speculator = MLPSpeculatorPreTrainedModel.from_pretrained(**kwargs)
                self.speculator.to(device=self.device)
        else:
            self.speculator = None

        if KV_CACHE_MANAGER_NUM_GPU_BLOCKS is not None:
            total_num_gpu_blocks = int(KV_CACHE_MANAGER_NUM_GPU_BLOCKS)
        else:
            total_num_gpu_blocks = None

        self.kv_cache_manager = PagedKVCacheManager(
            model_config.num_hidden_layers,
            model_config.num_attention_heads,
            model_config.hidden_size,
            kv_heads=model_config.num_key_value_heads,
            tensor_parallel_size=self.engine.world_size,
            dtype=dtype,
            device=self.device,
            total_num_gpu_blocks=total_num_gpu_blocks,
        )

        # log number of free blocks at init
        print("[PagedKVCacheManager] number of free blocks: %d" % (len(self.kv_cache_manager.free_blocks)))

    @property
    def batch_type(self) -> Type[PagedCausalLMBatch]:
        return self._batch_type

    @batch_type.setter
    def batch_type(self, value):
        self._batch_type = value

    @staticmethod
    def _process_tokens(
        input,
        batch: PagedCausalLMBatch,
        generated_tokens: List[TokenInfo],
        input_token_infos: List[InputTokens],
        decode_errors: List[GenerateError],
        prefill: bool = False,
        logits: torch.Tensor = None,
        cum_seqlens: torch.Tensor = None,
    ):
        for (i, next_tokens, scores, logprobs) in input:

            request = batch.requests[i]
            all_input_ids = batch.all_input_ids_tensor[i]

            if batch.input_lengths[i] == batch.total_lengths[i]:
                continue
            try:
                # Ensure tok view is 1st order, everything else is second.
                tok_view = next_tokens.view(-1)
                scores_view = scores.view(-1, scores.shape[-1])
                logprobs_view = logprobs.view(-1, logprobs.shape[-1]) if request.details.logprobs else None

                # TODO would be best to vectorize this also
                token_info = get_token_info(request, scores_view, tok_view, logprobs_view)

                generated_tokens.append(token_info)

                # Add to input tokens info list, if requested
                # Applies only to Prefill
                if prefill and request.details.input_toks:
                    input_length = batch.input_lengths[i]
                    start_index, end_index = cum_seqlens[i], cum_seqlens[i+1]
                    # we don't want to pass the last logits in to get_input_tokens_info
                    input_token_logits = logits[start_index:end_index-1, :]
                    input_token_infos.append(
                        get_input_tokens_info(
                            request,
                            all_input_ids[:input_length],
                            input_token_logits,
                        )
                    )

            except Exception as e:
                logging.exception(f"token decoding error for request #{request.id}")
                # Add to the errors to return
                decode_errors.append(GenerateError(
                    request_id=request.id, message=f"Token decoding error: {str(e)}"
                ))

            # Adjust output length for the next request
            all_input_ids[batch.input_lengths[i]] = token_info.token_id
            batch.input_lengths[i] += 1

    def _prefill(
        self,
        batch,
        generated_tokens,
        input_token_infos,
        decode_errors,
    ):
        bsize = batch.input_ids.shape[0]

        input_ids, position_ids, cache_data = prepare_inputs_for_prefill(
           batch.input_ids, batch.input_lengths, self.kv_cache_manager,
        )

        t0 = time.time_ns()
        output = self.model(
            input_ids,
            position_ids=position_ids,
            cache_data=cache_data,
            return_embeds=True,
        )
        t_forward_ns = time.time_ns()-t0
        logits, embeds = output

        embeds = embeds[cache_data.context_lengths[1:] - 1, :].unsqueeze(1)

        # Heterogeneous next token chooser expects last logits in the sequence
        next_input_ids, next_token_scores, next_token_logprobs = batch.next_token_chooser(
            input_ids=batch.all_input_ids_tensor[:, :batch.max_seqlen],
            scores=logits[cache_data.context_lengths[1:] - 1, :]
        )

        process_input = zip(
            range(bsize),
            next_input_ids,
            next_token_scores,
            next_token_logprobs,
        )

        self._process_tokens(
            process_input,
            batch,
            generated_tokens,
            input_token_infos,
            decode_errors,
            True,
            logits,
            cache_data.context_lengths
        )

        # update batch
        batch.input_ids = next_input_ids.view(-1, 1)
        batch.embeds = embeds
        batch.sequence_ids = cache_data.sequence_ids
        batch.max_seqlen = max(batch.input_lengths)

        return t_forward_ns

    def _next_token_with_speculation(
        self,
        batch,
        generated_tokens,
        input_token_infos,
        decode_errors,
        spec_ind,
    ):
        bsize = batch.input_ids.shape[0]

        (input_ids, position_ids, cache_data, this_flatting,
         unflat_indices, input_ids_unflat, child_sequence_ids_list) = prepare_inputs_with_speculation(
            batch.input_ids, batch.embeds, batch.sequence_ids, self.kv_cache_manager,
            self.speculator, spec_ind, batch.pad_token_id,
        )

        t0 = time.time_ns()
        logits, embeds = self.model(
            input_ids, position_ids=position_ids, cache_data=cache_data, return_embeds=True,
        )
        t_forward_ns = time.time_ns()-t0

        logits, new_embeds, new_sequence_ids, next_tokens = process_outputs_with_speculation(
            logits, embeds, self.kv_cache_manager, self.speculator, this_flatting,
            unflat_indices, input_ids_unflat, child_sequence_ids_list,
        )

        logits_full = logits.new_zeros(bsize, logits.shape[2])
        for i in range(bsize):
            if i not in spec_ind:
                logits_full[i, :] = logits[i,0,:]

        next_input_ids_full, next_token_scores_full, next_token_logprobs_full = batch.next_token_chooser(
            input_ids=batch.all_input_ids_tensor[:, :batch.max_seqlen], scores=logits_full
        )

        process_samples = zip(
            iter(range(bsize)),
            next_input_ids_full,
            next_token_scores_full,
            next_token_logprobs_full,
        )

        new_input_ids = torch.zeros_like(batch.input_ids)

        process_input = []
        for i, id, score, logprob in process_samples:
            if i not in spec_ind:
                process_input.append((i, id, score, logprob))
                new_input_ids[i] = id

        for i, token_ids in enumerate(next_tokens):
            # skip the non-speculative ones
            if i not in spec_ind:
                continue
            for j, id in enumerate(token_ids):
                this_logits = logits[i,j,:].view(-1)
                process_input.append((
                    i,
                    id,
                    this_logits,
                    torch.log_softmax(this_logits, dim=0) if batch.requests[i].details.logprobs else None
                ))

            # updated inputs
            new_input_ids[i] = token_ids[-1]

        self._process_tokens(
            iter(process_input),
            batch,
            generated_tokens,
            input_token_infos,
            decode_errors,
        )

        # update batch
        batch.input_ids = new_input_ids
        batch.embeds = new_embeds
        batch.sequence_ids = new_sequence_ids
        batch.max_seqlen = max(batch.input_lengths)

        return t_forward_ns

    def _next_token_without_speculation(
        self,
        batch,
        generated_tokens,
        input_token_infos,
        decode_errors,
    ):
        bsize = batch.input_ids.shape[0]

        input_ids, position_ids, cache_data = prepare_inputs_without_speculation(
            batch.input_ids, batch.embeds, batch.sequence_ids, self.kv_cache_manager,
        )

        t0 = time.time_ns()
        logits, embeds = self.model(
            input_ids, position_ids=position_ids, cache_data=cache_data, return_embeds=True,
        )
        t_forward_ns = time.time_ns()-t0

        next_input_ids, next_token_scores, next_token_logprobs = batch.next_token_chooser(
            input_ids=batch.all_input_ids_tensor[:, :batch.max_seqlen], scores=logits
        )

        process_input = zip(
            iter(range(bsize)),
            next_input_ids,
            next_token_scores,
            next_token_logprobs,
        )

        self._process_tokens(
            process_input,
            batch,
            generated_tokens,
            input_token_infos,
            decode_errors,
        )

        # update batch
        for i, token_id in enumerate(next_input_ids):
            batch.input_ids[i] = token_id
        batch.embeds = embeds.unsqueeze(1)
        batch.max_seqlen = max(batch.input_lengths)

        return t_forward_ns

    def generate_token(
        self, batch: PagedCausalLMBatch, first: bool = False, for_concat: bool = False,
    ) -> Tuple[List[TokenInfo], Optional[List[InputTokens]], List[GenerateError], int]:

        # Generated tokens
        generated_tokens: List[TokenInfo] = []
        input_token_infos: List[InputTokens] = []
        decode_errors: List[GenerateError] = []

        if first:
            t_forward_ns = self._prefill(
                batch,
                generated_tokens,
                input_token_infos,
                decode_errors,
            )
        else:
            bsize = batch.input_ids.shape[0]

            tokens_remaining = 0
            for i in range(len(batch.total_lengths)):
                tokens_remaining += batch.total_lengths[i] - batch.input_lengths[i]

            spec_ind = []
            for i, sample in enumerate(batch.next_token_chooser.do_sample):
                if not sample:
                    spec_ind.append(i)

            speculate = (
                self.speculator is not None and
                len(spec_ind) > 0 and
                bsize <= SPECULATOR_MAX_BATCH_SIZE and
                batch.next_token_chooser.repetition_processor is None and
                tokens_remaining < 0.25*len(self.kv_cache_manager.free_blocks)*self.kv_cache_manager.block_size
            )

            if speculate:
                t_forward_ns = self._next_token_with_speculation(
                    batch,
                    generated_tokens,
                    input_token_infos,
                    decode_errors,
                    spec_ind,
                )
            else:
                t_forward_ns = self._next_token_without_speculation(
                    batch,
                    generated_tokens,
                    input_token_infos,
                    decode_errors,
                )

        return generated_tokens, input_token_infos, decode_errors, t_forward_ns
