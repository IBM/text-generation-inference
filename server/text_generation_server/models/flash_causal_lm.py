import logging
import time
from operator import itemgetter

import torch
import torch.distributed

from torch.nn import functional as F

from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Union, Any

from text_generation_server.inference_engine import get_inference_engine_class
from text_generation_server.models import Model

from text_generation_server.models.types import Batch, GenerateError
from text_generation_server.pb import generate_pb2
from text_generation_server.pb.generate_pb2 import InputTokens
from text_generation_server.prompt_cache import PrefixCache
from text_generation_server.utils.hub import get_model_path
from text_generation_server.utils.token_types import TokenInfo
from text_generation_server.utils.tokens import (
    HeterogeneousNextTokenChooser, get_token_info, get_input_tokens_info,
)


@dataclass
class FlashCausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]

    # Decoder values
    # tensors have sequences from the batch concatenated
    # shape is [sum(seq_lengths)]
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    # shape is [sum(seq_lengths), embedding_size]
    inputs_embeds: Optional[torch.Tensor]
    # cumulative sequence lengths
    cu_seqlens: torch.Tensor
    # cumulative query sequence lengths, only used in decode
    cu_seqlens_q: Optional[torch.Tensor]
    # past key values, only used in decode
    past_key_values: Optional[torch.Tensor]
    # maximum of the input tokens across the batch (including prefix)
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
    ) -> Tuple[Optional["FlashCausalLMBatch"], List[GenerateError]]:
        errors = []
        batch_inputs = []
        requests = pb.requests

        # track indices of valid requests that have prefixes
        i = 0
        prefix_ids = {}
        # compute sequence lengths in this loop too
        #  if there is a prefix, input_lengths will include its length
        input_lengths = []
        total_lengths = []
        max_seqlen = 0
        # Cumulative length
        cu_seqlens = [0]
        cumulative_length = 0
        for r in requests:
            input_length = r.input_length
            # TODO: Also fail depending on the model type for ones that don't
            # have input_embeds implemented?
            if r.prefix_id:
                try:
                    prefix_embeds = prefix_cache.get(r.prefix_id)
                except Exception:
                    message = f"Prefix lookup error for request #{r.id}, prefix id {r.prefix_id}"
                    logging.error(message)
                    # Exclude this request from the batch, return an error
                    errors.append(GenerateError(request_id=r.id, message=message))
                    continue
                prefix_ids[i] = prefix_embeds
                input_length += prefix_embeds.shape[0]
            batch_inputs.append(r.inputs)
            input_lengths.append(input_length)
            max_seqlen = max(max_seqlen, input_length)
            total_lengths.append(input_length + r.max_output_length)
            cumulative_length += input_length
            cu_seqlens.append(cumulative_length)
            i += 1

        # remove errored requests
        if errors:
            requests = [r for r in pb.requests if not any(r.id == er.request_id for er in errors)]
            # early exit if no requests are valid
            if not requests:
                return None, errors

        # return as lists to avoid unnecessary padding;
        # sequences will be concatenated across the batch
        batch_tokenized_inputs = tokenizer(
            batch_inputs, truncation=True, max_length=max_seqlen, return_token_type_ids=False
        )["input_ids"]

        # Process inputs to generate the needed tensors
        input_ids = []
        position_ids = []
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
            position_ids.append(torch.arange(0, input_length))
        input_ids = torch.cat(input_ids)

        # convert all requests to embeddings if any request has a prefix_id
        if prefix_ids:
            inputs_embeds = embeddings_lookup(input_ids)
            input_ids = None
            # fill in the prefix embeddings into the space that we already
            # allocated due to the padding in input_ids
            for i, p in prefix_ids.items():
                start = cu_seqlens[i]
                prefix_length = p.shape[0]
                inputs_embeds[start:start+prefix_length, :] = p
        else:
            inputs_embeds = None

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
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=torch.cat(position_ids).to(device, non_blocking=True),
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
            cu_seqlens_q=None,
            max_seqlen=max_seqlen,
            past_key_values=None,
            input_lengths=input_lengths,
            total_lengths=total_lengths,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            pad_token_id=tokenizer.pad_token_id,
        ), errors

    @classmethod
    def concatenate(cls, batches: List["FlashCausalLMBatch"]) -> "FlashCausalLMBatch":
        # Batch attributes
        requests = []
        input_lengths = []
        total_lengths = []
        next_token_chooser_parameters = []
        ntc_current_tokens = []
        ntc_samplings = []
        ntc_return_logprobs = []

        device = batches[0].cu_seqlens_q.device

        # Batch tensors
        input_ids = []
        position_ids = []
        cu_seqlens = [torch.tensor([0], dtype=torch.int32, device=device)]
        max_seqlen = 0
        past_key_values = []

        # Allocate the all input IDs tensor
        new_batch_size = sum(len(b) for b in batches)
        max_total_length = max(mtl for b in batches for mtl in b.total_lengths)
        all_input_ids_tensor = batches[0].all_input_ids_tensor.new_full(
            (new_batch_size, max_total_length),
            batches[0].pad_token_id,
        )

        # Cumulative length
        cumulative_length = torch.tensor(0, device=device)
        start_index = 0
        for i, batch in enumerate(batches):
            requests.extend(batch.requests)
            input_lengths.extend(batch.input_lengths)
            total_lengths.extend(batch.total_lengths)

            next_token_chooser_parameters.extend(r.parameters for r in batch.requests)
            ntc_current_tokens.extend(batch.next_token_chooser.current_tokens)
            ntc_samplings.extend(batch.next_token_chooser.samplings)
            ntc_return_logprobs.extend(batch.next_token_chooser.return_logprobs)

            # Add cumulative lengths of all previous inputs
            cu_seqlens.append(batch.cu_seqlens[1:] + cumulative_length)

            input_ids.append(batch.input_ids)
            position_ids.append(batch.position_ids)
            past_key_values.append(
                batch.past_key_values if len(batch) != 1
                else batch.past_key_values[:, :batch.cu_seqlens[-1]]
            )
            batch.past_key_values = None

            end_index = start_index + len(batch)
            all_input_ids_tensor[
                start_index:end_index, :batch.all_input_ids_tensor.shape[1]
            ] = batch.all_input_ids_tensor
            start_index = end_index

            max_seqlen = max(max_seqlen, batch.max_seqlen)
            cumulative_length += batch.cu_seqlens[-1]

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

        return FlashCausalLMBatch(
            batch_id=batches[0].batch_id,
            requests=requests,
            input_ids=torch.cat(input_ids),
            inputs_embeds=None,
            position_ids=torch.cat(position_ids),
            cu_seqlens=torch.cat(cu_seqlens),
            cu_seqlens_q=torch.arange(len(requests) + 1, device=device, dtype=torch.int32),
            max_seqlen=max_seqlen,
            # Concat on dim=1 as first dim represents the model layers
            past_key_values=torch.cat(past_key_values, dim=1),
            input_lengths=input_lengths,
            total_lengths=total_lengths,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            pad_token_id=batches[0].pad_token_id,
        )

    def __len__(self):
        return len(self.requests)

    @classmethod
    def prune(
        cls, batch: "FlashCausalLMBatch", completed_ids: List[int]
    ) -> Optional["FlashCausalLMBatch"]:
        """Prune completed entries from a batch"""

        if not completed_ids:
            # Nothing to prune
            return batch

        # Compile list of indices to retain
        keep_indices = Model.get_indices_to_keep(batch.requests, completed_ids)
        new_size = len(keep_indices)

        # If the whole batch has finished, discard it
        if new_size == 0:
            return None

        if new_size == 1:
            index = keep_indices[0]

            def slice_list(l):
                return (l[index],)

            start, end = batch.cu_seqlens[index], batch.cu_seqlens[index+1]
            past_slice = batch.past_key_values[:, start:end]
            shape = list(batch.past_key_values.shape)
            # Preallocate single-batch tensor to maximum size
            shape[1] = batch.total_lengths[index]
            batch.past_key_values = batch.past_key_values.new_zeros(shape)
            batch.past_key_values[:, :(end - start)] = past_slice
        else:
            slice_list = itemgetter(*keep_indices)
            sliced = [
                batch.past_key_values[:, batch.cu_seqlens[i]: batch.cu_seqlens[i+1]]
                for i in keep_indices
            ]
            batch.past_key_values = torch.cat(sliced, dim=1)

        # TODO maybe a single loop for all these list slices
        batch.input_lengths = list(slice_list(batch.input_lengths))
        batch.total_lengths = slice_list(batch.total_lengths)
        batch.requests = slice_list(batch.requests)
        batch.next_token_chooser = batch.next_token_chooser.filter(keep_indices)

        batch.max_seqlen = max(batch.input_lengths)

        batch.input_ids = batch.input_ids[keep_indices]
        batch.position_ids = batch.position_ids[keep_indices]

        batch.all_input_ids_tensor = batch.all_input_ids_tensor[keep_indices, :max(batch.total_lengths)]

        if new_size == 1:
            batch.cu_seqlens = batch.cu_seqlens.new_tensor([0, batch.input_lengths[0]])
        else:
            # Recalculate cumulative seq lengths
            batch.cu_seqlens = batch.cu_seqlens[:new_size + 1]
            batch.cu_seqlens[1:] = batch.position_ids
            batch.cu_seqlens[1:].add_(1)
            torch.cumsum(batch.cu_seqlens, dim=0, out=batch.cu_seqlens)

        batch.cu_seqlens_q = batch.cu_seqlens_q[:new_size + 1]

        return batch

    # This needs to be re-evaluated. If we do it here, we need to ensure the tensor
    # is re-allocated in the case that the add-on batch only generates the prefill
    # token because no concatenation will occur.
    # def compact(self):
    #     """Copy tensors to free previously pruned/trimmed space"""
    #
    #     if len(self) == 1:
    #          self.past_key_values = self.past_key_values[:, :self.cu_seqlens[-1]].clone()


class FlashCausalLM(Model):
    def __init__(
        self,
        model_name: str,
        revision: str,
        deployment_framework: str,
        dtype: torch.dtype,
        quantize: Optional[str],
        model_config: Union[Any] = None,
        auto_model_class=None,
        max_sequence_length: Optional[int] = None,
    ):
        if not torch.cuda.is_available():
            raise NotImplementedError("FlashCausalLM is only available on GPU")

        self.present_pad = None

        model_path = get_model_path(model_name, revision)

        inference_engine = get_inference_engine_class(deployment_framework)(
            model_path, auto_model_class, dtype, quantize, model_config, max_sequence_length
        )

        super(FlashCausalLM, self).__init__(inference_engine, dtype)
        self.use_position_ids = True

        if self.model.config.pad_token_id is not None:
            self.tokenizer.pad_token_id = self.model.config.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            if self.model.config.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.model.config.eos_token_id
            elif self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    @property
    def batch_type(self) -> Type[FlashCausalLMBatch]:
        return FlashCausalLMBatch

    def generate_token(
        self, batch: FlashCausalLMBatch, first: bool = False, for_concat: bool = False,
    ) -> Tuple[List[TokenInfo], Optional[List[InputTokens]], List[GenerateError], int]:

        batch_size = len(batch)
        past_key_values = batch.past_key_values if first or batch_size > 1 \
            else batch.past_key_values[:, :batch.cu_seqlens[-1]]

        if first and batch_size == 1:
            # Preallocate past_key_value tensor for batch size == 1
            prealloc_length = batch.input_lengths[0] + (
                1 if for_concat else batch.requests[0].max_output_length
            )
        else:
            prealloc_length = None

        start_time = time.time_ns()
        out, present = self.model.forward(
            batch.input_ids,
            batch.position_ids,
            batch.cu_seqlens,
            batch.cu_seqlens_q,
            batch.max_seqlen,
            batch.inputs_embeds,
            past_key_values,
            prealloc_length,
        )
        forward_time_ns = time.time_ns() - start_time

        # Update present
        present_pad = self.present_pad
        if present_pad is None:
            present_pad = self.present_pad = present.new_zeros(present.shape[0], 1, *present.shape[2:])

        if batch_size > 1:
            new_past = []
            start_index = 0
            for i in range(1, batch_size + 1):
                end_index = batch.cu_seqlens[i]
                new_past.append(present[:, start_index:end_index])
                new_past.append(present_pad)
                start_index = end_index
            batch.past_key_values = torch.cat(new_past, dim=1)

        if first:
            if batch_size == 1:
                batch.past_key_values = present
            generated_tokens, input_token_infos, decode_errors = self._process_prefill(batch, out)
        else:
            generated_tokens, decode_errors = self._process_decode(batch, out)
            input_token_infos = None

        batch.cu_seqlens.add_(batch.cu_seqlens_q)
        batch.max_seqlen += 1

        return generated_tokens, input_token_infos, decode_errors, forward_time_ns

    def _process_prefill(
        self, batch: FlashCausalLMBatch, out,
    ) -> Tuple[List[TokenInfo], List[InputTokens], List[GenerateError]]:

        # Generated tokens
        generated_tokens: List[TokenInfo] = []
        input_token_infos: List[InputTokens] = []
        decode_errors: List[GenerateError] = []

        # Create position_ids tensor before incrementing input lengths
        batch.position_ids = batch.position_ids.new_tensor(batch.input_lengths)

        batch.input_ids = self._process_new_tokens(
            batch, out,
            generated_tokens,
            decode_errors,
            input_token_infos,
            True,
        )

        # Create final next batch tensors
        batch.inputs_embeds = None
        batch_size = len(batch)
        batch.cu_seqlens_q = torch.arange(
            batch_size + 1, device=self.device, dtype=torch.int32
        )

        return generated_tokens, input_token_infos, decode_errors

    def _process_decode(
        self, batch: FlashCausalLMBatch, out,
    ) -> Tuple[List[TokenInfo], List[GenerateError]]:
        # These are appended to in _process_new_tokens
        generated_tokens: List[TokenInfo] = []
        decode_errors: List[GenerateError] = []

        # position_ids are used in _process_new_tokens and must be incremented first
        batch.position_ids += 1
        batch.input_ids = self._process_new_tokens(
            batch, out, generated_tokens, decode_errors, None, False,
        )

        return generated_tokens, decode_errors

    def _process_new_tokens(
        self,
        batch: FlashCausalLMBatch,
        out,
        generated_tokens: List[TokenInfo],
        decode_errors: List[GenerateError],
        input_token_infos: List[InputTokens],
        prefill: bool,
    ):
        if prefill:
            # Prefill mode
            # out is of shape [cumulative_sequence_lengths, vocab_size]
            # where cumulative_sequence_lengths starts with 0
            logits = out[batch.cu_seqlens[1:] - 1, :]
        else:
            # Decode mode
            # out is of shape [batch_size, vocab_size] already (1 token per request in batch)
            logits = out

        next_token_ids, next_token_scores, next_token_logprobs = batch.next_token_chooser(
            input_ids=batch.all_input_ids_tensor[:, :batch.max_seqlen], scores=logits,
        )

        # add the next token ids to all_input_ids_tensor
        # Note (NH): It might be a bit better performance-wise to execute this
        # before the loop. Just in terms of it being more likely for things to
        # run in parallel (since cuda ops happen async when possible)
        batch.all_input_ids_tensor.scatter_(
            dim=1, index=batch.position_ids[:, None], src=next_token_ids[:, None]
        )

        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.cu_seqlens,
            next_token_ids,
            next_token_scores,
            next_token_logprobs,
            batch.all_input_ids_tensor
        )
        for i, (
            request,
            input_length,
            start_index, # cu_seqlens contains the start index for each request
            next_token,
            scores,
            logprobs,
            all_input_ids,
        ) in enumerate(iterator):
            try:
                # Ensure tok view is 1st order, everything else is second.
                tok_view = next_token.view(-1)
                scores_view = scores.view(-1, scores.shape[-1])
                logprobs_view = logprobs.view(-1, logprobs.shape[-1]) if request.details.logprobs else None

                # Get next token info
                token_info = get_token_info(request, scores_view, tok_view, logprobs_view)
                # Add to the tokens to return
                generated_tokens.append(token_info)

                # Add to input tokens info list, if requested
                # Applies only to Prefill
                if prefill and request.details.input_toks:
                    end_index = start_index + input_length
                    # we don't want to pass the last logits in to get_input_tokens_info
                    input_token_logits = out[start_index:end_index-1, :]
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
            batch.input_lengths[i] += 1

        return next_token_ids
