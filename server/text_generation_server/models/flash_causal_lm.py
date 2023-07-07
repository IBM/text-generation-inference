import logging
from operator import itemgetter

import torch
import torch.distributed

from torch.nn import functional as F

from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Union, Any

from text_generation_server.inference_engine import get_inference_engine_class
from text_generation_server.models import Model

from text_generation_server.models.types import Batch, TokenInfo, GenerateError
from text_generation_server.pb import generate_pb2
from text_generation_server.pb.generate_pb2 import InputTokens
from text_generation_server.prompt_cache import PrefixCache
from text_generation_server.utils.hub import get_model_path
from text_generation_server.utils.tokens import (
    NextTokenChooser, get_token_info, get_input_tokens_info,
)


@dataclass
class FlashCausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]

    # Decoder values
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    # cumulative sequence lengths
    cu_seqlens: torch.Tensor
    # cumulative query sequence lengths, only used in decode
    cu_seqlens_q: Optional[torch.Tensor]
    # past key values, only used in decode
    past_key_values: Optional[torch.Tensor]
    max_seqlen: int

    # All tokens
    all_input_ids_tensor: List[torch.Tensor]

    # Lengths of all generations present in the batch
    input_lengths: List[int]

    # Used for resizing preallocated kv-cache when pruning
    original_input_lengths: List[int]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]

    def get_id(self) -> int:
        return self.batch_id

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        embeddings_lookup: Optional,
        prefix_cache: Optional[PrefixCache],
        use_position_ids: bool = True,
    ) -> Tuple[Optional["FlashCausalLMBatch"], List[GenerateError]]:
        errors = []
        batch_inputs = []
        max_seqlen = 0
        for r in pb.requests:
            if r.prefix_id:
                message = f"Prompt prefixes not yet supported with flash attention (request #{r.id})"
                logging.error(message)
                # Exclude this request from the batch, return an error
                errors.append(GenerateError(request_id=r.id, message=message))
                continue
            batch_inputs.append(r.inputs)
            max_seqlen = max(max_seqlen, r.input_length)

        if errors:
            requests = [r for r in pb.requests if not any(r.id == er.request_id for er in errors)]
            if not requests:
                return None, errors

        batch_tokenized_inputs = tokenizer(
            batch_inputs, truncation=True, max_length=max_seqlen, return_token_type_ids=False
        )["input_ids"]

        input_ids = []
        position_ids = []
        cu_seqlens = [0]

        input_lengths = []
        all_input_ids_tensor = []

        next_token_choosers = []

        # Cumulative length
        cumulative_length = 0

        # Parse batch
        requests = pb.requests
        for r, tokenized_input in zip(requests, batch_tokenized_inputs):
            input_length = r.input_length

            tokenized_input = tokenized_input[-input_length:]

            input_lengths.append(input_length)

            tokenized_input = torch.tensor(tokenized_input, device=device)
            input_ids.append(tokenized_input)

            # Position ids
            position_ids.append(torch.arange(0, input_length, dtype=torch.int32))

            # Add cumulative lengths of all previous inputs
            cu_seqlens.append(cumulative_length + input_length)

            next_token_choosers.append(
                NextTokenChooser.from_pb(r.parameters, r.details.logprobs, tokenizer, device)
            )
            all_input_ids_tensor.append(F.pad(tokenized_input, (0, r.max_output_length)))

            cumulative_length += input_length

        input_ids = torch.cat(input_ids)
        position_ids = torch.cat(position_ids).to(device, non_blocking=True)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

        return cls(
            batch_id=pb.id,
            requests=requests,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            cu_seqlens_q=None,
            max_seqlen=max_seqlen,
            past_key_values=None,
            input_lengths=input_lengths,
            original_input_lengths=input_lengths.copy(),
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_choosers=next_token_choosers,
        ), errors

    @classmethod
    def concatenate(cls, batches: List["FlashCausalLMBatch"]) -> "FlashCausalLMBatch":
        # Batch attributes
        requests = []
        input_lengths = []
        original_input_lengths = []
        all_input_ids_tensor = []
        next_token_choosers = []

        device = batches[0].cu_seqlens_q.device

        # Batch tensors
        input_ids = []
        position_ids = []
        cu_seqlens = [torch.tensor([0], dtype=torch.int32, device=device)]
        max_seqlen = 0
        past_key_values = []

        # Cumulative length
        cumulative_length = torch.tensor(0, device=device)

        for i, batch in enumerate(batches):
            requests.extend(batch.requests)
            input_lengths.extend(batch.input_lengths)
            original_input_lengths.extend(batch.original_input_lengths)
            all_input_ids_tensor.extend(batch.all_input_ids_tensor)
            next_token_choosers.extend(batch.next_token_choosers)

            # Add cumulative lengths of all previous inputs
            cu_seqlens.append(batch.cu_seqlens[1:] + cumulative_length)

            input_ids.append(batch.input_ids)
            position_ids.append(batch.position_ids)
            past_key_values.append(
                batch.past_key_values if len(batch) != 1
                else batch.past_key_values[:, :batch.cu_seqlens[-1]]
            )
            batch.past_key_values = None

            max_seqlen = max(max_seqlen, batch.max_seqlen)

            cumulative_length += batch.cu_seqlens[-1]

        return FlashCausalLMBatch(
            batch_id=batches[0].batch_id,
            requests=requests,
            input_ids=torch.cat(input_ids),
            position_ids=torch.cat(position_ids),
            cu_seqlens=torch.cat(cu_seqlens),
            cu_seqlens_q=torch.arange(len(requests) + 1, device=device, dtype=torch.int32),
            max_seqlen=max_seqlen,
            # Concat on dim=1 as first dim represents the model layers
            past_key_values=torch.cat(past_key_values, dim=1),
            input_lengths=input_lengths,
            original_input_lengths=original_input_lengths,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_choosers=next_token_choosers,
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
            shape[1] = batch.original_input_lengths[index] + batch.requests[index].max_output_length
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
        batch.original_input_lengths = slice_list(batch.original_input_lengths)
        batch.requests = slice_list(batch.requests)
        batch.next_token_choosers = slice_list(batch.next_token_choosers)
        batch.all_input_ids_tensor = slice_list(batch.all_input_ids_tensor)

        batch.max_seqlen = max(batch.input_lengths)

        batch.input_ids = batch.input_ids[keep_indices]
        batch.position_ids = batch.position_ids[keep_indices]

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
        model_config: Union[Any] = None,
        auto_model_class=None,
    ):
        self.present_pad = None
        if dtype == torch.int8:
            raise NotImplementedError("FlashCausalLM does not support quantization")

        model_path = get_model_path(model_name, revision)

        if not torch.cuda.is_available():
            raise NotImplementedError("FlashCausalLM is only available on GPU")

        inference_engine = get_inference_engine_class(deployment_framework)(
            model_path, auto_model_class, dtype, model_config,
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
    ) -> Tuple[List[TokenInfo], Optional[List[InputTokens]], List[GenerateError]]:

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

        out, present = self.model.forward(
            batch.input_ids,
            batch.position_ids,
            batch.cu_seqlens,
            batch.cu_seqlens_q,
            batch.max_seqlen,
            past_key_values,
            prealloc_length,
        )

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

        return generated_tokens, input_token_infos, decode_errors

    def _process_prefill(
        self, batch: FlashCausalLMBatch, out,
    ) -> Tuple[List[TokenInfo], List[InputTokens], List[GenerateError]]:

        # New values for next forward
        next_batch_input_ids = []

        # Generated tokens
        generated_tokens: List[TokenInfo] = []
        input_token_infos: List[InputTokens] = []
        decode_errors: List[GenerateError] = []

        # Create position_ids tensor before incrementing input lengths
        batch.position_ids = batch.position_ids.new_tensor(batch.input_lengths)

        # For each member of the batch
        batch_size = len(batch)
        for i in range(batch_size):
            # Indexing metadata
            next_token_id = self._process_new_token(
                batch, i, out,
                generated_tokens,
                decode_errors,
                input_token_infos if batch.requests[i].details.input_toks else None,
                True,
            )

            next_batch_input_ids.append(next_token_id)

        # Create final next batch tensors
        batch.input_ids = torch.cat(next_batch_input_ids) \
            if batch_size > 1 else next_batch_input_ids[0].view(1)

        batch.cu_seqlens_q = torch.arange(
            batch_size + 1, device=self.device, dtype=torch.int32
        )

        return generated_tokens, input_token_infos, decode_errors

    def _process_decode(
        self, batch: FlashCausalLMBatch, out,
    ) -> Tuple[List[TokenInfo], List[GenerateError]]:

        # New values for next forward

        # Generated tokens
        generated_tokens: List[TokenInfo] = []
        decode_errors: List[GenerateError] = []

        # For each member of the batch
        for i in range(len(batch)):
            # Update input ids
            batch.input_ids[i] = self._process_new_token(
                batch, i, out, generated_tokens, decode_errors, None, False,
            )

        batch.position_ids += 1

        return generated_tokens, decode_errors

    def _process_new_token(
        self, batch, i, out, generated_tokens, decode_errors, input_token_infos, prefill,
    ):
        request = batch.requests[i]
        all_input_ids_tensor = batch.all_input_ids_tensor[i]
        input_length = batch.input_lengths[i]
        start_index = batch.cu_seqlens[i]
        end_index = start_index + input_length
        try:
            if prefill:
                # Prefill mode
                # out is of shape [cumulative_sequence_lengths, vocab_size]
                logits = out[start_index:end_index]
            else:
                # Decode mode
                # out is of shape [batch_size, vocab_size]
                logits = out[i].unsqueeze(0)

            # Select next token
            next_token_id, scores, logprobs = batch.next_token_choosers[i](
                all_input_ids_tensor[None, :input_length], logits[-1:, :]
            )
            next_token_id_item = next_token_id.item()

            # Get next token info
            token_info = get_token_info(request, scores, next_token_id, logprobs)

            # Add to input tokens info list, if requested
            # Applies only to first call for each batch
            if input_token_infos is not None:
                input_token_infos.append(
                    get_input_tokens_info(
                        request, all_input_ids_tensor[:input_length], logits[:-1, :]
                    )
                )

            # Add to the tokens to return
            generated_tokens.append(token_info)

        except Exception as e:
            logging.exception(f"token decoding error for request #{request.id}")
            next_token_id_item = self.tokenizer.pad_token_id
            next_token_id = all_input_ids_tensor.new_tensor([next_token_id_item])
            # Add to the errors to return
            decode_errors.append(GenerateError(
                request_id=request.id, message=f"Token decoding error: {str(e)}"
            ))

        # Append next token to all tokens
        all_input_ids_tensor[input_length] = next_token_id_item

        batch.input_lengths[i] = input_length + 1

        return next_token_id
