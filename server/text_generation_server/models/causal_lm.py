import logging
import time
from operator import itemgetter

import torch

from dataclasses import dataclass
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Union, Any

from text_generation_server.models.model import Model, CUDA_PAD_TO_MULT_OF_8
from text_generation_server.models.types import Batch, GenerateError
from text_generation_server.pb import generate_pb2
from text_generation_server.prompt_cache import PrefixCache
from text_generation_server.utils.hub import get_model_path
from text_generation_server.utils.token_types import TokenInfo, InputTokens
from text_generation_server.utils.tokens import HeterogeneousNextTokenChooser, get_token_info, get_input_tokens_info
from text_generation_server.inference_engine import get_inference_engine_class


@dataclass
class CausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]

    # Decoder values
    # Only one of input_ids and inputs_embeds will be non-None
    input_ids: Optional[torch.Tensor]
    # This is always None post-prefill
    inputs_embeds: Optional[torch.Tensor]
    attention_mask: torch.Tensor
    position_ids: Optional[torch.Tensor]
    past_key_values: Optional[List[Tuple]]

    # All tokens
    all_input_ids_tensor: torch.Tensor

    # Lengths of all generations present in the batch
    input_lengths: List[int]

    # Generation helpers
    next_token_chooser: HeterogeneousNextTokenChooser

    # Metadata used for padding
    max_sequence_length: int
    padding_right_offset: int
    max_remaining_tokens: List[int]
    pad_token_id: int

    # Metadata for the past_key_values cache

    # for BLOOM: the "head_dim" of the K tensor is transposed with the sequence
    # (i.e. K and V have different dims)
    keys_head_dim_last: bool = True
    # for GPTBigCode: the K/V tensors are merged into one tensor with dims
    # [batch, sequence, params]
    merged_kv_cache: bool = False

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
        use_position_ids: bool = False,
    ) -> Tuple[Optional["CausalLMBatch"], List[GenerateError]]:
        """Convert a text_generation.v1.Batch protobuf to a CausalLMBatch"""
        input_texts = []
        next_token_chooser_parameters = []
        input_lengths = []
        max_remaining_tokens = []
        return_logprobs = []
        prefix_ids = {}
        errors = []
        max_input_length = 0
        padding_right_offset = 0
        truncate_indices = []

        # Parse batch
        requests = pb.requests
        for i, r in enumerate(requests):
            input_length = r.input_length
            max_output_length = r.max_output_length
            if r.prefix_id:
                try:
                    prefix_embeds = prefix_cache.get(r.prefix_id)
                except Exception:
                    logging.exception(f"Prefix lookup error for request #{r.id}, prefix id {r.prefix_id}")
                    # Exclude this request from the batch, return error
                    errors.append(GenerateError(
                        request_id=r.id, message=f"Error retrieving prompt prefix '{r.prefix_id}'")
                    )
                    continue
                prefix_ids[i] = prefix_embeds
                # Add prefix length to user input length
                input_length += prefix_embeds.shape[0]
            input_texts.append(r.inputs)
            input_lengths.append(input_length)
            if r.truncate:
                truncate_indices.append(i)
            max_input_length = max(max_input_length, input_length)
            max_remaining_tokens.append(max_output_length)
            padding_right_offset = max(padding_right_offset, max_output_length)
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

        if errors:
            requests = [r for r in pb.requests if not any(r.id == er.request_id for er in errors)]
            if not requests:
                return None, errors

        batch_size = len(requests)

        # Tokenize batch
        tokenize_length = max_input_length
        # Pad to multiple of 8 for tensor core GPUs
        left_pad = 0
        if device.type == "cuda" and CUDA_PAD_TO_MULT_OF_8 and (mod := tokenize_length % 8) != 0:
            left_pad = 8 - mod
            tokenize_length += left_pad
        tokenized_inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenize_length,
            return_token_type_ids=False,
        ).to(device)
        all_input_ids = tokenized_inputs["input_ids"]

        # Allocate maximum attention_mask
        attention_mask = all_input_ids.new_zeros((batch_size, tokenize_length + padding_right_offset))
        # Copy tokenizer attention_mask into fully allocated attention_mask
        attention_mask[:, :tokenize_length] = tokenized_inputs["attention_mask"]

        # Mask out truncated tokens
        # (input_texts aren't truncated, only input_lengths are)
        if truncate_indices:
            add_bos_token = getattr(tokenizer, "add_bos_token", False)
            for i in truncate_indices:
                orig_input_length = requests[i].input_length
                attention_mask[i, :-orig_input_length-padding_right_offset] = 0
                all_input_ids[i, :-orig_input_length] = tokenizer.pad_token_id
                if add_bos_token:
                    # Ensure that first non-virtual token is set to BOS
                    all_input_ids[i, -orig_input_length] = tokenizer.bos_token_id

        # Padded all_input_ids_tensor; the maximum length of any sequence is the max
        # (padded) input sequence length + the max output length
        all_input_ids_tensor = all_input_ids.new_full(
            (batch_size, max_input_length + padding_right_offset),
            tokenizer.pad_token_id,
        )
        no_pad_input_ids = all_input_ids[:, left_pad:] if left_pad else all_input_ids
        all_input_ids_tensor[:, :no_pad_input_ids.shape[1]] = no_pad_input_ids

        if prefix_ids:
            # Get input embeddings
            inputs_embeds = embeddings_lookup(all_input_ids)
            for i, p in prefix_ids.items():
                input_length = input_lengths[i]
                orig_length = input_length - p.shape[0]
                # Insert prefix embeddings
                inputs_embeds[i, -input_length:-orig_length] = p
                # Update attention mask with virtual prefix tokens
                attention_mask[i, -input_length-padding_right_offset:-padding_right_offset] = 1
            input_ids = None
        else:
            input_ids = all_input_ids
            inputs_embeds = None

        if use_position_ids:
            # Fix up position ids
            sliced_attention_mask = attention_mask[:, :-padding_right_offset]
            position_ids = sliced_attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(sliced_attention_mask == 0, 1)
        else:
            position_ids = None

        return cls(
            batch_id=pb.id,
            requests=requests,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            all_input_ids_tensor=all_input_ids_tensor,
            input_lengths=input_lengths,
            max_remaining_tokens=max_remaining_tokens,
            next_token_chooser=next_token_chooser,
            max_sequence_length=max_input_length,
            padding_right_offset=padding_right_offset,
            pad_token_id=tokenizer.pad_token_id,
        ), errors

    @classmethod
    def concatenate(cls, batches: List["CausalLMBatch"]) -> "CausalLMBatch":
        """Concatenate multiple batches together by padding internal torch tensors"""

        # Used for padding
        total_batch_size = 0
        max_sequence_length = 0
        padding_right_offset = 0
        for batch in batches:
            total_batch_size += len(batch)
            max_sequence_length = max(max_sequence_length, batch.max_sequence_length)
            padding_right_offset = max(padding_right_offset, batch.padding_right_offset)

        # Batch attributes
        requests = []
        input_lengths = []
        max_remaining_tokens = []
        next_token_chooser_parameters = []
        ntc_current_tokens = []
        ntc_samplings = []
        ntc_return_logprobs = []

        # Batch tensors
        input_ids = None
        all_input_ids_tensor = None
        attention_mask = None
        position_ids = None
        past_key_values = []

        # We only concatenate batches that did at least one step
        if any(batch.past_key_values is None for batch in batches):
            raise ValueError("can only concatenate prefilled batches")

        # Check the past keys/values shape so that we can revert to that
        # shape after the convenience slicing if necessary
        three_dim_pkvs = not batches[0].merged_kv_cache and len(batches[0].past_key_values[0][0].shape) == 3

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0
        for i, batch in enumerate(batches):
            requests.extend(batch.requests)
            input_lengths.extend(batch.input_lengths)
            max_remaining_tokens.extend(batch.max_remaining_tokens)

            next_token_chooser_parameters.extend(r.parameters for r in batch.requests)
            ntc_current_tokens.extend(batch.next_token_chooser.current_tokens)
            ntc_samplings.extend(batch.next_token_chooser.samplings)
            ntc_return_logprobs.extend(batch.next_token_chooser.return_logprobs)

            # Slicing end index for this batch
            end_index = start_index + len(batch)

            # Create empty tensor
            # input_ids is always of shape [batch_size, 1]
            # We do not need to pad it
            if input_ids is None:
                input_ids = batch.input_ids.new_empty((total_batch_size, 1))
            # Copy to correct indices
            input_ids[start_index:end_index] = batch.input_ids

            # Create padded tensors for attention_mask and all_input_ids that
            # include space for the max output tokens
            if attention_mask is None:
                attention_mask = batch.attention_mask.new_zeros(
                    (total_batch_size, max_sequence_length + padding_right_offset),
                )
            if all_input_ids_tensor is None:
                all_input_ids_tensor = batch.all_input_ids_tensor.new_full(
                    (total_batch_size, max_sequence_length + padding_right_offset),
                    batches[0].pad_token_id,
                )

            # We need to slice the attention mask and all_input_ids_tensor to
            # remove padding from previous steps and to remove unused allocated space
            left_offset = max_sequence_length - batch.max_sequence_length
            batch_left_offset = -batch.max_sequence_length - batch.padding_right_offset
            attention_mask[
                start_index:end_index, left_offset:-padding_right_offset,
            ] = batch.attention_mask[
                :, batch_left_offset: -batch.padding_right_offset,
            ]

            all_input_ids_tensor[
                start_index:end_index, left_offset:-padding_right_offset,
            ] = batch.all_input_ids_tensor[
                :, :-batch.padding_right_offset,
            ]

            if batch.position_ids is not None:
                # Create empty tensor
                # position_ids is always of shape [batch_size, 1]
                if position_ids is None:
                    position_ids = batch.position_ids.new_empty((total_batch_size, 1))
                position_ids[start_index:end_index] = batch.position_ids

            # Ensure that we can update tensors in-place
            if type(batch.past_key_values[0]) == tuple:
                batch.past_key_values = [
                    [t.view(len(batch), -1, *t.shape[-2:]) for t in layer] for layer in batch.past_key_values
                ]
            elif three_dim_pkvs:
                for layer in batch.past_key_values:
                    for k, t in enumerate(layer):
                        layer[k] = t.view(len(batch), -1, *t.shape[-2:])

            start_index = end_index

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            pb=next_token_chooser_parameters,
            model_eos_token_id=batches[0].next_token_chooser.eos_token_id,
            model_pad_token_id=batches[0].next_token_chooser.pad_token_id,
            return_logprobs=ntc_return_logprobs,
            dtype=batches[0].next_token_chooser.dtype,
            device=batches[0].next_token_chooser.device,
            samplings=ntc_samplings,
            current_tokens=ntc_current_tokens,
        )

        first_past_kvs = batches[0].past_key_values
        if batches[0].merged_kv_cache:
            # For GPTBigCode the K and V tensors of the cache are merged into
            # one with dims [batch, sequence, params]
            _, _, cache_size = first_past_kvs[0].shape
            padded_past_shape = (
                total_batch_size,
                max_sequence_length - 1,
                cache_size,
            )
            # Iterate over attention layers
            for j in range(len(first_past_kvs)):
                padded_past = first_past_kvs[j].new_zeros(padded_past_shape)
                start_index = 0
                for batch in batches:
                    past = batch.past_key_values[j]
                    # Clear reference to the original tensor
                    batch.past_key_values[j] = None

                    # Slicing end index for this batch
                    end_index = start_index + len(batch)
                    # slice to remove the padding from previous batches
                    past_seq_len = batch.max_sequence_length - 1

                    # assumes the "head_dim" is last
                    padded_past[
                        start_index:end_index, -past_seq_len:, :
                    ] = past[:, -past_seq_len:, :]
                    del past

                    start_index = end_index

                past_key_values.append(padded_past)
    
        else:
            # Shenanigans to get dimensions because BLOOM outputs a past with a different shape
            # BLOOM Keys:   [batch_size * num_heads, head_dim, seq_length]
            # BLOOM Values: [batch_size * num_heads, seq_length, head_dim]
            _, num_heads, padded_sequence_length, head_dim = first_past_kvs[0][1].shape

            padded_past_values_shape = (
                total_batch_size,
                num_heads,
                max_sequence_length - 1,
                head_dim,
            )

            padded_past_keys_shape = padded_past_values_shape if batches[0].keys_head_dim_last \
                else (
                    # seq_length is last for BLOOM
                    total_batch_size,
                    num_heads,
                    head_dim,
                    max_sequence_length - 1,
                )

            # Iterate over attention layers
            for j in range(len(first_past_kvs)):
                padded_past_keys = first_past_kvs[j][0].new_zeros(padded_past_keys_shape)
                start_index = 0
                for batch in batches:
                    past_keys = batch.past_key_values[j][0]
                    # Clear reference to the original tensor
                    batch.past_key_values[j][0] = None

                    # Slicing end index for this batch
                    end_index = start_index + len(batch)
                    # We slice the keys to remove the padding from previous batches
                    past_seq_len = batch.max_sequence_length - 1
                    if batch.keys_head_dim_last:
                        padded_past_keys[
                            start_index:end_index, :, -past_seq_len:, :
                        ] = past_keys[:, :, -past_seq_len:, :]
                    else:
                        # BLOOM case
                        padded_past_keys[
                            start_index:end_index, :, :, -past_seq_len:
                        ] = past_keys[:, :, :, -past_seq_len:]
                    del past_keys

                    start_index = end_index

                padded_past_values = first_past_kvs[j][1].new_zeros(padded_past_values_shape)
                start_index = 0
                for batch in batches:
                    past_values = batch.past_key_values[j][1]
                    # Clear reference to the original tensor
                    batch.past_key_values[j][1] = None

                    # Slicing end index for this batch
                    end_index = start_index + len(batch)
                    # We slice the past values to remove the padding from previous batches
                    past_seq_len = batch.max_sequence_length - 1
                    padded_past_values[
                        start_index:end_index, :, -past_seq_len:, :
                    ] = past_values[:, :, -past_seq_len:, :]
                    del past_values

                    start_index = end_index

                if three_dim_pkvs:
                    # Revert reshaped past kv shape to what's expected by the model
                    padded_past_keys = padded_past_keys.reshape(-1, *padded_past_keys.shape[-2:])
                    padded_past_values = padded_past_values.reshape(-1, *padded_past_values.shape[-2:])

                past_key_values.append([padded_past_keys, padded_past_values])

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            input_ids=input_ids,
            inputs_embeds=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            all_input_ids_tensor=all_input_ids_tensor,
            input_lengths=input_lengths,
            next_token_chooser=next_token_chooser,
            max_remaining_tokens=max_remaining_tokens,
            max_sequence_length=max_sequence_length,
            padding_right_offset=padding_right_offset,
            pad_token_id=batches[0].pad_token_id,
            keys_head_dim_last=batches[0].keys_head_dim_last,
            merged_kv_cache=batches[0].merged_kv_cache,
        )

    @classmethod
    def prune(cls, batch: "CausalLMBatch", completed_ids: List[int]) -> Optional["CausalLMBatch"]:
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

        size_before = len(batch)

        # Apply indices to attention mask, past key values and other items that need to be cached

        # Check the past keys/values shape so that we can revert to that
        # shape after the convenience slicing if necessary
        three_dim_pkvs = not batch.merged_kv_cache and len(batch.past_key_values[0][0].shape) == 3

        #TODO maybe a single loop for all these list slices
        slice_list = itemgetter(*keep_indices) if new_size > 1 else lambda l: (l[keep_indices[0]],)
        batch.input_lengths = list(slice_list(batch.input_lengths))
        batch.max_remaining_tokens = list(slice_list(batch.max_remaining_tokens))
        batch.requests = slice_list(batch.requests)
        batch.next_token_chooser = batch.next_token_chooser.filter(keep_indices)

        batch.max_sequence_length = max(batch.input_lengths)
        new_padding_right_offset = max(batch.max_remaining_tokens)

        past_kv_length = batch.max_sequence_length - 1

        # Ensure that we can update layers in-place
        if type(batch.past_key_values) == tuple:
            batch.past_key_values = list(batch.past_key_values)

        for i, layer in enumerate(batch.past_key_values):
            if batch.merged_kv_cache:
                batch.past_key_values[i] = layer[keep_indices, -past_kv_length:, :]
            else:
                batch.past_key_values[i] = update_layer(
                    layer, size_before, keep_indices, past_kv_length, batch.keys_head_dim_last, three_dim_pkvs, 
                )

        batch.input_ids = batch.input_ids[keep_indices]
        batch.attention_mask = batch.attention_mask[
            keep_indices,
            -batch.padding_right_offset-batch.max_sequence_length:
            (batch.attention_mask.shape[1]-batch.padding_right_offset)+new_padding_right_offset,
        ]
        batch.all_input_ids_tensor = batch.all_input_ids_tensor[
            keep_indices,
            -batch.padding_right_offset-batch.max_sequence_length:
            (batch.all_input_ids_tensor.shape[1]-batch.padding_right_offset)+new_padding_right_offset,
        ]
        batch.padding_right_offset = new_padding_right_offset
        batch.position_ids = None if batch.position_ids is None \
            else batch.position_ids[keep_indices]

        return batch

def update_layer(layer, batch_size, keep_indices, past_kv_length, hdl, three_dim_pkv) -> List[torch.Tensor]:
    """Slice past kv layer with specific batch indices to keep"""

    # Force past to be of dim [batch_size, num_heads, ...] for easy indexing
    pk_shape, pv_shape = layer[0].shape[-2:], layer[1].shape[-2:]
    past_keys = layer[0].view(batch_size, -1, *pk_shape)
    if hdl:
        past_keys = past_keys[keep_indices, :, -past_kv_length:, :]
        pk_shape = (past_kv_length, pk_shape[1])
    else:
        past_keys = past_keys[keep_indices, :, :, -past_kv_length:]
        pk_shape = (pk_shape[0], past_kv_length)
    past_values = layer[1].view(batch_size, -1, *pv_shape)[keep_indices, :, -past_kv_length:, :]
    if three_dim_pkv:
        # Ensure tensors are reverted to expected shape post-slicing
        pv_shape = (past_kv_length, pv_shape[1])
        past_keys, past_values = past_keys.reshape(-1, *pk_shape), past_values.reshape(-1, *pv_shape)
    return [past_keys, past_values]


class CausalLM(Model):
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

        super(CausalLM, self).__init__(inference_engine, dtype)

        if self.model.config.pad_token_id is not None:
            self.tokenizer.pad_token_id = self.model.config.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            if self.model.config.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.model.config.eos_token_id
            elif self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Perform a forward pass to determine the structure of the past_key_values
        one_token = torch.tensor([[1]], device=inference_engine.get_device())
        _, past_key_values, _ = self.forward(input_ids=one_token, attention_mask=one_token)
        if torch.is_tensor(past_key_values[0]):
            self.batch_type = CombinedKVCausalLMBatch
        else:
            # check the ordering of the key tensor dimensions
            key_past, value_past = past_key_values[0]
            keys_head_dim_last = key_past.shape[-1] == value_past.shape[-1]
            self.batch_type = CausalLMBatch if keys_head_dim_last else KeysDimTransposedCausalLMBatch

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return self._batch_type

    @batch_type.setter
    def batch_type(self, value):
        self._batch_type = value

    def determine_pkv_types(self) -> Tuple[Type, Type]:
        one_token = torch.tensor([[1]], device=self.device)
        _, pkv, _ = self.forward(
            input_ids=one_token,
            attention_mask=one_token,
        )
        return type(pkv), type(pkv[0])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], int]:
        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids, past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )

        model_inputs["return_dict"] = True

        if position_ids is not None:
            # This can be incorrectly overwritten to None in prepare_inputs_for_generation
            model_inputs["position_ids"] = position_ids

        if inputs_embeds is not None:
            # Add embeddings - if non-None then input_ids should be None
            model_inputs["inputs_embeds"] = inputs_embeds

        # Model Forward
        start_time = time.time_ns()
        outputs = self.model.forward(**model_inputs)
        took_ns = time.time_ns() - start_time
        return outputs.logits, outputs.past_key_values, took_ns

    def generate_token(
        self, batch: CausalLMBatch, first: bool = False, for_concat: bool = False,
    ) -> Tuple[List[TokenInfo], Optional[List[InputTokens]], List[GenerateError], int]:
        # slice the attention mask to the correct shape
        attention_mask = batch.attention_mask[:, : -batch.padding_right_offset]

        logits, past, forward_time_ns = self.forward(
            batch.input_ids, attention_mask, batch.position_ids, batch.past_key_values, batch.inputs_embeds,
        )

        # Heterogeneous next token chooser expects last logits in the sequence
        next_input_ids, next_token_scores, next_token_logprobs = batch.next_token_chooser(
            input_ids=batch.all_input_ids_tensor[:, : -batch.padding_right_offset], scores=logits[:, -1, :]
        )

        # Generated tokens
        generated_tokens: List[TokenInfo] = []
        input_token_infos: List[InputTokens] = [] if first else None
        decode_errors: List[GenerateError] = []

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            logits,
            next_input_ids,
            next_token_scores,
            next_token_logprobs,
            batch.all_input_ids_tensor,
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            logits,
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

                # TODO would be best to vectorize this also
                token_info = get_token_info(request, scores_view, tok_view, logprobs_view)

                # Add to input tokens info list, if requested
                # Applies only to first call for each batch
                if first and request.details.input_toks:
                    input_token_infos.append(
                        get_input_tokens_info(
                            request,
                            all_input_ids[-input_length-batch.padding_right_offset: -batch.padding_right_offset],
                            logits[-input_length:-1, :],
                        )
                    )

                # Add to the tokens to return
                generated_tokens.append(token_info)

            except Exception as e:
                logging.exception(f"token decoding error for request #{request.id}")
                # Add to the errors to return
                decode_errors.append(GenerateError(
                    request_id=request.id, message=f"Token decoding error: {str(e)}"
                ))

            # Adjust input/output lengths for the next request
            batch.input_lengths[i] += 1
            batch.max_remaining_tokens[i] -= 1

        # Update attention_mask with padding as we added a new token to input_ids
        batch.attention_mask[:, -batch.padding_right_offset] = 1
        # Update all_input_ids with the newly generated tokens
        batch.all_input_ids_tensor[:, -batch.padding_right_offset] = next_input_ids

        if first and not for_concat:
            left_pad = batch.attention_mask.shape[1] - batch.padding_right_offset - batch.max_sequence_length
            if left_pad:
                # Trim pre-allocated tensors if we padded to multiple of 8. This
                # is important to be able to generate up to the model's token limit.
                batch.attention_mask = batch.attention_mask[:, left_pad:]
                # For a combined KV cache, past is a list of Tensors, not Tuples
                if torch.is_tensor(past[0]):
                    for cache in past:
                        cache.data = cache.data[..., left_pad:, :]
                else:
                    for key, value in past:
                        key.data = key.data[..., left_pad:, :] if batch.keys_head_dim_last else key.data[..., left_pad:]
                        value.data = value.data[..., left_pad:, :]

        if batch.position_ids is not None:
            batch.position_ids = batch.position_ids[:, -1:] + 1
        batch.input_ids = next_input_ids.view(-1, 1)
        batch.inputs_embeds = None
        batch.past_key_values = past
        batch.max_sequence_length += 1
        batch.padding_right_offset -= 1

        return generated_tokens, input_token_infos, decode_errors, forward_time_ns


class KeysDimTransposedCausalLMBatch(CausalLMBatch):
    @classmethod
    def from_pb(cls, *args, **kwargs) -> Tuple[Optional["CausalLMBatch"], List[GenerateError]]:
        batch, errors = super(KeysDimTransposedCausalLMBatch, cls).from_pb(*args, **kwargs)
        if batch is not None:
            batch.keys_head_dim_last = False
        return batch, errors

class CombinedKVCausalLMBatch(CausalLMBatch):
    @classmethod
    def from_pb(cls, *args, **kwargs) -> Tuple[Optional["CausalLMBatch"], List[GenerateError]]:
        batch, errors = super(CombinedKVCausalLMBatch, cls).from_pb(*args, **kwargs)
        if batch is not None:
            batch.merged_kv_cache = True
        return batch, errors
