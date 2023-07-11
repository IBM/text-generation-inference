import logging
from operator import itemgetter

import torch

from dataclasses import dataclass
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Union, Any

from text_generation_server.models.model import Model, CUDA_PAD_TO_MULT_OF_8
from text_generation_server.models.types import TokenInfo, Batch, InputTokens, GenerateError
from text_generation_server.pb import generate_pb2
from text_generation_server.prompt_cache import PrefixCache
from text_generation_server.utils.hub import get_model_path
from text_generation_server.utils.tokens import NextTokenChooser, get_token_info, get_input_tokens_info
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
    all_input_ids: List[torch.Tensor]

    # Lengths of all generations present in the batch
    input_lengths: List[int]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]

    # Metadata used for padding
    max_sequence_length: int
    padding_right_offset: int
    max_remaining_tokens: List[int]

    # Past metadata
    keys_head_dim_last: bool = True

    def get_id(self) -> int:
        return self.batch_id

    def __len__(self):
        return len(self.requests)

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        embeddings_lookup: Optional,
        prefix_cache: Optional[PrefixCache],
        use_position_ids: bool = False,
    ) -> Tuple[Optional["CausalLMBatch"], List[GenerateError]]:
        """Convert a text_generation.v1.Batch protobuf to a CausalLMBatch"""
        input_texts = []
        next_token_choosers = []
        input_lengths = []
        max_remaining_tokens = []
        prefix_ids = {}
        errors = []
        max_input_length = 0
        padding_right_offset = 0
        truncate_indices = []

        # Parse batch
        requests = pb.requests
        i = 0
        for r in requests:
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
            next_token_choosers.append(NextTokenChooser.from_pb(
                r.parameters, r.details.logprobs, tokenizer, device,
            ))
            i += 1

        if errors:
            requests = [r for r in pb.requests if not any(r.id == er.request_id for er in errors)]
            if not requests:
                return None, errors

        batch_size = len(requests)

        # Tokenize batch
        tokenize_length = max_input_length
        # Pad to multiple of 8 for tensor core GPUs
        if device.type == "cuda" and CUDA_PAD_TO_MULT_OF_8 and (mod := tokenize_length % 8) != 0:
            tokenize_length += 8 - mod
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

        # Mask out truncated tokens
        # (input_texts aren't truncated, only input_lengths are)
        for i in truncate_indices:
            input_length = input_lengths[i]
            attention_mask[i, :-input_length-padding_right_offset] = 0
            if inputs_embeds is not None:
                inputs_embeds[i, :-input_length, :] = 0
            else:
                input_ids[i, :-input_length] = tokenizer.pad_token_id

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
            all_input_ids=list(all_input_ids[:, -max_input_length:]),
            input_lengths=input_lengths,
            max_remaining_tokens=max_remaining_tokens,
            next_token_choosers=next_token_choosers,
            max_sequence_length=max_input_length,
            padding_right_offset=padding_right_offset,
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
        all_input_ids = []
        next_token_choosers = []
        max_remaining_tokens = []

        # Batch tensors
        input_ids = None
        attention_mask = None
        position_ids = None
        past_key_values = []

        # Check the past keys/values shape so that we can revert to that
        # shape after the convenience slicing if necessary
        three_dim_pkvs = batches[0].past_key_values is not None \
            and len(batches[0].past_key_values[0][0].shape) == 3

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0
        for i, batch in enumerate(batches):
            requests.extend(batch.requests)
            input_lengths.extend(batch.input_lengths)
            all_input_ids.extend(batch.all_input_ids)
            next_token_choosers.extend(batch.next_token_choosers)
            max_remaining_tokens.extend(batch.max_remaining_tokens)

            # Slicing end index for this batch
            end_index = start_index + len(batch)

            # We only concatenate batches that did at least one step
            if batch.past_key_values is None:
                raise ValueError("only concatenate prefilled batches")

            # Create empty tensor
            # input_ids is always of shape [batch_size, 1]
            # We do not need to pad it
            if input_ids is None:
                input_ids = batch.input_ids.new_empty((total_batch_size, 1))
            # Copy to correct indices
            input_ids[start_index:end_index] = batch.input_ids

            # Create padded tensor
            if attention_mask is None:
                attention_mask = batch.attention_mask.new_zeros(
                    (total_batch_size, max_sequence_length + padding_right_offset),
                )

            # We need to slice the attention mask to remove padding from previous steps
            # and to remove unused allocated space
            left_offset = max_sequence_length - batch.max_sequence_length
            batch_left_offset = (
                    batch.attention_mask.shape[1] - batch.max_sequence_length - batch.padding_right_offset
            )
            attention_mask[
                start_index:end_index, left_offset:-padding_right_offset,
            ] = batch.attention_mask[
                :, batch_left_offset : -batch.padding_right_offset,
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

        # Shenanigans to get dimensions because BLOOM outputs a past with a different shape
        # BLOOM Keys:   [batch_size * num_heads, head_dim, seq_length]
        # BLOOM Values: [batch_size * num_heads, seq_length, head_dim]
        first_past_kvs = batches[0].past_key_values
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
            all_input_ids=all_input_ids,
            input_lengths=input_lengths,
            next_token_choosers=next_token_choosers,
            max_remaining_tokens=max_remaining_tokens,
            max_sequence_length=max_sequence_length,
            padding_right_offset=padding_right_offset,
            keys_head_dim_last=batches[0].keys_head_dim_last,
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
        three_dim_pkvs = len(batch.past_key_values[0][0].shape) == 3

        #TODO maybe a single loop for all these list slices
        slice_list = itemgetter(*keep_indices) if new_size > 1 else lambda l: (l[keep_indices[0]],)
        batch.input_lengths = list(slice_list(batch.input_lengths))
        batch.max_remaining_tokens = list(slice_list(batch.max_remaining_tokens))
        batch.requests = slice_list(batch.requests)
        batch.all_input_ids = list(slice_list(batch.all_input_ids))
        batch.next_token_choosers = slice_list(batch.next_token_choosers)

        batch.max_sequence_length = max(batch.input_lengths)
        new_padding_right_offset = max(batch.max_remaining_tokens)

        past_kv_length = batch.max_sequence_length - 1

        # Ensure that we can update layers in-place
        if type(batch.past_key_values) == tuple:
            batch.past_key_values = list(batch.past_key_values)

        for i, layer in enumerate(batch.past_key_values):
            batch.past_key_values[i] = update_layer(
                layer, size_before, keep_indices, past_kv_length, batch.keys_head_dim_last, three_dim_pkvs
            )

        batch.input_ids = batch.input_ids[keep_indices]
        batch.attention_mask = batch.attention_mask[
            keep_indices,
            -batch.padding_right_offset-batch.max_sequence_length:
            (batch.attention_mask.shape[1]-batch.padding_right_offset)+new_padding_right_offset,
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
        self, model_name: str, revision: str, deployment_framework: str, dtype: torch.dtype, model_config: Union[Any] = None,
    ):
        model_path = get_model_path(model_name, revision)

        inference_engine = get_inference_engine_class(deployment_framework)(
            model_path, AutoModelForCausalLM, dtype, model_config,
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

        # Perform a forward pass to determine the ordering of past key attention tensor dimensions
        one_token = torch.tensor([[1]], device=inference_engine.get_device())
        _, past_key_values = self.forward(input_ids=one_token, attention_mask=one_token)
        key_past, value_past = past_key_values[0]
        keys_head_dim_last = key_past.shape[-1] == value_past.shape[-1]
        self.batch_type = CausalLMBatch if keys_head_dim_last else KeysDimTransposedCausalLMBatch

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return self._batch_type

    @batch_type.setter
    def batch_type(self, value):
        self._batch_type = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
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
        outputs = self.model.forward(**model_inputs)
        return (
            self.engine.process_logits(outputs.logits),
            outputs.past_key_values,
        )

    def generate_token(
        self, batch: CausalLMBatch, first: bool = False, for_concat: bool = False,
    ) -> Tuple[List[TokenInfo], Optional[List[InputTokens]], List[GenerateError]]:
        # slice the attention mask to the correct shape
        attention_mask = batch.attention_mask[:, : -batch.padding_right_offset]

        logits, past = self.forward(
            batch.input_ids, attention_mask, batch.position_ids, batch.past_key_values, batch.inputs_embeds,
        )

        # New values for next forward
        next_batch_input_ids = []

        # Generated tokens
        generated_tokens: List[TokenInfo] = []
        input_token_infos: List[InputTokens] = [] if first else None
        decode_errors: List[GenerateError] = []

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            logits,
            batch.next_token_choosers,
            batch.all_input_ids,
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            logits,
            next_token_chooser,
            all_input_ids,
        ) in enumerate(iterator):
            try:
                # Select next token
                next_token, scores, logprobs = next_token_chooser(
                    all_input_ids.unsqueeze(0), logits[-1:, :]
                )

                # Get next token info
                token_info = get_token_info(request, scores, next_token, logprobs)

                # Add to input tokens info list, if requested
                # Applies only to first call for each batch
                if first and request.details.input_toks:
                    input_token_infos.append(
                        get_input_tokens_info(
                            request,
                            all_input_ids[-input_length:],
                            logits[-input_length:-1, :],
                        )
                    )

                # Add to the tokens to return
                generated_tokens.append(token_info)

            except Exception as e:
                logging.exception(f"token decoding error for request #{request.id}")
                next_token = all_input_ids.new_tensor([self.tokenizer.pad_token_id])
                # Add to the errors to return
                decode_errors.append(GenerateError(
                    request_id=request.id, message=f"Token decoding error: {str(e)}"
                ))

            # Add to the next batch
            next_batch_input_ids.append(next_token)
            batch.all_input_ids[i] = torch.cat([all_input_ids, next_token])
            batch.input_lengths[i] += 1
            batch.max_remaining_tokens[i] -= 1

        # Update attention_mask with padding as we added a new token to input_ids
        batch.attention_mask[:, -batch.padding_right_offset] = 1

        if first and not for_concat:
            left_pad = batch.attention_mask.shape[1] - batch.padding_right_offset - batch.max_sequence_length
            if left_pad:
                # Trim attention mask and past kvs if we padded to multiple of 8. This is important to be able to
                # generate up to the model's token limit.
                batch.attention_mask = batch.attention_mask[:, left_pad:]
                for key, value in past:
                    key.data = key.data[..., left_pad:, :] if batch.keys_head_dim_last else key.data[..., left_pad:]
                    value.data = value.data[..., left_pad:, :]

        # Update position ids
        if batch.position_ids is not None:
            batch.position_ids = batch.position_ids[:, -1:] + 1
        batch.input_ids = torch.cat(next_batch_input_ids).view(len(batch), 1)
        batch.inputs_embeds = None
        batch.past_key_values = past
        batch.max_sequence_length += 1
        batch.padding_right_offset -= 1

        return generated_tokens, input_token_infos, decode_errors


class KeysDimTransposedCausalLMBatch(CausalLMBatch):
    @classmethod
    def from_pb(cls, *args, **kwargs) -> Tuple[Optional["CausalLMBatch"], List[GenerateError]]:
        batch, errors = super(KeysDimTransposedCausalLMBatch, cls).from_pb(*args, **kwargs)
        if batch is not None:
            batch.keys_head_dim_last = False
        return batch, errors
