import logging
from operator import itemgetter

import torch

from dataclasses import dataclass
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Union, Any

from transformers.modeling_outputs import BaseModelOutput

from text_generation_server.models.model import Model, CUDA_PAD_TO_MULT_OF_8
from text_generation_server.models.types import TokenInfo, Batch, InputTokens, GenerateError
from text_generation_server.pb import generate_pb2
from text_generation_server.prompt_cache import PrefixCache
from text_generation_server.utils.hub import get_model_path
from text_generation_server.utils.tokens import NextTokenChooser, get_token_info, NONES
from text_generation_server.inference_engine import get_inference_engine_class


@dataclass
class Seq2SeqLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]

    # Encoder values
    input_ids: Optional[torch.Tensor]
    inputs_embeds: Optional[torch.Tensor]
    attention_mask: torch.Tensor

    # Decoder values
    decoder_input_ids: Optional[torch.Tensor]
    decoder_inputs_embeds: Optional[torch.Tensor]
    decoder_attention_mask: Optional[torch.Tensor]
    encoder_last_hidden_state: Optional[torch.Tensor]

    # All tokens
    all_decoder_input_ids: List[torch.Tensor]

    # Seq2SeqLM keeps track of both encoder and decoder attention keys and values
    past_key_values: Optional[List[Tuple]]

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    decoder_input_lengths: List[int]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]

    # Metadata used for padding
    max_input_length: int
    max_decoder_input_length: int
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
    ) -> Tuple[Optional["Seq2SeqLMBatch"], List[GenerateError]]:
        """Convert a text_generation_server.v1.Batch protobuf to a Seq2SeqLMBatch"""
        input_texts = []
        next_token_choosers = []
        input_lengths = []
        decoder_input_lengths = []
        max_remaining_tokens = []
        encoder_prefix_ids = {}
        decoder_prefix_ids = {}
        errors = []
        max_input_length = 0
        max_decoder_input_length = 1
        padding_right_offset = 0
        truncate_indices = []

        # Parse batch
        requests = pb.requests
        i = 0
        for r in requests:
            input_length = r.input_length
            max_output_length = r.max_output_length
            decoder_input_length = 1
            if r.prefix_id:
                try:
                    encoder_prefix, decoder_prefix = prefix_cache.get(r.prefix_id)
                except Exception:
                    logging.exception(f"Prefix lookup error for request #{r.id}, prefix id {r.prefix_id}")
                    # Exclude this request from the batch, return error
                    errors.append(GenerateError(
                        request_id=r.id, message=f"Error retrieving prompt prefix '{r.prefix_id}'")
                    )
                    continue
                if encoder_prefix is not None:
                    encoder_prefix_ids[i] = encoder_prefix
                    # Add encoder prefix length to user input length
                    input_length += encoder_prefix.shape[0]
                if decoder_prefix is not None:
                    decoder_prefix_ids[i] = decoder_prefix
                    # Set decoder input length, this will include the BOS token
                    decoder_input_length = decoder_prefix.shape[0]
                    max_decoder_input_length = max(max_decoder_input_length, decoder_input_length)
            input_texts.append(r.inputs)
            input_lengths.append(input_length)
            if r.truncate:
                truncate_indices.append(i)
            decoder_input_lengths.append(decoder_input_length)
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
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]

        if encoder_prefix_ids:
            # Get input embeddings
            inputs_embeds = embeddings_lookup(input_ids)
            for i, ep in encoder_prefix_ids.items():
                input_length = input_lengths[i]
                orig_length = input_length - ep.shape[0]
                # Insert prefix embeddings
                inputs_embeds[i, -input_length:-orig_length] = ep
                # Update attention mask with virtual prefix tokens
                attention_mask[i, -input_length:] = 1
        else:
            inputs_embeds = None

        # Mask out truncated tokens
        # (input_texts aren't truncated, only input_lengths are)
        for i in truncate_indices:
            input_length = input_lengths[i]
            attention_mask[i, :-input_length] = 0
            input_ids[i, :-input_length] = tokenizer.pad_token_id
            if inputs_embeds is not None:
                inputs_embeds[i, :-input_length, :] = 0

        if decoder_prefix_ids:
            # Construct decoder embeddings and attention mask
            start_tok_embedding = prefix_cache.decoder_start_tok_embedding
            decoder_inputs_embeds = start_tok_embedding.new_zeros(
                (batch_size, max_decoder_input_length, start_tok_embedding.shape[1])
            )
            decoder_inputs_embeds[:, -1] = prefix_cache.decoder_start_tok_embedding
            decoder_attention_mask = attention_mask.new_zeros(
                (batch_size, max_decoder_input_length + padding_right_offset)
            )
            decoder_attention_mask[:, -1-padding_right_offset] = 1

            for i, dp in decoder_prefix_ids.items():
                # Update decoder embedding and attention mask
                length = decoder_input_lengths[i]
                decoder_inputs_embeds[i, -length:] = dp
                decoder_attention_mask[i, -length-padding_right_offset:-padding_right_offset] = 1

            # These do not actually get passed to the model
            decoder_input_ids = input_ids.new_zeros((batch_size, max_decoder_input_length))
            decoder_input_ids[:, -1] = tokenizer.bos_token_id
        else:
            decoder_inputs_embeds = None
            decoder_attention_mask = None
            # Each decoder sequence only contains the bos_token
            # so decoder_input_ids is a torch tensor of size [batch_size, 1]
            decoder_input_ids = input_ids.new_full((batch_size, 1), tokenizer.bos_token_id)

        return cls(
            batch_id=pb.id,
            requests=requests,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
            all_decoder_input_ids=list(decoder_input_ids),
            encoder_last_hidden_state=None,
            past_key_values=None,
            input_lengths=input_lengths,
            decoder_input_lengths=decoder_input_lengths,
            max_remaining_tokens=max_remaining_tokens,
            next_token_choosers=next_token_choosers,
            max_input_length=max_input_length,
            max_decoder_input_length=max_decoder_input_length,
            padding_right_offset=padding_right_offset,
        ), errors

    @classmethod
    def concatenate(cls, batches: List["Seq2SeqLMBatch"]) -> "Seq2SeqLMBatch":
        """Concatenate multiple batches together by padding internal torch tensors"""

        # Used for padding
        total_batch_size = 0
        max_input_length = 0
        max_decoder_input_length = 0
        padding_right_offset = 0
        for batch in batches:
            total_batch_size += len(batch)
            max_input_length = max(max_input_length, batch.max_input_length)
            max_decoder_input_length = max(
                max_decoder_input_length, batch.max_decoder_input_length
            )
            padding_right_offset = max(padding_right_offset, batch.padding_right_offset)

        # Batch attributes
        requests = []
        input_lengths = []
        decoder_input_lengths = []
        max_remaining_tokens = []
        next_token_choosers = []
        all_decoder_input_ids = []

        # Batch tensors
        attention_mask = None
        decoder_input_ids = None
        decoder_attention_mask = None
        encoder_last_hidden_state = None
        past_key_values = []

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0

        for i, batch in enumerate(batches):
            # Extend all list attributes
            requests.extend(batch.requests)
            input_lengths.extend(batch.input_lengths)
            decoder_input_lengths.extend(batch.decoder_input_lengths)
            max_remaining_tokens.extend(batch.max_remaining_tokens)
            next_token_choosers.extend(batch.next_token_choosers)
            all_decoder_input_ids.extend(batch.all_decoder_input_ids)

            # Slicing end index for this batch
            end_index = start_index + len(batch)

            # We only concatenate batches that did at least one step
            if batch.encoder_last_hidden_state is None:
                raise ValueError("Batch encoder_last_hidden_state cannot be None")

            # Create padded tensor
            if attention_mask is None:
                attention_mask = batch.attention_mask.new_zeros(
                    (total_batch_size, max_input_length),
                )
            # Copy to correct indices
            attention_mask[
                start_index:end_index, -batch.max_input_length :
            ] = batch.attention_mask[:, -batch.max_input_length :]

            # Create padded tensor
            if decoder_input_ids is None:
                decoder_input_ids = batch.decoder_input_ids.new_zeros((total_batch_size, 1))
            # Copy to correct indices
            decoder_input_ids[start_index:end_index] = batch.decoder_input_ids

            # Create padded tensor
            if decoder_attention_mask is None:
                # As decoder_attention_mask might not exist, we use `batch.attention_mask` for device here
                decoder_attention_mask = batch.attention_mask.new_zeros(
                    (total_batch_size, max_decoder_input_length + padding_right_offset),
                )
            # If the decoder mask does not exist yet, all generations started at the same time and we never concatenated
            # this batch. All generations are of length `batch.max_decoder_input_length`.
            left_offset = max_decoder_input_length - batch.max_decoder_input_length
            if batch.decoder_attention_mask is None:
                decoder_attention_mask[start_index:end_index, left_offset:-padding_right_offset] = 1
            # If it exists, we need to index
            else:
                batch_left_offset = (
                    batch.decoder_attention_mask.shape[1]
                    - batch.max_decoder_input_length - batch.padding_right_offset
                )
                decoder_attention_mask[
                    start_index:end_index, left_offset:-padding_right_offset,
                ] = batch.decoder_attention_mask[
                    :, batch_left_offset : -batch.padding_right_offset,
                ]

            # Create padded tensor
            if encoder_last_hidden_state is None:
                encoder_last_hidden_state = batch.encoder_last_hidden_state.new_zeros(
                    (
                        total_batch_size,
                        max_input_length,
                        batch.encoder_last_hidden_state.shape[-1],
                    ),
                )

            # Copy to correct indices
            encoder_last_hidden_state[
                start_index:end_index, -batch.max_input_length :, :
            ] = batch.encoder_last_hidden_state[:, -batch.max_input_length :, :]
            batch.encoder_last_hidden_state = None

            # Ensure that we can update tensors in-place
            if type(batch.past_key_values[0]) == tuple:
                batch.past_key_values = [[t for t in layer] for layer in batch.past_key_values]

            start_index = end_index

        # Determine shapes for new past kv tensors
        first_past_kvs = batches[0].past_key_values
        _, num_heads, _, head_dim = first_past_kvs[0][1].shape
        first_dims = total_batch_size, num_heads

        padded_dec_values_shape = (*first_dims, max_decoder_input_length - 1,  head_dim)
        padded_enc_values_shape = (*first_dims, max_input_length, head_dim)

        if batches[0].keys_head_dim_last:
            padded_dec_keys_shape = padded_dec_values_shape
            padded_enc_keys_shape = padded_enc_values_shape
        else:
            padded_dec_keys_shape = (*first_dims, head_dim, max_decoder_input_length - 1)
            padded_enc_keys_shape = (*first_dims, head_dim, max_input_length)

        padded_shapes = (
            padded_dec_keys_shape,
            padded_dec_values_shape,
            padded_enc_keys_shape,
            padded_enc_values_shape,
        )

        # Iterate over attention layers
        for j in range(len(first_past_kvs)):
            past_key_values.append([])

            # Decoder and encoder past
            for k in range(4):
                # Initialize tensors
                padded_past_values = first_past_kvs[j][k].new_zeros(padded_shapes[k])
                past_key_values[j].append(padded_past_values)

                start_index = 0
                for batch in batches:
                    t = batch.past_key_values[j][k]
                    # Clear reference to the original tensor
                    batch.past_key_values[j][k] = None
                    # Slicing end index for this batch
                    end_index = start_index + len(batch)
                    # We slice the past keys and values to remove the padding from previous batches
                    past_seq_len = batch.max_decoder_input_length - 1 if k < 2 else batch.max_input_length
                    if batch.keys_head_dim_last or k == 1 or k == 3:
                        padded_past_values[
                            start_index:end_index, :, -past_seq_len:, :
                        ] = t[:, :, -past_seq_len:, :]
                    else:
                        padded_past_values[
                            start_index:end_index, :, :, -past_seq_len:
                        ] = t[:, :, :, -past_seq_len:]
                    del t

                    start_index = end_index

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            input_ids=None,
            inputs_embeds=None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=None,
            decoder_attention_mask=decoder_attention_mask,
            encoder_last_hidden_state=encoder_last_hidden_state,
            all_decoder_input_ids=all_decoder_input_ids,
            past_key_values=past_key_values,
            input_lengths=input_lengths,
            decoder_input_lengths=decoder_input_lengths,
            max_remaining_tokens=max_remaining_tokens,
            next_token_choosers=next_token_choosers,
            max_input_length=max_input_length,
            max_decoder_input_length=max_decoder_input_length,
            padding_right_offset=padding_right_offset,
            keys_head_dim_last=batches[0].keys_head_dim_last,
        )

    @classmethod
    def prune(cls, batch: "Seq2SeqLMBatch", completed_ids: List[int]) -> Optional["Seq2SeqLMBatch"]:
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

        #TODO maybe a single loop for all these list slices
        slice_list = itemgetter(*keep_indices) if new_size > 1 else lambda l: (l[keep_indices[0]],)
        batch.input_lengths = slice_list(batch.input_lengths)
        batch.decoder_input_lengths = list(slice_list(batch.decoder_input_lengths))
        batch.max_remaining_tokens = list(slice_list(batch.max_remaining_tokens))
        batch.requests = slice_list(batch.requests)
        batch.next_token_choosers = slice_list(batch.next_token_choosers)
        batch.all_decoder_input_ids = list(slice_list(batch.all_decoder_input_ids))

        batch.max_input_length = max(batch.input_lengths)
        batch.max_decoder_input_length = max(batch.decoder_input_lengths)
        new_padding_right_offset = max(batch.max_remaining_tokens)

        batch.decoder_attention_mask = batch.decoder_attention_mask[
           keep_indices,
           -(batch.padding_right_offset + batch.max_decoder_input_length)
           :(batch.decoder_attention_mask.shape[1]-batch.padding_right_offset)+new_padding_right_offset,
        ] if batch.decoder_attention_mask is not None else None

        batch.encoder_last_hidden_state = batch.encoder_last_hidden_state[
            keep_indices, -batch.max_input_length:
        ] if batch.encoder_last_hidden_state is not None else None

        batch.input_ids = batch.input_ids[keep_indices] if batch.input_ids is not None else None
        batch.attention_mask = batch.attention_mask[keep_indices, -batch.max_input_length:]
        batch.decoder_input_ids = batch.decoder_input_ids[keep_indices]

        # Ensure that past_key_values tensors can be updated in-place
        if type(batch.past_key_values[0]) == tuple:
            batch.past_key_values = [list(layer) for layer in batch.past_key_values]

        decoder_past_seq_len = batch.max_decoder_input_length - 1
        for layer in batch.past_key_values:
            # Decoder and encoder keys
            if batch.keys_head_dim_last:
                layer[0] = layer[0][keep_indices, :, -decoder_past_seq_len:, :]
                layer[2] = layer[2][keep_indices, :, -batch.max_input_length:, :]
            else:
                layer[0] = layer[0][keep_indices, :, :, -decoder_past_seq_len:]
                layer[2] = layer[2][keep_indices, :, :, -batch.max_input_length:]
            # Decoder and encoder values
            layer[1] = layer[1][keep_indices, :, -decoder_past_seq_len:, :]
            layer[3] = layer[3][keep_indices, :, -batch.max_input_length:, :]

        batch.padding_right_offset = new_padding_right_offset

        return batch


class Seq2SeqLM(Model):
    def __init__(
        self, model_name: str, revision: str, deployment_framework: str, dtype: torch.dtype, model_config: Union[Any] = None,
    ):
        model_path = get_model_path(model_name, revision)

        inference_engine = get_inference_engine_class(deployment_framework)(
            model_path, AutoModelForSeq2SeqLM, dtype, model_config,
        )
        super(Seq2SeqLM, self).__init__(inference_engine, dtype)

        bos_token_id = self.model.config.decoder_start_token_id
        if bos_token_id is None:
            bos_token_id = self.model.config.bos_token_id
        if bos_token_id is not None:
            self.tokenizer.bos_token_id = bos_token_id

        # Perform a forward pass to determine the ordering of past key attention tensor dimensions
        one_token = torch.tensor([[bos_token_id]], device=inference_engine.get_device())
        _, _, past_key_values = self.forward(
            input_ids=one_token,
            attention_mask=torch.ones_like(one_token),
            decoder_input_ids=one_token,
        )
        key_past, value_past, _, _ = past_key_values[0]
        keys_head_dim_last = key_past.shape[-1] == value_past.shape[-1]
        self.batch_type = Seq2SeqLMBatch if keys_head_dim_last else KeysDimTransposedSeq2SeqLMBatch

    @property
    def batch_type(self) -> Type[Seq2SeqLMBatch]:
        return self._batch_type

    @batch_type.setter
    def batch_type(self, value):
        self._batch_type = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        if inputs_embeds is not None:
            input_ids = None
        if decoder_inputs_embeds is not None:
            decoder_input_ids = None

        outputs = self.model.forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return (
            self.engine.process_logits(outputs.logits),
            outputs.encoder_last_hidden_state,
            outputs.past_key_values,
        )

    def generate_token(
        self, batch: Seq2SeqLMBatch, first: bool = False, for_concat: bool = False,
    ) -> Tuple[List[TokenInfo], Optional[List[InputTokens]], List[GenerateError]]:
        # slice to the correct shape
        decoder_attention_mask = None if batch.decoder_attention_mask is None \
            else batch.decoder_attention_mask[:, : -batch.padding_right_offset]

        encoder_outputs = None if batch.encoder_last_hidden_state is None \
            else BaseModelOutput(last_hidden_state=batch.encoder_last_hidden_state)

        logits, encoder_last_hidden_state, past = self.forward(
            batch.input_ids,
            batch.attention_mask,
            batch.decoder_input_ids,
            decoder_attention_mask,
            encoder_outputs,
            batch.past_key_values,
            batch.inputs_embeds,
            batch.decoder_inputs_embeds,
        )

        # New values for next forward
        next_batch_decoder_input_ids = []

        # Generated tokens
        generated_tokens: List[TokenInfo] = []
        input_token_infos: List[InputTokens] = [] if first else None
        decode_errors: List[GenerateError] = []

        # Zipped iterator
        iterator = zip(
            batch.requests,
            logits,
            batch.next_token_choosers,
            batch.all_decoder_input_ids,
            batch.decoder_input_lengths,
            batch.input_ids if first else NONES,
            batch.input_lengths if first else NONES,
        )

        # For each member of the batch
        for i, (
            request,
            logits,
            next_token_chooser,
            all_decoder_input_ids,
            decoder_input_length,
            input_ids,
            input_length,
        ) in enumerate(iterator):
            try:
                # Select next token
                next_token, scores, logprobs = next_token_chooser(
                    all_decoder_input_ids[None, -decoder_input_length:], logits
                )

                # Return latest token
                token_info = get_token_info(request, scores, next_token, logprobs)

                # Return input tokens if requested
                # Note this could all be handled in the router for seq2seq models
                # but we use the same path as causal_lm to keep the logic consistent
                if first and request.details.input_toks:
                    logprob = float('nan') if request.details.logprobs else 0.0
                    input_token_infos.append(InputTokens(
                        request_id=request.id,
                        tokens=[
                            TokenInfo(token_id=int(tid), logprob=logprob)
                            for tid in input_ids[-input_length:]
                        ],
                    ))

                # Add to the tokens to return
                generated_tokens.append(token_info)

            except Exception as e:
                logging.exception(f"token decoding error for request #{request.id}")
                next_token = all_decoder_input_ids.new_tensor([self.tokenizer.pad_token_id])
                # Add to the errors to return
                decode_errors.append(GenerateError(
                    request_id=request.id, message=f"Token decoding error: {str(e)}"
                ))

            # Append next token to decoder tokens
            next_batch_decoder_input_ids.append(next_token)
            batch.all_decoder_input_ids[i] = torch.cat([all_decoder_input_ids, next_token])
            batch.decoder_input_lengths[i] += 1
            batch.max_remaining_tokens[i] -= 1

        # Update decoder_attention_mask as we added a new token to input_ids
        if batch.decoder_attention_mask is not None:
            batch.decoder_attention_mask[:, -batch.padding_right_offset] = 1

        batch.input_ids = None
        batch.inputs_embeds = None
        batch.decoder_input_ids = torch.cat(next_batch_decoder_input_ids).view(-1, 1)
        batch.decoder_inputs_embeds = None
        batch.encoder_last_hidden_state = encoder_last_hidden_state
        batch.past_key_values = past
        batch.max_decoder_input_length += 1
        batch.padding_right_offset -= 1

        return generated_tokens, input_token_infos, decode_errors


class KeysDimTransposedSeq2SeqLMBatch(Seq2SeqLMBatch):
    @classmethod
    def from_pb(cls, *args, **kwargs) -> Tuple[Optional["Seq2SeqLMBatch"], List[GenerateError]]:
        batch, errors = super(KeysDimTransposedSeq2SeqLMBatch, cls).from_pb(*args, **kwargs)
        if batch is not None:
            batch.keys_head_dim_last = False
        return batch, errors
