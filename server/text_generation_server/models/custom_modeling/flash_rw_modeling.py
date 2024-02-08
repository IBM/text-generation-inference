import torch
import torch.distributed

from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from typing import Optional

from text_generation_server.utils.flash_attn import attention
from text_generation_server.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelHead,
    FastLayerNorm,
    PositionRotaryEmbedding,
    get_linear,
)


def load_row(config, prefix: str, weights, bias: bool):
    weight = weights.get_multi_weights_row(prefix, quantize=config.quantize)

    if bias and weights.process_group.rank() == 0:
        # Rank is only on the first rank process
        bias = weights.get_tensor(f"{prefix}.bias")
    else:
        bias = None

    linear = get_linear(weight, bias, config.quantize)
    if config.parallel_attn:
        return linear
    else:
        return TensorParallelRowLinear(linear, process_group=weights.process_group)


class RWConfig(PretrainedConfig):
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }

    def __init__(
        self,
        model_type="RefinedWeb",
        vocab_size=250880,
        hidden_size=64,
        num_hidden_layers=None,
        num_attention_heads=None,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_kv_heads=None,
        multi_query=False,
        alibi=False,
        new_decoder_architecture=None,
        bias=False,
        parallel_attn=False,
        **kwargs,
    ):
        if alibi:
            raise NotImplementedError(
                "alibi is not supported by this version of the model"
            )

        self.model_type = model_type
        self.alibi = False
        self.rotary = True

        self.vocab_size = vocab_size
        # Backward compatibility with n_embed kwarg
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = (
            num_hidden_layers
            if num_hidden_layers is not None
            else kwargs.pop("n_layer", 2)
        )
        self.n_head = (
            num_attention_heads
            if num_attention_heads is not None
            else kwargs.pop("n_head", 8)
        )
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bias = bias
        self.parallel_attn = parallel_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        if num_kv_heads is not None:
            self.n_head_kv = num_kv_heads
        else:
            old_n_head_kv = kwargs.pop("n_head_kv", None)
            if old_n_head_kv is not None:
                self.n_head_kv = old_n_head_kv
            else:
                self.n_head_kv = 1 if multi_query else self.n_head

        if new_decoder_architecture is not None:
            self.new_decoder_architecture = new_decoder_architecture
        else:
            self.new_decoder_architecture = (model_type == "RefinedWeb")

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class FlashRWAttention(torch.nn.Module):
    def __init__(
        self,
        config,
        prefix,
        weights,
    ):
        super().__init__()
        self.num_heads = config.n_head
        self.num_heads_kv = config.n_head_kv
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(
            dim=self.head_size, base=10000.0, device=weights.device
        )
        self.softmax_scale = self.head_size ** (-0.5)

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()

        self.query_key_value = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.query_key_value",
            weights=weights,
            bias=config.bias,
        )
        self.dense = load_row(
            config, prefix=f"{prefix}.dense", weights=weights, bias=config.bias
        )

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        qkv = self.query_key_value(hidden_states)

        # Split query from key_value
        query, kv = qkv.split(
            [self.head_size * self.num_heads, 2 * self.head_size * self.num_heads_kv],
            dim=1,
        )

        # Prepare query and key_value for indexing
        query = query.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_heads_kv, self.head_size)

        # Inplace rotary
        self.rotary_emb(query, cos, sin)
        self.rotary_emb(kv[:, 0], cos, sin)

        # Prefill
        if layer_past_present_indices is None:
            # Copy to layer past
            layer_past[...] = kv

            # flash attention
            attn_output = attention(
                query,
                torch.select(kv, dim=1, index=0),
                torch.select(kv, dim=1, index=1),
                cu_seqlens,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
            # Add present to the layer_past tensor at the correct indices
            layer_past[layer_past_present_indices] = kv

            # flash attention
            attn_output = attention(
                query,
                layer_past[:, 0],
                layer_past[:, 1],
                cu_seqlens,
                max_s,
                self.softmax_scale,
                cu_seqlens_q,
                1,
                False,
            )

        return self.dense(attn_output.reshape(-1, self.num_heads * self.head_size))


class FlashRWLargeAttention(torch.nn.Module):
    def __init__(
        self,
        config,
        prefix,
        weights,
    ):
        super().__init__()

        hidden_size = config.hidden_size
        num_heads = config.n_head
        num_groups = config.n_head_kv

        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.num_groups = num_groups
        self.num_heads = num_heads // self.num_groups

        self.rotary_emb = PositionRotaryEmbedding.static(
            self.head_size, base=10000.0, device=weights.device
        )
        self.softmax_scale = self.head_size ** (-0.5)

        process_group = weights.process_group

        if process_group.size() > self.num_groups:
            raise NotImplementedError(
                "Tensor Parallelism is not implemented for world_size > n groups"
            )
        if self.num_groups % process_group.size() != 0:
            raise NotImplementedError(
                f"Tensor Parallelism is not implemented for {self.num_groups} not divisible by {process_group.size()}"
            )
        self.num_groups = self.num_groups // process_group.size()

        self.query_key_value = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.query_key_value",
            weights=weights,
            bias=config.bias,
        )
        self.dense = load_row(
            config, prefix=f"{prefix}.dense", weights=weights, bias=config.bias
        )

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(-1, self.num_groups, self.num_heads + 2, self.head_size)

        # Split on group dimension
        query, kv = qkv.split(
            [self.num_heads, 2],
            dim=2,
        )
        # Merge groups and heads
        query = query.reshape(-1, self.num_groups * self.num_heads, self.head_size)

        # Inplace rotary
        self.rotary_emb(query, cos, sin)
        self.rotary_emb(kv[:, :, 0], cos, sin)

        # Prefill
        if layer_past_present_indices is None:
            # Copy to layer past
            layer_past[...] = kv

            # flash attention
            attn_output = attention(
                query,
                torch.select(kv, dim=2, index=0),
                torch.select(kv, dim=2, index=1),
                cu_seqlens,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
            # Add present to the layer_past tensor at the correct indices
            layer_past[layer_past_present_indices] = kv

            # flash attention
            attn_output = attention(
                query,
                layer_past[:, :, 0],
                layer_past[:, :, 1],
                cu_seqlens,
                max_s,
                self.softmax_scale,
                cu_seqlens_q,
                1,
                False,
            )

        return self.dense(
            attn_output.view(-1, self.num_groups * self.num_heads * self.head_size)
        )


class FlashMLP(nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        self.act = torch.nn.functional.gelu

        self.dense_h_to_4h = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.dense_h_to_4h", weights=weights, bias=config.bias
        )
        self.dense_4h_to_h = load_row(
            config, prefix=f"{prefix}.dense_4h_to_h", weights=weights, bias=config.bias
        )

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class FlashRWLayer(nn.Module):
    def __init__(
        self,
        layer_id,
        config,
        weights,
    ):
        super().__init__()

        parallel_attn = config.parallel_attn
        self.parallel_attn = parallel_attn

        prefix = f"transformer.h.{layer_id}"

        self.input_layernorm = FastLayerNorm.load(
            prefix=f"{prefix}.input_layernorm",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )
        self.self_attention = FlashRWAttention(
            config,
            prefix=f"{prefix}.self_attention",
            weights=weights,
        )
        self.post_attention_layernorm = (
            FastLayerNorm.load(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )
            if not parallel_attn
            else None
        )

        self.mlp = FlashMLP(
            config,
            prefix=f"{prefix}.mlp",
            weights=weights,
        )

        self.process_group = weights.process_group

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        if self.parallel_attn:
            ln_hidden_states, residual = self.input_layernorm(hidden_states, residual)

            attn_output = self.self_attention(
                ln_hidden_states,
                cos,
                sin,
                cu_seqlens,
                max_s,
                layer_past,
                layer_past_present_indices,
                cu_seqlens_q,
            )

            mlp_output = self.mlp(ln_hidden_states)
            intermediate = mlp_output + attn_output

            if self.process_group.size() > 1:
                torch.distributed.all_reduce(intermediate, group=self.process_group)

            return intermediate, residual
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.self_attention(
                hidden_states,
                cos,
                sin,
                cu_seqlens,
                max_s,
                layer_past,
                layer_past_present_indices,
                cu_seqlens_q,
            )

            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

            mlp_output = self.mlp(hidden_states)

            return mlp_output, residual


class FlashRWLargeLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"transformer.h.{layer_id}"
        self.ln_attn = FastLayerNorm.load(
            prefix=f"{prefix}.ln_attn",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )
        self.ln_mlp = FastLayerNorm.load(
            prefix=f"{prefix}.ln_mlp",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )

        self.self_attention = FlashRWLargeAttention(
            config,
            prefix=f"{prefix}.self_attention",
            weights=weights,
        )
        assert config.parallel_attn, "This version doesn't support non parallel_attn"

        self.mlp = FlashMLP(config, prefix=f"{prefix}.mlp", weights=weights)

        self.process_group = weights.process_group

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        ln_attn, residual = self.ln_attn(hidden_states, residual)
        ln_mlp, _ = self.ln_mlp(residual)

        # Self attention.
        attn_output = self.self_attention(
            ln_attn,
            cos,
            sin,
            cu_seqlens,
            max_s,
            layer_past,
            layer_past_present_indices,
            cu_seqlens_q,
        )

        # MLP.
        mlp_output = self.mlp(ln_mlp)

        intermediate = attn_output + mlp_output

        if self.process_group.size() > 1:
            torch.distributed.all_reduce(intermediate, group=self.process_group)

        return intermediate, residual


class FlashRWPreTrainedModel(PreTrainedModel):
    config_class = RWConfig


class FlashRWModel(FlashRWPreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.config = config

        self.word_embeddings = TensorParallelEmbedding(
            prefix="transformer.word_embeddings", weights=weights
        )
        if config.new_decoder_architecture:  # "RefinedWeb"
            self.h = nn.ModuleList(
                [
                    FlashRWLargeLayer(layer_id, config, weights)
                    for layer_id in range(config.num_hidden_layers)
                ]
            )
            self.cache_size = (
                self.h[0].self_attention.num_groups,
                2,
                self.h[0].self_attention.head_size,
            )
        else:  # "RefinedWebModel"
            self.h = nn.ModuleList(
                [
                    FlashRWLayer(layer_id, config, weights)
                    for layer_id in range(config.num_hidden_layers)
                ]
            )
            self.cache_size = (
                2,
                self.h[0].self_attention.num_heads_kv,
                self.h[0].self_attention.head_size,
            )

        self.ln_f = FastLayerNorm.load(
            prefix="transformer.ln_f",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )

        self.head_size = self.h[0].self_attention.head_size

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        cu_seqlens_q,
        max_s,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values=None,
        pre_allocate_past_size: Optional[int] = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.word_embeddings(input_ids)

        # Prefill
        if past_key_values is None:
            # Create past tensor
            past_key_values = hidden_states.new_empty(
                (
                    len(self.h),
                    len(hidden_states)
                    if pre_allocate_past_size is None
                    else pre_allocate_past_size,
                    *self.cache_size,
                )
            )
            layer_past_present_indices = None
            slice_past_index = len(hidden_states)
        # Decode
        else:
            # Create indices from cumulative sequence lengths
            layer_past_present_indices = cu_seqlens[1:] - 1
            slice_past_index = None

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.h[0].self_attention.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
        )

        residual = None
        for i, layer in enumerate(self.h):
            # We added padding that we now need to slice
            layer_past_key_values = (
                past_key_values[i]
                if slice_past_index is None
                else past_key_values[i, :slice_past_index]
            )

            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlens,
                max_s,
                layer_past_key_values,
                layer_past_present_indices,
                cu_seqlens_q,
            )

        hidden_states, _ = self.ln_f(hidden_states, residual)

        return hidden_states, past_key_values


class FlashRWForCausalLM(FlashRWPreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)

        self.transformer = FlashRWModel(config, weights)

        self.lm_head = TensorParallelHead.load(
            config, prefix="lm_head", weights=weights
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer.word_embeddings

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        cu_seqlens_q,
        max_s,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        pre_allocate_past_size: Optional[int] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
    ):
        hidden_states, present = self.transformer(
            input_ids,
            position_ids,
            cu_seqlens,
            cu_seqlens_q,
            max_s,
            inputs_embeds,
            past_key_values,
            pre_allocate_past_size,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states)
        return logits, present
