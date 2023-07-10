# port of models described in RW
# We use the bloom model as a starting point for these model.
# Please refer to the bloom models for usage instructions.

#import paddle
import paddle
#from paddle import nn
from paddle import Tensor, nn

from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from paddle.nn import functional as F

import math
import warnings
from typing import Optional, Tuple, Union, List

from .. import PretrainedModel
from ...utils.converter import StateDictNameMapping, init_name_mappings
from paddlenlp.utils.log import logger
#from transformers.utils import logging

from .configuration import (
    RWConfig,
    RW_PRETRAINED_INIT_CONFIGURATION,
)


# NOTE(Hesslow): Unfortunately we did not fuse matmul and bias during training, this means that there's one additional quantization to bfloat16 between the operations.
# In order not to degrade the quality of our HF-port, we keep these characteristics in the final model.
class Linear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        ret = input @ self.weight
        if self.bias is None:
            return ret
        else:
            return ret + self.bias


from einops import rearrange

# rotary pos emb helpers (paddle.jit.script does not seem to support staticmethod...)
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=x1.ndim - 1)  # dim=-1 triggers a bug in paddle < 1.8.0


class RotaryEmbedding(paddle.nn.Layer):
    """Implementation of RotaryEmbedding from GPT-NeoX.
    This implementation is design to operate on queries and keys that are compatible with
    [batch_size, n_heads_per_partition, seq_len, head_dim] (e.g. MinGPTAttention format).
    """

    def __init__(
        self,
        head_dim: int,
        base=10000,
    ):
        super().__init__()
        # head_dim must be an even number
        inv_freq = 1.0 / (base ** (paddle.arange(0, head_dim, 2).astype("float32") / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.head_dim = head_dim
        self.seq_len_cached = None
        self.batch_size_cached = None
        self.cos_cached: Tensor | None = None
        self.sin_cached: Tensor | None = None

    def cos_sin(
        self,
        seq_len: int,
        dtype=paddle.bfloat16,
    ) -> Tensor:
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = paddle.arange(seq_len, dtype=self.inv_freq.dtype)
            freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
            emb = paddle.concat((freqs, freqs), axis=-1)

            if dtype in [paddle.float16, paddle.bfloat16]:
                #emb = emb.float()
                emb = paddle.cast(emb, dtype)

            self.cos_cached = emb.cos()[None, :, :]
            self.sin_cached = emb.sin()[None, :, :]

            self.cos_cached = paddle.cast(self.cos_cached, dtype)
            self.sin_cached = paddle.cast(self.sin_cached, dtype)

        return self.cos_cached, self.sin_cached

    def forward(self, q, k):
        batch, seq_len, head_dim = q.shape
        cos, sin = self.cos_sin(seq_len, q.dtype)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def _make_causal_mask(
    input_ids_shape: paddle.shape, past_key_values_length: int
):
    batch_size, target_length = input_ids_shape
    mask = paddle.empty((target_length, target_length + past_key_values_length), dtype=paddle.bool)
    # ONNX doesn't support `Tensor.triu` properly, thus we use this workaround
    seq_ids = paddle.arange(target_length)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(shape=(batch_size, 1, target_length, target_length + past_key_values_length))
    return expanded_mask


def _expand_mask(mask: Tensor, tgt_length: int):
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(paddle.cast(mask[:, None, None, :], "bool"))
    return expanded_mask.expand(shape=(batch_size, 1, tgt_length, src_length))


def build_alibi_tensor(attention_mask: Tensor, num_heads: int, dtype: paddle.dtype) -> Tensor:
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))

    base = paddle.to_tensor(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=paddle.float32)
    #base = Tensor(
    #    2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=paddle.float32
    #)
    powers = paddle.arange(1, 1 + closest_power_of_2, dtype=paddle.float32)
    slopes = paddle.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = Tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=paddle.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = paddle.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=paddle.int32)
        slopes = paddle.concat([slopes, paddle.pow(extra_base, extra_powers)], axis=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(axis=-1) - 1) * attention_mask)[:, None, :]
    alibi = paddle.cast(slopes[..., None], "bfloat16") * arange_tensor
    return paddle.cast(alibi.reshape([batch_size * num_heads, 1, seq_length]), dtype)


def dropout_add(x: Tensor, residual: Tensor, prob: float, training: bool) -> Tensor:
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class Attention(nn.Layer):
    def __init__(self, config: RWConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.maybe_rotary = RotaryEmbedding(config.head_dim) if config.rotary else lambda q, k: (q, k)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor

        self.query_key_value = Linear(
            self.hidden_size,
            3 * self.hidden_size if not config.multi_query else (self.hidden_size + 2 * self.head_dim),
            bias_attr=config.bias,
        )
        self.multi_query = config.multi_query
        self.dense = Linear(self.hidden_size, self.hidden_size, bias_attr=config.bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv = config.n_head if not self.multi_query else 1

    def _split_heads(self, fused_qkv: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
        storage as `fused_qkv`

        Args:
            fused_qkv (`Tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        if not self.multi_query:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.reshape([batch_size, seq_length, self.num_heads, 3, self.head_dim])
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        else:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.reshape([batch_size, seq_length, self.num_heads + 2, self.head_dim])
            return fused_qkv[..., :-2, :], fused_qkv[..., -2, :].unsqueeze(-2), fused_qkv[..., -1, :].unsqueeze(-2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        Merge heads together over the last dimenstion

        Args:
            x: (`Tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            Tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First reshape to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.reshape([batch_size, self.num_heads, seq_length, self.head_dim])

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.transpose([0, 2, 1, 3])

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape([batch_size, seq_length, self.num_heads * self.head_dim])

    def forward(
        self,
        hidden_states: Tensor,
        alibi: Tensor,
        attention_mask: Tensor,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        head_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        i: int=0,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

        # [batch_size, seq_length, num_heads, head_dim]
        query_layer = query_layer.transpose([0, 2, 1, 3]).reshape([batch_size * self.num_heads, q_length, self.head_dim])
        key_layer = key_layer.transpose([0, 2, 1, 3]).reshape([
            batch_size * self.num_kv,
            q_length,
            self.head_dim,
        ])
        value_layer = value_layer.transpose([0, 2, 1, 3]).reshape([batch_size * self.num_kv, q_length, self.head_dim])

        query_layer, key_layer = self.maybe_rotary(query_layer, key_layer)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = paddle.concat((past_key, key_layer), axis=1)
            value_layer = paddle.concat((past_value, value_layer), axis=1)

        # if use layer_past, kv_length != q_length
        _, kv_length, _ = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        if alibi is None:
            query_layer_ = query_layer.reshape([batch_size, self.num_heads, q_length, self.head_dim])
            key_layer_ = key_layer.reshape([batch_size, self.num_kv, kv_length, self.head_dim])
            value_layer_ = value_layer.reshape([batch_size, self.num_kv, kv_length, self.head_dim])

            attn_output = query_layer_ @ key_layer_.transpose([0, 1, 3, 2])
            attention_scores = attn_output.reshape([batch_size, self.num_heads, q_length, kv_length])


            # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
            input_dtype = attention_scores.dtype
            # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
            if input_dtype == paddle.float16 or input_dtype == paddle.bfloat16:
                attention_scores = paddle.cast(attention_scores, paddle.float32)
            # attn_weights = paddle.masked_fill(attention_scores, attention_mask, paddle.finfo(attention_scores.dtype).min)
            #attention_scores = paddle.where(attention_mask, mask_value, attention_scores)
            attention_scores = paddle.where(attention_mask > 0, paddle.full_like(attention_scores, -1e38), attention_scores)
            attention_probs = F.softmax(
                attention_scores * self.inv_norm_factor,
                axis=-1,
                dtype=hidden_states.dtype,
            )
            # [batch_size, num_heads, q_length, kv_length]
            attention_probs = self.attention_dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # matmul: [batch_size, num_heads, q_length, head_dim]
            context_layer = attention_probs @ value_layer_

            # change reshape [batch_size , q_length,  num_heads * head_dim]
            context_layer = context_layer.transpose([0, 2, 1, 3])
            context_layer = context_layer.reshape([batch_size, q_length, -1])

            output_tensor = self.dense(context_layer)

            outputs = (output_tensor, present)
            if output_attentions:
                outputs += (attention_probs,)

            return outputs
        else:
            query_layer_ = query_layer.reshape([batch_size, self.num_heads, q_length, self.head_dim])
            key_layer_ = key_layer.reshape([batch_size, self.num_kv, kv_length, self.head_dim])
            value_layer_ = value_layer.reshape([batch_size, self.num_kv, kv_length, self.head_dim])

            alibi = alibi.reshape([batch_size, self.num_heads, 1, -1])

            attention_scores = query_layer_ @ key_layer_.transpose([0, 1, 3, 2])

            # change reshape to [batch_size, num_heads, q_length, kv_length]
            #attention_scores = matmul_result.reshape([batch_size, self.num_heads, q_length, kv_length])

            #mask_value = paddle.to_tensor(-1e9, dtype=attention_scores.dtype)
            attention_mask_float = paddle.zeros_like(attention_mask, dtype=attention_scores.dtype)
            attention_mask_float = paddle.where(attention_mask, paddle.full_like(attention_scores, -1e9), attention_mask_float)

            # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
            input_dtype = attention_scores.dtype
            # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
            if input_dtype == paddle.float16 or input_dtype == paddle.bfloat16:
                attention_scores = paddle.cast(attention_scores, paddle.float32)
            # attn_weights = paddle.masked_fill(attention_scores, attention_mask, paddle.finfo(attention_scores.dtype).min)
            attention_probs = F.softmax(
                (attention_scores + alibi) * self.inv_norm_factor + attention_mask_float,
                axis=-1,
                dtype=hidden_states.dtype,
            )
            # [batch_size, num_heads, q_length, kv_length]
            attention_probs = self.attention_dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask


            # matmul: [batch_size, num_heads, q_length, kv_length] * [batch_size, num_kv, kv_length, head_dim]
            context_layer = attention_probs @ value_layer_

            # change reshape [batch_size x num_heads, q_length, head_dim]
            context_layer = context_layer.reshape([batch_size * self.num_heads, q_length, self.head_dim])

            # change reshape [batch_size, num_heads, q_length, head_dim]
            context_layer = self._merge_heads(context_layer)

            output_tensor = self.dense(context_layer)

            outputs = (output_tensor, present)
            if output_attentions:
                outputs += (attention_probs,)

            return outputs


class MLP(nn.Layer):
    def __init__(self, config: RWConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.dense_h_to_4h = Linear(hidden_size, 4 * hidden_size, bias_attr=config.bias)
        self.act = nn.GELU()
        self.dense_4h_to_h = Linear(4 * hidden_size, hidden_size, bias_attr=config.bias)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x


class DecoderLayer(nn.Layer):
    def __init__(self, config: RWConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNorm(hidden_size, epsilon=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = Attention(config)

        if not config.parallel_attn:
            # unused if parallel attn
            self.post_attention_layernorm = LayerNorm(hidden_size, epsilon=config.layer_norm_epsilon)

        self.mlp = MLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

        self.config = config

    def forward(
        self,
        hidden_states: Tensor=None,
        alibi: Tensor=None,
        attention_mask: Tensor=None,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        head_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        i: int=0,
    ):

        layernorm_output = self.input_layernorm(hidden_states)
        residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            i=i,
        )

        attention_output = attn_outputs[0]

        if not self.config.parallel_attn:
            residual = dropout_add(attention_output, residual, self.config.attention_dropout, training=self.training)
            layernorm_output = self.post_attention_layernorm(residual)

        outputs = attn_outputs[1:]

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        if self.config.parallel_attn:
            mlp_output += attention_output

        output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class RWPreTrainedModel(PretrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RWConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]

    pretrained_init_configuration = RW_PRETRAINED_INIT_CONFIGURATION

    @classmethod
    def _get_name_mappings(cls, config: RWConfig) -> List[StateDictNameMapping]:
        mappings = [
            "word_embeddings.weight",
            "ln_f.weight",
            "ln_f.bias",
        ]

        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [
                    f"h.{layer_index}.input_layernorm.weight",
                    f"h.{layer_index}.input_layernorm.weight",
                ],
                [
                    f"h.{layer_index}.input_layernorm.bias",
                    f"h.{layer_index}.input_layernorm.bias",
                ],
                [
                    f"h.{layer_index}.self_attention.query_key_value.weight",
                    f"h.{layer_index}.self_attention.query_key_value.weight",
                    "transpose",
                ],
                [
                    f"h.{layer_index}.self_attention.query_key_value.bias",
                    f"h.{layer_index}.self_attention.query_key_value.bias",
                ],
                [
                    f"h.{layer_index}.self_attention.dense.weight",
                    f"h.{layer_index}.self_attention.dense.weight",
                    "transpose",
                ],
                [
                    f"h.{layer_index}.self_attention.dense.bias",
                    f"h.{layer_index}.self_attention.dense.bias",
                ],
                [
                    f"h.{layer_index}.mlp.dense_h_to_4h.weight",
                    f"h.{layer_index}.mlp.dense_h_to_4h.weight",
                    "transpose",
                ],
                [
                    f"h.{layer_index}.mlp.dense_h_to_4h.bias",
                    f"h.{layer_index}.mlp.dense_h_to_4h.bias",
                ],
                [
                    f"h.{layer_index}.mlp.dense_4h_to_h.weight",
                    f"h.{layer_index}.mlp.dense_4h_to_h.weight",
                    "transpose",
                ],
                [
                    f"h.{layer_index}.mlp.dense_4h_to_h.bias",
                    f"h.{layer_index}.mlp.dense_4h_to_h.bias",
                ],
            ]
            mappings.extend(layer_mappings)

        init_name_mappings(mappings)
        # Other than RWModel, other architectures will prepend model prefix
        if config.architectures is not None and "RWModel" not in config.architectures:
            for mapping in mappings:
                mapping[0] = "transformer." + mapping[0]
                if len(mapping) > 1 and mapping[1] != None:
                    mapping[1] = "transformer." + mapping[1]

        #if cls.__name__ != "RobertaModel":
        #    for mapping in mappings:
        #        mapping[1] = "roberta." + mapping[1]

        if config.architectures is not None:
            if "RWForCausalLM" in config.architectures:
                mappings.extend(
                    [
                        "lm_head.weight",
                        "lm_head.bias",
                    ]
                )
            if "RWForTokenClassification" in config.architectures:
                mappings.extend(
                    [
                        ["classifier.weight", None, "transpose"],
                        ["classifier.bias", None],
                    ]
                )
            if "RWForQuestionAnswering" in config.architectures:
                mappings.extend(
                    [
                        ["qa_outputs.weight", None, "transpose"],
                        ["qa_outputs.bias", None],
                    ]
                )
        init_name_mappings(mappings)
        return [StateDictNameMapping(*mapping) for mapping in mappings]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, layer: nn.Layer):
        """Initialize the weights."""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            layer.weight.set_value(
                paddle.tensor.normal(mean=0.0, std=self.config.initializer_range, shape=layer.weight.shape)
            )
            if getattr(layer, "bias", None) is not None:
                layer.weight.set_value(paddle.zeros(shape=layer.weight.shape, dtype=paddle.get_default_dtype()))
        #TODO: laynorm initialization

        #if isinstance(module, nn.Linear) or isinstance(module, Linear):
        #    # Slightly different from the TF version which uses truncated_normal for initialization
        #    # cf https://github.com/pypaddle/pypaddle/pull/5617
        #    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        #    if module.bias is not None:
        #        module.bias.data.zero_()
        #elif isinstance(module, nn.Embedding):
        #    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        #    if module.padding_idx is not None:
        #        module.weight.data[module.padding_idx].zero_()
        #elif isinstance(module, LayerNorm):
        #    module.bias.data.zero_()
        #    module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module: nn.Layer, value: bool = False):
        if isinstance(module, RWModel):
            module.gradient_checkpointing = value

    @staticmethod
    def _convert_to_standard_cache(
        past_key_value: Tuple[Tuple[Tensor, Tensor]], batch_size: int
    ) -> Tuple[Tuple[Tensor, Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        num_heads, ...]))
        """
        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].reshape([batch_size, num_heads, head_dim, seq_length]),
                layer_past[1].reshape([batch_size, num_heads, seq_length, head_dim]),
            )
            for layer_past in past_key_value
        )

    @staticmethod
    def _convert_to_rw_cache(
        past_key_value: Tuple[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tuple[Tensor, Tensor]]:
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].reshape([batch_size_times_num_heads, head_dim, seq_length]),
                layer_past[1].reshape([batch_size_times_num_heads, seq_length, head_dim]),
            )
            for layer_past in past_key_value
        )


class RWModel(RWPreTrainedModel):
    def __init__(self, config: RWConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.alibi = config.alibi

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        # Transformer blocks
        self.h = nn.LayerList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, epsilon=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        #self.post_init()
        #TODO:

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ):
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        #device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: Tensor):
        self.word_embeddings = new_embeddings

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(shape=(num_hidden_layers, -1, -1, -1, -1))
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"

        head_mask = paddle.cast(head_mask, dtype=self.dtype)
        return head_mask

    def get_head_mask(
        self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.
        Args:
            head_mask (`paddle.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            `paddle.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **deprecated_arguments,
    ) -> Union[Tuple[Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = paddle.ones((batch_size, seq_length_with_past))

        if self.alibi:
            alibi = build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)
        else:
            alibi = None

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)

                    return custom_forward

            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                    i=i,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CausalLMHead(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        ret = input @ self.weight.T
        return ret

class RWForCausalLM(RWPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config: RWConfig):
        super().__init__(config)
        self.transformer = RWModel(config)
        self.lm_head = CausalLMHead(config.vocab_size, config.hidden_size, bias_attr=False)

        # Initialize weights and apply final processing
        #self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: Tensor):
        self.lm_head = new_embeddings

    def prepare_attention_mask_for_generation(self, input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype("int64")
        else:
            attention_mask = paddle.ones_like(input_ids, dtype="int64")
        return attention_mask

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # the cache may be in the stardard format (e.g. in contrastive search), convert to our's format if needed
            if past[0][0].shape[0] == input_ids.shape[0]:
                past = self._convert_to_rw_cache(past)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`paddle.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape([batch_size * seq_length, vocab_size]), shift_labels.reshape([batch_size * seq_length])
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[Tensor]], beam_idx: Tensor) -> Tuple[Tuple[Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(tuple(past_state.index_select(0, beam_idx) for past_state in layer_past) for layer_past in past)

#class RWForSequenceClassification(RWPreTrainedModel):
#    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]
#
#    def __init__(self, config: RWConfig):
#        super().__init__(config)
#        self.num_labels = config.num_labels
#        self.transformer = RWModel(config)
#        self.score = nn.Linear(config.hidden_size, config.num_labels)
#
#        # Initialize weights and apply final processing
#        #self.post_init()
#
#    def forward(
#        self,
#        input_ids=None,
#        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
#        attention_mask: Optional[Tensor] = None,
#        head_mask: Optional[Tensor] = None,
#        inputs_embeds: Optional[Tensor] = None,
#        labels: Optional[Tensor] = None,
#        use_cache: Optional[bool] = None,
#        output_attentions: Optional[bool] = None,
#        output_hidden_states: Optional[bool] = None,
#        return_dict: Optional[bool] = None,
#        **deprecated_arguments,
#    ) -> Union[Tuple[Tensor], SequenceClassifierOutputWithPast]:
#        r"""
#        labels (`paddle.LongTensor` of shape `(batch_size,)`, *optional*):
#            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#        """
#        if deprecated_arguments.pop("position_ids", False) is not False:
#            # `position_ids` could have been `Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
#            warnings.warn(
#                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
#                " passing `position_ids`.",
#                FutureWarning,
#            )
#        if len(deprecated_arguments) > 0:
#            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")
#
#        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#        transformer_outputs = self.transformer(
#            input_ids,
#            past_key_values=past_key_values,
#            attention_mask=attention_mask,
#            head_mask=head_mask,
#            inputs_embeds=inputs_embeds,
#            use_cache=use_cache,
#            output_attentions=output_attentions,
#            output_hidden_states=output_hidden_states,
#            return_dict=return_dict,
#        )
#
#        hidden_states = transformer_outputs[0]
#        #[bz, seq, num_labels]
#        logits = self.score(hidden_states)
#
#        if input_ids is not None:
#            batch_size = input_ids.shape[0]
#        else:
#            batch_size = inputs_embeds.shape[0]
#
#        if self.config.pad_token_id is None and batch_size != 1:
#            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
#        if self.config.pad_token_id is None:
#            sequence_lengths = -1
#        else:
#            if input_ids is not None:
#                sequence_lengths = paddle.not_equal(input_ids, paddle.full(shape=input_ids.shape, fill_value=self.config.pad_token_id, dtype=input_ids.dtype)).sum(axis=-1) - 1
#            else:
#                sequence_lengths = -1
#                logger.warning(
#                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
#                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
#                )
#
#        import pdb; pdb.set_trace()
#        if isinstance(sequence_length, Tensor):
#            # [bz, 1]
#            sequence_lengths_idx = sequence_lengths.unsqueeze(1)
#            # [1, num_labels]
#            right_index = paddle.arange(self.config.num_labels).unsqueeze(0)
#            # idx: [bz, num_labels] 
#            pooled_logits = logits[:, sequence_lengths]
#
#        loss = None
#        if labels is not None:
#            if self.config.problem_type is None:
#                if self.num_labels == 1:
#                    self.config.problem_type = "regression"
#                elif self.num_labels > 1 and (labels.dtype == paddle.long or labels.dtype == paddle.int):
#                    self.config.problem_type = "single_label_classification"
#                else:
#                    self.config.problem_type = "multi_label_classification"
#
#            if self.config.problem_type == "regression":
#                loss_fct = MSELoss()
#                if self.num_labels == 1:
#                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
#                else:
#                    loss = loss_fct(pooled_logits, labels)
#            elif self.config.problem_type == "single_label_classification":
#                loss_fct = CrossEntropyLoss()
#                loss = loss_fct(pooled_logits, labels)
#            elif self.config.problem_type == "multi_label_classification":
#                loss_fct = BCEWithLogitsLoss()
#                loss = loss_fct(pooled_logits, labels)
#        if not return_dict:
#            output = (pooled_logits,) + transformer_outputs[1:]
#            return ((loss,) + output) if loss is not None else output
#
#        return SequenceClassifierOutputWithPast(
#            loss=loss,
#            logits=pooled_logits,
#            past_key_values=transformer_outputs.past_key_values,
#            hidden_states=transformer_outputs.hidden_states,
#            attentions=transformer_outputs.attentions,
#        )


class RWForTokenClassification(RWPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config: RWConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = RWModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        #self.post_init()

    def forward(
        self,
        input_ids=None,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[Tensor], TokenClassifierOutput]:
        r"""
        labels (`paddle.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            batch_size, seq_length = labels.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape([batch_size * seq_length, self.num_labels]), labels.reshape([batch_size * seq_length]))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class RWForQuestionAnswering(RWPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = RWModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        #self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`paddle.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`paddle.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        #[bz, seq, 2]
        logits = self.qa_outputs(sequence_output)
        #[bz, seq, 1] //
        start_logits, end_logits = paddle.unstack(x=logits, axis=-1)
        #[bz, seq]
        start_logits = start_logits.squeeze(-1)
        #[bz, seq]
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            #import pdb; pdb.set_trace()
            # If we are on multi-GPU, split add a dimension
            #[bz,]
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            #[bz,]
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]   #  seq_len
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
