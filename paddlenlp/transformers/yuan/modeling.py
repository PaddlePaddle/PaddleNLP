# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modeling class for Yuan2.0 model"""

import copy
import math
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute
from paddle.nn import CrossEntropyLoss

from ...transformers.conversion_utils import StateDictNameMapping, init_name_mappings
from ...transformers.model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...transformers.model_utils import PretrainedModel
from ...utils.log import logger
from ..activations import ACT2FN
from .configuration import YuanConfig

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        ColumnSequenceParallelLinear,
        RowSequenceParallelLinear,
    )
except:
    pass

__all__ = [
    "YuanModel",
    "YuanPretrainedModel",
    "YuanForCausalLM",
]


class YuanRMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        """
        YuanRMSNorm is equivalent to LlamaRMSNorm
        """
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Assign(paddle.ones([hidden_size])),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = paddle.cast(hidden_states, "float32")
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * paddle.cast(hidden_states, input_dtype)


class LocalizedFiltering(paddle.nn.Layer):
    """
    Mega's Exponential Moving Average layer, largely left unmodified from the original repo with the exception of
    variable names and moving away from the stateful representation of incremental decoding state. See
    "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(self, hidden_size):
        super().__init__()

        self.embed_dim = hidden_size
        self.lf_conv2d_group = 1
        self.lf_conv2d_num_pad = 1

        self.conv1 = paddle.nn.Conv2D(
            self.embed_dim,
            self.embed_dim // 2,
            (2, 1),
            stride=(1, 1),
            padding=(self.lf_conv2d_num_pad, 0),
            groups=self.lf_conv2d_group,
        )
        self.conv2 = paddle.nn.Conv2D(
            self.embed_dim // 2,
            self.embed_dim,
            (2, 1),
            stride=(1, 1),
            padding=(self.lf_conv2d_num_pad, 0),
            groups=self.lf_conv2d_group,
        )
        self.output_layernorm = YuanRMSNorm(self.embed_dim)

    def _train_forward(self, inputs):
        inputs = paddle.transpose(inputs, perm=[1, 0, *range(2, len(inputs.shape))])
        seq_len, bsz, embed_dim = inputs.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Unexpected embedding dimension received: input is {embed_dim}, model expects {self.embed_dim}"
            )
        residual = inputs

        inputs = paddle.transpose(paddle.reshape(inputs, [seq_len, 1, bsz, embed_dim]), [2, 3, 0, 1])
        output1 = self.conv1(inputs)
        output1 = output1[:, :, :seq_len, :]

        output2 = self.conv2(output1)
        output2 = paddle.transpose(output2[:, :, :seq_len, :], [2, 3, 0, 1])
        output2 = paddle.reshape(output2, [seq_len, bsz, embed_dim])
        assert output2.shape == residual.shape

        lf_output = self.output_layernorm(output2 + residual)
        lf_output = paddle.transpose(lf_output, [1, 0, *range(2, len(lf_output.shape))])
        return lf_output

    def _inference_forward(self, inputs, before_hidden_states):

        if before_hidden_states is None:
            inputs = inputs.transpose([1, 0, *range(2, len(inputs.shape))])
            seq_len, bsz, embed_dim = inputs.shape
            if embed_dim != self.embed_dim:
                raise ValueError(
                    f"Unexpected embedding dimension received: input is {embed_dim}, model expects {self.embed_dim}"
                )
            residual = inputs

            inputs = paddle.transpose(paddle.reshape(inputs, [seq_len, 1, bsz, embed_dim]), [2, 3, 0, 1])
            output1 = self.conv1(inputs)
            output1 = output1[:, :, :seq_len, :]

            output2 = self.conv2(output1)
            output2 = paddle.transpose(output2[:, :, :seq_len, :], [2, 3, 0, 1])
            output2 = paddle.reshape(output2, [seq_len, bsz, embed_dim])
            assert output2.shape == residual.shape

            lf_output = self.output_layernorm(output2 + residual)
            lf_output = paddle.transpose(lf_output, [1, 0, *range(2, len(lf_output.shape))])
            return lf_output
        else:
            inputs = paddle.transpose(inputs, [1, 0, *range(2, len(inputs.shape))])
            before_hidden_states = paddle.transpose(
                before_hidden_states, [1, 0, *range(2, len(before_hidden_states.shape))]
            )
            residual = inputs

            seq_len, bsz, embed_dim = inputs.shape
            seq_len_before, _, _ = before_hidden_states.shape

            assert seq_len == 1 and seq_len_before == 2

            inputs = paddle.concat((before_hidden_states, inputs), axis=0)
            inputs = paddle.transpose(paddle.reshape(inputs, [3, 1, bsz, embed_dim]), [2, 3, 0, 1])

            output1 = self.conv1(inputs)
            output2 = self.conv2(output1[:, :, 1:-1, :])
            output2 = output2[:, :, 1:-1, :]
            output2 = paddle.reshape(output2, [1, bsz, embed_dim])
            assert output2.shape == residual.shape

            lf_output = self.output_layernorm(output2 + residual)
            lf_output = paddle.transpose(lf_output, [1, 0, *range(2, len(lf_output.shape))])

        return lf_output

    def forward(self, inputs, before_hidden_states) -> paddle.Tensor:
        assert self.lf_conv2d_num_pad == 1
        if self.training:
            lf_output = self._train_forward(inputs)
        else:
            lf_output = self._inference_forward(inputs, before_hidden_states)

        return lf_output


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: paddle.shape, dtype: paddle.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = paddle.full((tgt_len, tgt_len), paddle.to_tensor(paddle.finfo(dtype).min))
    mask_cond = paddle.arange(mask.size(-1))
    mask_cond = paddle.add(mask_cond, 1)
    mask_cond_reshaped = paddle.reshape(mask_cond, [mask.size(-1), 1])
    mask = paddle.where(mask_cond < mask_cond_reshaped, paddle.zeros_like(mask), mask)
    mask = paddle.cast(mask, dtype)
    if past_key_values_length > 0:
        mask = paddle.concat([paddle.zeros([tgt_len, past_key_values_length], dtype=dtype), mask], zeros=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: paddle.Tensor, dtype: paddle.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
    expanded_mask = paddle.to_tensor(expanded_mask, dtype=dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(paddle.cast(inverted_mask, paddle.bool), paddle.finfo(dtype).min)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class YuanPretrainedModel(PretrainedModel):
    config_class = YuanConfig
    base_model_prefix = "yuan"
    supports_gradient_checkpointing = True
    _no_split_modules = ["YuanDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    @classmethod
    def _get_name_mappings(cls, config: YuanConfig) -> List[StateDictNameMapping]:
        mappings: List[StateDictNameMapping] = []
        model_mappings = [
            ["embed_tokens.weight"],
            ["norm.weight"],
        ]
        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [f"layers.{layer_index}.self_attn.q_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.k_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.v_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.o_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.rotary_emb.inv_freq"],
                [f"layers.{layer_index}.mlp.gate_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.down_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.up_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.input_layernorm.weight"],
                [f"layers.{layer_index}.post_attention_layernorm.weight"],
                [f"layers.{layer_index}.self_attn.lf_gate.conv1.bias"],
                [f"layers.{layer_index}.self_attn.lf_gate.conv1.weight"],
                [f"layers.{layer_index}.self_attn.lf_gate.conv2.bias"],
                [f"layers.{layer_index}.self_attn.lf_gate.conv2.weight"],
                [f"layers.{layer_index}.self_attn.lf_gate.output_layernorm.weight"],
            ]
            model_mappings.extend(layer_mappings)

        init_name_mappings(mappings=model_mappings)

        if "YuanModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "model." + mapping[0]
                mapping[1] = "yuan." + mapping[1]
            model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: YuanConfig, is_split=True):

        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}

            base_actions = {
                "lm_head.weight": partial(fn, is_column=True),
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
            }

            if not config.vocab_size % config.tensor_parallel_degree == 0:
                base_actions.pop("lm_head.weight")
                base_actions.pop("embed_tokens.weight")
            # Column Linear

            base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
            # if we have enough num_key_value_heads to split, then split it.
            if config.num_attention_heads % config.tensor_parallel_degree == 0:
                base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)

            base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings

    @classmethod
    def _get_fuse_or_split_param_mappings(cls, config: YuanConfig, is_fuse=False):
        def convert_qk_keys_fn(fused_params, tensor_parallel_degree):
            concat_fn = np.concatenate
            split_fn = np.split
            if isinstance(fused_params, paddle.Tensor):
                concat_fn = paddle.concat
                split_fn = paddle.split

            q_weight, k_weight = split_fn(fused_params, 2, axis=-1)

            hidden_size = q_weight.shape[-1]
            step = 1
            if tensor_parallel_degree > 1:
                assert hidden_size // tensor_parallel_degree, "hidden_size must be divisible by tensor_parallel_degree"
                step = hidden_size // tensor_parallel_degree

            q_slices = [q_weight[:, i : i + step] for i in range(0, hidden_size, step)]
            k_slices = [k_weight[:, i : i + step] for i in range(0, hidden_size, step)]
            q1 = concat_fn(q_slices[0::2], -1)
            q2 = concat_fn(k_slices[0::2], -1)
            k1 = concat_fn(q_slices[1::2], -1)
            k2 = concat_fn(k_slices[1::2], -1)

            return concat_fn([q1, q2], -1), concat_fn([k1, k2], -1)

        def fuse_qk_keys_fn(fuse_params):
            concat_fn = np.concatenate
            if isinstance(fuse_params[0], paddle.Tensor):
                concat_fn = paddle.concat
            return concat_fn(fuse_params, axis=-1)

        # last key is fused key, other keys are to be fused.

        final_actions = {}
        if config.tensor_parallel_degree <= 1:
            return final_actions

        if is_fuse:
            fuse_qk_keys = (
                "layers.0.self_attn.q_proj.weight",  # base param key
                "layers.0.self_attn.k_proj.weight",  # base param key
                "layers.0.self_attn.qk_proj.weight",  # new param key
            )

            for i in range(config.num_hidden_layers):
                keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_qk_keys])
                final_actions[keys] = partial(fuse_qk_keys_fn)
        else:
            split_qk_keys = (
                "layers.0.self_attn.q_proj.weight",  # new param key
                "layers.0.self_attn.k_proj.weight",  # new param key
                "layers.0.self_attn.qk_proj.weight",  # base param key
            )

            for i in range(config.num_hidden_layers):
                keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in split_qk_keys])
                final_actions[keys] = partial(convert_qk_keys_fn, tensor_parallel_degree=config.tensor_parallel_degree)

        return final_actions

    def _init_weights(self, layer):
        """Initialization hook"""
        if self.config.tensor_parallel_degree > 1:
            rng_tracker = get_rng_state_tracker().rng_state
        if isinstance(
            layer,
            (
                nn.Linear,
                nn.Embedding,
                mpu.VocabParallelEmbedding,
                mpu.ColumnParallelLinear,
                mpu.RowParallelLinear,
                ColumnSequenceParallelLinear,
                RowSequenceParallelLinear,
            ),
        ):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                if layer.weight.is_distributed:
                    with rng_tracker():
                        layer.weight.set_value(
                            paddle.tensor.normal(
                                mean=0.0,
                                std=self.config.initializer_range
                                if hasattr(self.config, "initializer_range")
                                else self.llama.config.initializer_range,
                                shape=layer.weight.shape,
                            )
                        )
                else:
                    layer.weight.set_value(
                        paddle.tensor.normal(
                            mean=0.0,
                            std=self.config.initializer_range
                            if hasattr(self.config, "initializer_range")
                            else self.llama.config.initializer_range,
                            shape=layer.weight.shape,
                        )
                    )

        with paddle.no_grad():
            if isinstance(layer, YuanMLP):
                factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
                layer.down_proj.weight.scale_(factor)
            if isinstance(layer, YuanAttention):
                factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
                layer.o_proj.weight.scale_(factor)

    def _post_init(self, *args, **kwargs):
        with paddle.no_grad():
            self.init_weights()


class YuanRotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):

        """
        YuanRotaryEmbedding is equivalent to LlamaRotaryEmbedding in transformers v4.36
        """

        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # [dim / 2]
        self.inv_freq = 1.0 / (self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim))
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype=self.inv_freq.dtype)
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]
        return (
            paddle.cast(cos, x.dtype),
            paddle.cast(sin, x.dtype),
        )


class YuanMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        if config.sequence_parallel:
            ColumnParallelLinear = ColumnSequenceParallelLinear
            RowParallelLinear = RowSequenceParallelLinear
        else:
            ColumnParallelLinear = fleet.meta_parallel.ColumnParallelLinear
            RowParallelLinear = fleet.meta_parallel.RowParallelLinear
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act
        if config.tensor_parallel_degree > 1:

            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )

            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=False)
        self.act_fn = ACT2FN[self.hidden_act]

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.act_fn(self.up_proj(x)))


class YuanAttention(nn.Layer):

    """Localized Filtering-based Attention 'YUAN 2.0: A Large Language Model with Localized Filtering-based Attention' paper"""

    def __init__(self, config: YuanConfig):
        super().__init__()
        self.config = config
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.causal_mask = config.causal_mask
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        self.use_flash_attention = config.use_flash_attention
        self.kv_indices = None
        self.tp_degree = config.tensor_parallel_degree
        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree

            if self.num_key_value_heads % config.tensor_parallel_degree == 0:
                self.num_key_value_heads = self.num_key_value_heads // config.tensor_parallel_degree
            else:
                assert False
        if config.sequence_parallel:
            ColumnParallelLinear = ColumnSequenceParallelLinear
            RowParallelLinear = RowSequenceParallelLinear
        else:
            ColumnParallelLinear = fleet.meta_parallel.ColumnParallelLinear
            RowParallelLinear = fleet.meta_parallel.RowParallelLinear
        self.dropout = 0.0

        if config.tensor_parallel_degree > 1:
            self.o_proj = RowParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
                input_is_parallel=True,
            )
            self.v_proj = ColumnParallelLinear(
                self.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                has_bias=False,
                gather_output=False,
            )
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
                gather_output=False,
            )
            self.k_proj = ColumnParallelLinear(
                self.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                has_bias=False,
                gather_output=False,
            )
        else:
            self.o_proj = nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias_attr=False,
            )
            self.v_proj = nn.Linear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, bias_attr=False)
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias_attr=False)
            self.k_proj = nn.Linear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, bias_attr=False)
        self.rotary_emb = YuanRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        self.lf_gate = LocalizedFiltering(self.hidden_size)

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        return tensor.reshape([bsz, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        before_hidden_states = None
        is_first_step = False
        if use_cache:
            if past_key_value is None:
                inference_hidden_states_memory = paddle.empty(
                    [bsz, 2, hidden_states.shape[2]], dtype=hidden_states.dtype
                )
                is_first_step = True
            else:
                before_hidden_states = past_key_value[2]

        if use_cache:
            if is_first_step:
                if q_len >= 2:
                    inference_hidden_states_memory = hidden_states[:, -2:, :]
                else:
                    inference_hidden_states_memory[:, :, :] = 0
                    inference_hidden_states_memory[:, -1:, :] = hidden_states[:, -1:, :]
            else:
                hidden_states_tmp = before_hidden_states[:, -1:, :]
                inference_hidden_states_memory = copy.deepcopy(
                    paddle.concat((hidden_states_tmp, hidden_states), axis=1)
                )

        value_states = (
            self.v_proj(hidden_states).reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        )
        hidden_states = self.lf_gate(hidden_states, before_hidden_states)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        qk_states = paddle.concat([query_states, key_states], axis=-1)
        qk_states = qk_states.reshape([bsz, q_len, self.num_heads, int(qk_states.shape[-1] // self.num_heads)])
        (query_states, key_states) = paddle.chunk(qk_states, 2, axis=-1)
        query_states = query_states.transpose([0, 2, 1, *range(3, len(query_states.shape))])
        key_states = key_states.transpose([0, 2, 1, *range(3, len(key_states.shape))])

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states, inference_hidden_states_memory) if use_cache else None

        if self.use_flash_attention:
            attn_weights = None
            query_states = query_states.transpose([0, 2, 1, *range(3, len(query_states.shape))])
            key_states = key_states.transpose([0, 2, 1, *range(3, len(key_states.shape))])
            value_states = value_states.transpose([0, 2, 1, *range(3, len(value_states.shape))])

            batch_size = query_states.shape[0]

            output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
            )
            # attn_output = rearrange(output[0], '(b s) ... -> b s ...', b=batch_size)
            seq_length = output[0].shape[0] // batch_size
            new_shape = (batch_size, seq_length) + tuple(output[0].shape[1:])
            attn_output = paddle.reshape(output[0], new_shape)
        else:
            attn_weights = paddle.matmul(
                query_states, key_states.transpose([0, 1, 3, 2, *range(4, len(key_states.shape))])
            ) / math.sqrt(self.head_dim)

            if attn_weights.shape != [bsz, self.num_heads, q_len, kv_seq_len]:
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.shape}"
                )
            if attention_mask is not None:
                if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = paddle.maximum(
                    attn_weights, paddle.to_tensor(paddle.finfo(attn_weights.dtype).min, attn_weights.dtype)
                )

            # upcast attention to fp32
            attn_weights = paddle.cast(
                nn.functional.softmax(attn_weights, axis=-1, dtype=paddle.float32), query_states.dtype
            )
            attn_output = paddle.matmul(attn_weights, value_states)

            if attn_output.shape != [bsz, self.num_heads, q_len, self.head_dim]:
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.shape}"
                )

            attn_output = attn_output.transpose([0, 2, 1, *range(3, len(attn_output.shape))])

        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = attn_output.reshape([bsz, q_len, -1])
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class YuanDecoderLayer(nn.Layer):
    def __init__(self, config: YuanConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = YuanAttention(config=config)
        self.mlp = YuanMLP(config)
        self.input_layernorm = YuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = YuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class YuanModel(YuanPretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`YuanDecoderLayer`]

    Args:
        config: YuanConfig
    """

    def __init__(self, config: YuanConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # TODO: control it by config
        self.eod_token = config.eod_token
        self.reset_attention_mask = config.reset_attention_mask
        self.reset_position_ids = config.reset_position_ids
        self.enable_recompute = False
        self.recompute_granularity = config.recompute_granularity
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []
        if config.tensor_parallel_degree > 1:
            self.embed_tokens = mpu.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.LayerList([YuanDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = YuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self._post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            expanded_attn_mask = paddle.to_tensor(expanded_attn_mask)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def _prepare_decoder_attention_mask_training(
        self, input_id, inputs_embeds, eod_token, reset_mask_flag, reset_attention_mask=True, reset_position_ids=True
    ):

        micro_batch_size, seq_length = input_id.shape
        attention_mask = paddle.tril(paddle.ones((micro_batch_size, seq_length, seq_length), dtype=self.config.dtype))
        attention_mask = paddle.reshape(attention_mask, (micro_batch_size, 1, seq_length, seq_length))

        position_ids = paddle.arange(seq_length, dtype=paddle.int64)
        position_ids = position_ids.unsqueeze(0).expand_as(input_id)

        if reset_position_ids:
            position_ids = position_ids.clone()

        if reset_position_ids or reset_attention_mask:
            # Loop through the batches:
            for b in range(micro_batch_size):

                # Find indecies where EOD token is.
                eod_index = position_ids[b, input_id[b] == eod_token]

                # Detach indecies from positions if going to modify positions.
                if reset_position_ids:
                    eod_index = eod_index.detach()
                # Loop through EOD indecies:
                prev_index = 0
                for j in range(eod_index.shape[0]):
                    i = eod_index[j]
                    # Mask attention loss.
                    if reset_attention_mask:
                        attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                    # Reset positions.
                    if reset_position_ids:
                        position_ids[b, (i + 1) :] -= i + 1 - prev_index
                        prev_index = i + 1

        inverted_mask = 1 - attention_mask
        output_attn_mask = inverted_mask.masked_fill(
            paddle.cast(inverted_mask, "bool"), paddle.finfo(inputs_embeds.dtype).min
        )
        if reset_mask_flag:
            output_attn_mask = output_attn_mask[:, :, -1:, :]
        return output_attn_mask, position_ids

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        position_ids: Optional[Tensor],
        attention_mask: Tensor,
        output_attentions: bool,
        past_key_value: Tensor,
        use_cache: bool,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            use_reentrant=self.config.recompute_use_reentrant,
        )

        return hidden_states

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids1 = copy.deepcopy(input_ids)
        reset_mask_flag = False
        if past_key_values:
            input_ids = input_ids[:, -1:]
            if use_cache:
                reset_mask_flag = True
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = paddle.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=paddle.int64
            )
            position_ids = paddle.unsqueeze(position_ids, axis=0).reshape([-1, seq_length])
        else:
            position_ids = paddle.reshape(position_ids, [-1, seq_length])
            position_ids = paddle.cast(position_ids, dtype="int64")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if self.training or self.reset_position_ids:
            attention_mask, _ = self._prepare_decoder_attention_mask_training(
                input_ids1,
                inputs_embeds,
                self.eod_token,
                reset_mask_flag,
                self.reset_attention_mask,
                self.reset_position_ids,
            )

        else:
            if attention_mask is None:
                attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            has_gradient = not hidden_states.stop_gradient
            if (
                self.enable_recompute
                and idx not in self.no_recompute_layers
                and has_gradient
                and self.recompute_granularity == "full"
            ):
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class YuanForCausalLM(YuanPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.eod_token = config.eod_token
        self.sep_token = config.sep_token
        self.use_loss_mask = config.use_loss_mask
        self.yuan = YuanModel(config)
        if config.sequence_parallel:
            ColumnParallelLinear = ColumnSequenceParallelLinear
        else:
            ColumnParallelLinear = fleet.meta_parallel.ColumnParallelLinear

        if config.tensor_parallel_degree > 1:
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                has_bias=False,
                gather_output=True,
            )
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)
        # Initialize weights and apply final processing
        self._post_init()

    def get_input_embeddings(self):
        return self.yuan.embed_tokens

    def set_input_embeddings(self, value):
        self.yuan.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.yuan = decoder

    def get_decoder(self):
        return self.yuan

    def get_loss_mask(self, input_ids, labels, eod_token, sep_token):
        micro_batch_size, seq_length = input_ids.shape

        loss_mask = paddle.ones(input_ids.shape, dtype=paddle.float32)

        position_ids = paddle.arange(seq_length, dtype=paddle.int64)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        """modify loss_mask to only calculate the loss of the answer (separated with [SEP])"""

        for b in range(micro_batch_size):
            eod_indexs = position_ids[b, input_ids[b] == eod_token]
            sep_indexs = position_ids[b, input_ids[b] == sep_token]

            if len(eod_indexs) == 0 or len(sep_indexs) == 0:
                loss_mask[b] = 1.0
            else:
                if eod_indexs[0] > sep_indexs[0]:
                    loss_mask[b, 0 : sep_indexs[0]] = 0

                    if len(eod_indexs) == len(sep_indexs):
                        for ii, eod_index in enumerate(eod_indexs):
                            start_index = eod_index
                            if ii == (len(sep_indexs) - 1):
                                stop_index = seq_length
                            else:
                                stop_index = sep_indexs[ii + 1]
                            loss_mask[b, start_index:stop_index] = 0.0
                    else:
                        if len(eod_indexs) > len(sep_indexs):
                            loss_mask[b, :] = 1.0
                        else:
                            for ii, eod_index in enumerate(eod_indexs):
                                start_index = eod_index
                                stop_index = sep_indexs[ii + 1]

                                loss_mask[b, start_index:stop_index] = 0.0

                elif eod_indexs[0] < sep_indexs[0]:

                    if len(eod_indexs) == len(sep_indexs):
                        for ii, eod_index in enumerate(eod_indexs):
                            start_index = eod_index
                            stop_index = sep_indexs[ii]
                            loss_mask[b, start_index:stop_index] = 0.0

                    else:
                        if len(eod_indexs) < len(sep_indexs):
                            loss_mask[b, :] = 1.0
                        else:
                            for ii, eod_index in enumerate(eod_indexs):
                                start_index = eod_index
                                if ii >= len(sep_indexs):
                                    stop_index = seq_length
                                else:
                                    stop_index = sep_indexs[ii]
                                loss_mask[b, start_index:stop_index] = 0.0

        loss_mask[input_ids == eod_token] = 1.0
        return loss_mask

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, YuanForCausalLM

        >>> model = YuanForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = True
        outputs = self.yuan(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            if self.use_loss_mask:
                loss_mask = self.get_loss_mask(input_ids, labels, self.eod_token, self.sep_token)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :, :].contiguous()
            shift_labels = labels[..., :].contiguous()
            # Flatten the tokens
            if self.use_loss_mask:
                loss_fct = CrossEntropyLoss(reduction="none")
                shift_logits = paddle.reshape(shift_logits, [-1, self.config.vocab_size])
                shift_labels = paddle.reshape(shift_labels, [-1])
                # Enable model parallelism
                shift_labels = paddle.to_tensor(shift_labels)
                loss = loss_fct(shift_logits, shift_labels)

                loss = paddle.sum(loss * loss_mask) / loss_mask.sum()
            else:
                loss_fct = CrossEntropyLoss()
                shift_logits = paddle.reshape(shift_logits, [-1, self.config.vocab_size])
                shift_labels = paddle.reshape(shift_labels, [-1])
                # Enable model parallelism
                shift_labels = paddle.to_tensor(shift_labels)
                loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        position_ids = None
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = paddle.cast(attention_mask, dtype="int64").cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
