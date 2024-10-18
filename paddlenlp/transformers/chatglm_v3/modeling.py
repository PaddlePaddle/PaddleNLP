# encoding=utf-8
# Copyright (c) 2023 ChatGLM2-6B Model Team and PaddlePaddle Authors. All Rights Reserved.
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

import paddle.distributed.fleet.meta_parallel as mpu
import paddle
from paddle import Tensor, nn
from typing import Optional, Tuple, List
from functools import partial
from paddlenlp.utils.tools import get_env_device
import math
from paddle.distributed.fleet.utils import recompute
from paddlenlp.transformers import PretrainedModel
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddlenlp.transformers.llama.modeling import LlamaPretrainingCriterion, LlamaMLP,LlamaAttention, LlamaLMHead, LlamaRMSNorm, assign_kv_heads
from paddlenlp.utils.converter import StateDictNameMapping, init_name_mappings
from paddlenlp.transformers.model_outputs import ModelOutput
from .configuration import ChatGLMv3Config
from paddlenlp.generation.logits_process import LogitsProcessorList
from paddlenlp.transformers.long_sequence_strategies import LongSequenceStrategies
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.generation.utils import get_unfinished_flag 
from paddlenlp.generation.logits_process import TopKProcess, TopPProcess
import paddle.nn.functional as F
import paddle.distributed as dist
from paddlenlp.transformers.segment_parallel_utils import ReshardLayer
from paddlenlp.transformers import linear_utils
from paddlenlp.transformers.linear_utils import Linear
from paddlenlp.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        ColumnSequenceParallelLinear,
        GatherOp,
        RowSequenceParallelLinear,
        ScatterOp,
        mark_as_sequence_parallel_parameter,
    )
except:
    pass

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding

except ImportError:
    fused_rotary_position_embedding = None

from paddle.distributed import fleet
import warnings
from paddlenlp.utils.log import logger
import os
try:
    if get_env_device() in ["npu", "gcu"]:
        for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
            if lib.endswith(".so"):
                paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(lib)
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None

from paddlenlp.transformers.llama import fusion_ops

rms_norm_fused = fusion_ops.rms_norm_fused

try:
    from paddle.incubate.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
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
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [1, seqlen, 1, dim]
        self.cos_cached = freqs.cos()[None, :, None, :]
        self.sin_cached = freqs.sin()[None, :, None, :]
        self.cos_sin_table = None if get_env_device() != "gcu" else paddle.concat([freqs.cos(), freqs.sin()], axis=-1)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        return (
            cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
            sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
        )

    def get_fused_cos_sin(self, x, seq_len=None):
        if self.cos_sin_table is not None and self.cos_sin_table.dtype != x.dtype:
            return self.cos_sin_table.cast(x.dtype)
        else:
            return self.cos_sin_table


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of paddle.repeat_interleave(hidden_states, n_rep, axis=1). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    try:
        hidden_states = paddle.repeat_interleave(hidden_states.to("cpu"), n_rep, -2)
    except:
        hidden_states = paddle.cast(hidden_states.to("cpu"), "float32")
        hidden_states = paddle.repeat_interleave(hidden_states.to("cpu"), n_rep, -2)
    return paddle.cast(hidden_states, "float16").to(get_env_device())
def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )

def get_use_casual_mask():
    """Get the value of the 'USE_CASUAL_MASK' environment variable."""
    return os.getenv("USE_CASUAL_MASK", "False") == "True"

def build_alibi_tensor(
    bool_attention_mask: Tensor, num_heads: int, dtype: paddle.dtype, tensor_parallel_degree=1
) -> Tensor:
    batch_size, seq_length = bool_attention_mask.shape[0], bool_attention_mask.shape[-1]
    slopes = paddle.to_tensor(_get_interleave(num_heads), dtype="float32")
    alibi = slopes.unsqueeze(axis=[1, 2]) * paddle.arange(seq_length, dtype="float32").unsqueeze(axis=[0, 1]).expand(
        [num_heads, -1, -1]
    )
    alibi = alibi.reshape(shape=(1, num_heads, 1, seq_length)).expand([batch_size, -1, -1, -1])
    return paddle.cast(alibi, dtype)


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    # [bsz, n_head, q_len, kv_seq_len]
    shape = x.shape
    #  [bsz, 1, q_len, kv_seq_len]
    shape[1] = 1
    mask = paddle.full(shape, paddle.finfo(x.dtype).min, dtype=x.dtype)
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def assign_kv_heads(num_kv_heads: int, num_gpus: int):
    # Initialize the assignment list
    """
    Assign kv heads to different GPUs in the Tensor Parallel Setup

    Examples:
        assign_kv_heads(num_kv_heads=1, num_gpus=2): [[0], [0]]
        assign_kv_heads(num_kv_heads=2, num_gpus=2): [[0], [1]]
        assign_kv_heads(num_kv_heads=4, num_gpus=2): [[0,1], [2,3]]
        assign_kv_heads(num_kv_heads=1, num_gpus=4): [[0],[0],[0],[0]]
        assign_kv_heads(num_kv_heads=2, num_gpus=4): [[0],[0],[1],[1]]
        assign_kv_heads(num_kv_heads=4, num_gpus=4): [[0],[1],[2],[3]]
    """
    assignment_list = [[] for _ in range(num_gpus)]
    # Case 1: more heads than cards
    if num_kv_heads > num_gpus:
        num_heads_per_card = num_kv_heads // num_gpus
        for i in range(num_gpus):
            for j in range(num_heads_per_card):
                assignment_list[i].append(i * num_heads_per_card + j)
    # Case 2: more cards than heads. each card get only 1 head.
    else:
        num_card_per_heads = num_gpus // num_kv_heads
        for i in range(num_kv_heads):
            for j in range(num_card_per_heads):
                assignment_list[i * num_card_per_heads + j].append(i)
    return assignment_list

def parallel_matmul(x: Tensor, y: Tensor, tensor_parallel_output=True):
    is_fleet_init = True
    tensor_parallel_degree = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
    except:
        is_fleet_init = False

    if paddle.in_dynamic_mode():
        y_is_distributed = y.is_distributed
    else:
        y_is_distributed = tensor_parallel_degree > 1

    if is_fleet_init and tensor_parallel_degree > 1 and y_is_distributed:
        # if not running under distributed.launch, it will raise AttributeError: 'Fleet' object has no attribute '_hcg'
        input_parallel = paddle.distributed.collective._c_identity(x, group=model_parallel_group)
        logits = paddle.matmul(input_parallel, y, transpose_y=False)

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)

    else:
        logits = paddle.matmul(x, y, transpose_y=False)
        return logits


def scaled_dot_product_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    alibi=None,
    sequence_parallel=False,
    reshard_layer=None,
    npu_is_casual=False,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, _, _ = value_states.shape

    if config.use_flash_attention and flash_attention:
        return fusion_ops.fusion_flash_attention(
            query_states,
            config,
            key_states,
            value_states,
            attention_mask,
            output_attentions,
            alibi,
            sequence_parallel,
            reshard_layer,
            npu_is_casual,
        )

        # Paddle Flash Attention input [ bz, seqlen, nhead, head_dim]
        # Torch Flash Attention input [ bz, nhead, seqlen, head_dim]

    else:
        if config.context_parallel_degree > 1:
            raise ValueError("Context parallel requires `use_flash_attention=True`")

        #  [ bz, seqlen, nhead, head_dim] -> [bs, nhead, seq_len, head_dim]
        query_states = paddle.transpose(query_states, [0, 2, 1, 3])
        # merge with the next tranpose
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        # matmul and devide by sqrt(head_dim)
        attn_weights = paddle.matmul(query_states / math.sqrt(head_dim), key_states.transpose([0, 1, 3, 2]))
        # logger.info(f"qk result: {attn_weights}")
        # then add alibi bias
        if alibi is not None:
            alibi = alibi.reshape([bsz, num_heads, 1, -1])
            attn_weights = attn_weights + alibi

        if attn_weights.shape != [bsz, num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        # In sep mode, the attenion mask should be created in the runtime.
        if reshard_layer is not None:
            attention_mask = None

        # NOTE: we only call get_triangle_upper_mask under PP setup
        # we just make it triangle_upper_mask
        if attention_mask is None:
            attention_mask = get_triangle_upper_mask(attn_weights)
        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )

        attn_weights = attn_weights + attention_mask
        if not paddle.in_dynamic_mode():
            attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
        else:
            with paddle.amp.auto_cast(False):
                attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)

        attn_output = paddle.matmul(attn_weights, value_states)

        # logger.info(f"qkv result: {attn_output}")
        attn_output = attn_output.transpose([0, 2, 1, 3])

        if reshard_layer is not None:
            attn_output = reshard_layer(
                attn_output,
                split_axis=1,
                concat_axis=2,
            )
            q_len = q_len // config.sep_parallel_degree
            num_heads = num_heads * config.sep_parallel_degree

        if sequence_parallel:
            attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])
        else:
            attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return (attn_output, attn_weights) if output_attentions else attn_output


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def is_casual_mask(attention_mask):
    """
    Upper triangular of attention_mask equals to attention_mask is casual
    """
    return (paddle.triu(attention_mask) == attention_mask).all().item()


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make casual mask used for self-attention
    """
    batch_size, target_length = input_ids_shape  # target_length: seq_len

    if get_env_device() == "npu":
        mask = paddle.tril(paddle.ones((target_length, target_length))).astype("int32")
    else:
        mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))

    if past_key_values_length > 0:
        # [tgt_len, tgt_len + past_len]
        mask = paddle.concat([paddle.ones([target_length, past_key_values_length], dtype="bool"), mask], axis=-1)

    # [bs, 1, tgt_len, tgt_len + past_len]
    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    if get_env_device() == "npu":
        mask = mask[:, None, None, :].astype(dtype)
    else:
        mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):

    if position_ids is None:
        # Note: Only for LlamaForCausalLMPipe model pretraining
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        bs, sq, _, dim_2 = cos.shape
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        q_embed = q.reshape([bs, sq, -1, dim_2, 2])
        k_embed = k.reshape([bs, sq, -1, dim_2, 2])
        q_embed = paddle.stack(
            [
                q_embed[..., 0] * cos - q_embed[..., 1] * sin,
                q_embed[..., 1] * cos + q_embed[..., 0] * sin,
            ],
            -1,
        )
        k_embed = paddle.stack(
            [
                k_embed[..., 0] * cos - k_embed[..., 1] * sin,
                k_embed[..., 1] * cos + k_embed[..., 0] * sin,
            ],
            -1,
        )

    q_embed = q_embed.reshape([bs, sq, -1, dim_2*2])
    k_embed = k_embed.reshape([bs, sq, -1, dim_2*2])
    return q_embed, k_embed


class ChatGLMv3PretrainedModel(PretrainedModel):
    config_class = ChatGLMv3Config
    base_model_prefix = "chatglmv3"
    @classmethod
    def _get_name_mappings(cls, config: ChatGLMv3Config) -> List[StateDictNameMapping]:
        mappings = [
            "rotary_pos_emb.inv_freq",
        ]

        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [
                    f"layers.{layer_index}.input_layernorm.weight",
                    f"layers.{layer_index}.input_layernorm.weight",
                ],
                [
                    f"layers.{layer_index}.self_attention.query_key_value.weight",
                    f"layers.{layer_index}.self_attention.query_key_value.weight",
                    # "transpose",
                ],
                [
                    f"layers.{layer_index}.self_attention.k_proj.weight",
                    f"layers.{layer_index}.self_attention.k_proj.weight",
                    # "transpose",
                ],
                [
                    f"layers.{layer_index}.self_attention.q_proj.weight",
                    f"layers.{layer_index}.self_attention.q_proj.weight",
                    # "transpose",
                ],
                [
                    f"layers.{layer_index}.self_attention.v_proj.weight",
                    f"layers.{layer_index}.self_attention.v_proj.weight",
                    # "transpose",
                ],
                [
                    f"layers.{layer_index}.self_attention.query_key_value.bias",
                    f"layers.{layer_index}.self_attention.query_key_value.bias",
                ],
                [
                    f"layers.{layer_index}.self_attention.v_proj.bias",
                    f"layers.{layer_index}.self_attention.v_proj.bias",
                ],
                [
                    f"layers.{layer_index}.self_attention.k_proj.bias",
                    f"layers.{layer_index}.self_attention.k_proj.bias",
                ],
                [
                    f"layers.{layer_index}.self_attention.q_proj.bias",
                    f"layers.{layer_index}.self_attention.q_proj.bias",
                ],
                [
                    f"layers.{layer_index}.self_attention.dense.weight",
                    f"layers.{layer_index}.self_attention.dense.weight",
                ],
                [
                    f"layers.{layer_index}.post_attention_layernorm.weight",
                    f"layers.{layer_index}.post_attention_layernorm.weight",
                ],
                [
                    f"layers.{layer_index}.mlp.dense_h_to_4h.weight",
                    f"layers.{layer_index}.mlp.dense_h_to_4h.weight",
                ],
                [
                    f"layers.{layer_index}.mlp.dense_4h_to_h.weight",
                    f"layers.{layer_index}.mlp.dense_4h_to_h.weight",
                ],
            ]

            mappings.extend(layer_mappings)

        init_name_mappings(mappings)

        if config.architectures is not None:
            if "ChatGLMv3Model" not in config.architectures:
                for mapping in mappings:
                    mapping[0] = "transformer." + mapping[0]
                    if len(mapping) > 1 and mapping[1] is not None:
                        mapping[1] = "transformer." + mapping[1]

            mappings.append(["output_layer.weight","output_layer.weight","transpose",])
            mappings.append(["transformer.word_embeddings.weight","transformer.word_embeddings.weight",])
            mappings.append(["transformer.final_layernorm.weight","transformer.final_layernorm.weight",])

        init_name_mappings(mappings)
        return [StateDictNameMapping(*mapping) for mapping in mappings]

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: ChatGLMv3Config, is_split=True):

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
                "output_layer.weight": partial(fn, is_column=True),
                # Row Linear
                "transformer.word_embeddings.weight": partial(fn, is_column=False),
                "layers.0.self_attention.dense.weight": partial(fn, is_column=False),
                "layers.0.mlp.dense_4h_to_h.weight": partial(fn, is_column=False),
            }

            if not config.vocab_size % config.tensor_parallel_degree == 0:
                base_actions.pop("output_layer.weight")
                base_actions.pop("word_embeddings.weight")
            # Column Linear
            if config.fuse_attention_qkv:
                base_actions["layers.0.self_attention.query_key_value.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attention.query_key_value.bias"] = partial(fn, is_column=True)
            else:
                base_actions["layers.0.self_attention.q_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attention.q_proj.bias"] = partial(fn, is_column=True)
                # if we have enough num_key_value_heads to split, then split it.
                if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                    base_actions["layers.0.self_attention.k_proj.weight"] = partial(fn, is_column=True)
                    base_actions["layers.0.self_attention.v_proj.weight"] = partial(fn, is_column=True)
                    base_actions["layers.0.self_attention.k_proj.bias"] = partial(fn, is_column=True)
                    base_actions["layers.0.self_attention.v_proj.bias"] = partial(fn, is_column=True)

            if config.fuse_attention_ffn:
                base_actions["layers.0.mlp.dense_h_to_4h.weight"] = partial(
                    fn, is_column=True, is_naive_2fuse=True
                )
            else:
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
    def _get_fuse_or_split_param_mappings(cls, config: ChatGLMv3Config, is_fuse=False):
        # return parameter fuse utils
        from paddlenlp.transformers.conversion_utils import split_or_fuse_func

        fn = split_or_fuse_func(is_fuse=is_fuse)

        # last key is fused key, other keys are to be fused.
        fuse_qkv_keys = (
            "layers.0.self_attention.q_proj.weight",
            "layers.0.self_attention.k_proj.weight",
            "layers.0.self_attention.v_proj.weight",
            "layers.0.self_attention.query_key_value.weight",
        )
        fuse_qkv_bias = (
            "layers.0.self_attention.q_proj.bias",
            "layers.0.self_attention.k_proj.bias",
            "layers.0.self_attention.v_proj.bias",
            "layers.0.self_attention.query_key_value.bias",
        )

        fuse_gate_up_keys = (
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.dense_h_to_4h.weight",
        )
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
        fuse_attention_qkv = getattr(config, "fuse_attention_qkv", False)
        fuse_attention_ffn = getattr(config, "fuse_attention_ffn", False)

        final_actions = {}
        if is_fuse:
            if fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_qkv_keys])
                    final_actions[keys] = partial(
                        fn, is_qkv=True, num_heads=num_heads, num_key_value_heads=num_key_value_heads
                    )
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_qkv_bias])
                    final_actions[keys] = partial(
                        fn, is_qkv=True, num_heads=num_heads, num_key_value_heads=num_key_value_heads
                    )

            if fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_gate_up_keys])
                    final_actions[keys] = fn
        else:
            if not fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_qkv_keys])
                    final_actions[keys] = partial(
                        fn, split_nums=3, is_qkv=True, num_heads=num_heads, num_key_value_heads=num_key_value_heads
                    )
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_qkv_bias])
                    final_actions[keys] = partial(
                        fn, split_nums=3, is_qkv=True, num_heads=num_heads, num_key_value_heads=num_key_value_heads
                    )
            if not fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_gate_up_keys])
                    final_actions[keys] = partial(fn, split_nums=2)
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
                LlamaLMHead,
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
                                else self.transformer.config.initializer_range,
                                shape=layer.weight.shape,
                            )
                        )
                else:
                    layer.weight.set_value(
                        paddle.tensor.normal(
                            mean=0.0,
                            std=self.config.initializer_range
                            if hasattr(self.config, "initializer_range")
                            else self.transformer.config.initializer_range,
                            shape=layer.weight.shape,
                        )
                    )
        # Layer.apply is DFS https://github.com/PaddlePaddle/Paddle/blob/a6f5021fcc58b21f4414bae6bf4731ef6971582c/python/paddle/nn/layer/layers.py#L527-L530
        # sublayer is init first
        # scale RowParallelLinear weight
        with paddle.no_grad():
            if isinstance(layer, LlamaMLP):
                factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
                layer.dense_4h_to_h.weight.scale_(factor)
            if isinstance(layer, LlamaAttention):
                factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
                layer.dense.weight.scale_(factor)


class LlamaMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tensor_parallel_degree = config.tensor_parallel_degree
        self.fuse_attention_ffn = config.fuse_attention_ffn

        if config.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
            RowParallelLinear = linear_utils.RowSequenceParallelLinear
        else:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear
            RowParallelLinear = linear_utils.RowParallelLinear

        if config.tensor_parallel_degree > 1:
            if config.fuse_attention_ffn:
                self.dense_h_to_4h = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size * 2,
                    gather_output=False,
                    has_bias=False,
                )
            else:
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

            self.dense_4h_to_h = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            if config.fuse_attention_ffn:
                self.dense_h_to_4h = Linear(self.hidden_size, self.intermediate_size * 2, bias_attr=False)
            else:
                self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
                self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias_attr=False)

            self.dense_4h_to_h = Linear(self.intermediate_size, self.hidden_size, bias_attr=False)

    def forward(self, x):
        if self.fuse_attention_ffn:
            # FIXME(yangjianbang): use paddle's native swiglu
            if get_env_device() == "xpu":
                try:
                    import paddle_xpu_nn  # noqa: F821

                    out = self.dense_h_to_4h(x)
                    out = paddle_xpu_nn.xpu_swiglu(out, axis=-1, turn=True)
                    out = self.dense_4h_to_h(out)
                    return out
                except ImportError:
                    gate_out, up_out = paddle.chunk(self.dense_h_to_4h(x), chunks=2, axis=-1)
                    out = self.dense_4h_to_h(F.silu(gate_out) * up_out)
                    return out

            x = swiglu(self.dense_h_to_4h(x))
        else:
            x = swiglu(self.gate_proj(x), self.up_proj(x))
        out = self.dense_4h_to_h(x)
        return out



class Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: ChatGLMv3Config, layerwise_recompute: bool = False):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.head_dim = self.hidden_size // config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads
        assert config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.gqa_or_mqa = config.num_attention_heads != config.num_key_value_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.seq_length = config.seq_length
        self.sequence_parallel = config.sequence_parallel

        self.fuse_attention_qkv = config.fuse_attention_qkv

        self.kv_indices = None
        # Note that we will actually perform a recompute only if both enable_recompute and layerwise_recompute are set to True
        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity
        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree

            if self.num_key_value_heads % config.tensor_parallel_degree == 0:
                self.num_key_value_heads = self.num_key_value_heads // config.tensor_parallel_degree
            else:
                if self.fuse_attention_qkv:
                    # TODO(Yuang): support fusion for kv when kv heads cannot be divided by mp
                    raise ValueError(
                        f"fuse_attention_qkv can't be True when num_key_value_heads {config.num_key_value_heads} % tensor_parallel_degree {config.tensor_parallel_degree} != 0"
                    )
                logger.warning(
                    f"Get num_key_value_heads: {self.num_key_value_heads}, can't split to tensor_parallel_degree: {config.tensor_parallel_degree}, so we don't spilt key value weight."
                )
                self.kv_indices = paddle.to_tensor(
                    assign_kv_heads(self.num_key_value_heads, config.tensor_parallel_degree)[
                        config.tensor_parallel_rank
                    ]
                )

        self.use_fused_rope = config.use_fused_rope
        if self.use_fused_rope and get_env_device() not in ["npu", "xpu", "gcu"]:
            if "gpu" not in paddle.device.get_device() or fused_rotary_position_embedding is None:
                warnings.warn(
                    "Enable fuse rope in the config, but fuse rope is not available. "
                    "Will disable fuse rope. Try using latest gpu version of Paddle."
                )
                self.use_fused_rope = False

        if config.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
            RowParallelLinear = linear_utils.RowSequenceParallelLinear
        else:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear
            RowParallelLinear = linear_utils.RowParallelLinear

        if config.tensor_parallel_degree > 1:
            if self.fuse_attention_qkv:
                self.query_key_value = ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size + 2 * self.config.num_key_value_heads * self.head_dim,
                    has_bias=True,
                    gather_output=False,
                )
            else:
                self.q_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    has_bias=True,
                    gather_output=False,
                )
                if self.kv_indices is None:
                    self.k_proj = ColumnParallelLinear(
                        self.hidden_size,
                        self.config.num_key_value_heads * self.head_dim,
                        has_bias=True,
                        gather_output=False,
                    )
                    self.v_proj = ColumnParallelLinear(
                        self.hidden_size,
                        self.config.num_key_value_heads * self.head_dim,
                        has_bias=True,
                        gather_output=False,
                    )
                else:
                    self.k_proj = Linear(
                        self.hidden_size,
                        self.config.num_key_value_heads * self.head_dim,
                        bias_attr=True,
                    )
                    self.v_proj = Linear(
                        self.hidden_size,
                        self.config.num_key_value_heads * self.head_dim,
                        bias_attr=True,
                    )

        else:
            if self.fuse_attention_qkv:
                self.query_key_value = Linear(
                    self.hidden_size,
                    self.hidden_size + 2 * self.config.num_key_value_heads * self.head_dim,
                    bias_attr=True,
                )
            else:
                self.q_proj = Linear(
                    self.hidden_size,
                    self.hidden_size,
                    bias_attr=True,
                )
                self.k_proj = Linear(
                    self.hidden_size,
                    self.config.num_key_value_heads * self.head_dim,
                    bias_attr=True,
                )
                self.v_proj = Linear(
                    self.hidden_size,
                    self.config.num_key_value_heads * self.head_dim,
                    bias_attr=True,
                )

        if config.tensor_parallel_degree > 1:
            self.dense = RowParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
                input_is_parallel=True,
            )
        else:
            self.dense = Linear(
                self.hidden_size,
                self.hidden_size,
                bias_attr=False,
            )

        if config.rope:
            if config.use_long_sequence_strategies:
                self.rotary_emb = LongSequenceStrategies.build_long_sequence_strategy(
                    config.long_sequence_strategy_type,
                    config.long_sequence_strategy_name,
                    **config.long_sequence_init_args,
                )
            else:
                self._init_rope()

        self.reshard_layer = None
        if config.sep_parallel_degree > 1:
            assert self.num_key_value_heads % config.sep_parallel_degree == 0
            assert self.num_heads % config.sep_parallel_degree == 0
            self.reshard_layer = ReshardLayer()

        self.config = config

    def _init_rope(self):
        self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
            )

    def forward(
        self,
        hidden_states,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        alibi: Optional[paddle.Tensor] = None,
        npu_is_casual: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # [bs, seq_len, num_head * head_dim] -> [seq_len / n, bs, num_head * head_dim] (n is model parallelism)

        if self.fuse_attention_qkv:
            # print(f"hidden_states {hidden_states}")
            mix_layer = self.query_key_value(hidden_states)
            # print(f"mix_layer {mix_layer}")
            # NOTE for GQA attention fusion (compatible with MHA and MQA):
            # The weight for query_key_value is in shape like [hidden_size, hidden_size + 2 * num_kv_heads * head_dim].
            # After the projection, the mix_layer is in shape like [b, s, hidden_size + 2 * num_kv_heads * head_dim].
            # Reshape the mix_layer into a shape like [b, s, num_kv_heads, (num_groups + 2) * head_dim],
            # where num_groups = num_q_heads // num_kv_heads.
            # Split the mix_layer on the last axis into three sections [num_groups * head_dim, head_dim, head_dim]
            # to represent the q, k and v respectively.
            # The q is in the shape like [b, s, num_kv_heads, num_groups * head_dim].
            # The k and v are in the shape like [b, s, num_kv_heads, head_dim].
            # Under MHA, the q is ready for the following calculation since num_kv_heads == num_q_heads,
            # But for the GQA or MQA, q should be reshaped into [b, s, num_q_heads, head_dim].
            if self.reshard_layer is not None:
                if self.sequence_parallel:
                    assert self.seq_length % self.config.sep_parallel_degree == 0
                    mix_layer = paddle.reshape_(
                        mix_layer,
                        [
                            -1,
                            self.seq_length // self.config.sep_parallel_degree,
                            self.num_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim,
                        ],
                    )
                # [bs, seq_len / sep, num_head, head_dim] -> [bs, seq_len, num_head / sep, head_dim]
                mix_layer = self.reshard_layer(
                    mix_layer,
                    split_axis=2,
                    concat_axis=1,
                )
                mix_layer = paddle.reshape_(
                    mix_layer, [0, self.seq_length, -1, (self.num_key_value_groups + 2) * self.head_dim]
                )  # [bs, seq_len, num_head/k, 3*head_dim], k is sep degree
            else:
                if self.sequence_parallel:
                    target_shape = [
                        -1,
                        self.seq_length,
                        self.num_key_value_heads,
                        (self.num_key_value_groups + 2) * self.head_dim,
                    ]  #[b, s, 2, (16+2)*128]
                else:
                    target_shape = [0, 0, self.num_key_value_heads, (self.num_key_value_groups + 2) * self.head_dim]  #[b, s, 2, (16+2)*128]
                mix_layer = paddle.reshape_(mix_layer, target_shape)
            query_states, key_states, value_states = paddle.split(
                mix_layer,
                num_or_sections=[self.num_key_value_groups * self.head_dim, self.head_dim, self.head_dim],
                axis=-1,
            )  # [bs, seq_len, num_kv_heads, 3*head_dim]
            if self.gqa_or_mqa:
                query_states = paddle.reshape_(query_states, [0, 0, self.num_heads, self.head_dim])
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if self.reshard_layer is not None:
                if self.sequence_parallel:
                    assert self.seq_length % self.config.sep_parallel_degree == 0
                    query_states = paddle.reshape(
                        query_states,
                        [-1, self.seq_length // self.config.sep_parallel_degree, self.num_heads * self.head_dim],
                    )
                    key_states = paddle.reshape(
                        key_states,
                        [
                            -1,
                            self.seq_length // self.config.sep_parallel_degree,
                            self.num_key_value_heads * self.head_dim,
                        ],
                    )
                    value_states = paddle.reshape(
                        value_states,
                        [
                            -1,
                            self.seq_length // self.config.sep_parallel_degree,
                            self.num_key_value_heads * self.head_dim,
                        ],
                    )
                query_states = self.reshard_layer(
                    query_states,
                    split_axis=2,
                    concat_axis=1,
                )
                key_states = self.reshard_layer(
                    key_states,
                    split_axis=2,
                    concat_axis=1,
                )
                value_states = self.reshard_layer(
                    value_states,
                    split_axis=2,
                    concat_axis=1,
                )
                query_states = paddle.reshape(
                    query_states, [0, self.seq_length, -1, self.head_dim]
                )  # [bs, seq_len, num_head/k, head_dim], k is sep degree
                key_states = paddle.reshape(key_states, [0, self.seq_length, -1, self.head_dim])
                value_states = paddle.reshape(value_states, [0, self.seq_length, -1, self.head_dim])
            else:
                if self.sequence_parallel:
                    target_query_shape = [-1, self.seq_length, self.num_heads, self.head_dim]
                    target_key_value_shape = [-1, self.seq_length, self.num_key_value_heads, self.head_dim]
                else:
                    target_query_shape = [0, 0, self.num_heads, self.head_dim]
                    target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
                query_states = query_states.reshape(shape=target_query_shape)
                key_states = key_states.reshape(shape=target_key_value_shape)
                value_states = value_states.reshape(shape=target_key_value_shape)

        kv_seq_len = key_states.shape[-3]

        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]

        if self.config.rope:
            if self.reshard_layer is not None:
                batch_size, seq_length, _, _ = query_states.shape
                position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))
            if self.config.context_parallel_degree > 1:
                batch_size, seq_length, _, _ = query_states.shape
                group = fleet.get_hybrid_communicate_group().get_sep_parallel_group()
                chunk_size = seq_length // 2
                chunk_num = group.nranks * 2
                rank = group.rank
                first_chunk_ids = paddle.arange(rank * chunk_size, (rank + 1) * chunk_size, dtype="int64")
                second_chunk_ids = paddle.arange(
                    (chunk_num - rank - 1) * chunk_size, (chunk_num - rank) * chunk_size, dtype="int64"
                )
                position_ids = paddle.concat([first_chunk_ids, second_chunk_ids]).expand((batch_size, seq_length))
            if self.use_fused_rope:
                query_states, key_states = fusion_ops.fusion_rope(
                    query_states,
                    key_states,
                    value_states,
                    hidden_states,
                    position_ids,
                    past_key_value,
                    self.rotary_emb,
                    self.config.context_parallel_degree,
                )

            else:
                if self.config.context_parallel_degree > 1:
                    kv_seq_len *= self.config.context_parallel_degree
                if self.config.use_long_sequence_strategies:
                    cos, sin = self.rotary_emb(seq_len=kv_seq_len)
                    cos = cos[None, :, None, :]
                    sin = sin[None, :, None, :]
                    cos, sin = (
                        cos.cast(value_states.dtype) if cos.dtype != value_states.dtype else cos,
                        sin.cast(value_states.dtype) if sin.dtype != value_states.dtype else sin,
                    )
                else:
                    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)

        past_key_value = (key_states, value_states) if use_cache else None
        if self.kv_indices is not None:
            key_states = paddle.index_select(key_states, self.kv_indices, axis=2)
            value_states = paddle.index_select(value_states, self.kv_indices, axis=2)

        # TODO(wj-Mcat): use broadcast strategy when n_kv_heads = 1
        # repeat k/v heads if n_kv_heads < n_heads
        # paddle version > 2.6 or develop support flash-attn with gqa/mqa
        paddle_version = float(paddle.__version__[:3])
        if not self.config.use_flash_attention or ((paddle_version != 0.0) and (paddle_version <= 2.6)):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
        if (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "core_attn"
        ):
            outputs = recompute(
                scaled_dot_product_attention,
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                alibi,
                self.sequence_parallel,
                reshard_layer=self.reshard_layer,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            outputs = scaled_dot_product_attention(
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                alibi,
                self.sequence_parallel,
                reshard_layer=self.reshard_layer,
                npu_is_casual=npu_is_casual,
            )
        if output_attentions:
            attn_output, attn_weights = outputs
        else:
            attn_output = outputs

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        attn_output = self.dense(attn_output)
        # logger.info(f"dense； {attn_output}")
        if not output_attentions:
            attn_weights = None

        outputs = (attn_output,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs



class DecoderLayer(nn.Layer):
    def __init__(self, config, layerwise_recompute: bool = False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attention = Attention(config, layerwise_recompute)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)
        self.sequence_parallel = config.sequence_parallel
        # Note that we will actually perform a recompute only if both enable_recompute and layerwise_recompute are set to True
        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        alibi: Optional[paddle.Tensor] = None,
        npu_is_casual: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `cache` key value states are returned and can be used to speed up decoding
                (see `cache`).
            cache (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """

        # [bs * seq_len, embed_dim] -> [seq_len * bs / n, embed_dim] (sequence_parallel)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # logger.info(f"self.input_layernorm： {hidden_states}")

        # Self Attention
        has_gradient = not hidden_states.stop_gradient
        if (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "full_attn"
        ):
            outputs = recompute(
                self.self_attention,
                hidden_states,
                position_ids,
                past_key_value,
                attention_mask,
                output_attentions,
                use_cache,
                alibi,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            outputs = self.self_attention(
                hidden_states,
                position_ids,
                past_key_value,
                attention_mask,
                output_attentions,
                use_cache,
                alibi,
                npu_is_casual=npu_is_casual,
            )

        if type(outputs) is tuple:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        if output_attentions:
            self_attn_weights = outputs[1]

        if use_cache:
            present_key_value = outputs[2 if output_attentions else 1]

        hidden_states = residual + hidden_states
        # logger.info(f"the 1st residual + hidden_states： {hidden_states}")
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # logger.info(f"self.post_attention_layernorm： {hidden_states}")
        hidden_states = self.mlp(hidden_states)

        # logger.info(f" self.mlp： {hidden_states}")
        hidden_states = residual + hidden_states
        # logger.info(f" the 2st residual + hidden_states： {hidden_states}")

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class ChatGLMv3Model(ChatGLMv3PretrainedModel):

    def __init__(self, config: ChatGLMv3Config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.sequence_parallel = config.sequence_parallel
        self.recompute_granularity = config.recompute_granularity
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []
        self.config = config

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        if config.tensor_parallel_degree > 1 and config.vocab_size % config.tensor_parallel_degree == 0:
            self.word_embeddings = mpu.VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.word_embeddings = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        self.layers = nn.LayerList(
            [DecoderLayer(config, i not in self.no_recompute_layers) for i in range(config.num_hidden_layers)]
        )
        self.final_layernorm = LlamaRMSNorm(config)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length, dtype):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(attention_mask, dtype, tgt_length=input_shape[-1])
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape, past_key_values_length=past_key_values_length
                    )
                    if get_env_device() == "npu":
                        expanded_attn_mask = expanded_attn_mask.astype("bool") & combined_attention_mask.astype("bool")
                    else:
                        expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        if get_env_device() == "npu":
            x = paddle.to_tensor(0.0, dtype="float32")
            y = paddle.to_tensor(paddle.finfo(dtype).min, dtype="float32")
            expanded_attn_mask = expanded_attn_mask.astype("float32")
            expanded_attn_mask = paddle.where(expanded_attn_mask, x, y).astype(dtype)
        elif get_env_device() in ["xpu", "gcu"]:
            x = paddle.to_tensor(0.0, dtype=dtype)
            y = paddle.to_tensor(paddle.finfo(dtype).min, dtype=dtype)
            expanded_attn_mask = expanded_attn_mask.astype(dtype)
            expanded_attn_mask = paddle.where(expanded_attn_mask, x, y).astype(dtype)
        else:
            expanded_attn_mask = paddle.where(expanded_attn_mask, 0.0, paddle.finfo(dtype).min).astype(dtype)
        return expanded_attn_mask

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
        alibi=None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            alibi,
            use_reentrant=self.config.recompute_use_reentrant,
        )

        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        **kwargs,
    ):
        if self.sequence_parallel and use_cache:
            raise ValueError("We currently only support sequence parallel without cache.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        # NOTE: to make cache can be clear in-time
        past_key_values = list(past_key_values)

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = past_key_values[0][0].shape[1]
            seq_length_with_past += cache_length
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if self.config.context_parallel_degree > 1 and (attention_mask is not None or self.config.alibi):
            raise NotImplementedError("Ring FlashAttention dosen't support attention_mask or alibi")
        # embed positions
        if attention_mask is None:
            # [bs, seq_len]
            attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
        if self.config.alibi:
            if self.config.use_long_sequence_strategies:
                alibi_layer = LongSequenceStrategies.build_long_sequence_strategy(
                    self.config.long_sequence_strategy_type,
                    self.config.long_sequence_strategy_name,
                    **self.config.long_sequence_init_args,
                )
                alibi = alibi_layer(attention_mask, self.config.num_attention_heads, dtype=inputs_embeds.dtype)
            else:
                alibi = build_alibi_tensor(attention_mask, self.config.num_attention_heads, dtype=inputs_embeds.dtype)
            if self.config.tensor_parallel_degree > 1:
                block_size = self.config.num_attention_heads // self.config.tensor_parallel_degree
                alibi = alibi[
                    :,
                    self.config.tensor_parallel_rank
                    * block_size : (self.config.tensor_parallel_rank + 1)
                    * block_size,
                ]
                alibi = alibi.reshape([batch_size * block_size, 1, seq_length_with_past])
            else:
                alibi = alibi.reshape([batch_size * self.config.num_attention_heads, 1, seq_length_with_past])
        else:
            alibi = None

        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))

        use_casual_mask = get_use_casual_mask()

        if use_casual_mask:
            attention_mask = None
        else:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), cache_length, inputs_embeds.dtype
            )  # [bs, 1, seq_len, seq_len]

        is_casual = False

        if self.config.use_flash_attention and get_env_device() != "gcu":
            if use_casual_mask:
                is_casual = True
            else:
                is_casual = is_casual_mask(attention_mask)
            if get_env_device() != "npu":
                if is_casual and alibi is None:
                    attention_mask = None
            else:
                attention_mask = None if attention_mask is None else attention_mask.astype("bool")
        alibi = None if alibi is None else alibi.astype("bool")
        hidden_states = inputs_embeds
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(self.layers):
            # logger.info(f"**************************************the {idx} decoder layer begin**************************************")
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
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    alibi=alibi,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    alibi=alibi,
                    npu_is_casual=is_casual,
                )

            # NOTE: clear outdate cache after it has been used for memory saving
            past_key_value = past_key_values[idx] = None
            if type(layer_outputs) is tuple:
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            # logger.info(f"**************************************the {idx} decoder layer end**************************************")

        hidden_states = self.final_layernorm(hidden_states)
        # logger.info(f"self.final_layernorm{self.final_layernorm}")

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
        )



class ChatGLMv3ForCausalLM(ChatGLMv3PretrainedModel):
    enable_to_static_method = True
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = ChatGLMv3Model(config)
        self.output_layer = LlamaLMHead(config)
        self.criterion = LlamaPretrainingCriterion(config)

    def get_input_embeddings(self):
        return self.transformer.word_embeddings

    def set_input_embeddings(self, value):
        self.transformer.word_embeddings = value

    def get_output_embeddings(self):
        return self.output_layer

    def set_output_embeddings(self, new_embeddings):
        self.output_layer = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    def prepare_inputs_for_generation(
        self, input_ids, use_cache=False, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        position_ids = kwargs.get("position_ids", paddle.arange(seq_length).expand((batch_size, seq_length)))
        attention_mask = kwargs.get("attention_mask", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(axis=-1)
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
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _get_model_inputs_spec(self, dtype: str):
        return {
            "input_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "attention_mask": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "position_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        }

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, CausalLMOutputWithCrossAttentions) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat([position_ids, position_ids[..., -1:] + 1], axis=-1)

        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = paddle.concat(
                [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype)], axis=-1
            )

        return model_kwargs

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.transformer(
            input_ids,  # [bs, seq_len]
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # [bs, seq_len, dim]

        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is togather with ParallelCrossEntropy
        tensor_parallel_output = (
            self.config.tensor_parallel_output and labels is not None and self.config.tensor_parallel_degree > 1
        )

        logits = self.output_layer(hidden_states, tensor_parallel_output=tensor_parallel_output)
        # logger.info(f"self.output_layer{self.output_layer}")
        loss = None

        if labels is not None:
            loss = self.criterion(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def sample(
        self,
        input_ids,
        logits_processors,
        max_length,
        pad_token_id,
        eos_token_id,
        top_k=None,
        top_p=None,
        temperature=None,
        min_tokens_to_keep=1,
        stopping_criteria=None,
        streamer=None,
        fast_ptq_sampling=False,
        trunc_input=True,
        synced_gpus=False,
        **model_kwargs
    ):
        model_kwargs["use_cache"] = model_kwargs.get("use_cache", True)

        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        # max_length will be convert to MaxLengthCriteria
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            # logger.warning(
            #    "`max_length` is deprecated in this function, use"
            #    " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead."
            # )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        generate_end = False
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = paddle.to_tensor(0.0 if generate_end else 1.0)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # NOTE: to decrease ref-count and clear outdate cache in-time
            model_kwargs["cache"] = None
            model_kwargs["past_key_values"] = None
            outputs = self(**model_inputs)
            if synced_gpus and generate_end:
                continue  # don't waste resources running the code we don't need

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs

            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # sample
            origin_probs = F.softmax(logits)
            origin_probs = paddle.log(origin_probs)
            if temperature is not None and temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits)
            if paddle.device.is_compiled_with_custom_device("npu"):
                probs = paddle.cast(probs.cpu(), "float32")

            if top_k is not None and top_k != 0:
                probs = TopKProcess(probs, top_k, min_tokens_to_keep)
            if top_p is not None and top_p < 1.0:
                probs = TopPProcess(probs, top_p, min_tokens_to_keep)
            if paddle.device.is_compiled_with_custom_device("gcu"):
                probs = paddle.cast(probs, "float32")

            # multinomial already support fp16 and bf16 currently, fix issue: https://github.com/PaddlePaddle/Paddle/issues/51852
            next_tokens = paddle.multinomial(probs)

            if self.config.tensor_parallel_degree > 1:
                # Maybe no need to broadcast if seed is set correclty.
                from paddle.distributed import fleet

                try:
                    hcg = fleet.get_hybrid_communicate_group()
                    group = hcg.get_model_parallel_group()
                    src = group.get_model_parallel_group_src_rank()
                except:
                    group, src = None, 0
                paddle.distributed.broadcast(next_tokens, src=src, group=group)

            next_scores = paddle.index_sample(origin_probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)
            if streamer is not None:
                if self.config.tensor_parallel_rank == 0:
                    streamer.put(next_tokens.cpu())

            if stopping_criteria(input_ids, scores):
                generate_end = True

            if eos_token_id is not None:
                unfinished_flag = get_unfinished_flag(input_ids, unfinished_flag, eos_token_id)
                if not paddle.any(unfinished_flag):
                    generate_end = True

            # Stop when there is a </s> in all sentences
            if generate_end and not synced_gpus:
                break

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )
            if fast_ptq_sampling:
                break

        if streamer is not None:
            streamer.end()

        return input_ids[:, origin_len:] if trunc_input else input_ids, scores
