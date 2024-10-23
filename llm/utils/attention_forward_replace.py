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

from typing import Optional, Tuple

import paddle

import paddlenlp

group_size_ratio = 1 / 4


def qwen2_ssa_forward(
    self,
    hidden_states,
    position_ids: Optional[Tuple[paddle.Tensor]] = None,
    past_key_value: Optional[Tuple[paddle.Tensor]] = None,
    attention_mask: Optional[paddle.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
    from paddlenlp.transformers.qwen2.modeling import (
        apply_rotary_pos_emb,
        fused_rotary_position_embedding,
        recompute,
        repeat_kv,
        scaled_dot_product_attention,
    )

    bsz, q_len, _ = hidden_states.shape
    group_size = int(q_len * group_size_ratio)

    if q_len % group_size != 0:
        raise ValueError(f"q_len: {q_len} cannot be divided by group_size: {group_size}")

    query_states = self.q_proj(hidden_states)  # [4, 8192, 1536]
    key_states = self.k_proj(hidden_states)  # [4, 8192, 256]
    value_states = self.v_proj(hidden_states)  # [4, 8192, 256]

    if self.sequence_parallel:
        target_query_shape = [-1, self.seq_length, self.num_heads, self.head_dim]
        target_key_value_shape = [-1, self.seq_length, self.num_key_value_heads, self.head_dim]
    else:
        target_query_shape = [0, 0, self.num_heads, self.head_dim]
        target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]

    perm0 = [0, 2, 1, 3]
    query_states = query_states.reshape(shape=target_query_shape)
    key_states = key_states.reshape(shape=target_key_value_shape)
    value_states = value_states.reshape(shape=target_key_value_shape)
    kv_seq_len = key_states.shape[-3]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-3]
    if self.use_fused_rope:
        assert past_key_value is None, "fuse rotary not support cache kv for now"
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states, _ = fused_rotary_position_embedding(
            query_states,
            key_states,
            v=None,
            sin=sin,
            cos=cos,
            position_ids=position_ids,
            use_neox_rotary_style=False,
        )
    else:
        # value_states [4, 8192, 32, 128]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)  # [1, 4096, 1, 128]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        key_states = paddle.concat([past_key_value[0], key_states], axis=1)
        value_states = paddle.concat([past_key_value[1], value_states], axis=1)
    past_key_value = (key_states, value_states) if use_cache else None
    # [bsz, seq_len, num_heads, head_dim]
    key_states = repeat_kv(key_states, self.num_key_value_groups)  # [4, 8192, 12, 128]
    value_states = repeat_kv(value_states, self.num_key_value_groups)  # [4, 8192, 12, 128]
    query_states = query_states.transpose(perm=perm0)
    key_states = key_states.transpose(perm=perm0)
    value_states = value_states.transpose(perm=perm0)

    # qkv shoule be of size (bsz, num_heads, q_len, head_dim)
    # def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):  # 有inplace操作
    #     qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(shifts=(-group_size // 2), axis=2)
    #     qkv = qkv.transpose(perm=perm0).reshape([bsz * (q_len // group_size), group_size, num_heads, head_dim])
    #     return qkv

    def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):  # 无inplace操作
        rolled_qkv = qkv[:, num_heads // 2 :].roll(shifts=(-group_size // 2), axis=2)
        new_qkv = paddle.concat([qkv[:, : num_heads // 2], rolled_qkv], axis=2)
        new_qkv = new_qkv.transpose(perm=perm0).reshape([bsz * (q_len // group_size), group_size, num_heads, head_dim])
        return new_qkv

    query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    key_states = shift(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    value_states = shift(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)

    # bsz, group_size, num_heads, head_dim
    has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
    attention_mask = None
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
            self.training,
            self.sequence_parallel,
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
            self.training,
            self.sequence_parallel,
        )
    if output_attentions:
        attn_output, attn_weights = outputs
    else:
        attn_output = outputs
    assert attn_output.shape == [
        bsz * (q_len // group_size),
        group_size,
        self.head_dim * self.num_heads,
    ], f"attn_output shape is wrong of: {attn_output.shape}"
    attn_output = attn_output.reshape([bsz, q_len, self.num_heads, self.head_dim])
    # shift back
    # attn_output[:, :, self.num_heads // 2:] = attn_output[:, :, self.num_heads // 2:].roll(shifts=group_size // 2, axis=1)
    rolled_part = attn_output[:, :, self.num_heads // 2 :].roll(shifts=group_size // 2, axis=1)
    new_attn_output = paddle.concat([attn_output[:, :, : self.num_heads // 2], rolled_part], axis=2)
    attn_output = new_attn_output.reshape([bsz, q_len, self.num_heads * self.head_dim])

    attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None
    outputs = (attn_output,)

    if output_attentions:
        outputs += (attn_weights,)

    if use_cache:
        outputs += past_key_value
    if type(outputs) is tuple and len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def llama2_ssa_forward(
    self,
    hidden_states,
    position_ids: Optional[Tuple[paddle.Tensor]] = None,
    past_key_value: Optional[Tuple[paddle.Tensor]] = None,
    attention_mask: Optional[paddle.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    alibi: Optional[paddle.Tensor] = None,
    attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    npu_is_casual: bool = False,
    **kwargs,
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
    from paddlenlp.transformers.llama.modeling import (
        apply_rotary_pos_emb,
        fused_rotary_position_embedding,
        recompute,
        repeat_kv,
        scaled_dot_product_attention,
    )

    bsz, q_len, _ = hidden_states.shape
    group_size = int(q_len * group_size_ratio)
    if q_len % group_size != 0:
        raise ValueError(f"q_len: {q_len} cannot be divided by group_size: {group_size}")

    query_states = self.q_proj(hidden_states)  # [4, 8192, 1536]
    key_states = self.k_proj(hidden_states)  # [4, 8192, 256]
    value_states = self.v_proj(hidden_states)  # [4, 8192, 256]

    if self.sequence_parallel:
        target_query_shape = [-1, self.seq_length, self.num_heads, self.head_dim]
        target_key_value_shape = [-1, self.seq_length, self.num_key_value_heads, self.head_dim]
    else:
        target_query_shape = [0, 0, self.num_heads, self.head_dim]
        target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]

    perm0 = [0, 2, 1, 3]
    query_states = query_states.reshape(shape=target_query_shape)
    key_states = key_states.reshape(shape=target_key_value_shape)
    value_states = value_states.reshape(shape=target_key_value_shape)
    kv_seq_len = key_states.shape[-3]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-3]
    if self.use_fused_rope:
        assert past_key_value is None, "fuse rotary not support cache kv for now"
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states, _ = fused_rotary_position_embedding(
            query_states,
            key_states,
            v=None,
            sin=sin,
            cos=cos,
            position_ids=position_ids,
            use_neox_rotary_style=False,
        )
    else:
        # value_states [4, 8192, 32, 128]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)  # [1, 4096, 1, 128]

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    if past_key_value is not None:
        key_states = paddle.concat([past_key_value[0], key_states], axis=1)
        value_states = paddle.concat([past_key_value[1], value_states], axis=1)
    past_key_value = (key_states, value_states) if use_cache else None
    # [bsz, seq_len, num_heads, head_dim]
    key_states = repeat_kv(key_states, self.num_key_value_groups)  # [4, 8192, 12, 128]
    value_states = repeat_kv(value_states, self.num_key_value_groups)  # [4, 8192, 12, 128]

    key_states = key_states.transpose(perm=perm0)
    value_states = value_states.transpose(perm=perm0)
    query_states = query_states.transpose(perm=perm0)

    # qkv shoule be of size (bsz, num_heads, q_len, head_dim)
    # def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):  # 有inplace操作
    #     qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(shifts=(-group_size // 2), axis=2)
    #     qkv = qkv.transpose(perm=perm0).reshape([bsz * (q_len // group_size), group_size, num_heads, head_dim])
    #     return qkv

    def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):  # 无inplace操作
        rolled_qkv = qkv[:, num_heads // 2 :].roll(shifts=(-group_size // 2), axis=2)
        new_qkv = paddle.concat([qkv[:, : num_heads // 2], rolled_qkv], axis=2)
        new_qkv = new_qkv.transpose(perm=perm0).reshape([bsz * (q_len // group_size), group_size, num_heads, head_dim])
        return new_qkv

    query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    key_states = shift(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    value_states = shift(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    # bsz, group_size, num_heads, head_dim
    has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
    attention_mask = None
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
            attn_mask_startend_row_indices,
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
            attn_mask_startend_row_indices,
            self.sequence_parallel,
            reshard_layer=self.reshard_layer,
            npu_is_casual=npu_is_casual,
        )
    if output_attentions:
        attn_output, attn_weights = outputs
    else:
        attn_output = outputs
    assert attn_output.shape == [
        bsz * (q_len // group_size),
        group_size,
        self.head_dim * self.num_heads,
    ], f"attn_output shape is wrong of: {attn_output.shape}"
    attn_output = attn_output.reshape([bsz, q_len, self.num_heads, self.head_dim])
    # shift back
    # attn_output[:, :, self.num_heads // 2:] = attn_output[:, :, self.num_heads // 2:].roll(shifts=group_size // 2, axis=1)
    rolled_part = attn_output[:, :, self.num_heads // 2 :].roll(shifts=group_size // 2, axis=1)
    new_attn_output = paddle.concat([attn_output[:, :, : self.num_heads // 2], rolled_part], axis=2)
    attn_output = new_attn_output.reshape([bsz, q_len, self.num_heads * self.head_dim])

    attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None
    outputs = (attn_output,)

    if output_attentions:
        outputs += (attn_weights,)

    if use_cache:
        outputs += past_key_value
    if type(outputs) is tuple and len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def qwen2_origin_forward(
    self,
    hidden_states,
    position_ids: Optional[Tuple[paddle.Tensor]] = None,
    past_key_value: Optional[Tuple[paddle.Tensor]] = None,
    attention_mask: Optional[paddle.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
    from paddlenlp.transformers.qwen2.modeling import (
        apply_rotary_pos_emb,
        fused_rotary_position_embedding,
        recompute,
        repeat_kv,
        scaled_dot_product_attention,
    )

    """Input shape: Batch x Time x Channel"""
    # [bs, seq_len, num_head * head_dim] -> [seq_len / n, bs, num_head * head_dim] (n is model parallelism)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

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
    if self.use_fused_rope:
        assert past_key_value is None, "fuse rotary not support cache kv for now"
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states, _ = fused_rotary_position_embedding(
            query_states,
            key_states,
            v=None,
            sin=sin,
            cos=cos,
            position_ids=position_ids,
            use_neox_rotary_style=False,
        )
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # [bs, seq_len, num_head, head_dim]
    if past_key_value is not None:
        key_states = paddle.concat([past_key_value[0], key_states], axis=1)
        value_states = paddle.concat([past_key_value[1], value_states], axis=1)
    past_key_value = (key_states, value_states) if use_cache else None

    # TODO(wj-Mcat): use broadcast strategy when n_kv_heads = 1
    # repeat k/v heads if n_kv_heads < n_heads
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
            self.training,
            self.sequence_parallel,
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
            self.training,
            self.sequence_parallel,
        )
    if output_attentions:
        attn_output, attn_weights = outputs
    else:
        attn_output = outputs

    # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
    # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
    attn_output = self.o_proj(attn_output)

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


def replace_qwen2_attn(use_flash_attn=False, use_full=False, inference=False):
    """
    Replace the forward function of qwen2 attention with the SS attention.
    """
    paddlenlp.transformers.qwen2.modeling.Qwen2Attention.forward = qwen2_ssa_forward


def replace_llama2_attn(use_flash_attn=False, use_full=False, inference=False):
    """
    Replace the forward function of llama2 attention with the SS attention.
    """
    paddlenlp.transformers.llama.modeling.LlamaAttention.forward = llama2_ssa_forward
