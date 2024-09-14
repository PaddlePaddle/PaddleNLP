"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""

import paddle

if paddle.is_compiled_with_cuda():
    import paddlenlp_ops as custom_ops


def get_block_shape(
    sequence_lengths_stage: paddle.Tensor,
    sequence_lengths_remove: paddle.Tensor,
    max_len: paddle.Tensor,
    cum_offsets: paddle.Tensor,
    block_shape_q: int,
    group_size: int,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    batch_ids, tile_ids_per_batch, num_blocks_x_cpu = custom_ops.get_block_shape(
        sequence_lengths_stage, sequence_lengths_remove, max_len, cum_offsets, block_shape_q, group_size
    )
    return (batch_ids, tile_ids_per_batch, num_blocks_x_cpu)


def split_kv_block(
    sequence_lengths_stage: paddle.Tensor,
    sequence_lengths_remove: paddle.Tensor,
    max_len: paddle.Tensor,
    cum_offsets: paddle.Tensor,
    padding_len: int,
    num_rows_per_block: int,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    batch_ids, tile_ids_per_batch, num_blocks_cpu = custom_ops.get_block_shape(
        sequence_lengths_stage, sequence_lengths_remove, max_len, cum_offsets, padding_len, num_rows_per_block
    )
    return (batch_ids, tile_ids_per_batch, num_blocks_cpu)


def encoder_write_cache_with_rope(
    qkv: paddle.Tensor,
    rotary_emb: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    padding_offsets: paddle.Tensor,
    cum_offsets: paddle.Tensor,
    block_table: paddle.Tensor,
    batch_ids: paddle.Tensor,
    tile_ids_per_batch: paddle.Tensor,
    num_blocks_cpu: paddle.Tensor,
    max_enc_len: paddle.Tensor,
    qkv_out: paddle.Tensor,
    cache_k: paddle.Tensor,
    cache_v: paddle.Tensor,
    qkv_out_scales: paddle.Tensor,
    qkv_biases: paddle.Tensor,
    cache_k_scale: paddle.Tensor,
    cache_v_scale: paddle.Tensor,
    cache_k_zp: paddle.Tensor,
    cache_v_zp: paddle.Tensor,
    cache_quant_type_str: str,  # none, cache_int8, cache_int4
    max_input_len: int,
    num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> paddle.Tensor:
    return custom_ops.encoder_write_cache_with_rope(
        qkv,
        rotary_emb,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        padding_offsets,
        cum_offsets,
        block_table,
        batch_ids,
        tile_ids_per_batch,
        num_blocks_cpu,
        max_enc_len,
        qkv_out,
        cache_k,
        cache_v,
        qkv_out_scales,
        qkv_biases,
        cache_k_scale,
        cache_v_scale,
        cache_k_zp,
        cache_v_zp,
        cache_quant_type_str,
        max_input_len,
        num_heads,
        kv_num_heads,
        head_dim,
    )


def decoder_write_cache_with_rope(
    qkv: paddle.Tensor,
    rotary_emb: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    padding_offsets: paddle.Tensor,
    cum_offsets: paddle.Tensor,
    block_table: paddle.Tensor,
    max_dec_len: paddle.Tensor,
    qkv_out: paddle.Tensor,
    cache_k: paddle.Tensor,
    cache_v: paddle.Tensor,
    qkv_out_scales: paddle.Tensor,
    qkv_biases: paddle.Tensor,
    cache_k_scale: paddle.Tensor,
    cache_v_scale: paddle.Tensor,
    cache_k_zp: paddle.Tensor,
    cache_v_zp: paddle.Tensor,
    cache_quant_type_str: str,  # none, cache_int8, cache_int4
    max_input_len: int,
    num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> paddle.Tensor:
    return custom_ops.decoder_write_cache_with_rope(
        qkv,
        rotary_emb,
        seq_lens_encoder,
        seq_lens_decoder,
        padding_offsets,
        cum_offsets,
        block_table,
        max_dec_len,
        qkv_out,
        cache_k,
        cache_v,
        qkv_out_scales,
        qkv_biases,
        cache_k_scale,
        cache_v_scale,
        cache_k_zp,
        cache_v_zp,
        cache_quant_type_str,
        max_input_len,
        num_heads,
        kv_num_heads,
        head_dim,
    )


def append_attention(
    qkv: paddle.Tensor,
    cache_k: paddle.Tensor,
    cache_v: paddle.Tensor,
    seq_lens_q: paddle.Tensor,
    seq_lens_kv: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    padding_offsets: paddle.Tensor,
    cum_offsets: paddle.Tensor,
    block_table: paddle.Tensor,
    batch_ids: paddle.Tensor,
    tile_ids_per_batch: paddle.Tensor,
    num_blocks_x_cpu: paddle.Tensor,
    max_enc_len: paddle.Tensor,
    max_dec_len: paddle.Tensor,
    out_tmp: paddle.Tensor,
    attn_mask: paddle.Tensor,
    cache_k_scale: paddle.Tensor,
    cache_v_scale: paddle.Tensor,
    cache_k_zps: paddle.Tensor,
    cache_v_zps: paddle.Tensor,
    shift_bias: paddle.Tensor,
    smooth_weight: paddle.Tensor,
    cache_quant_type_str: str,
    max_input_len: int,
    block_shape_q: int,
    num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    in_scale: float,
    mode: str,  # encoder(encoder and append) or decoder
    max_partition_size: int,
    encoder_max_partition_size: int,
    speculate_max_draft_token_num: int = 5,
    causal: bool = True,
    is_decoder: bool = False,
    enable_prefill: bool = True,
) -> paddle.Tensor:
    return custom_ops.append_attention(
        qkv,
        cache_k,
        cache_v,
        seq_lens_q,
        seq_lens_kv,
        seq_lens_encoder,
        padding_offsets,
        cum_offsets,
        block_table,
        batch_ids,
        tile_ids_per_batch,
        num_blocks_x_cpu,
        max_enc_len,
        max_dec_len,
        out_tmp,
        attn_mask,
        cache_k_scale,
        cache_v_scale,
        cache_k_zps,
        cache_v_zps,
        shift_bias,
        smooth_weight,
        cache_quant_type_str,
        max_input_len,
        block_shape_q,
        num_heads,
        kv_num_heads,
        head_dim,
        in_scale,
        mode,
        max_partition_size,
        encoder_max_partition_size,
        speculate_max_draft_token_num,
        causal,
        is_decoder,
        enable_prefill,
    )


# # append w8a8c8 attention demo
def compute_append_attn(
    qkv,
    cache_k,
    cache_v,
    rotary_embs,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    padding_offsets,
    cum_offsets,
    block_tables,
    encoder_batch_ids,
    encoder_tile_ids_per_batch,
    encoder_num_blocks,
    kv_batch_ids,
    kv_tile_ids_per_batch,
    kv_num_blocks,
    decoder_batch_ids,
    decoder_tile_ids_per_batch,
    decoder_num_blocks,
    max_enc_len,
    max_dec_len,
    qkv_out_scales,
    qkv_biases,
    k_quant_scales,
    v_quant_scales,
    k_dequant_scales,
    v_dequant_scales,
    cache_k_zps,
    cache_v_zps,
    linear_shifts,
    linear_smooths,
    cache_quant_type_str,
    max_input_len,
    num_heads,
    kv_num_heads,
    head_dim,
    out_linear_in_scale,
    encoder_block_shape_q=128,
    decoder_block_shape_q=128,
    block_size=64,
    max_partition_size=1024,
    encoder_max_partition_size=8192,
    speculate_max_draft_token_num=5,
    causal=True,
    is_decoder=False,
    enable_prefill=True,
):

    qkv_out = paddle.empty_like(qkv, dtype=paddle.get_default_dtype())
    fmha_out = paddle.empty(
        [qkv.shape[0], num_heads * head_dim], dtype="int8" if out_linear_in_scale > 0.0 else qkv_out.dtype
    )

    # paddle.device.synchronize()
    _ = encoder_write_cache_with_rope(
        qkv,
        rotary_embs,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        padding_offsets,
        cum_offsets,
        block_tables,
        encoder_batch_ids,
        encoder_tile_ids_per_batch,
        encoder_num_blocks,
        max_enc_len,
        qkv_out,
        cache_k,
        cache_v,
        qkv_out_scales,
        qkv_biases,
        k_quant_scales,
        v_quant_scales,
        cache_k_zps,  # C4 zero points
        cache_v_zps,  # C4 zero points
        cache_quant_type_str,
        max_input_len,
        num_heads,
        kv_num_heads,
        head_dim,
    )
    # paddle.device.synchronize()
    _ = append_attention(
        qkv_out,
        cache_k,
        cache_v,
        seq_lens_this_time,
        seq_lens_decoder,
        seq_lens_encoder,
        padding_offsets,
        cum_offsets,
        block_tables,
        encoder_batch_ids,  # from get_block_shape
        encoder_tile_ids_per_batch,  # from get_block_shape
        encoder_num_blocks,  # from get_block_shape
        max_enc_len,
        max_dec_len,
        fmha_out,
        None,  # attn_mask
        k_dequant_scales,
        v_dequant_scales,
        cache_k_zps,  # C4 zero points
        cache_v_zps,  # C4 zero points
        linear_shifts,
        linear_smooths,
        cache_quant_type_str,  # one of "none "cache_int8", "cache_int4"
        max_input_len,
        encoder_block_shape_q,  #
        num_heads,
        kv_num_heads,
        head_dim,
        out_linear_in_scale,
        "encoder",
        max_partition_size,  # max_partition_size
        encoder_max_partition_size,  # encoder_max_partition_size
        speculate_max_draft_token_num,
        causal,
        is_decoder,
        enable_prefill,
    )
    # paddle.device.synchronize()

    _ = decoder_write_cache_with_rope(
        qkv,
        rotary_embs,
        seq_lens_this_time,
        seq_lens_decoder,
        seq_lens_encoder,
        padding_offsets,
        block_tables,
        max_dec_len,
        qkv_out,
        cache_k,
        cache_v,
        qkv_out_scales,
        qkv_biases,
        k_quant_scales,
        v_quant_scales,
        cache_k_zps,  # C4 zero points
        cache_v_zps,  # C4 zero points
        cache_quant_type_str,
        max_input_len,
        num_heads,
        head_dim,
        kv_num_heads,
    )
    # paddle.device.synchronize()
    _ = append_attention(
        qkv_out,
        cache_k,
        cache_v,
        seq_lens_this_time,
        seq_lens_decoder,
        seq_lens_encoder,
        padding_offsets,
        cum_offsets,
        block_tables,
        decoder_batch_ids,  # from get_block_shape
        decoder_tile_ids_per_batch,  # from get_block_shape
        decoder_num_blocks,  # from get_block_shape
        max_enc_len,
        max_dec_len,
        fmha_out,
        None,
        k_dequant_scales,
        v_dequant_scales,
        cache_k_zps,  # C4 zero points
        cache_v_zps,  # C4 zero points
        linear_shifts,
        linear_smooths,
        cache_quant_type_str,  # one of "none "cache_int8", "cache_int4"
        max_input_len,
        decoder_block_shape_q,  # decoder_block_shape_q
        num_heads,
        kv_num_heads,
        head_dim,
        out_linear_in_scale,
        "decoder",
        max_partition_size,  # max_partition_size
        encoder_max_partition_size,  # encoder_max_partition_size
        speculate_max_draft_token_num,
        causal,
        is_decoder,
        enable_prefill,
    )
    # paddle.device.synchronize()
    return fmha_out
