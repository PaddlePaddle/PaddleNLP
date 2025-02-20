# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from typing import List, Optional

import paddle

import paddlenlp.custom_ops._C as _C


def append_attention(
    qkv: paddle.Tensor,
    key_cache: paddle.Tensor,
    value_cache: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    padding_offsets: paddle.Tensor,
    cum_offsets: paddle.Tensor,
    block_tables: paddle.Tensor,
    encoder_batch_ids: paddle.Tensor,
    encoder_tile_ids_per_batch: paddle.Tensor,
    encoder_num_blocks: paddle.Tensor,
    kv_batch_ids: paddle.Tensor,
    kv_tile_ids_per_batch: paddle.Tensor,
    kv_num_blocks: paddle.Tensor,
    decoder_batch_ids: paddle.Tensor,
    decoder_tile_ids_per_batch: paddle.Tensor,
    decoder_num_blocks: paddle.Tensor,
    max_enc_len_this_time: paddle.Tensor,
    max_dec_len_this_time: paddle.Tensor,
    max_len_kv: paddle.Tensor,
    rotary_embs: Optional[paddle.Tensor],
    attn_mask: Optional[paddle.Tensor],
    qkv_bias: Optional[paddle.Tensor],
    qkv_out_scales: Optional[paddle.Tensor],
    cache_k_quant_scales: Optional[paddle.Tensor],
    cache_v_quant_scales: Optional[paddle.Tensor],
    cache_k_dequant_scales: Optional[paddle.Tensor],
    cache_v_dequant_scales: Optional[paddle.Tensor],
    cache_k_zp: Optional[paddle.Tensor],
    cache_v_zp: Optional[paddle.Tensor],
    out_linear_shifts: Optional[paddle.Tensor],
    out_linear_smooths: Optional[paddle.Tensor],
    compute_type: str,
    cache_quant_type: str,
    use_neox_rotary_style: bool,
    max_input_length: int,
    softmax_scale: float,
    quant_max_bound: float,
    quant_min_bound: float,
    out_linear_in_scale: float,
    speculate_max_draft_token_num: int,
    causal: bool,
    speculate_decoder: bool,
) -> List[paddle.Tensor]:

    return _C.append_attention(
        qkv,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
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
        max_enc_len_this_time,
        max_dec_len_this_time,
        max_len_kv,
        rotary_embs,
        attn_mask,
        qkv_bias,
        qkv_out_scales,
        cache_k_quant_scales,
        cache_v_quant_scales,
        cache_k_dequant_scales,
        cache_v_dequant_scales,
        cache_k_zp,
        cache_v_zp,
        out_linear_shifts,
        out_linear_smooths,
        compute_type,
        cache_quant_type,
        use_neox_rotary_style,
        max_input_length,
        softmax_scale,
        quant_max_bound,
        quant_min_bound,
        out_linear_in_scale,
        speculate_max_draft_token_num,
        causal,
        speculate_decoder,
    )


def avx_weight_only(x: paddle.Tensor, weight: paddle.Tensor, alog: str, trans: bool) -> List[paddle.Tensor]:

    return _C.avx_weight_only(x, weight, alog, trans)


def dequant_int8(intput: paddle.Tensor, out_scale: paddle.Tensor, dtype: str) -> List[paddle.Tensor]:

    return _C.dequant_int8(intput, out_scale, dtype)


def draft_model_postprocess(
    base_model_draft_tokens: paddle.Tensor,
    base_model_seq_lens_this_time: paddle.Tensor,
    base_model_seq_lens_encoder: paddle.Tensor,
    base_model_stop_flags: paddle.Tensor,
) -> List[paddle.Tensor]:
    return _C.draft_model_postprocess(
        base_model_draft_tokens, base_model_seq_lens_this_time, base_model_seq_lens_encoder, base_model_stop_flags
    )


def draft_model_preprocess(
    draft_tokens: paddle.Tensor,
    input_ids: paddle.Tensor,
    stop_flags: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    step_idx: paddle.Tensor,
    first_token_record: paddle.Tensor,
    not_need_stop: paddle.Tensor,
    accept_tokens: paddle.Tensor,
    accept_num: paddle.Tensor,
    base_model_seq_lens_encoder: paddle.Tensor,
    base_model_seq_lens_decoder: paddle.Tensor,
    base_model_step_idx: paddle.Tensor,
    base_model_stop_flags: paddle.Tensor,
    base_model_draft_tokens: paddle.Tensor,
    max_draft_token: int,
    truncate_first_token: bool,
) -> List[paddle.Tensor]:
    return _C.draft_model_preprocess(
        draft_tokens,
        input_ids,
        stop_flags,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        step_idx,
        first_token_record,
        not_need_stop,
        accept_tokens,
        accept_num,
        base_model_seq_lens_encoder,
        base_model_seq_lens_decoder,
        base_model_step_idx,
        base_model_stop_flags,
        base_model_draft_tokens,
        max_draft_token,
        truncate_first_token,
    )


def draft_model_set_value_by_flags(
    draft_tokens: paddle.Tensor,
    pre_ids_all: paddle.Tensor,
    stop_flags: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    step_idx: paddle.Tensor,
) -> paddle.Tensor:
    return _C.draft_model_set_value_by_flags(
        draft_tokens, pre_ids_all, stop_flags, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, step_idx
    )


def draft_model_update(
    inter_next_tokens: paddle.Tensor,
    draft_tokens: paddle.Tensor,
    pre_ids: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    step_idx: paddle.Tensor,
    output_cum_offsets: paddle.Tensor,
    stop_flags: paddle.Tensor,
    not_need_stop: paddle.Tensor,
    max_dec_len: paddle.Tensor,
    end_ids: paddle.Tensor,
    base_model_draft_tokens: paddle.Tensor,
    max_seq_len: int,
    substep: int,
) -> List[paddle.Tensor]:
    return _C.draft_model_update(
        inter_next_tokens,
        draft_tokens,
        pre_ids,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        step_idx,
        output_cum_offsets,
        stop_flags,
        not_need_stop,
        max_dec_len,
        end_ids,
        base_model_draft_tokens,
        max_seq_len,
        substep,
    )


def eagle_get_base_model_hidden_states(
    input: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    stop_flags: paddle.Tensor,
    accept_nums: paddle.Tensor,
    base_model_seq_lens_this_time: paddle.Tensor,
    base_model_seq_lens_encoder: paddle.Tensor,
    actual_draft_token_num: int,
) -> List[paddle.Tensor]:
    return _C.eagle_get_base_model_hidden_states(
        input,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        stop_flags,
        accept_nums,
        base_model_seq_lens_this_time,
        base_model_seq_lens_encoder,
        actual_draft_token_num,
    )


def eagle_get_self_hidden_states(
    input: paddle.Tensor,
    last_seq_lens_this_time: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    step_idx: paddle.Tensor,
) -> List[paddle.Tensor]:
    return _C.eagle_get_self_hidden_states(input, last_seq_lens_this_time, seq_lens_this_time, step_idx)


def encode_rotary_qk(
    q: paddle.Tensor,
    kv: paddle.Tensor,
    rotary_emb: paddle.Tensor,
    seq_lens: paddle.Tensor,
    rotary_emb_dims: int,
    use_neox: bool,
) -> None:

    _C.encode_rotary_qk(q, kv, rotary_emb, seq_lens, rotary_emb_dims, use_neox)


def flash_attn_bwd(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    out: paddle.Tensor,
    softmax_lse: paddle.Tensor,
    seed_offset: paddle.Tensor,
    attn_mask: Optional[paddle.Tensor],
    out_grad: paddle.Tensor,
    dropout: float,
    causal: bool,
) -> List[paddle.Tensor]:

    return _C.flash_attn_bwd(q, k, v, out, softmax_lse, seed_offset, attn_mask, out_grad, dropout, causal)


def cutlass_fp8_fp8_fp8_dual_gemm_fused(
    x: paddle.Tensor,
    y0: paddle.Tensor,
    y1: paddle.Tensor,
    bias0: Optional[paddle.Tensor],
    bias1: Optional[paddle.Tensor],
    transpose_x: bool,
    transpose_y: bool,
    scale0: float,
    scale1: float,
    scale_out: float,
    act: str,
) -> List[paddle.Tensor]:

    return _C.cutlass_fp8_fp8_fp8_dual_gemm_fused(
        x, y0, y1, bias0, bias1, transpose_x, transpose_y, scale0, scale1, scale_out, act
    )


def cutlass_fp8_fp8_half_gemm_fused(
    x: paddle.Tensor,
    y: paddle.Tensor,
    bias: Optional[paddle.Tensor],
    transpose_x: bool,
    transpose_y: bool,
    scale: float,
    output_type: str,
    act: str,
) -> List[paddle.Tensor]:

    return _C.cutlass_fp8_fp8_half_gemm_fused(x, y, bias, transpose_x, transpose_y, scale, output_type, act)


def fused_get_rotary_embedding(
    input_ids: paddle.Tensor,
    position_ids: paddle.Tensor,
    head_dim_shape_tensor: paddle.Tensor,
    prompt_num: int,
    theta: float,
    use_neox: bool,
) -> List[paddle.Tensor]:

    return _C.fused_get_rotary_embedding(input_ids, position_ids, head_dim_shape_tensor, prompt_num, theta, use_neox)


def fused_rotary_position_encoding(
    query: paddle.Tensor,
    key: paddle.Tensor,
    position_ids: paddle.Tensor,
    cos_sin_cache: paddle.Tensor,
    head_size: int,
    is_neox: bool,
) -> List[paddle.Tensor]:
    return _C.fused_rotary_position_encoding(query, key, position_ids, cos_sin_cache, head_size, is_neox)


def gemm_dequant(x: paddle.Tensor, y: paddle.Tensor, scale: paddle.Tensor, out_dtype: str) -> List[paddle.Tensor]:

    return _C.gemm_dequant(x, y, scale, out_dtype)


def get_block_shape_and_split_kv_block(
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    max_enc_len_this_time: paddle.Tensor,
    max_dec_len_this_time: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    cum_offsets: paddle.Tensor,
    group_size: int,
    block_size: int,
    decoder_step_token_num: int,
) -> List[paddle.Tensor]:

    return _C.get_block_shape_and_split_kv_block(
        seq_lens_encoder,
        seq_lens_decoder,
        max_enc_len_this_time,
        max_dec_len_this_time,
        seq_lens_this_time,
        cum_offsets,
        group_size,
        block_size,
        decoder_step_token_num,
    )


def get_output(x: paddle.Tensor, rank_id: int, wait_flag: float) -> None:

    _C.get_output(x, rank_id, wait_flag)


def get_padding_offset(
    input_ids: paddle.Tensor, cum_offsets: paddle.Tensor, token_num: paddle.Tensor, seq_len: paddle.Tensor
) -> List[paddle.Tensor]:

    return _C.get_padding_offset(input_ids, cum_offsets, token_num, seq_len)


def get_padding_offset_v2(
    input_ids: paddle.Tensor,
    cum_offsets: paddle.Tensor,
    token_num: paddle.Tensor,
    seq_len: paddle.Tensor,
    draft_tokens: Optional[paddle.Tensor],
    seq_lens_encoder: Optional[paddle.Tensor],
) -> List[paddle.Tensor]:

    return _C.get_padding_offset_v2(input_ids, cum_offsets, token_num, seq_len, draft_tokens, seq_lens_encoder)


def get_position_ids(
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    position_ids: paddle.Tensor,
) -> List[paddle.Tensor]:
    return _C.get_position_ids(seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, position_ids)


def get_token_penalty_multi_scores(
    pre_ids: paddle.Tensor,
    logits: paddle.Tensor,
    penalty_scores: paddle.Tensor,
    frequency_scores: paddle.Tensor,
    presence_scores: paddle.Tensor,
    cur_len: paddle.Tensor,
    min_len: paddle.Tensor,
    eos_token_id: paddle.Tensor,
) -> List[paddle.Tensor]:

    return _C.get_token_penalty_multi_scores(
        pre_ids, logits, penalty_scores, frequency_scores, presence_scores, cur_len, min_len, eos_token_id
    )


def ngram_match(
    input_ids: paddle.Tensor,
    input_ids_len: paddle.Tensor,
    pre_ids: paddle.Tensor,
    step_idx: paddle.Tensor,
    draft_token_num: paddle.Tensor,
    draft_tokens: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    max_dec_len: paddle.Tensor,
    real_batch_size: int,
    max_ngram_size: int,
    max_draft_tokens: int,
) -> None:

    _C.ngram_match(
        input_ids,
        input_ids_len,
        pre_ids,
        step_idx,
        draft_token_num,
        draft_tokens,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        max_dec_len,
        real_batch_size,
        max_ngram_size,
        max_draft_tokens,
    )


def qkv_transpose_split(
    qkv: paddle.Tensor,
    padding_offset: paddle.Tensor,
    seq_lens: paddle.Tensor,
    input_ids: paddle.Tensor,
    num_head: int,
    head_size: int,
) -> List[paddle.Tensor]:

    return _C.qkv_transpose_split(qkv, padding_offset, seq_lens, input_ids, num_head, head_size)


def quant_int8(
    intput: paddle.Tensor,
    shift: paddle.Tensor,
    smooth: paddle.Tensor,
    scale: float,
    round_type: int,
    max_bound: float,
    min_bound: float,
) -> List[paddle.Tensor]:

    return _C.quant_int8(intput, shift, smooth, scale, round_type, max_bound, min_bound)


def rebuild_padding(
    tmp_out: paddle.Tensor, padding_offset: paddle.Tensor, seq_lens: paddle.Tensor, input_ids: paddle.Tensor
) -> List[paddle.Tensor]:

    return _C.rebuild_padding(tmp_out, padding_offset, seq_lens, input_ids)


def rebuild_padding_v2(
    tmp_out: paddle.Tensor,
    cum_offsets: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    output_padding_offset: paddle.Tensor,
    max_input_length: int,
) -> List[paddle.Tensor]:

    return _C.rebuild_padding_v2(
        tmp_out, cum_offsets, seq_lens_decoder, seq_lens_encoder, output_padding_offset, max_input_length
    )


def save_output(x: paddle.Tensor, not_need_stop: paddle.Tensor, rank_id: int) -> None:

    _C.save_output(x, not_need_stop, rank_id)


def save_with_output(
    x: paddle.Tensor, batch_idx: paddle.Tensor, step_idx: paddle.Tensor, file_path: str, rank_id: int
) -> List[paddle.Tensor]:

    return _C.save_with_output(x, batch_idx, step_idx, file_path, rank_id)


def set_preids_token_penalty_multi_scores(
    pre_ids: paddle.Tensor,
    input_ids: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    step_idx: paddle.Tensor,
    stop_flags: paddle.Tensor,
    logits: paddle.Tensor,
    penalty_scores: paddle.Tensor,
    frequency_scores: paddle.Tensor,
    presence_scores: paddle.Tensor,
    temperatures: paddle.Tensor,
    bad_tokens: paddle.Tensor,
    cur_len: paddle.Tensor,
    min_len: paddle.Tensor,
    eos_token_id: paddle.Tensor,
) -> None:

    _C.set_preids_token_penalty_multi_scores(
        pre_ids,
        input_ids,
        seq_lens_encoder,
        seq_lens_decoder,
        step_idx,
        stop_flags,
        logits,
        penalty_scores,
        frequency_scores,
        presence_scores,
        temperatures,
        bad_tokens,
        cur_len,
        min_len,
        eos_token_id,
    )


def set_stop_value_multi_ends(
    topk_ids: paddle.Tensor, stop_flags: paddle.Tensor, end_ids: paddle.Tensor, mode: int
) -> List[paddle.Tensor]:

    return _C.set_stop_value_multi_ends(topk_ids, stop_flags, end_ids, mode)


def set_value_by_flags_and_idx(
    pre_ids_all: paddle.Tensor, pre_ids_now: paddle.Tensor, step_idx: paddle.Tensor, stop_flags: paddle.Tensor
) -> List[paddle.Tensor]:

    return _C.set_value_by_flags_and_idx(pre_ids_all, pre_ids_now, step_idx, stop_flags)


def speculate_clear_accept_nums(accept_num: paddle.Tensor, seq_lens_decoder: paddle.Tensor) -> List[paddle.Tensor]:
    return _C.speculate_clear_accept_nums(accept_num, seq_lens_decoder)


def speculate_get_output(x: paddle.Tensor, rank_id: int, wait_flag: bool) -> None:

    _C.speculate_get_output(x, rank_id, wait_flag)


def speculate_get_output_padding_offset(
    output_cum_offsets_tmp: paddle.Tensor,
    out_token_num: paddle.Tensor,
    seq_lens_output: paddle.Tensor,
    max_seq_len: int,
) -> List[paddle.Tensor]:

    return _C.speculate_get_output_padding_offset(output_cum_offsets_tmp, out_token_num, seq_lens_output, max_seq_len)


def speculate_get_seq_lens_output(
    seq_lens_this_time: paddle.Tensor, seq_lens_encoder: paddle.Tensor, seq_lens_decoder: paddle.Tensor
) -> List[paddle.Tensor]:

    return _C.speculate_get_seq_lens_output(seq_lens_this_time, seq_lens_encoder, seq_lens_decoder)


def speculate_get_token_penalty_multi_scores(
    pre_ids: paddle.Tensor,
    logits: paddle.Tensor,
    penalty_scores: paddle.Tensor,
    frequency_scores: paddle.Tensor,
    presence_scores: paddle.Tensor,
    temperatures: paddle.Tensor,
    bad_tokens: paddle.Tensor,
    cur_len: paddle.Tensor,
    min_len: paddle.Tensor,
    eos_token_id: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    output_padding_offset: paddle.Tensor,
    output_cum_offsets: paddle.Tensor,
    max_seq_len: int,
) -> None:

    _C.speculate_get_token_penalty_multi_scores(
        pre_ids,
        logits,
        penalty_scores,
        frequency_scores,
        presence_scores,
        temperatures,
        bad_tokens,
        cur_len,
        min_len,
        eos_token_id,
        seq_lens_this_time,
        output_padding_offset,
        output_cum_offsets,
        max_seq_len,
    )


def speculate_save_output(
    accept_tokens: paddle.Tensor, accept_num: paddle.Tensor, not_need_stop: paddle.Tensor, rank_id: int
) -> None:

    _C.speculate_save_output(accept_tokens, accept_num, not_need_stop, rank_id)


def speculate_set_value_by_flags_and_idx(
    pre_ids_all: paddle.Tensor,
    accept_tokens: paddle.Tensor,
    accept_num: paddle.Tensor,
    stop_flags: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    step_idx: paddle.Tensor,
) -> None:

    _C.speculate_set_value_by_flags_and_idx(
        pre_ids_all,
        accept_tokens,
        accept_num,
        stop_flags,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        step_idx,
    )


def speculate_update(
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    not_need_stop: paddle.Tensor,
    draft_tokens: paddle.Tensor,
    actual_draft_token_nums: paddle.Tensor,
    accept_tokens: paddle.Tensor,
    accept_num: paddle.Tensor,
    stop_flags: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    is_block_step: paddle.Tensor,
) -> List[paddle.Tensor]:
    return _C.speculate_update(
        seq_lens_encoder,
        seq_lens_decoder,
        not_need_stop,
        draft_tokens,
        actual_draft_token_nums,
        accept_tokens,
        accept_num,
        stop_flags,
        seq_lens_this_time,
        is_block_step,
    )


def speculate_verify(
    accept_tokens: paddle.Tensor,
    accept_num: paddle.Tensor,
    step_idx: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    stop_flags: paddle.Tensor,
    draft_tokens: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    verify_tokens: paddle.Tensor,
    verify_scores: paddle.Tensor,
    max_dec_len: paddle.Tensor,
    end_tokens: paddle.Tensor,
    is_block_step: paddle.Tensor,
    output_cum_offsets: paddle.Tensor,
    actual_candidate_len: paddle.Tensor,
    actual_draft_token_nums: paddle.Tensor,
    topp: paddle.Tensor,
    max_seq_len: int,
    verify_window: int,
    enable_topp: bool,
) -> List[paddle.Tensor]:
    return _C.speculate_verify(
        accept_tokens,
        accept_num,
        step_idx,
        seq_lens_encoder,
        seq_lens_decoder,
        stop_flags,
        draft_tokens,
        seq_lens_this_time,
        verify_tokens,
        verify_scores,
        max_dec_len,
        end_tokens,
        is_block_step,
        output_cum_offsets,
        actual_candidate_len,
        actual_draft_token_nums,
        topp,
        max_seq_len,
        verify_window,
        enable_topp,
    )


def step_paddle(
    stop_flags: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    ori_seq_lens_encoder: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    block_tables: paddle.Tensor,
    encoder_block_lens: paddle.Tensor,
    is_block_step: paddle.Tensor,
    step_block_list: paddle.Tensor,
    step_lens: paddle.Tensor,
    recover_block_list: paddle.Tensor,
    recover_lens: paddle.Tensor,
    need_block_list: paddle.Tensor,
    need_block_len: paddle.Tensor,
    used_list_len: paddle.Tensor,
    free_list: paddle.Tensor,
    free_list_len: paddle.Tensor,
    input_ids: paddle.Tensor,
    pre_ids: paddle.Tensor,
    step_idx: paddle.Tensor,
    next_tokens: paddle.Tensor,
    first_token_ids: paddle.Tensor,
    block_size: int,
    encoder_decoder_block_num: int,
    speculate_step_token_num: int,
) -> List[paddle.Tensor]:

    return _C.step_paddle(
        stop_flags,
        seq_lens_this_time,
        ori_seq_lens_encoder,
        seq_lens_encoder,
        seq_lens_decoder,
        block_tables,
        encoder_block_lens,
        is_block_step,
        step_block_list,
        step_lens,
        recover_block_list,
        recover_lens,
        need_block_list,
        need_block_len,
        used_list_len,
        free_list,
        free_list_len,
        input_ids,
        pre_ids,
        step_idx,
        next_tokens,
        first_token_ids,
        block_size,
        encoder_decoder_block_num,
        speculate_step_token_num,
    )


def top_p_candidates(
    probs: paddle.Tensor,
    top_p: paddle.Tensor,
    output_padding_offset: paddle.Tensor,
    candidates_len: int,
    max_seq_len: int,
) -> List[paddle.Tensor]:

    return _C.top_p_candidates(probs, top_p, output_padding_offset, candidates_len, max_seq_len)


def top_p_sampling_reject(probs: paddle.Tensor, top_p: paddle.Tensor, seed: int) -> List[paddle.Tensor]:

    return _C.top_p_sampling_reject(probs, top_p, seed)


def transpose_remove_padding(
    input: paddle.Tensor, seq_lens: paddle.Tensor, padding_offset: paddle.Tensor
) -> List[paddle.Tensor]:

    return _C.transpose_remove_padding(input, seq_lens, padding_offset)


def update_inputs_v2(
    stop_flags: paddle.Tensor,
    step_idx: paddle.Tensor,
    not_need_stop: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    max_dec_len: paddle.Tensor,
    input_ids: paddle.Tensor,
    stop_nums: paddle.Tensor,
    next_tokens: paddle.Tensor,
    is_block_step: paddle.Tensor,
    end_ids: paddle.Tensor,
    kwargs_next_tokens: paddle.Tensor,
) -> None:

    _C.update_inputs_v2(
        stop_flags,
        step_idx,
        not_need_stop,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        max_dec_len,
        input_ids,
        stop_nums,
        next_tokens,
        is_block_step,
        end_ids,
        kwargs_next_tokens,
    )


def write_cache_kv(
    input_k: paddle.Tensor, input_v: paddle.Tensor, cache_kv: paddle.Tensor, sequence_lengths: paddle.Tensor
) -> None:

    _C.write_cache_kv(input_k, input_v, cache_kv, sequence_lengths)


def xft_greedy_search(probs: paddle.Tensor) -> List[paddle.Tensor]:

    return _C.xft_greedy_search(probs)


def xft_transformer(
    input: paddle.Tensor,
    ln1Gamma: List[paddle.Tensor],
    qkvWeight: List[paddle.Tensor],
    attnOutWeight: List[paddle.Tensor],
    ln2Gamma: List[paddle.Tensor],
    gateWeight: List[paddle.Tensor],
    upWeight: List[paddle.Tensor],
    downWeight: List[paddle.Tensor],
    pastSeqLen: paddle.Tensor,
    currentSeqLen: paddle.Tensor,
    step: paddle.Tensor,
    hiddensize: int,
    totalLayer: int,
    computeType: str,
    cacheDtype: str,
    activation: str,
    normType: str,
    attHeadDim: int,
    attHeadNum: int,
    kvHeadNum: int,
    maxPositions: int,
    maxPosEmbed: int,
    intermediateSize: int,
) -> List[paddle.Tensor]:

    return _C.xft_transformer(
        input,
        ln1Gamma,
        qkvWeight,
        attnOutWeight,
        ln2Gamma,
        gateWeight,
        upWeight,
        downWeight,
        pastSeqLen,
        currentSeqLen,
        step,
        hiddensize,
        totalLayer,
        computeType,
        cacheDtype,
        activation,
        normType,
        attHeadDim,
        attHeadNum,
        kvHeadNum,
        maxPositions,
        maxPosEmbed,
        intermediateSize,
    )
