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

from .utils import custom_dispatch


@custom_dispatch
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
    pass


@custom_dispatch
def avx_weight_only(x: paddle.Tensor, weight: paddle.Tensor, alog: str, trans: bool) -> paddle.Tensor:
    pass


@custom_dispatch
def decode_mla_write_cache(
    kv_nope: paddle.Tensor,
    kv_pe: paddle.Tensor,
    kv_cache: paddle.Tensor,
    seq_lens: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    padding_offsets: paddle.Tensor,
    cum_offsets: paddle.Tensor,
    block_tables: paddle.Tensor,
    cache_quant_type_str: str,
    max_seq_len: int,
    speculate_decoder: bool,
) -> paddle.Tensor:
    pass


@custom_dispatch
def dequant_int8(intput: paddle.Tensor, out_scale: paddle.Tensor, dtype: str) -> paddle.Tensor:
    pass


@custom_dispatch
def draft_model_postprocess(
    base_model_draft_tokens: paddle.Tensor,
    base_model_seq_lens_this_time: paddle.Tensor,
    base_model_seq_lens_encoder: paddle.Tensor,
    base_model_stop_flags: paddle.Tensor,
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
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
    pass


@custom_dispatch
def draft_model_set_value_by_flags(
    draft_tokens: paddle.Tensor,
    pre_ids_all: paddle.Tensor,
    stop_flags: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    step_idx: paddle.Tensor,
) -> paddle.Tensor:
    pass


@custom_dispatch
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
    pass


@custom_dispatch
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
) -> paddle.Tensor:
    pass


@custom_dispatch
def eagle_get_self_hidden_states(
    input: paddle.Tensor,
    last_seq_lens_this_time: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    step_idx: paddle.Tensor,
) -> paddle.Tensor:
    pass


@custom_dispatch
def encode_rotary_qk(
    q: paddle.Tensor,
    kv: paddle.Tensor,
    rotary_emb: paddle.Tensor,
    seq_lens: paddle.Tensor,
    rotary_emb_dims: int,
    use_neox: bool,
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
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
    pass


@custom_dispatch
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
) -> paddle.Tensor:
    pass


@custom_dispatch
def cutlass_fp8_fp8_half_gemm_fused(
    x: paddle.Tensor,
    y: paddle.Tensor,
    bias: Optional[paddle.Tensor],
    transpose_x: bool,
    transpose_y: bool,
    scale: float,
    output_type: str,
    act: str,
) -> paddle.Tensor:
    pass


@custom_dispatch
def cutlass_fp8_fp8_half_block_gemm_fused(
    x: paddle.Tensor,
    y: paddle.Tensor,
    x_scale: paddle.Tensor,
    y_scale: paddle.Tensor,
    bias: paddle.Tensor,
    trans_x: bool,
    trans_y: bool,
    output_type: str,
    activation_type: str,
) -> paddle.Tensor:
    pass


@custom_dispatch
def cutlass_fp8_fp8_half_gemm_ptr_scale_fused(
    x: paddle.Tensor,
    y: paddle.Tensor,
    x_scale: paddle.Tensor,
    y_scale: paddle.Tensor,
    bias: paddle.Tensor,
    trans_x: bool,
    trans_y: bool,
    output_type: str,
) -> paddle.Tensor:
    pass


@custom_dispatch
def fused_get_rotary_embedding(
    input_ids: paddle.Tensor,
    position_ids: paddle.Tensor,
    head_dim_shape_tensor: paddle.Tensor,
    prompt_num: int,
    theta: float,
    use_neox: bool,
) -> paddle.Tensor:
    pass


@custom_dispatch
def fused_rotary_position_encoding(
    query: paddle.Tensor,
    key: paddle.Tensor,
    position_ids: paddle.Tensor,
    cos_sin_cache: paddle.Tensor,
    head_size: int,
    is_neox: bool,
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def gemm_dequant(x: paddle.Tensor, y: paddle.Tensor, scale: paddle.Tensor, out_dtype: str) -> paddle.Tensor:
    pass


@custom_dispatch
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
    pass


@custom_dispatch
def get_output(x: paddle.Tensor, rank_id: int, wait_flag: float) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def get_padding_offset(
    input_ids: paddle.Tensor, cum_offsets: paddle.Tensor, token_num: paddle.Tensor, seq_len: paddle.Tensor
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def get_padding_offset_v2(
    input_ids: paddle.Tensor,
    cum_offsets: paddle.Tensor,
    token_num: paddle.Tensor,
    seq_len: paddle.Tensor,
    draft_tokens: Optional[paddle.Tensor],
    seq_lens_encoder: Optional[paddle.Tensor],
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def get_position_ids(
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    position_ids: paddle.Tensor,
) -> paddle.Tensor:
    pass


@custom_dispatch
def get_position_ids_and_mask_encoder_batch(
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    position_ids: paddle.Tensor,
    mask_encoder_batch: paddle.Tensor,
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def get_token_penalty_multi_scores(
    pre_ids: paddle.Tensor,
    logits: paddle.Tensor,
    penalty_scores: paddle.Tensor,
    frequency_scores: paddle.Tensor,
    presence_scores: paddle.Tensor,
    cur_len: paddle.Tensor,
    min_len: paddle.Tensor,
    eos_token_id: paddle.Tensor,
) -> paddle.Tensor:
    pass


@custom_dispatch
def get_token_penalty_multi_scores_v2(
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
) -> paddle.Tensor:
    pass


@custom_dispatch
def group_quant(
    x: paddle.Tensor, group_size: int, transpose_scale: bool, quant_max_bound: float, quant_min_bound: float
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def multi_head_latent_attention(
    query: paddle.Tensor,
    key_cache: paddle.Tensor,
    value_cache: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
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
    decoder_num_blocks_cpu: paddle.Tensor,
    max_enc_len_this_time: paddle.Tensor,
    max_dec_len_this_time: paddle.Tensor,
    max_len_kv: paddle.Tensor,
    attn_mask: paddle.Tensor,
    query_bias: paddle.Tensor,
    query_out_scales: paddle.Tensor,
    cache_k_quant_scales: paddle.Tensor,
    cache_v_quant_scales: paddle.Tensor,
    cache_k_dequant_scales: paddle.Tensor,
    cache_v_dequant_scales: paddle.Tensor,
    cache_k_zp: paddle.Tensor,
    cache_v_zp: paddle.Tensor,
    out_linear_shifts: paddle.Tensor,
    out_linear_smooths: paddle.Tensor,
    compute_type: str,
    cache_quant_type: str,
    nope_size: int,
    max_input_length: int,
    softmax_scale: float,
    quant_max_bound: float,
    quant_min_bound: float,
    out_linear_in_scale: float,
    speculate_max_draft_token_num: int,
    causal: bool,
    speculate_decoder: bool,
) -> paddle.Tensor:
    pass


@custom_dispatch
def noaux_tc(
    scores: paddle.Tensor,
    scores_with_bias: paddle.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    routed_scaling_factor: float,
) -> paddle.Tensor:
    pass


@custom_dispatch
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
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def prefill_mla_write_cache(
    kv_nope: paddle.Tensor,
    kv_pe: paddle.Tensor,
    kv_cache: paddle.Tensor,
    seq_lens: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    padding_offsets: paddle.Tensor,
    cum_offsets: paddle.Tensor,
    block_tables: paddle.Tensor,
    cache_quant_type_str: str,
    max_seq_len: int,
) -> paddle.Tensor:
    pass


@custom_dispatch
def preprocess_for_moe(topk_ids: paddle.Tensor, num_experts: int, block_size: int) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def qkv_transpose_split(
    qkv: paddle.Tensor,
    padding_offset: paddle.Tensor,
    seq_lens: paddle.Tensor,
    input_ids: paddle.Tensor,
    num_head: int,
    head_size: int,
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def quant_int8(
    intput: paddle.Tensor,
    shift: paddle.Tensor,
    smooth: paddle.Tensor,
    scale: float,
    round_type: int,
    max_bound: float,
    min_bound: float,
) -> paddle.Tensor:
    pass


@custom_dispatch
def rebuild_padding(
    tmp_out: paddle.Tensor, padding_offset: paddle.Tensor, seq_lens: paddle.Tensor, input_ids: paddle.Tensor
) -> paddle.Tensor:
    pass


@custom_dispatch
def rebuild_padding_v2(
    tmp_out: paddle.Tensor,
    cum_offsets: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    output_padding_offset: paddle.Tensor,
    max_input_length: int,
) -> paddle.Tensor:
    pass


@custom_dispatch
def sage_attention(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    km: paddle.Tensor,
    seq_len_this_time: paddle.Tensor,
    vm: paddle.Tensor,
    sm_scale: float,
    qk_quant_gran: str,
    pv_accum_dtype: str,
    tensor_layout: int,
    is_causal: bool,
    smooth_k: bool,
    smooth_v: bool,
    return_lse: bool,
) -> paddle.Tensor:
    pass


@custom_dispatch
def sage_attention_dsk(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    km: paddle.Tensor,
    seq_len_this_time: paddle.Tensor,
    vm: paddle.Tensor,
    sm_scale: float,
    qk_quant_gran: str,
    pv_accum_dtype: str,
    tensor_layout: int,
    is_causal: bool,
    smooth_k: bool,
    smooth_v: bool,
    return_lse: bool,
) -> paddle.Tensor:
    pass


@custom_dispatch
def save_output(x: paddle.Tensor, not_need_stop: paddle.Tensor, rank_id: int) -> paddle.Tensor:
    pass


@custom_dispatch
def save_with_output(
    x: paddle.Tensor, batch_idx: paddle.Tensor, step_idx: paddle.Tensor, file_path: str, rank_id: int
) -> paddle.Tensor:
    pass


@custom_dispatch
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
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def set_stop_value_multi_ends(
    topk_ids: paddle.Tensor, stop_flags: paddle.Tensor, end_ids: paddle.Tensor, mode: int
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def set_stop_value_multi_ends_v2(
    topk_ids: paddle.Tensor,
    stop_flags: paddle.Tensor,
    seq_lens: paddle.Tensor,
    end_ids: paddle.Tensor,
    next_tokens: paddle.Tensor,
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def set_value_by_flags_and_idx(
    pre_ids_all: paddle.Tensor, pre_ids_now: paddle.Tensor, step_idx: paddle.Tensor, stop_flags: paddle.Tensor
) -> paddle.Tensor:
    pass


@custom_dispatch
def set_value_by_flags_and_idx_v2(
    pre_ids_all: paddle.Tensor,
    input_ids: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    step_idx: paddle.Tensor,
    stop_flags: paddle.Tensor,
) -> paddle.Tensor:
    pass


@custom_dispatch
def speculate_clear_accept_nums(accept_num: paddle.Tensor, seq_lens_decoder: paddle.Tensor) -> paddle.Tensor:
    pass


@custom_dispatch
def speculate_get_output(x: paddle.Tensor, rank_id: int, wait_flag: bool) -> paddle.Tensor:
    pass


@custom_dispatch
def speculate_get_output_padding_offset(
    output_cum_offsets_tmp: paddle.Tensor,
    out_token_num: paddle.Tensor,
    seq_lens_output: paddle.Tensor,
    max_seq_len: int,
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def speculate_get_seq_lens_output(
    seq_lens_this_time: paddle.Tensor, seq_lens_encoder: paddle.Tensor, seq_lens_decoder: paddle.Tensor
) -> paddle.Tensor:
    pass


@custom_dispatch
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
) -> paddle.Tensor:
    pass


@custom_dispatch
def speculate_save_output(
    accept_tokens: paddle.Tensor, accept_num: paddle.Tensor, not_need_stop: paddle.Tensor, rank_id: int
) -> paddle.Tensor:
    pass


@custom_dispatch
def speculate_set_value_by_flags_and_idx(
    pre_ids_all: paddle.Tensor,
    accept_tokens: paddle.Tensor,
    accept_num: paddle.Tensor,
    stop_flags: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    step_idx: paddle.Tensor,
) -> paddle.Tensor:
    pass


@custom_dispatch
def speculate_step_paddle(
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
    accept_num: paddle.Tensor,
    block_size: int,
    encoder_decoder_block_num: int,
    max_draft_tokens: int,
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
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
    pass


@custom_dispatch
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
    pass


@custom_dispatch
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
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def top_p_candidates(
    probs: paddle.Tensor,
    top_p: paddle.Tensor,
    output_padding_offset: paddle.Tensor,
    candidates_len: int,
    max_seq_len: int,
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
def top_p_sampling_reject(probs: paddle.Tensor, top_p: paddle.Tensor, seed: int) -> paddle.Tensor:
    pass


@custom_dispatch
def transpose_remove_padding(
    input: paddle.Tensor, seq_lens: paddle.Tensor, padding_offset: paddle.Tensor
) -> paddle.Tensor:
    pass


@custom_dispatch
def tune_cublaslt_gemm(
    k: paddle.Tensor,
    n: paddle.Tensor,
    m_start: int,
    m_end: int,
    dtype: str,
    is_test: bool,
    is_read_from_file: bool,
    path: str,
) -> None:
    pass


@custom_dispatch
def update_inputs(
    stop_flags: paddle.Tensor,
    not_need_stop: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    input_ids: paddle.Tensor,
    stop_nums: paddle.Tensor,
    next_tokens: paddle.Tensor,
    is_block_step: paddle.Tensor,
) -> List[paddle.Tensor]:
    pass


@custom_dispatch
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
    pass


@custom_dispatch
def write_cache_kv(
    input_k: paddle.Tensor, input_v: paddle.Tensor, cache_kv: paddle.Tensor, sequence_lengths: paddle.Tensor
) -> paddle.Tensor:
    pass


@custom_dispatch
def xft_greedy_search(probs: paddle.Tensor) -> paddle.Tensor:
    pass


@custom_dispatch
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
) -> paddle.Tensor:
    pass
