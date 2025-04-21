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
import time

import numpy as np
import paddle
import paddlenlp_ops

np.random.seed(2024)
paddle.seed(2024)


def div_up(a, b):
    return (a + b - 1) // b


def alloc_diff(x, y):
    diff = (x - y).abs().max()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    assert diff < 5e-3
    assert cos_diff < 5e-3


def get_padding_offset(bsz, max_seq_len, seq_lens_this_time):
    cum_offsets_now = paddle.cumsum(max_seq_len - seq_lens_this_time)
    cum_offsets = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cum_offsets[1:] = cum_offsets_now
    token_num = paddle.sum(seq_lens_this_time)
    padding_offsets = paddle.zeros(shape=(token_num), dtype="int32")
    cu_seqlens_q = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cu_seqlens_k = paddle.zeros(shape=(bsz + 1), dtype="int32")
    for i in range(bsz):
        seq_len_now = seq_lens_this_time[i]
        cum_offset = cum_offsets[i]
        for j in range(seq_len_now):
            padding_offsets[i * max_seq_len - cum_offset + j] = cum_offset
        cum_seq_len = (i + 1) * max_seq_len - cum_offsets[i + 1]
        cu_seqlens_q[i + 1] = cum_seq_len
        cu_seqlens_k[i + 1] = cum_seq_len
    return padding_offsets, cum_offsets[:-1], cu_seqlens_q, cu_seqlens_k


RUN_TIME = 1
WARM_UP = 0
BLOCK_SIZE = 64
HEAD_DIM_QK = 576
HEAD_DIM_V = 512
PE_SIZE = 64
MAX_LENGTH = 8192
MAX_DEC_LEN = 1024
NUM_Q_HEAD = 8
NUM_KV_HEAD = 1
dtype = "bfloat16"
DRAFT_TOTAL_TOKEN_NUM = 1
CAUSAL = True

SPECULATE_DECODER = False


def ref_attention(q_all, p_compressed_kv, compressed_kv, p_key_pe, key_pe, bsz, cache_length, softmax_scale):
    q = q_all.reshape([bsz, DRAFT_TOTAL_TOKEN_NUM, NUM_Q_HEAD, HEAD_DIM_QK])
    p_compressed_kv = p_compressed_kv.reshape((bsz, cache_length, HEAD_DIM_V))
    compressed_kv = compressed_kv.reshape((bsz, DRAFT_TOTAL_TOKEN_NUM, HEAD_DIM_V))
    p_key_pe = p_key_pe.reshape((bsz, cache_length, PE_SIZE))
    key_pe = key_pe.reshape((bsz, DRAFT_TOTAL_TOKEN_NUM, PE_SIZE))

    p_k = paddle.concat([p_compressed_kv, p_key_pe], axis=-1).reshape((bsz, cache_length, HEAD_DIM_QK))
    k = paddle.concat([compressed_kv, key_pe], axis=-1).reshape((bsz, DRAFT_TOTAL_TOKEN_NUM, HEAD_DIM_QK))
    k = paddle.concat([p_k, k], axis=1).transpose([0, 2, 1])
    v = paddle.concat([p_compressed_kv, compressed_kv], axis=1).reshape(
        (bsz, cache_length + DRAFT_TOTAL_TOKEN_NUM, HEAD_DIM_V)
    )

    out = paddle.zeros(shape=[bsz, DRAFT_TOTAL_TOKEN_NUM, NUM_Q_HEAD, HEAD_DIM_V], dtype=dtype)
    for i in range(DRAFT_TOTAL_TOKEN_NUM):
        for j in range(NUM_Q_HEAD):

            query = q[:, i, j, :].reshape([-1, 1, HEAD_DIM_QK])
            query = query.scale(softmax_scale)
            cu_k = k[:, :, : cache_length + i + 1]
            score = paddle.bmm(x=query, y=cu_k)
            score = paddle.nn.functional.softmax(score, axis=-1)
            cu_v = v[:, : cache_length + i + 1, :]
            sub_out = paddle.bmm(score, cu_v).reshape([-1, 1, 1, HEAD_DIM_V])
            out[:, i : i + 1, j : j + 1, :] = sub_out

    out = out.reshape([-1, NUM_Q_HEAD, HEAD_DIM_V])
    return out


def fake_prefill(input_length, compressed_kv, key_pe, latent_cache_shape, block_tables):

    seq_lens_enc = [
        input_length,
    ] * bsz
    seq_lens_dec = [
        0,
    ] * bsz
    seq_lens_this_time = [
        input_length,
    ] * bsz
    max_enc_len_this_time = max(seq_lens_enc)
    max_dec_len_this_time = max(seq_lens_dec)
    max_enc_len_this_time = paddle.to_tensor([max_enc_len_this_time], "int32", place=paddle.CPUPlace())
    max_dec_len_this_time = paddle.to_tensor([max_dec_len_this_time], "int32", place=paddle.CPUPlace())

    seq_lens_encoder = paddle.to_tensor(seq_lens_enc, "int32")
    seq_lens_this_time = paddle.to_tensor(seq_lens_this_time, "int32")
    seq_lens_decoder = paddle.to_tensor(seq_lens_dec, "int32")
    padding_offsets, cum_offsets, cu_seqlens_q, cu_seqlens_k = get_padding_offset(bsz, MAX_LENGTH, seq_lens_this_time)
    latent_cache = paddle.zeros(shape=latent_cache_shape).astype(dtype)
    # import pdb; pdb.set_trace()

    paddlenlp_ops.prefill_mla_write_cache(
        compressed_kv,
        key_pe,
        latent_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        padding_offsets,
        cum_offsets,
        block_tables,
        "none",
        MAX_LENGTH,
    )

    return latent_cache


def test_append_c16_attention(cache_length, bsz):
    # prepare input
    seq_lens_enc = [
        0,
    ] * bsz
    seq_lens_dec = [
        cache_length,
    ] * bsz
    seq_lens_this_time = [
        DRAFT_TOTAL_TOKEN_NUM,
    ] * bsz
    max_enc_len_this_time = max(seq_lens_enc)
    max_dec_len_this_time = max(seq_lens_dec)
    max_enc_len_this_time = paddle.to_tensor([max_enc_len_this_time], "int32", place=paddle.CPUPlace())
    max_dec_len_this_time = paddle.to_tensor([max_dec_len_this_time], "int32", place=paddle.CPUPlace())
    token_num = sum(seq_lens_this_time)
    block_num_per_seq = div_up(MAX_LENGTH, BLOCK_SIZE)
    max_block_num = block_num_per_seq * bsz
    free_list = list(range(max_block_num - 1, -1, -1))
    block_tables = paddle.ones(shape=(bsz, block_num_per_seq), dtype="int32") * (-1)
    for i in range(bsz):
        need_block_num = div_up(seq_lens_dec[i] + MAX_DEC_LEN, BLOCK_SIZE)
        for j in range(need_block_num):
            block_id = free_list.pop()
            block_tables[i, j] = block_id
    seq_lens_encoder = paddle.to_tensor(seq_lens_enc, "int32")
    seq_lens_this_time = paddle.to_tensor(seq_lens_this_time, "int32")
    seq_lens_decoder = paddle.to_tensor(seq_lens_dec, "int32")
    padding_offsets, cum_offsets, cu_seqlens_q, cu_seqlens_k = get_padding_offset(bsz, MAX_LENGTH, seq_lens_this_time)
    q_varlen_shape = [token_num, NUM_Q_HEAD * HEAD_DIM_QK]
    latent_cache_shape = (
        max_block_num,
        NUM_KV_HEAD,
        BLOCK_SIZE,
        HEAD_DIM_QK,
    )

    (
        encoder_batch_ids,
        encoder_tile_ids_per_batch,
        encoder_num_blocks,
        kv_batch_ids,
        kv_tile_ids_per_batch,
        kv_num_blocks,
        decoder_batch_ids,
        decoder_tile_ids_per_batch,
        decoder_num_blocks_device,
        decoder_num_blocks,
        decoder_chunk_size,
        max_len_kv,
    ) = paddlenlp_ops.get_block_shape_and_split_kv_block(
        seq_lens_encoder,
        seq_lens_decoder,
        max_enc_len_this_time,
        max_dec_len_this_time,
        seq_lens_this_time,
        cum_offsets,
        NUM_Q_HEAD // NUM_KV_HEAD,
        BLOCK_SIZE,
        DRAFT_TOTAL_TOKEN_NUM,
    )
    softmax_scale = HEAD_DIM_QK ** (-0.5)

    # fake_prefill to prepare latent_cache
    p_compressed_kv_shape = [bsz * cache_length, NUM_KV_HEAD * HEAD_DIM_V]
    p_key_pe_shape = [cache_length * bsz, NUM_KV_HEAD * PE_SIZE]
    p_compressed_kv = paddle.randn(shape=p_compressed_kv_shape).astype(dtype)
    p_key_pe = paddle.randn(shape=p_key_pe_shape).astype(dtype)
    latent_cache = fake_prefill(cache_length, p_compressed_kv, p_key_pe, latent_cache_shape, block_tables)

    # dec
    query = paddle.randn(shape=q_varlen_shape).astype(dtype)
    compressed_kv_shape = [token_num, NUM_KV_HEAD, HEAD_DIM_V]
    key_pe_shape = [token_num, NUM_KV_HEAD, PE_SIZE]
    compressed_kv = paddle.rand(shape=compressed_kv_shape).astype(dtype)
    key_pe = paddle.rand(shape=key_pe_shape).astype(dtype)
    paddlenlp_ops.decode_mla_write_cache(
        compressed_kv,
        key_pe,
        latent_cache,
        seq_lens_decoder,
        seq_lens_encoder,
        padding_offsets,
        cum_offsets,
        block_tables,
        "none",
        MAX_LENGTH,
        SPECULATE_DECODER,
    )

    ref_out = ref_attention(query, p_compressed_kv, compressed_kv, p_key_pe, key_pe, bsz, cache_length, softmax_scale)
    paddle.device.synchronize()
    s_time = 0
    for i in range(RUN_TIME + WARM_UP):
        if i == WARM_UP:
            s_time = time.time()
        out = paddlenlp_ops.multi_head_latent_attention(
            query,
            latent_cache,
            latent_cache,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            cu_seqlens_q,
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
            decoder_num_blocks_device,
            decoder_num_blocks,
            decoder_chunk_size,
            max_enc_len_this_time,
            max_dec_len_this_time,
            max_len_kv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "bf16",
            "none",  # cache_quant_type
            HEAD_DIM_V,
            MAX_LENGTH,
            softmax_scale,
            0.0,
            0.0,
            0.0,  # out_linear_in_scale
            DRAFT_TOTAL_TOKEN_NUM,  # DRAFT_TOTAL_TOKEN_NUM
            CAUSAL,
            SPECULATE_DECODER,
        )
        paddle.device.synchronize()

    e_time = time.time()
    out = out.reshape([-1, NUM_Q_HEAD, HEAD_DIM_V])
    alloc_diff(ref_out, out)
    print(
        "dec bsz:{}, num_q_head:{}, cache_length:{}, cost_time:{}ms".format(
            bsz, NUM_Q_HEAD, cache_length, (e_time - s_time) / RUN_TIME * 1000
        )
    )


if __name__ == "__main__":
    print("CUDA_VERSION >=12.8 is required to compile and run mla benchmark")
    for cache_length in [1023, 2047]:
        for bsz in [1, 16, 32, 64, 128, 256]:
            test_append_c16_attention(cache_length, bsz)
