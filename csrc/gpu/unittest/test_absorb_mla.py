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
import math
import time

import numpy as np
import paddle
import paddlenlp_ops

np.random.seed(2024)
paddle.seed(2024)


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


run_time = 1
warm_up = 0
block_size = 64
head_dim_qk = 576
nope_size = 512
pe_size = 64
head_dim_v = 512
max_dec_len = 1
num_q_head = 8

num_kv_head = 1
dtype = "bfloat16"
# prefill
max_length = 8192
# bsz = 1
# cache_length = 2048


def mqa_attention(query, p_compressed_kv, compressed_kv, p_key_pe, key_pe, token_num, softmax_scale):
    q = query.reshape([-1, num_q_head, head_dim_qk])
    p_compressed_kv = p_compressed_kv.reshape((bsz, cache_length, nope_size))
    compressed_kv = compressed_kv.reshape((bsz, 1, nope_size))
    p_key_pe = p_key_pe.reshape((bsz, cache_length, pe_size))
    key_pe = key_pe.reshape((bsz, 1, pe_size))

    p_k = paddle.concat([p_compressed_kv, p_key_pe], axis=-1).reshape((bsz, cache_length, head_dim_qk))
    k = paddle.concat([compressed_kv, key_pe], axis=-1).reshape((bsz, 1, head_dim_qk))
    k = paddle.concat([p_k, k], axis=1).transpose([0, 2, 1])
    v = paddle.concat([p_compressed_kv, compressed_kv], axis=1).reshape((bsz, cache_length + 1, head_dim_v))
    # attn_mask = paddle.zeros(shape=[seq_len, seq_len], dtype="float32")
    # for i in range(seq_len):
    #     for j in range(seq_len):
    #         if i <= j:
    #             attn_mask[i][j] = -10000
    out = paddle.zeros(shape=[bsz, num_q_head, head_dim_v], dtype=dtype)
    # print("cache_length: ", cache_length)
    print("q: ", q.shape)
    print("k: ", k.shape)
    print("v: ", v.shape)
    # q = q[1:2]
    # k = k[1:2]
    # v = v[1:2]
    # k = k[:, :, :512]
    # v = v[:, :512, :]
    # k = k[:, :, 512:1024]
    # v = v[:, 512:1024, :]
    print("q: ", q)
    print("k: ", k)
    print("v: ", v)
    for i in range(num_q_head):
        query = q[:, i, :].reshape([-1, 1, head_dim_qk])
        print("query: ", query[-1])
        score = paddle.bmm(x=query, y=k)  # [bsz, 1, seq_lens]
        print("score: ", score[-1])
        row_max = paddle.max(score, -1).reshape([-1, 1, 1])
        print("row_max: ", row_max[-1])
        row_max_scale = paddle.max(score, -1).scale(softmax_scale / 1).reshape([-1, 1, 1]) # softmax_scale / math.log(2, math.e)
        print("row_max scale: ", row_max_scale[-1])
        score = score - row_max
        print("score sub max: ", score[-1])
        score = score.scale(softmax_scale / 1)
        print("score2: ", score[-1])
        score = paddle.exp(score)
        # tmp_score = paddle.zeros_like(score)
        # tmp_score[:] = 2
        # print("tmp_score: ", tmp_score[-1])
        # score = paddle.pow(tmp_score, score)
        print("score sub max exp: ", score[-1])
        d_sum = paddle.sum(score, axis=-1, keepdim=True).reshape([-1, 1, 1])
        print("d_sum: ", d_sum[-1])
        sub_out = paddle.bmm(score, v)
        print("sub_out: ", sub_out[-1])
        sub_out /= d_sum
        print("norm_res: ", sub_out[-1])

        # query = query.scale(softmax_scale)
        # score = paddle.bmm(x=query, y=k)
        # score = paddle.nn.functional.softmax(score, axis=-1)
        # sub_out = paddle.bmm(score, v).reshape([-1, 1, head_dim_v])
        # # if (i == 6):
        # print("sub_out ref: ", sub_out)
        # import pdb; pdb.set_trace()

        out[:, i : i + 1, :] = sub_out

    out = out.reshape([-1, num_q_head * head_dim_v])
    return out


def prefill(input_length, compressed_kv, key_pe, latent_cache_shape, block_tables):

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
    padding_offsets, cum_offsets, cu_seqlens_q, cu_seqlens_k = get_padding_offset(bsz, max_length, seq_lens_this_time)
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
        max_length,
    )

    return latent_cache


def test_append_c16_attention(cache_length, bsz):
    # prefill

    seq_lens_enc = [
        0,
    ] * bsz
    seq_lens_dec = [
        cache_length,
    ] * bsz
    seq_lens_this_time = [
        1,
    ] * bsz
    max_enc_len_this_time = max(seq_lens_enc)
    max_dec_len_this_time = max(seq_lens_dec)
    max_enc_len_this_time = paddle.to_tensor([max_enc_len_this_time], "int32", place=paddle.CPUPlace())
    max_dec_len_this_time = paddle.to_tensor([max_dec_len_this_time], "int32", place=paddle.CPUPlace())
    token_num = sum(seq_lens_this_time)
    block_num_per_seq = (max_length + block_size - 1) // block_size
    max_block_num = block_num_per_seq * bsz
    free_list = list(range(max_block_num - 1, -1, -1))
    block_tables = paddle.ones(shape=(bsz, block_num_per_seq), dtype="int32") * (-1)
    for i in range(bsz):
        need_block_num = (seq_lens_dec[i] + max_dec_len + block_size - 1) // block_size
        for j in range(need_block_num):
            block_id = free_list.pop()
            block_tables[i, j] = block_id
    seq_lens_encoder = paddle.to_tensor(seq_lens_enc, "int32")
    seq_lens_this_time = paddle.to_tensor(seq_lens_this_time, "int32")
    seq_lens_decoder = paddle.to_tensor(seq_lens_dec, "int32")
    padding_offsets, cum_offsets, cu_seqlens_q, cu_seqlens_k = get_padding_offset(bsz, max_length, seq_lens_this_time)
    q_varlen_shape = [token_num, num_q_head * head_dim_qk]
    latent_cache_shape = (
        max_block_num,
        num_kv_head,
        block_size,
        head_dim_qk,
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
        max_len_kv,
    ) = paddlenlp_ops.get_block_shape_and_split_kv_block(
        seq_lens_encoder,
        seq_lens_decoder,
        max_enc_len_this_time,
        max_dec_len_this_time,
        seq_lens_this_time,
        cum_offsets,
        num_q_head // num_kv_head,
        block_size,
        1,
    )
    softmax_scale = head_dim_qk ** (-0.5)
    print("softmax_scale: ", softmax_scale)

    # prefill
    p_compressed_kv_shape = [bsz * cache_length, num_kv_head * nope_size]
    p_key_pe_shape = [cache_length * bsz, num_kv_head * pe_size]
    p_compressed_kv = paddle.randn(shape=p_compressed_kv_shape).astype(dtype)
    p_key_pe = paddle.randn(shape=p_key_pe_shape).astype(dtype)
    latent_cache = prefill(cache_length, p_compressed_kv, p_key_pe, latent_cache_shape, block_tables)
    # print("latent_cache-2: ", latent_cache[0])
    # print("latent_cache-1: ", latent_cache[1])
    # dec
    query = paddle.randn(shape=q_varlen_shape).astype(dtype)
    compressed_kv_shape = [token_num, num_kv_head, nope_size]
    key_pe_shape = [token_num, num_kv_head, pe_size]
    compressed_kv = paddle.rand(shape=compressed_kv_shape).astype(dtype)
    key_pe = paddle.rand(shape=key_pe_shape).astype(dtype)
    # print("compressed_kv: ", compressed_kv)
    print("key_pe: ", key_pe)
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
        max_length,
        False,
    )
    print("latent_cache0: ", latent_cache.shape)
    # print("latent_cache_v0: ", latent_cache[132][:, :, :512])
    # print("latent_cache1: ", latent_cache[1])
    inputs = [
        query,
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
        max_enc_len_this_time,
        max_dec_len_this_time,
        max_len_kv,
    ]
    inputs_name = [
        "query",
        "seq_lens_encoder",
        "seq_lens_decoder",
        "seq_lens_this_time",
        "cu_seqlens_q",
        "padding_offsets",
        "cum_offsets",
        "block_tables",
        "encoder_batch_ids",
        "encoder_tile_ids_per_batch",
        "encoder_num_blocks",
        "kv_batch_ids",
        "kv_tile_ids_per_batch",
        "kv_num_blocks",
        "decoder_batch_ids",
        "decoder_tile_idss_per_batch",
        "decoder_num_blocks_device",
        "decoder_num_blocks",
        "max_enc_len_this_time",
        "max_dec_len_this_time",
        "max_len_kv",
    ]
    for i in range(len(inputs_name)):
        if "query" == inputs_name[i]:
            print(f"{inputs_name[i]}: {inputs[i].reshape([-1, num_q_head, head_dim_qk])}")
        else:
            print(f"{inputs_name[i]}: {inputs[i]}")
    paddle.device.synchronize()
    s_time = 0
    for i in range(run_time + warm_up):
        if i == warm_up:
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
            max_enc_len_this_time,
            max_dec_len_this_time,
            max_len_kv,
            None,  # attn_mask
            None,
            None,  # qkv_scale
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
            head_dim_v,
            max_length,
            softmax_scale / 1,
            0.0,
            0.0,
            0.0,  # out_linear_in_scale
            1,  # speculate_max_draft_token_num
            True,  # causal
            False,  # speculate_decoder
        )
        # paddle.device.synchronize()
        out1 = paddlenlp_ops.multi_head_latent_attention(
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
            max_enc_len_this_time,
            max_dec_len_this_time,
            max_len_kv,
            None,  # attn_mask
            None,
            None,  # qkv_scale
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
            head_dim_v,
            max_length,
            softmax_scale / 1,
            0.0,
            0.0,
            0.0,  # out_linear_in_scale
            1,  # speculate_max_draft_token_num
            True,  # causal
            False,  # speculate_decoder
        )
        out2 = paddlenlp_ops.multi_head_latent_attention(
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
            max_enc_len_this_time,
            max_dec_len_this_time,
            max_len_kv,
            None,  # attn_mask
            None,
            None,  # qkv_scale
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
            head_dim_v,
            max_length,
            softmax_scale / 1,
            0.0,
            0.0,
            0.0,  # out_linear_in_scale
            1,  # speculate_max_draft_token_num
            True,  # causal
            False,  # speculate_decoder
        )
        paddle.device.synchronize()

    e_time = time.time()
    base_out = mqa_attention(query, p_compressed_kv, compressed_kv, p_key_pe, key_pe, token_num, softmax_scale)
    base_out = base_out.reshape([-1, num_q_head, head_dim_v])
    out = out.reshape([-1, num_q_head, head_dim_v])
    out1 = out1.reshape([-1, num_q_head, head_dim_v])
    out2 = out2.reshape([-1, num_q_head, head_dim_v])
    print("out: ", out)
    print("out1: ", out1)
    print("out2: ", out2)
    print("base_out: ", base_out)

    diff = base_out - out
    max_diff1 = diff.abs().max()
    print(f"max diff all: {max_diff1}")
    max_diff1 = diff.abs().argmax()
    print(f"max diff all: {max_diff1}")

    # if (bsz > 1):
    #     diff1 = base_out[0] - out[0]
    #     max_diff1 = diff1.abs().max()
    #     print(f"max diff bs1: {max_diff1}")
    #     diff2 = base_out[1] - out[1]
    #     max_diff2 = diff2.abs().max()
    #     print(f"max diff bs2: {max_diff2}")

    diff = out1 - out
    max_diff = diff.abs().max()
    print(f"max diff self: {max_diff}")
    max_diff = diff.abs().argmax()
    print(f"max diff arg self: {max_diff}")

    diff = out2 - out
    max_diff = diff.abs().max()
    print(f"max diff self: {max_diff}")
    max_diff = diff.abs().argmax()
    print(f"max diff arg self: {max_diff}")
    # print(
    #     "dec bsz:{}, num_q_head:{}, cache_length:{}, cost_time:{}ms".format(
    #         bsz, num_q_head, cache_length, (e_time - s_time) / run_time * 1000
    #     )
    # )


if __name__ == "__main__":
    # for cache_length in [1024, 2048]:
    #   for bsz in [1, 8, 32, 96, 128, 256]:
    for cache_length in [1024]:
        for bsz in [128]:
            test_append_c16_attention(cache_length, bsz)
