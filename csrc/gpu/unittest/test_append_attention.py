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

import unittest

import numpy as np
import paddle
from paddlenlp_ops import append_attention
import time


paddle.seed(1)

class RopeEmbedding:
    def _rotary_position_embedding(self, seq_len, head_dim, dtype):
        pos_seq = paddle.arange(0, seq_len, 1, dtype=dtype)
        indices = paddle.arange(0, head_dim, 2, dtype=dtype)
        indices = 1 / 10000 ** (indices / head_dim)

        sinusoid_inp = pos_seq.unsqueeze(1) * indices.unsqueeze(0)
        pos_emb = paddle.concat(
            [paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)], axis=-1
        )
        pos_emb = paddle.reshape(pos_emb, (1, 1, seq_len, head_dim))
        pos_emb.stop_gradient = True
        return pos_emb
    def _apply_rope(self, rp, q, k, v=None, is_causal=False):
        # sin [sequence_length, embed_size_per_head//2]
        # cos [sequence_length, embed_size_per_head//2]
        sin, cos = paddle.chunk(rp, 2, axis=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = paddle.reshape(paddle.stack([sin, sin], axis=-1), rp.shape)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = paddle.reshape(paddle.stack([cos, cos], axis=-1), rp.shape)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_q = paddle.reshape(
            paddle.stack([-q[:, :, :, 1::2], q[:, :, :, 0::2]], axis=-1),
            paddle.shape(q),
        )
        query = paddle.add(
            paddle.multiply(q, cos_pos), paddle.multiply(rotate_half_q, sin_pos)
        )
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_k = paddle.reshape(
            paddle.stack([-k[:, :, :, 1::2], k[:, :, :, 0::2]], axis=-1),
            paddle.shape(k),
        )
        key = paddle.add(
            paddle.multiply(k, cos_pos), paddle.multiply(rotate_half_k, sin_pos)
        )
        if v is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_v = paddle.reshape(
                paddle.stack([-v[:, :, :, 1::2], v[:, :, :, 0::2]], axis=-1),
                paddle.shape(v),
            )
            value = paddle.add(
                paddle.multiply(v, cos_pos),
                paddle.multiply(rotate_half_v, sin_pos),
            )
            return query, key, value
        return query, key



def create_attn_mask(
    mask_type,
    batch_size,
    seq_lens,
    pre_cache_length=0,
):
    max_seq_len = max(seq_lens)
    mask = paddle.zeros(
        # [batch_size, 1, max_seq_len, max_seq_len + pre_cache_length],
        [batch_size, 1, max_seq_len, max_seq_len ],
        dtype=mask_type,
    )
    mask[:, :, :, :pre_cache_length] = 1
    for i in range(batch_size):
        seq_len = seq_lens[i]
        mask[i, 0, :seq_len, :seq_len] = (
            paddle.tril(paddle.ones(shape=(seq_len, seq_len), dtype=mask_type))
            - 1
        ) * 1e4
    return mask

def block_cache_to_naive_cache(
    cache_k, cache_v, bsz, block_tables, cache_seq_len
):
    _, num_head, blocksize, dim_head = cache_k.shape
    out_cache_k = paddle.zeros(
        shape=[bsz, num_head, cache_seq_len, dim_head], dtype=cache_k.dtype
    )
    out_cache_v = paddle.zeros(
        shape=[bsz, num_head, cache_seq_len, dim_head], dtype=cache_v.dtype
    )
    for i in range(bsz):
        for j in range(cache_seq_len):
            out_cache_k[i, :, j, :] = cache_k[
                block_tables[i, j // blocksize], :, j % blocksize, :
            ]
            out_cache_v[i, :, j, :] = cache_v[
                block_tables[i, j // blocksize], :, j % blocksize, :
            ]
    return out_cache_k, out_cache_v

def naive_attention_impl(
    query,
    key,
    value,
    cache_k=None,
    cache_v=None,
    pre_cache_k=None,
    pre_cache_v=None,
    mask=None,
    scale=1.0,
    cache_k_dequant_scales=None,
    cache_v_dequant_scales=None,
    use_cachekv_int8="None",
):
    batch = query.shape[0]
    heads = query.shape[1]
    seq_len = query.shape[2]
    head_dim = query.shape[3]
    kv_head = key.shape[1]

    key = key.reshape([batch, kv_head, 1, seq_len, head_dim])
    key = paddle.tile(key, [1, 1, heads // kv_head, 1, 1])
    key = key.reshape([batch, heads, seq_len, head_dim])

    # if use_cachekv_int8 == "dynamic":
    #     unsqueeze_shape = [2, 3]
    # elif use_cachekv_int8 == "static":
    #     unsqueeze_shape = [0, 2, 3]
    # if pre_cache_k is not None:
    #     pre_cache_k = pre_cache_k.reshape([batch, kv_head, 1, -1, head_dim])
    #     pre_cache_k = paddle.tile(pre_cache_k, [1, 1, heads // kv_head, 1, 1])
    #     pre_cache_k = pre_cache_k.reshape([batch, heads, -1, head_dim])
    #     key = paddle.concat([pre_cache_k, key], axis=2)
    if cache_k is not None:
        # if cache_k_dequant_scales is not None:
        #     dequant_cache_k = (
        #         (cache_k.astype('float32') - 128.0)
        #         * cache_k_dequant_scales.unsqueeze(unsqueeze_shape)
        #     ).astype(key.dtype)
        #     dequant_cache_k = dequant_cache_k.reshape(
        #         [batch, kv_head, 1, -1, head_dim]
        #     )
        #     dequant_cache_k = paddle.tile(
        #         dequant_cache_k, [1, 1, heads // kv_head, 1, 1]
        #     )
        #     dequant_cache_k = dequant_cache_k.reshape(
        #         [batch, heads, -1, head_dim]
        #     )
        #     key = paddle.concat([dequant_cache_k, key], axis=2)
        # else:
        cache_k = cache_k.reshape([batch, kv_head, 1, -1, head_dim])
        cache_k = paddle.tile(cache_k, [1, 1, heads // kv_head, 1, 1])
        cache_k = cache_k.reshape([batch, heads, -1, head_dim])
        key = paddle.concat([cache_k, key], axis=2)

    value = value.reshape([batch, kv_head, 1, seq_len, head_dim])
    value = paddle.tile(value, [1, 1, heads // kv_head, 1, 1])
    value = value.reshape([batch, heads, seq_len, head_dim])

    # if pre_cache_v is not None:
    #     pre_cache_v = pre_cache_v.reshape([batch, kv_head, 1, -1, head_dim])
    #     pre_cache_v = paddle.tile(pre_cache_v, [1, 1, heads // kv_head, 1, 1])
    #     pre_cache_v = pre_cache_v.reshape([batch, heads, -1, head_dim])
    #     value = paddle.concat([pre_cache_v, value], axis=2)
    if cache_v is not None:
        # if cache_v_dequant_scales is not None:
        #     dequant_cache_v = (
        #         (cache_v.astype('float32') - 128.0)
        #         * cache_v_dequant_scales.unsqueeze(unsqueeze_shape)
        #     ).astype(value.dtype)
        #     dequant_cache_v = dequant_cache_v.reshape(
        #         [batch, kv_head, 1, -1, head_dim]
        #     )
        #     dequant_cache_v = paddle.tile(
        #         dequant_cache_v, [1, 1, heads // kv_head, 1, 1]
        #     )
        #     dequant_cache_v = dequant_cache_v.reshape(
        #         [batch, heads, -1, head_dim]
        #     )
        #     value = paddle.concat([dequant_cache_v, value], axis=2)
        # else:
        cache_v = cache_v.reshape([batch, kv_head, 1, -1, head_dim])
        cache_v = paddle.tile(cache_v, [1, 1, heads // kv_head, 1, 1])
        cache_v = cache_v.reshape([batch, heads, -1, head_dim])
        value = paddle.concat([cache_v, value], axis=2)

    qk_res = paddle.matmul(query, key, transpose_y=True)
    attention = qk_res * scale
    if mask is not None:
        attention = attention + mask
    softmax_result = paddle.nn.functional.softmax(attention, -1)
    result = paddle.matmul(softmax_result, value)
    return result

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

def remove_padding(seq_lens, cu_seq_lens, inputs, token_num):
    bsz, num_head, seq_len, dim_head = inputs.shape
    output = paddle.zeros(
        shape=[token_num, num_head * dim_head], dtype=inputs.dtype
    )
    inputs = inputs.transpose([0, 2, 1, 3]).reshape([bsz, seq_len, -1])
    for i in range(bsz):
        seq_len_now = seq_lens[i]
        start_idx = cu_seq_lens[i]
        end_idx = cu_seq_lens[i + 1]
        output[start_idx:end_idx, :] = inputs[i, :seq_len_now, :]
    return output

def get_qkv_and_qkv_concat_tensor(bs,q_num_head,kv_num_head,seq_len,dim_head,place,dtype):
    query = np.random.random([bs,q_num_head,seq_len,dim_head])
    q = paddle.to_tensor(
        query, place=place, dtype=dtype, stop_gradient=False
    )
    key = np.random.random([bs,kv_num_head,seq_len,dim_head])
    k = paddle.to_tensor(
        key, place=place, dtype=dtype, stop_gradient=False
    )
    value = np.random.random([bs,kv_num_head,seq_len,dim_head])
    v = paddle.to_tensor(
        value, place=place, dtype=dtype, stop_gradient=False
    )
    token_num=bs*seq_len

    qkv = paddle.concat(
            [
                q.transpose([0, 2, 1, 3]).reshape(
                    [token_num, q_num_head*dim_head]
                ),
                k.transpose([0, 2, 1, 3]).reshape(
                    [token_num, kv_num_head*dim_head]
                ),
                v.transpose([0, 2, 1, 3]).reshape(
                    [token_num, kv_num_head*dim_head]
                ),
            ],
            axis=1,
        ).reshape([token_num, -1])
    return q, k, v, qkv


class TestAppendGroupQueryAttnRoPE(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestAppendGroupQueryAttnRoPE"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 1
        self.q_num_head = 12
        self.kv_num_head = 2
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 128
        self.q_hid_dim = self.q_num_head * self.dim_head
        self.kv_hid_dim = self.kv_num_head * self.dim_head
        self.blocksize = 64
        self.use_neox_rotary_style = False
        #max_seq_len = self.seq_len + self.max_dec_len
        self.max_seq_len=self.seq_len + self.max_dec_len
        self.softmax_scale=self.dim_head**-0.5
        self.block_num_per_seq = (
            self.seq_len + self.max_dec_len + self.blocksize - 1
        ) // self.blocksize
        self.rope = RopeEmbedding()
        self.max_block_num = self.block_num_per_seq * self.batch_size
        self.free_list = list(range(self.max_block_num - 1, -1, -1))
        
        self.seq_lens_enc = [
            self.seq_len,
        ] * self.batch_size
        self.seq_lens_dec = [
            0,
        ] * self.batch_size
        self.max_enc_len_this_time = max(self.seq_lens_enc)
        self.max_dec_len_this_time = max(self.seq_lens_dec)
        self.seq_lens_encoder = paddle.to_tensor(
            self.seq_lens_enc,
            "int32",
        )
        self.seq_lens_decoder = paddle.to_tensor(
            self.seq_lens_dec,
            "int32",
        )
        self.max_enc_len_this_time = paddle.to_tensor([self.max_enc_len_this_time], "int32", place=paddle.CPUPlace())
        self.max_dec_len_this_time = paddle.to_tensor([self.max_dec_len_this_time], "int32", place=paddle.CPUPlace())
        self.seq_lens_this_time = self.seq_lens_encoder
       
        self.cache_shape = (
            self.max_block_num,
            self.kv_num_head,
            self.blocksize,
            self.dim_head,
        )
        self.dtype = 'float16'

        self.scale = 1.0 / np.sqrt(self.dim_head)
        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.block_tables = paddle.zeros(
            shape=(self.batch_size, self.block_num_per_seq), dtype="int32"
        )
        for i in range(self.batch_size):
            need_block_num = (
                self.seq_len + self.max_dec_len + self.blocksize - 1
            ) // self.blocksize
            for j in range(need_block_num):
                self.block_tables[i, j] = self.free_list.pop()
        (
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
        ) = get_padding_offset(
            self.batch_size, self.seq_len, self.seq_lens_this_time
        )
        self.token_num = self.padding_offset.shape[0]
        
    def get_rotary_position_embedding(self, position_ids, head_dim):
        bsz, max_seq_len = position_ids.shape[:2]
        rot_emb = paddle.zeros(
            (2, bsz, max_seq_len, 1, head_dim // 2), dtype="float32"
        )
        inv_freq = 10000 ** (
            -paddle.arange(0, head_dim, 2, dtype="float32") / head_dim
        )

        # shape: [B, S, D/2]
        freqs = paddle.einsum(
            "ij,k->ijk", position_ids.cast("float32"), inv_freq
        )
        # shape: [B, S, D/2]
        emb = paddle.stack([freqs], axis=-1).reshape(
            (bsz, max_seq_len, head_dim // 2)
        )
        # shape: [B, S, 1, D]
        emb = paddle.unsqueeze(emb, 2)

        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)
        return rot_emb
    
    def ref_append_attention(self,naive_cache_k=None,naive_cache_v=None,attn_mask=None):
        paddle.disable_static()
        self.token_num=self.seq_len*self.batch_size
        q, k, v, qkv = get_qkv_and_qkv_concat_tensor(
            self.batch_size,
            self.q_num_head,
            self.kv_num_head,
            self.seq_len,
            self.dim_head,
            self.place,
            self.dtype
        )
        out_ = naive_attention_impl(
            q, k, v, naive_cache_k, naive_cache_v, None, None, attn_mask, self.scale
        )
        out_ = remove_padding(
            self.seq_lens_this_time, self.cu_seqlens_q, out_, self.token_num
        )
        speculate_max_draft_token_num=1
        from paddlenlp_ops import get_block_shape_and_split_kv_block

        (
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            decoder_num_blocks_cpu,
            max_len_kv,
        ) = get_block_shape_and_split_kv_block(
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.max_enc_len_this_time,
            self.max_dec_len_this_time,
            self.seq_lens_this_time,
            self.cum_offset,
            self.q_num_head // self.kv_num_head,
            self.blocksize,
            speculate_max_draft_token_num,
        )

        from paddlenlp_ops import append_attention

        #Warm up
        WARM_UP = 1
        RUN_TIME = 2
        for i in range(WARM_UP+RUN_TIME):
            if i == WARM_UP:
                paddle.device.synchronize()
                start_time = time.time()
            out = append_attention(
                qkv,
                self.cache_k,
                self.cache_v,
                self.seq_lens_encoder,
                self.seq_lens_decoder,
                self.seq_lens_this_time,
                self.padding_offset,
                self.cum_offset,
                self.block_tables,
                encoder_batch_ids,
                encoder_tile_ids_per_batch,
                encoder_num_blocks,
                kv_batch_ids,
                kv_tile_ids_per_batch,
                kv_num_blocks,
                decoder_batch_ids,
                decoder_tile_ids_per_batch,
                decoder_num_blocks_cpu,
                self.max_enc_len_this_time,
                self.max_dec_len_this_time,
                max_len_kv,
                None,
                None,  # attn_mask
                None,  # qkv_bias
                None,  # qkv_out_scales
                None, # cache_k_quant_scales
                None, # cache_v_quant_scales
                None, # cache_k_dequant_scales
                None, # cache_v_dequant_scales
                None, # cache_k_zp 这看不懂
                None, # cache_v_zp 这看不懂
                None, # out_shifts 看不懂
                None, # out_smooths 看不懂
                "fp16",
                "none", # cache_quant_type
                self.use_neox_rotary_style,
                self.max_seq_len,
                self.softmax_scale,
                0.0, #quant_min_bound
                0.0, #quant_max_bound
                0.0, # out_linear_in_scale
                speculate_max_draft_token_num,#speculate_max_draft_token_num
                True,  # causal
                False,  # speculate_decoder
            )[0]
        paddle.device.synchronize()
        end_time = time.time()
        print(
        "[append-attn ut]  cost_time:{}ms".format(
            (end_time - start_time) / RUN_TIME * 1000
        )
        )

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=5e-03,
            atol=1e-02,
        )
        

    def test_all(self):
        tmp_position_ids = paddle.arange(
            self.seq_len + self.max_dec_len
        ).reshape((1, -1))
        self.rope_emb = self.get_rotary_position_embedding(
            tmp_position_ids, self.dim_head
        )
        self.attention_mask = create_attn_mask(
            self.dtype,
            self.batch_size,
            [
                self.seq_len,
            ]
            * self.batch_size,
        )
        #encoder
        # self.seq_lens_encoder,self.seq_lens_decoder,self.max_enc_len_this_time,self.max_dec_len_this_time=get_encoder_decoder_len(self.batch_size,self.seq_len)
        self.seq_lens_this_time = self.seq_lens_encoder
        self.ref_append_attention(attn_mask=self.attention_mask)

        naive_cache_k, naive_cache_v = block_cache_to_naive_cache(
            self.cache_k,
            self.cache_v,
            self.batch_size,
            self.block_tables,
            self.seq_len,
        )

        self.seq_lens_decoder[:] = self.seq_lens_encoder
        self.seq_lens_encoder[:] = 0
        self.seq_lens_this_time[:] = 1
        self.seq_lens_enc = [
            0,
        ] * self.batch_size
        self.seq_lens_dec = [
            self.seq_len,
        ] * self.batch_size
        self.max_enc_len_this_time = max(self.seq_lens_enc)
        self.max_dec_len_this_time = max(self.seq_lens_dec)
        self.max_enc_len_this_time = paddle.to_tensor([self.max_enc_len_this_time], "int32", place=paddle.CPUPlace())
        self.max_dec_len_this_time = paddle.to_tensor([self.max_dec_len_this_time], "int32", place=paddle.CPUPlace())
        # self.attention_mask = make_causal_mask(
        #     self.dtype,
        #     self.batch_size,
        #     1,
        #     naive_cache_k.shape[2]
        # )
        self.seq_len = 1
        (
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
        ) = get_padding_offset(self.batch_size, 1, self.seq_lens_this_time)
        self.ref_append_attention(naive_cache_k,naive_cache_v,None)

    
if __name__ == '__main__':
    unittest.main()