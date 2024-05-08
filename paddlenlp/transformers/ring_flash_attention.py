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

# paddlenlp/transformers/ring_attention.py

import random

import numpy as np
import paddle
import paddle.distributed as dist
from custom_setup_ops import flash_attn_bwd
from paddle import _C_ops
from paddle.autograd.py_layer import PyLayer
from paddle.nn.functional.flash_attention import scaled_dot_product_attention


class RingCommunicator:
    def __init__(self, group, local_key, local_value):
        self._k_buffer = [paddle.zeros_like(local_key) for _ in range(2)]
        self._v_buffer = [paddle.zeros_like(local_value) for _ in range(2)]

        self._k_buffer[0] = local_key.clone()
        self._v_buffer[0] = local_value.clone()

        self._next_buffer_idx = 0

        self.group = group
        self.group_rank = group.rank
        self.send_rank = self.group.ranks[(self.group_rank + 1) % self.group.world_size]
        self.recv_rank = self.group.ranks[(self.group_rank - 1) % self.group.world_size]

        self._reqs = []

    def wait(self):
        # for req in self._reqs:
        #     req.wait()
        # self._reqs = None
        paddle.device.synchronize()

    def add_to_buffers(self, key, value):
        if key.shape != self._k_buffer[self._next_buffer_idx].shape:
            self._k_buffer[self._next_buffer_idx][:, : key.shape[1], :, :] += key
            self._v_buffer[self._next_buffer_idx][:, : key.shape[1], :, :] += value
        else:
            self._k_buffer[self._next_buffer_idx] += key
            self._v_buffer[self._next_buffer_idx] += value

    def get_buffers(self):
        return self._k_buffer[self._next_buffer_idx], self._v_buffer[self._next_buffer_idx]

    def send_recv(self):
        send_k_op = dist.P2POp(dist.isend, self._k_buffer[self._next_buffer_idx], self.send_rank, self.group)
        send_v_op = dist.P2POp(dist.isend, self._v_buffer[self._next_buffer_idx], self.send_rank, self.group)
        recv_k_op = dist.P2POp(dist.irecv, self._k_buffer[(self._next_buffer_idx + 1) % 2], self.recv_rank, self.group)
        recv_v_op = dist.P2POp(dist.irecv, self._v_buffer[(self._next_buffer_idx + 1) % 2], self.recv_rank, self.group)

        self._next_buffer_idx = (self._next_buffer_idx + 1) % 2

        ops = [send_k_op, send_v_op, recv_k_op, recv_v_op]

        self._reqs = dist.batch_isend_irecv(ops)


def update_out_and_lse(old_out, old_lse, block_out, block_lse, second_chunk_only=False):
    if second_chunk_only:
        second_chunk_out = old_out[:, old_out.shape[1] // 2 :, :, :]
        second_chunk_lse = old_lse[:, old_lse.shape[1] // 2 :, :, :]
        second_chunk_out, second_chunk_lse = update_out_and_lse(
            second_chunk_out, second_chunk_lse, block_out, block_lse
        )
        old_out[:, old_out.shape[1] // 2 :, :, :] = second_chunk_out
        old_lse[:, old_lse.shape[1] // 2 :, :, :] = second_chunk_lse
        return old_out, old_lse
    else:
        lse = paddle.log(1 + paddle.exp(block_lse - old_lse)) + old_lse
        return old_out * paddle.exp(old_lse - lse) + block_out * paddle.exp(block_lse - lse), lse


def get_chunk_id(rank, cp_size):
    return rank, (2 * cp_size - 1 - rank)


def concat_masks(attn_masks_list, rank, cp_size):
    assert len(attn_masks_list) == 2 * cp_size
    first_chunk_id, second_chunk_id = get_chunk_id(rank, cp_size)
    return paddle.concat([attn_masks_list[first_chunk_id], attn_masks_list[second_chunk_id]], axis=3)


def balanced_ring_flash_attention_fwd_func(
    group,
    local_query,
    local_key,
    local_value,
    fixed_seed_offset=None,
    attn_mask=None,
    dropout=0.0,
    is_causal=False,
    training=True,
):
    cp_size = group.world_size
    rank = group.rank

    comm_buffer = RingCommunicator(group, local_key, local_value)
    local_q_seq_len = local_query.shape[1]

    if attn_mask is not None:
        attn_masks_list = paddle.split(attn_mask, num_or_sections=cp_size * 2, axis=3)
    if is_causal:
        local_query_second_chunk = local_query[:, local_q_seq_len // 2 :, :, :].clone().contiguous()
    for step in range(cp_size):
        block_k, block_v = comm_buffer.get_buffers()

        if step != cp_size - 1:
            comm_buffer.send_recv()

        if not is_causal:
            # out [bs, seq, nhead, headdim]
            # lse [bs, nhead, seq]
            block_out, _, block_lse, _ = _C_ops.flash_attn(
                local_query,
                block_k,
                block_v,
                fixed_seed_offset,
                None if attn_mask is None else concat_masks(attn_masks_list, (group.rank - step) % cp_size, cp_size),
                dropout,
                False,
                False,
                not training,
                "",
            )
            block_lse = paddle.unsqueeze(paddle.transpose(block_lse, [0, 2, 1]), axis=-1)

            if step == 0:
                out, lse = block_out, block_lse
            else:
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            if step == 0:
                block_out, _, block_lse, _ = _C_ops.flash_attn(
                    local_query, block_k, block_v, fixed_seed_offset, None, dropout, True, False, not training, ""
                )
                block_lse = paddle.unsqueeze(paddle.transpose(block_lse, [0, 2, 1]), axis=-1)
                out, lse = block_out, block_lse
            elif step > rank:
                block_out, _, block_lse, _ = _C_ops.flash_attn(
                    local_query_second_chunk,
                    block_k,
                    block_v,
                    fixed_seed_offset,
                    None,
                    dropout,
                    False,
                    False,
                    not training,
                    "",
                )
                block_lse = block_lse[:, :, 0 : (local_q_seq_len // 2)]
                block_lse = paddle.unsqueeze(paddle.transpose(block_lse, [0, 2, 1]), axis=-1)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse, True)
            else:
                block_out, _, block_lse, _ = _C_ops.flash_attn(
                    local_query,
                    block_k[:, : local_q_seq_len // 2, :, :],
                    block_v[:, : local_q_seq_len // 2, :, :],
                    fixed_seed_offset,
                    None,
                    dropout,
                    False,
                    False,
                    not training,
                    "",
                )
                block_lse = paddle.unsqueeze(paddle.transpose(block_lse, [0, 2, 1]), axis=-1)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        # if step != cp_size - 1:
        #     comm_buffer.wait()
        paddle.device.synchronize()

    out = out.to(local_query.dtype)
    lse = paddle.transpose(paddle.squeeze(lse, axis=-1), [0, 2, 1])
    return out, lse


def balanced_ring_flash_attention_bwd_func(
    group,
    out_grad,
    local_query,
    local_key,
    local_value,
    local_out,
    lse,
    fixed_seed_offset,
    attn_mask,
    dropout=0.0,
    is_causal=False,
):
    cp_size = group.world_size
    rank = group.rank

    local_q_seq_len = local_query.shape[1]

    query_grad_buffer = paddle.zeros_like(local_query).to("float32")
    key_grad_buffer = paddle.zeros_like(local_key).to("float32")
    value_grad_buffer = paddle.zeros_like(local_value).to("float32")

    kv_comm_buffer = RingCommunicator(group, local_key, local_value)
    grad_comm_buffer = RingCommunicator(group, key_grad_buffer, value_grad_buffer)

    if is_causal:
        local_query_second_chunk = local_query[:, local_q_seq_len // 2 :, :, :].clone().contiguous()
        local_out_second_chunk = local_out[:, local_q_seq_len // 2 :, :, :].clone().contiguous()
        lse_second_chunk = lse[:, :, local_q_seq_len // 2 :].clone().contiguous()
        out_grad_second_chunk = out_grad[:, local_q_seq_len // 2 :, :, :].clone().contiguous()

    if attn_mask is not None:
        attn_masks_list = paddle.split(attn_mask, num_or_sections=cp_size * 2, axis=3)

    for step in range(cp_size):
        block_k, block_v = kv_comm_buffer.get_buffers()

        if step != cp_size - 1:
            kv_comm_buffer.send_recv()

        if not is_causal:
            block_q_grad, block_k_grad, block_v_grad = flash_attn_bwd(
                local_query,
                block_k,
                block_v,
                local_out,
                lse,
                fixed_seed_offset,
                None if attn_mask is None else concat_masks(attn_masks_list, (group.rank - step) % cp_size, cp_size),
                out_grad,
                dropout,
                False,
            )
            query_grad_buffer += block_q_grad
        else:
            if step == 0:
                block_q_grad, block_k_grad, block_v_grad = flash_attn_bwd(
                    local_query, block_k, block_v, local_out, lse, fixed_seed_offset, None, out_grad, dropout, True
                )
                query_grad_buffer += block_q_grad
            elif step > rank:
                block_q_grad, block_k_grad, block_v_grad = flash_attn_bwd(
                    local_query_second_chunk,
                    block_k,
                    block_v,
                    local_out_second_chunk,
                    lse_second_chunk,
                    fixed_seed_offset,
                    None,
                    out_grad_second_chunk,
                    dropout,
                    False,
                )
                query_grad_buffer[:, local_q_seq_len // 2 :, :, :] += block_q_grad
            else:
                block_q_grad, block_k_grad, block_v_grad = flash_attn_bwd(
                    local_query,
                    block_k[:, : local_q_seq_len // 2, :, :],
                    block_v[:, : local_q_seq_len // 2, :, :],
                    local_out,
                    lse,
                    fixed_seed_offset,
                    None,
                    out_grad,
                    dropout,
                    False,
                )
                query_grad_buffer += block_q_grad

        # if step != cp_size - 1:
        #     kv_comm_buffer.wait()
        # if step != 0:
        #     grad_comm_buffer.wait()
        paddle.device.synchronize()

        grad_comm_buffer.add_to_buffers(block_k_grad, block_v_grad)
        grad_comm_buffer.send_recv()

    grad_comm_buffer.wait()
    key_grad_buffer, value_grad_buffer = grad_comm_buffer.get_buffers()

    dtype = local_query.dtype
    return query_grad_buffer.to(dtype), key_grad_buffer.to(dtype), value_grad_buffer.to(dtype)


class RingFlashAttention(PyLayer):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        group=None,
        fixed_seed_offset=None,
        attn_mask=None,
        dropout=0.0,
        is_causal=False,
        training=True,
    ):
        if dropout > 0.0:
            raise NotImplementedError("Dropout is not supported in ring attention yet.")
        if group is None:
            group = dist.fleet.get_hybrid_communicate_group().get_cp_parallel_group()
        if attn_mask is not None:
            is_causal = False

        out, lse = balanced_ring_flash_attention_fwd_func(
            group, query, key, value, fixed_seed_offset, attn_mask, dropout, is_causal, training
        )
        ctx.save_for_backward(query, key, value, out, lse, attn_mask)
        ctx.group = group
        ctx.fixed_seed_offset = fixed_seed_offset
        ctx.dropout = dropout
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, out_grad):
        query, key, value, out, lse, attn_mask = ctx.saved_tensor()
        group = ctx.group
        fixed_seed_offset = ctx.fixed_seed_offset
        dropout = ctx.dropout
        is_causal = ctx.is_causal

        if fixed_seed_offset is None:
            fixed_seed_offset = paddle.to_tensor([0, 0], place=paddle.CPUPlace(), dtype=paddle.int64).contiguous()

        query_grad, key_grad, value_grad = balanced_ring_flash_attention_bwd_func(
            group, out_grad, query, key, value, out, lse, fixed_seed_offset, attn_mask, dropout, is_causal
        )
        if attn_mask is not None and not attn_mask.stop_gradient:
            return query_grad, key_grad, value_grad, None
        else:
            return query_grad, key_grad, value_grad


import unittest


class TestRingFlashAttention(unittest.TestCase):
    def setUp(self):
        paddle.distributed.init_parallel_env()
        self.group = paddle.distributed.new_group(range(paddle.distributed.get_world_size()), backend="nccl")
        self.degree = self.group.world_size
        self.rank = self.group.rank

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def generate_full_data(self, batch_size, seq_len, num_head, head_dim):
        query = (paddle.randn([batch_size, seq_len, num_head, head_dim], dtype=paddle.float32)).to("gpu", "float16")
        key = (paddle.randn([batch_size, seq_len, num_head, head_dim], dtype=paddle.float32)).to("gpu", "float16")
        value = (paddle.randn([batch_size, seq_len, num_head, head_dim], dtype=paddle.float32)).to("gpu", "float16")

        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False

        return query, key, value

    def split_belanced_data(self, input):
        sliced_datas = paddle.split(input, num_or_sections=self.degree * 2, axis=1)
        sliced_data0, sliced_data1 = sliced_datas[self.rank], sliced_datas[self.degree * 2 - 1 - self.rank]
        return paddle.concat([sliced_data0, sliced_data1], axis=1).detach()

    def single_test(self, bsz, seq_len_per_device, head_num, head_dim, is_causal, use_mask):
        query, key, value = self.generate_full_data(bsz, seq_len_per_device * self.degree, head_num, head_dim)

        local_query = self.split_belanced_data(query)
        local_key = self.split_belanced_data(key)
        local_value = self.split_belanced_data(value)

        local_query.stop_gradient = False
        local_key.stop_gradient = False
        local_value.stop_gradient = False

        if use_mask:
            mask_shape = (1, 1, query.shape[1], query.shape[1])
            mask = np.random.random(mask_shape)
            attn_mask = paddle.to_tensor(mask, place=query.place, dtype=query.dtype)
            attn_mask = paddle.ones(mask_shape).to(query.dtype)
            attn_mask_list = paddle.split(attn_mask, axis=2, num_or_sections=self.degree * 2)
            first_chunk_id, second_chunk_id = get_chunk_id(self.rank, self.degree)
            local_attn_mask = paddle.concat([attn_mask_list[first_chunk_id], attn_mask_list[second_chunk_id]], axis=2)
        else:
            attn_mask = None
            local_attn_mask = None

        local_out = RingFlashAttention.apply(
            local_query, local_key, local_value, self.group, is_causal=is_causal, attn_mask=local_attn_mask
        )
        ref_out = scaled_dot_product_attention(query, key, value, is_causal=is_causal, attn_mask=attn_mask)
        ref_local_out = self.split_belanced_data(ref_out)
        np.testing.assert_allclose(local_out.numpy(), ref_local_out.numpy(), rtol=5e-03, atol=1e-03)

        local_out.backward()
        ref_out.backward()

        ref_local_query_grad = self.split_belanced_data(query.grad)
        ref_local_key_grad = self.split_belanced_data(key.grad)
        ref_local_value_grad = self.split_belanced_data(value.grad)

        np.testing.assert_allclose(local_query.grad.numpy(), ref_local_query_grad.numpy(), rtol=5e-03, atol=1e-03)
        np.testing.assert_allclose(local_key.grad.numpy(), ref_local_key_grad.numpy(), rtol=5e-03, atol=1e-03)
        np.testing.assert_allclose(local_value.grad.numpy(), ref_local_value_grad.numpy(), rtol=5e-03, atol=1e-03)

    def test_normal_flash_attention(self):
        self.single_test(1, 256, 1, 256, False, False)

    def test_masked_flash_attention(self):
        self.single_test(1, 256, 1, 256, False, True)

    def test_casual_flash_attention(self):
        self.single_test(1, 256, 1, 256, True, False)


if __name__ == "__main__":
    unittest.main()
# python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 ring_flash_attention.py
