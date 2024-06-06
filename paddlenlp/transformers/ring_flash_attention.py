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

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import _C_ops
from paddle.autograd.py_layer import PyLayer

try:
    from paddlenlp_ops import flash_attn_bwd
except (ImportError, ModuleNotFoundError):
    from paddlenlp.utils.log import logger

    logger.warning(
        "if you run ring_flash_attention.py, please ensure you install "
        "the paddlenlp_ops by following the instructions "
        "provided at https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md"
    )


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
        # TODO(zhangyuqin1998)：batch_isend_irecv异步流下，无法wait，需要修复。对性能有影响。
        paddle.device.synchronize()

    def add_to_buffers(self, key, value):
        if key.shape != self._k_buffer[self._next_buffer_idx].shape:
            k_buffer_chunk = paddle.slice(
                self._k_buffer[self._next_buffer_idx], axes=[1], starts=[0], ends=[key.shape[1]]
            )
            v_buffer_chunk = paddle.slice(
                self._v_buffer[self._next_buffer_idx], axes=[1], starts=[0], ends=[value.shape[1]]
            )
            k_buffer_chunk += key
            v_buffer_chunk += value
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
    if old_out is None and old_lse is None:
        return block_out.to("float32"), block_lse.to("float32")

    if second_chunk_only:
        second_chunk_out_ = paddle.slice(old_out, axes=[1], starts=[old_out.shape[1] // 2], ends=[old_out.shape[1]])
        second_chunk_lse_ = paddle.slice(old_lse, axes=[1], starts=[old_lse.shape[1] // 2], ends=[old_lse.shape[1]])
        second_chunk_out, second_chunk_lse = update_out_and_lse(
            second_chunk_out_, second_chunk_lse_, block_out, block_lse
        )
        paddle.assign(second_chunk_out, second_chunk_out_)
        paddle.assign(second_chunk_lse, second_chunk_lse_)
        return old_out, old_lse
    else:
        block_out, block_lse = block_out.to("float32"), block_lse.to("float32")
        with paddle.amp.auto_cast(enable=False, dtype="bfloat16"):
            lse = old_lse - F.log_sigmoid(old_lse - block_lse)
            return old_out - (old_out - block_out) * F.sigmoid(block_lse - old_lse), lse


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

    out, lse, k_cache, v_cache = None, None, dict(), dict()

    if attn_mask is not None:
        attn_masks_list = paddle.split(attn_mask, num_or_sections=cp_size * 2, axis=3)
    if is_causal:
        local_query_second_chunk = paddle.slice(
            local_query, axes=[1], starts=[local_q_seq_len // 2], ends=[local_q_seq_len]
        )
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
            block_lse = paddle.unsqueeze_(paddle.transpose_(block_lse, [0, 2, 1]), axis=-1)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            # block_k and block_v is from rank (group.rank - step) % cp_size
            if step == 0:
                block_out, _, block_lse, _ = _C_ops.flash_attn(
                    local_query, block_k, block_v, fixed_seed_offset, None, dropout, True, False, not training, ""
                )
                block_lse = paddle.unsqueeze_(paddle.transpose_(block_lse, [0, 2, 1]), axis=-1)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
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
                block_lse = paddle.slice(block_lse, axes=[1], starts=[0], ends=[local_q_seq_len // 2])
                block_lse = paddle.unsqueeze_(paddle.transpose_(block_lse, [0, 2, 1]), axis=-1)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse, True)
            else:
                block_k = paddle.slice(block_k, axes=[1], starts=[0], ends=[local_q_seq_len // 2])
                block_v = paddle.slice(block_v, axes=[1], starts=[0], ends=[local_q_seq_len // 2])
                block_out, _, block_lse, _ = _C_ops.flash_attn(
                    local_query,
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
                block_lse = paddle.unsqueeze_(paddle.transpose_(block_lse, [0, 2, 1]), axis=-1)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
                k_cache[step] = block_k
                v_cache[step] = block_v

        # TODO(zhangyuqin1998)：batch_isend_irecv异步流下，无法wait，需要修复。对性能有影响。
        paddle.device.synchronize()

    out = out.to(local_query.dtype)
    lse = paddle.transpose_(paddle.squeeze_(lse, axis=-1), [0, 2, 1])
    return out, lse, k_cache, v_cache


def balanced_ring_flash_attention_bwd_func(
    group,
    k_cache,
    v_cache,
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

    query_grad_buffer = paddle.zeros_like(local_query)
    key_grad_buffer = paddle.zeros_like(local_key)
    value_grad_buffer = paddle.zeros_like(local_value)

    kv_comm_buffer = RingCommunicator(group, local_key, local_value)
    grad_comm_buffer = RingCommunicator(group, key_grad_buffer, value_grad_buffer)

    if is_causal:
        local_query_second_chunk = paddle.slice(
            local_query, axes=[1], starts=[local_q_seq_len // 2], ends=[local_q_seq_len]
        )
        local_out_second_chunk = paddle.slice(
            local_out, axes=[1], starts=[local_q_seq_len // 2], ends=[local_q_seq_len]
        )
        lse_second_chunk = paddle.slice(lse, axes=[2], starts=[local_q_seq_len // 2], ends=[local_q_seq_len])
        out_grad_second_chunk = paddle.slice(out_grad, axes=[1], starts=[local_q_seq_len // 2], ends=[local_q_seq_len])
        query_grad_buffer_second_chunk = paddle.slice(
            query_grad_buffer, axes=[1], starts=[local_q_seq_len // 2], ends=[local_q_seq_len]
        )

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
                query_grad_buffer_second_chunk += block_q_grad
            else:
                block_q_grad, block_k_grad, block_v_grad = flash_attn_bwd(
                    local_query,
                    k_cache[step],
                    v_cache[step],
                    local_out,
                    lse,
                    fixed_seed_offset,
                    None,
                    out_grad,
                    dropout,
                    False,
                )
                query_grad_buffer += block_q_grad

        # TODO(zhangyuqin1998)：batch_isend_irecv异步流下，无法wait，需要修复。对性能有影响。
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
            group = dist.fleet.get_hybrid_communicate_group().get_sep_parallel_group()
        if attn_mask is not None:
            is_causal = False

        out, lse, k_cache, v_cache = balanced_ring_flash_attention_fwd_func(
            group, query, key, value, fixed_seed_offset, attn_mask, dropout, is_causal, training
        )
        ctx.save_for_backward(query, key, value, out, lse, attn_mask, k_cache, v_cache)
        ctx.group = group
        ctx.fixed_seed_offset = fixed_seed_offset
        ctx.dropout = dropout
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, out_grad):
        query, key, value, out, lse, attn_mask, k_cache, v_cache = ctx.saved_tensor()
        group = ctx.group
        fixed_seed_offset = ctx.fixed_seed_offset
        dropout = ctx.dropout
        is_causal = ctx.is_causal

        if fixed_seed_offset is None:
            fixed_seed_offset = paddle.to_tensor([0, 0], place=paddle.CPUPlace(), dtype=paddle.int64)

        query_grad, key_grad, value_grad = balanced_ring_flash_attention_bwd_func(
            group,
            k_cache,
            v_cache,
            out_grad,
            query,
            key,
            value,
            out,
            lse,
            fixed_seed_offset,
            attn_mask,
            dropout,
            is_causal,
        )
        if attn_mask is not None and not attn_mask.stop_gradient:
            return query_grad, key_grad, value_grad, None
        else:
            return query_grad, key_grad, value_grad
