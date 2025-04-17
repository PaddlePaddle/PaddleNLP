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

"""
this code is modified from https://github.com/DAMO-NLP-SG/Inf-CLIP/blob/main/inf_cl/ring.py
"""
import paddle
import paddle.autograd
import paddle.distributed
import paddle.distributed as dist
import paddle.nn.functional as F

from .flash import _cal_flash_loss, _flash_prob_backward, _flash_prob_forward


def init_dp_sd_comm_group():
    hcg = dist.fleet.get_hybrid_communicate_group()
    dp_world_size = hcg.get_data_parallel_world_size()
    sd_world_size = hcg.get_sharding_parallel_world_size()

    if dp_world_size > 1 and sd_world_size > 1:
        dp_sd_group, dp_sd_comm_group = hcg.create_fuse_group(["data", "sharding"])
    elif dp_world_size > 1:
        dp_sd_group, dp_sd_comm_group = hcg._dp_group, hcg.get_data_parallel_group()
    elif sd_world_size > 1:
        dp_sd_group, dp_sd_comm_group = hcg._sharding_group, hcg.get_sharding_parallel_group()

    hcg._dp_sd_group = dp_sd_group
    hcg._dp_sd_comm_group = dp_sd_comm_group
    return dp_sd_group, dp_sd_comm_group


class RingComm:
    def __init__(self, group):
        self.group = group
        self._ops = []
        self._reqs = None
        self.group_rank = group.rank
        self.world_size = group.world_size
        self.send_rank = self.group.ranks[(self.group_rank + 1) % self.world_size]
        self.recv_rank = self.group.ranks[(self.group_rank - 1) % self.world_size]

    def send_recv(self, to_send, recv_tensor=None):
        if recv_tensor is None:
            res = paddle.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, self.group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, self.group)

        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []


class RingProb(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        group=None,
    ):
        if group is None:
            hcg = dist.fleet.get_hybrid_communicate_group()
            if not hasattr(hcg, "_dp_sd_group") and not hasattr(hcg, "_dp_sd_comm_group"):
                init_dp_sd_comm_group()
            group = hcg._dp_sd_comm_group

        assert group is not None, "Communication group must be specified!"

        k = k.contiguous()
        comm = RingComm(group)

        colle = [q, k]

        lse = None
        next_k = None
        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k: paddle.Tensor = comm.send_recv(k)
                comm.commit()

            # vanilla lse
            qk = paddle.einsum("mhd,nhd->mn", q, k)
            block_lse = paddle.log(paddle.exp(qk).sum(axis=-1))

            if step == 0:
                lse = block_lse
            else:
                lse = lse - F.sigmoid(lse - block_lse).log()

            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k

        # this should be out_padded
        colle.append(lse)
        ctx.save_for_backward(*colle)
        ctx.group = group
        return lse

    @staticmethod
    def backward(ctx, dlse):
        q, k, lse = ctx.saved_tensor()
        k_comm = RingComm(ctx.group)
        d_k_comm = RingComm(ctx.group)
        dq, dk = None, None
        next_dk = None

        block_dq_buffer = paddle.empty(q.shape, dtype=paddle.float32)
        block_dk_buffer = paddle.empty(k.shape, dtype=paddle.float32)

        next_dk, next_k = None, None

        for step in range(k_comm.world_size):
            if step + 1 != k_comm.world_size:
                next_k = k_comm.send_recv(k)
                k_comm.commit()

            # vanilla gradient calculation
            qk = paddle.einsum("mhd,nhd->mn", q, k)
            qk_grad = paddle.exp(qk - lse[:, None]).cast("float32")
            qk_grad = qk_grad * dlse[:, None]
            block_dq_buffer = paddle.einsum("mn,nhd->mhd", qk_grad, k.cast("float32"))
            block_dk_buffer = paddle.einsum("nm,mhd->nhd", qk_grad.T, q.cast("float32"))

            if step == 0:
                dq = block_dq_buffer
                dk = block_dk_buffer
            else:
                dq += block_dq_buffer
                d_k_comm.wait()
                dk = block_dk_buffer + next_dk

            if step + 1 != k_comm.world_size:
                k_comm.wait()
                k = next_k

            next_dk = d_k_comm.send_recv(dk)
            d_k_comm.commit()

        d_k_comm.wait()

        return dq, next_dk


class InfProb(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, q, k, group):
        if group is None:
            hcg = dist.fleet.get_hybrid_communicate_group()
            if not hasattr(hcg, "_dp_sd_group") and not hasattr(hcg, "_dp_sd_comm_group"):
                init_dp_sd_comm_group()
            group = hcg._dp_sd_comm_group

        assert group is not None, "Communication group must be specified!"

        k = k.contiguous()
        comm = RingComm(group)

        colle = [q, k]

        lse = None
        next_k = None
        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k: paddle.Tensor = comm.send_recv(k)
                comm.commit()

            # flash lse
            block_lse = _flash_prob_forward(q, k)

            if step == 0:
                lse = block_lse
            else:
                lse = lse - F.sigmoid(lse - block_lse).log()

            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k

        # this should be out_padded
        colle.append(lse)
        ctx.save_for_backward(*colle)
        ctx.group = group
        return lse

    @staticmethod
    def backward(ctx, dlse):
        q, k, lse = ctx.saved_tensor()
        k_comm = RingComm(ctx.group)
        d_k_comm = RingComm(ctx.group)
        dq, dk = None, None
        next_dk = None

        block_dq_buffer = paddle.empty(q.shape, dtype=paddle.float32)
        block_dk_buffer = paddle.empty(k.shape, dtype=paddle.float32)

        next_dk, next_k = None, None

        for step in range(k_comm.world_size):
            if step + 1 != k_comm.world_size:
                next_k = k_comm.send_recv(k)
                k_comm.commit()

            # flash gradient calculation
            block_dq_buffer, block_dk_buffer = _flash_prob_backward(q, k, lse, dlse)

            if step == 0:
                dq = block_dq_buffer
                dk = block_dk_buffer
            else:
                dq += block_dq_buffer
                d_k_comm.wait()
                dk = block_dk_buffer + next_dk

            if step + 1 != k_comm.world_size:
                k_comm.wait()
                k = next_k

            next_dk = d_k_comm.send_recv(dk)
            d_k_comm.commit()

        d_k_comm.wait()

        return dq, next_dk


def _cal_ring_loss(q, k, labels, head_dim=256):
    bq = q.shape[0]
    bk = k.shape[0]
    q = q.reshape([bq, -1, head_dim]).cast("float32")
    k = k.reshape([bk, -1, head_dim]).cast("float32")

    lse = RingProb.apply(q, k, None)
    numerator = paddle.einsum("mhd,mhd->m", q, k[labels, ...])
    loss = -numerator + lse

    return loss


def _cal_inf_loss(q, k, labels, head_dim=256):
    bq = q.shape[0]
    bk = k.shape[0]
    q = q.reshape([bq, -1, head_dim]).cast("float32")
    k = k.reshape([bk, -1, head_dim]).cast("float32")

    lse = InfProb.apply(q, k, None)
    numerator = paddle.einsum("mhd,mhd->m", q, k[labels, ...])
    loss = -numerator + lse

    return loss


class GradientGather(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, dx):
        dist.all_reduce(dx)
        return dx


def cal_ring_loss(q, k, labels=None, scale=None, head_dim=256):
    """The paddle implementation of the ring-cl.

    Args:
        q (paddle.Tensor): The column tensor in contrastive loss. The shape is [B, D].
        k (paddle.Tensor): The row tensor in contrastive loss. The shape is [B, D].
        labels (paddle.Tensor, optional): In CLIP loss, the labels are the indices of the positive pairs. The shape is [B]. When setting to None, the labels are the range of [0, B). Defaults to None.
        scale (paddle.Tensor, optional): The scale tensor of the query tensor. Defaults to None.
        head_dim (int, optional): The head dimension. (must be 16, 32, 64, 128 or 256). Defaults to 256.

    """

    if labels is None:
        labels = paddle.arange(q.shape[0])
    if scale is not None and scale != 1.0:
        scale = GradientGather.apply(scale)
        q = scale * q

    if paddle.distributed.is_initialized():
        return _cal_ring_loss(q, k, labels, head_dim).mean()
    else:
        return _cal_flash_loss(q, k, labels, head_dim).mean()


def cal_inf_loss(q, k, labels=None, scale=None, head_dim=256):
    """The triton implementation of the inf-cl.

    Args:
        q (paddle.Tensor): The column tensor in contrastive loss. The shape is [B, D].
        k (paddle.Tensor): The row tensor in contrastive loss. The shape is [B, D].
        labels (paddle.Tensor, optional): In CLIP loss, the labels are the indices of the positive pairs. The shape is [B]. When setting to None, the labels are the range of [0, B). Defaults to None.
        scale (paddle.Tensor, optional): The scale tensor of the query tensor. Defaults to None.
        head_dim (int, optional): The head dimension. (must be 16, 32, 64, 128 or 256). Defaults to 256.

    """

    if labels is None:
        labels = paddle.arange(q.shape[0])
    if scale is not None and scale != 1.0:
        scale = GradientGather.apply(scale)
        q = scale * q
    if paddle.distributed.is_initialized():
        return _cal_inf_loss(q, k, labels, head_dim).mean()
    else:
        return _cal_flash_loss(q, k, labels, head_dim).mean()


if __name__ == "__main__":
    import time

    import numpy as np

    strategy = paddle.distributed.fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 2,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1,
        "sep_degree": 1,
    }
    paddle.distributed.fleet.init(is_collective=True, strategy=strategy)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Parameters
    dtype = paddle.float32
    num_heads = 3  # Number of attention heads
    seq_length_q = 32768  # Sequence length
    seq_length_k = 32768
    d_model = 256  # Dimension of each head (must be 16, 32, 64, or 128)

    # Randomly initialize inputs
    q = paddle.rand((seq_length_q // world_size, num_heads * d_model), dtype=dtype)
    k = paddle.rand((seq_length_k // world_size, num_heads * d_model), dtype=dtype)
    l = paddle.ones([], dtype=dtype) * np.log(1 / 0.07)
    l.stop_gradient = False  # Logit scale

    q = F.normalize(q, p=2, axis=-1)
    q.stop_gradient = False  # Query
    k = F.normalize(k, p=2, axis=-1)
    k.stop_gradient = False  # Key

    q1 = q.clone().detach()
    q1.stop_gradient = False
    k1 = k.clone().detach()
    k1.stop_gradient = False
    l1 = l.clone().detach()
    l1.stop_gradient = False

    for i in range(1000):
        # A. local torch gradient
        start = time.time()
        # A.1. gather q, k
        gathered_q = [paddle.zeros_like(q) for _ in range(world_size)]
        gathered_k = [paddle.zeros_like(k) for _ in range(world_size)]
        dist.all_gather(gathered_q, q)
        dist.all_gather(gathered_k, k)
        gathered_q[rank] = q
        gathered_k[rank] = k
        all_q = paddle.concat(gathered_q, axis=0)
        all_k = paddle.concat(gathered_k, axis=0)
        # A.2. calculating qk logits
        qk = paddle.einsum("md,nd->mn", l.exp() * all_q, all_k)
        kq = qk.T
        _labels = paddle.arange(seq_length_q)
        # A.3. calculating loss
        loss_i2t = F.cross_entropy(qk, _labels, reduction="mean")
        loss_t2i = F.cross_entropy(kq, _labels, reduction="mean")
        # A.4. scaling loss to normal value
        scale_factor = all_q.shape[0] / q.shape[0]
        loss = (loss_i2t + loss_t2i) * 0.5 * scale_factor
        loss.backward()
        show_loss = loss.detach().clone()
        dist.all_reduce(show_loss)
        show_loss = show_loss / (world_size * scale_factor)
        end = time.time()

        dist.barrier()

        # B. triton implementation
        start1 = time.time()
        # labels = torch.arange(seq_length_q // world_size).to(q.device)
        loss1_i2t = cal_inf_loss(q1, k1, scale=l1.exp())
        loss1_t2i = cal_inf_loss(k1, q1, scale=l1.exp())
        loss1 = (loss1_i2t + loss1_t2i).mean() * 0.5
        loss1.backward()
        end1 = time.time()

        dist.barrier()

        if rank == 0:
            print(rank, end - start, end1 - start1, loss, show_loss, loss1)
            print(l.grad, l1.grad, paddle.max(paddle.abs(q.grad - q1.grad)), paddle.max(paddle.abs(k.grad - k1.grad)))

        set_to_zero = False
        q.clear_gradient(set_to_zero)
        k.clear_gradient(set_to_zero)
        l.clear_gradient(set_to_zero)
        q1.clear_gradient(set_to_zero)
        k1.clear_gradient(set_to_zero)
        l1.clear_gradient(set_to_zero)
