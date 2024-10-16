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
from __future__ import annotations

from typing import Tuple

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F

from ..utils.log import logger


class MoEGateMixin:
    def gate_score_func(self, logits: paddle.Tensor) -> paddle.Tensor:
        # [..., hidden_dim] -> [..., num_experts]
        with paddle.amp.auto_cast(False):
            scoring_func = getattr(self, "scoring_func", None)
            if scoring_func == "softmax":
                scores = F.softmax(logits.cast("float32"), axis=-1)
            elif scoring_func == "sigmoid":
                scores = F.sigmoid(logits.cast("float32"))
            elif scoring_func == "tanh":
                scores = F.tanh(logits.cast("float32"))
            elif scoring_func == "relu":
                scores = F.relu(logits.cast("float32"))
            elif scoring_func == "gelu":
                scores = F.gelu(logits.cast("float32"))
            elif scoring_func == "leaky_relu":
                scores = F.leaky_relu(logits.cast("float32"))
            else:
                logger.warning(f"insupportable scoring function for MoE gating: {scoring_func}, use softmax instead")
                scores = F.softmax(logits.cast("float32"), axis=-1)
        return scores

    def gumbel_rsample(self, logits: paddle.Tensor) -> paddle.Tensor:
        gumbel = paddle.distribution.gumbel.Gumbel(0, 1)
        return gumbel.rsample(logits.shape)

    def uniform_sample(self, logits: paddle.Tensor) -> paddle.Tensor:
        uniform = paddle.distribution.uniform.Uniform(0, 1)
        return uniform.sample(logits.shape)

    @paddle.no_grad()
    def _one_hot_to_float(self, x, num_classes):
        if x.dtype not in (paddle.int32, paddle.int64):
            x = paddle.cast(x, paddle.int64)
        return F.one_hot(x, num_classes=num_classes).cast(paddle.float32)

    @paddle.no_grad()
    def _one_hot_to_int64(self, x, num_classes):
        if x.dtype not in (paddle.int32, paddle.int64):
            x = paddle.cast(x, paddle.int64)
        return F.one_hot(x, num_classes=num_classes).cast(paddle.int64)

    @paddle.no_grad()
    def _capacity(self, gates: paddle.Tensor, capacity_factor: float, min_capacity: int) -> paddle.Tensor:
        """Calculate the capacity for each expert based on the gates and capacity factor.

        Args:
            gates (paddle.Tensor): A tensor of shape [num_tokens, num_experts] representing the probability distribution
                over experts for each token.
            capacity_factor (float): A scalar float value representing the capacity factor for each expert.
            min_capacity (int): A scalar integer value representing the minimum capacity for each expert.

        Returns:
            int: A tensor value representing the calculated capacity for each expert.
        """
        assert gates.ndim == 2, f"gates should be 2D, but got {gates.ndim}, {gates.shape}"
        # gates has shape of SE
        num_tokens = gates.shape[0]
        num_experts = gates.shape[1]
        capacity = int((num_tokens // num_experts) * capacity_factor)
        if capacity < min_capacity:
            capacity = min_capacity
        assert capacity > 0, f"requires capacity > 0, capacity_factor: {capacity_factor}, input_shape: {gates.shape}"

        return capacity

    def _cal_aux_loss(self, gates, mask):
        """
        计算辅助损失

        Args:
            gates (paddle.Tensor): 表示每个expert的输出概率。形状为[batch_size，num_experts]
            mask (paddle.Tensor): 表示每个样本是否属于某个expert。形状为[batch_size，num_experts]

        Returns:
            paddle.Tensor: 辅助损失值。

        """
        me = paddle.mean(gates, axis=0)
        ce = paddle.mean(mask.cast("float32"), axis=0)
        if self.global_aux_loss:
            me_list, ce_list = [], []
            dist.all_gather(me_list, me, group=self.group)
            dist.all_gather(ce_list, ce, group=self.group)

            me_list[self.rank] = me
            ce_list[self.rank] = ce
            me = paddle.stack(me_list).mean(0)
            ce = paddle.stack(ce_list).mean(0)
        aux_loss = paddle.sum(me * ce) * float(self.num_experts)
        return aux_loss

    def _cal_z_loss(self, logits) -> paddle.Tensor:
        """
        计算z损失
        Args:
            logits (paddle.paddle.Tensor): 模型输出。形状为[batch_size, num_experts]
        Returns:
            paddle.paddle.Tensor: z损失值。
        """
        l_zloss = logits.exp().sum(1).log().square().mean()
        return l_zloss

    def _cal_orthogonal_loss(self) -> paddle.Tensor:
        """Gate weight orthogonal loss.

        Returns:
            Paddle.Tensor: orthogonal loss
        """
        weight = F.normalize(self.weight, axis=0)
        orthogonal_loss = paddle.mean(paddle.square(paddle.matmul(weight.T, weight) - paddle.eye(self.num_experts)))
        return orthogonal_loss


class PretrainedMoEGate(nn.Layer, MoEGateMixin):
    def __init__(self, num_experts, expert_hidden_size, **kwargs):
        super(PretrainedMoEGate, self).__init__()

        self.num_experts = num_experts
        self.expert_hidden_size = expert_hidden_size

        # force keep in float32 when using amp
        self._cast_to_low_precision = False

        self.capacity_factor = kwargs["capacity_factor"] if hasattr(kwargs, "capacity_factor") else 1.0  # fmt:skip
        self.eval_capacity_factor = kwargs["eval_capacity_factor"] if hasattr(kwargs, "eval_capacity_factor") else 1.0  # fmt:skip
        self.min_capacity = kwargs["min_capacity"] if hasattr(kwargs, "min_capacity") else 1.0  # fmt:skip

        self.group = kwargs["group"] if hasattr(kwargs, "group") else None
        self.global_aux_loss = kwargs["global_aux_loss"] if hasattr(kwargs, "global_aux_loss") else False
        if self.global_aux_loss:
            assert self.group is not None, "group is required when global_aux_loss is True"
            self.rank = dist.get_rank(self.group)

        self.expert_drop = kwargs["expert_drop"] if hasattr(kwargs, "expert_drop") else False
        self.noisy_gate_policy = kwargs["noisy_gate_policy"] if hasattr(kwargs, "noisy_gate_policy") else None
        self.drop_tokens = kwargs["drop_tokens"] if hasattr(kwargs, "drop_tokens") else True
        self.use_rts = kwargs["use_rts"] if hasattr(kwargs, "use_rts") else True
        self.top2_2nd_expert_sampling = (
            kwargs["top2_2nd_expert_sampling"] if hasattr(kwargs, "top2_2nd_expert_sampling") else True
        )
        self.drop_policy = kwargs["drop_policy"] if hasattr(kwargs, "drop_policy") else "probs"
        self.top_k = kwargs["top_k"] if hasattr(kwargs, "top_k") else 1

    def topk_navie(self, scores: paddle.Tensor, k: int) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """_summary_

        Args:
            scores (paddle.Tensor): [bsz*seq_len, n_experts]
            k (int): select the top k experts

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: topk_weight, topk_idx
            topk_weight: [bsz*seq_len, k]
            topk_idx: [bsz*seq_len, k]
        """
        topk_weight, topk_idx = paddle.topk(scores, k=k, axis=-1, sorted=False)
        return topk_weight, topk_idx

    def topk_group(
        self, scores: paddle.Tensor, k: int, n_group: int, topk_group: int
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """_summary_

        Args:
            scores (paddle.Tensor): [bsz*seq_len, n_experts]
            k (int): select the top k experts in each group
            n_groups (int): the number of groups for all experts
            topk_group (int): the number of groups selected

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: topk_weight, topk_idx
            topk_weight: [bsz*seq_len, k]
            topk_idx: [bsz*seq_len, k]

        Note: the group size is normal greater than the number of k
        """
        bsz_seq_len, n_experts = scores.shape
        assert n_experts % n_group == 0, "n_experts must be divisible by n_groups"

        group_scores = scores.reshape([0, n_group, -1]).max(axis=-1)  # [n, n_group]
        group_idx = paddle.topk(group_scores, k=topk_group, axis=-1, sorted=False)[1]  # [n, top_k_group]
        group_mask = paddle.zeros_like(group_scores).put_along_axis(group_idx, paddle.to_tensor(1.0), axis=-1)  # fmt:skip
        score_mask = (
            group_mask.unsqueeze(-1).expand([bsz_seq_len, n_group, n_experts // n_group]).reshape([bsz_seq_len, -1])
        )  # [n, e]
        tmp_scores = scores * score_mask  # [n, e]
        topk_weight, topk_idx = paddle.topk(tmp_scores, k=k, axis=-1, sorted=False)

        return topk_weight, topk_idx

    def top1gating(
        self,
        logits: paddle.Tensor,
        used_token: paddle.Tensor = None,
    ) -> Tuple[int, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Implements Top1Gating on logits."""
        if self.noisy_gate_policy == "RSample":
            logits += self.gumbel_rsample(logits.shape)

        gates = self.gate_score_func(logits=logits)
        capacity = self._capacity(gates, self.capacity_factor, self.min_capacity)

        # Create a mask for 1st's expert per token
        # noisy gating
        indices1_s = paddle.argmax(logits if self.noisy_gate_policy == "RSample" else gates, axis=1)  # 仅保存最大值位置
        mask1 = self._one_hot_to_float(indices1_s, num_classes=self.num_experts)  # 将最大值位置转换为one-hot向量 [s, e]

        # mask only used tokens
        if used_token is not None:
            mask1 = paddle.einsum("s,se->se", used_token, mask1)  # 将used_token与mask1进行逐元素相乘，得到新的mask1

        # gating decisions
        exp_counts = paddle.sum(mask1, axis=0)  # 计算每个专家的token数量

        # if we don't want to drop any tokens
        if not self.drop_tokens:
            new_capacity = paddle.max(exp_counts)  # 计算每个专家的token数量
            # Communicate across expert processes to pick the maximum capacity.
            if self.group is not None:
                dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=self.group)  # 在专家进程之间进行最大值计算
            # Make sure the capacity value does not exceed the number of tokens.
            capacity = int(min(new_capacity, paddle.tensor(mask1.size(0))))

        l_aux = self._cal_aux_loss(gates, mask1)
        l_zloss = self._cal_z_loss(logits)

        # Random Token Selection
        if self.use_rts:
            mask1_rand = mask1 * self.uniform_sample(mask1)
        else:
            mask1_rand = mask1

        assert (
            logits.shape[0] >= self.min_capacity
        ), "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

        _, top_idx = paddle.topk(mask1_rand, k=capacity, axis=0)  # 选择top_capacity个token

        # 将mask1中的元素与top_idx进行逐元素相乘，得到新的mask1
        new_mask1 = mask1 * paddle.zeros_like(mask1).put_along_axis(top_idx, paddle.to_tensor(1.0), axis=0)
        mask1 = new_mask1

        # Compute locations in capacity buffer
        locations1 = paddle.cumsum(mask1, axis=0) - 1  # 计算每个token在mask1中的位置

        # Store the capacity location for each token
        locations1_s = paddle.sum(locations1 * mask1, axis=1).cast(paddle.int64)  # 计算每个token在mask1中的位置

        # Normalize gate probabilities
        mask1_float = mask1.cast(paddle.float32)
        gates = gates / gates * mask1_float

        locations1_sc = self._one_hot_to_float(locations1_s, capacity)
        combine_weights = paddle.einsum("se,sc->sec", gates, locations1_sc)
        dispatch_mask = combine_weights.cast(paddle.bool).detach()

        return capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss

    def top2gating(
        self,
        logits: paddle.Tensor,
    ) -> Tuple[int, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Args:
            logits: [S, E]，形状为 [seq_len, num_experts]，用于计算top2 gate。
            cap: 表示每个token可以分发的最大数量的超参数。

        Returns:
            tuple:
                - capacity: 每个token可分发的最大数量。
                - dispatch_masks: 用于dispatching的mask。
                - combine_weights：用于combining的权重。
                - scatter_indexes: 用于scattering的索引。
                - loss_aux: aux loss。
                - loss_z: z loss。
        """
        """Implements Top2Gating on logits."""
        # everything is in fp32 in this function
        gates = self.gate_score_func(logits=logits)

        # Create a mask for 1st's expert per token.
        indices1_s = paddle.argmax(gates, axis=1)  # [S, 1]
        mask1 = self._one_hot_to_int64(indices1_s, self.num_experts)  # [S, E]

        if self.top2_2nd_expert_sampling:
            # Create a mask for 2nd's expert per token using Gumbel-max trick.
            # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
            logits += self.gumbel_rsample(logits)

        # Replace top-expert with min value
        logits_except1 = logits.masked_fill(mask1.cast(paddle.bool), float("-inf"))  # [S, E]
        indices2_s = paddle.argmax(logits_except1, axis=1)  # [S, 1]
        mask2 = self._one_hot_to_int64(indices2_s, self.num_experts)  # [S, E]

        # Note: mask1 and mask2 can be combined to form a single mask.
        # mask = paddle.concat([mask1, mask2], axis=0)
        # locations = paddle.cumsum(mask, axis=0) - 1
        # locations1, locations2 = locations.split(2, axis=0)
        # Compute locations in capacity buffer.
        locations1 = paddle.cumsum(mask1, axis=0) - 1  # [S, E]
        locations2 = paddle.cumsum(mask2, axis=0) - 1  # [S, E]
        # Update 2nd's location by accounting for locations of 1st.
        locations2 += paddle.sum(mask1, axis=0, keepdim=True)

        l_aux = self._cal_aux_loss(gates, mask1)
        l_zloss = self._cal_z_loss(logits)

        # gating decisions
        exp_counts = paddle.sum(mask1 + mask2, axis=0)
        if self.drop_tokens:
            # Calculate configured capacity and remove locations outside capacity from mask
            capacity = self._capacity(gates, self.capacity_factor, self.min_capacity)
            # Remove locations outside capacity from mask.
            mask1 *= (locations1 < capacity).cast(paddle.int64)
            mask2 *= (locations2 < capacity).cast(paddle.int64)
        else:
            # Do not drop tokens - set capacity according to current expert assignments
            new_capacity = paddle.max(exp_counts)
            if self.group is not None:
                dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=self.group)
            capacity = int(new_capacity)

        # Store the capacity location for each token.
        locations1_s = paddle.sum(locations1 * mask1, axis=1)
        locations2_s = paddle.sum(locations2 * mask2, axis=1)

        # Normalize gate probabilities
        mask1_float = mask1.cast(paddle.float32)
        mask2_float = mask2.cast(paddle.float32)
        gates1_s = paddle.einsum("se,se->s", gates, mask1_float)
        gates2_s = paddle.einsum("se,se->s", gates, mask2_float)
        denom_s = gates1_s + gates2_s
        # Avoid divide-by-zero
        denom_s = paddle.clip(denom_s, min=paddle.finfo(denom_s.dtype).eps)
        gates1_s /= denom_s
        gates2_s /= denom_s

        # Calculate combine_weights and dispatch_mask
        gates1 = paddle.einsum("s,se->se", gates1_s, mask1_float)
        gates2 = paddle.einsum("s,se->se", gates2_s, mask2_float)
        locations1_sc = self._one_hot_to_float(locations1_s, capacity)
        locations2_sc = self._one_hot_to_float(locations2_s, capacity)
        combine1_sec = paddle.einsum("se,sc->sec", gates1, locations1_sc)
        combine2_sec = paddle.einsum("se,sc->sec", gates2, locations2_sc)
        combine_weights = combine1_sec + combine2_sec
        dispatch_mask = combine_weights.cast(paddle.bool)

        return capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss

    def topkgating(
        self,
        gates: paddle.Tensor,
    ) -> Tuple[int, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Implements TopKGating on logits."""
        # get topk gates
        top_gate, top_idx = paddle.topk(gates, k=self.top_k, axis=1)
        # get topk mask
        mask = paddle.zeros_like(gates).put_along_axis(top_idx, paddle.to_tensor(1.0), axis=1)
        exp_counts = paddle.sum(mask, axis=0)

        l_aux = self._cal_aux_loss(gates, mask)

        if self.drop_tokens:
            # Calculate configured capacity and remove locations outside capacity from mask
            capacity = self._capacity(gates, self.capacity_factor * self.top_k, self.min_capacity)

            # update mask and locations by capacity
            if self.drop_policy == "probs":
                topk_masked_gates = paddle.zeros_like(gates).put_along_axis(top_idx, top_gate, axis=1)
                capacity_probs, capacity_indices = paddle.topk(topk_masked_gates, k=capacity, axis=0, sorted=False)
                capacity_mask = paddle.zeros_like(gates).put_along_axis(capacity_indices, paddle.to_tensor(1.0), axis=0)  # fmt:skip
                mask = mask * capacity_mask
                locations = paddle.cumsum(mask, axis=0) - 1

            elif self.drop_policy == "position":
                locations = paddle.cumsum(mask, axis=0) - 1
                mask *= (locations < capacity).cast(paddle.int64)
            else:
                raise ValueError(f"Invalid drop_policy: {self.drop_policy}")
        else:
            # Do not drop tokens - set capacity according to current expert assignments
            new_capacity = paddle.max(exp_counts)
            if self.group is not None:
                dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=self.group)
            capacity = int(new_capacity)

        # normalize gates
        gates_masked = gates * mask
        gates_s = paddle.sum(gates_masked, axis=-1, keepdim=True)
        denom_s = paddle.clip(gates_s, min=paddle.finfo(gates_masked.dtype).eps)
        gates_masked = gates_masked / denom_s

        # dispatch_mask
        locations_sc = self._one_hot_to_float(locations * mask, num_classes=capacity)
        combine_weights = paddle.einsum("se,sec->sec", gates_masked, locations_sc)
        dispatch_mask = combine_weights.cast(paddle.bool)

        return capacity, combine_weights, dispatch_mask, exp_counts, l_aux

    def forward(self, hidden_states):
        raise NotImplementedError("Please implement the forward function.")


class TopKGate(PretrainedMoEGate):
    def __init__(
        self,
        num_experts,
        expert_hidden_size,
        weight_attr=None,
        bias_attr=None,
        top_k=2,
        capacity_factor=1.0,
        eval_capacity_factor=1.0,
        scoring_func="softmax",
        scaling_attr=None,
    ):
        super().__init__(num_experts, expert_hidden_size, weight_attr, bias_attr)
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.scaling_attr = scaling_attr

    def forward(
        self,
        hidden_states: paddle.Tensor,
        used_token: paddle.Tensor = None,
    ):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape([-1, hidden_size])
        logits = F.linear(x=paddle.cast(hidden_states, paddle.float32), weight=self.weight, bias=self.bias)
        if self.top_k == 1:
            gate_output = self.top1gating(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity,
                used_token,
                self.noisy_gate_policy if self.training else None,
                self.drop_tokens,
            )
        elif self.top_k == 2:
            gate_output = self.top2gating(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity,
                self.drop_tokens,
                self.top2_2nd_expert_sampling,
            )
        else:
            gate_output = self.topkgating(
                logits,
                self.top_k,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity,
                self.drop_tokens,
            )

        return gate_output
