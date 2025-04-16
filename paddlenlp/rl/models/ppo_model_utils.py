# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
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
"""Utilities for score models."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import paddle
import paddle.distributed
import paddle.distributed as dist
import paddle.incubate.nn.functional as PF
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import ParallelCrossEntropy

from ...transformers.llama.modeling import (
    LlamaPretrainingCriterion as PretrainingCriterion,
)
from ...transformers.model_outputs import ModelOutput


@dataclass
class PolicyOutput(ModelOutput):
    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    # logits_entropy: Optional[paddle.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class ValueOutput(ModelOutput):
    loss: Optional[paddle.Tensor] = None
    value: paddle.Tensor = None
    reward: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None


def merge_fwd_labels(loss_cls):
    """
    PipelineParallel and trainer.criterion both use labels as tuple, thus wrap.
    """
    ori_fwd = loss_cls.forward

    def loss_fwd(self, predict, labels):
        return ori_fwd(self, predict, *labels)

    fwd_params = inspect.signature(ori_fwd).parameters
    # forward(self, predict, label1, label2, ...)
    loss_cls.label_names = list(fwd_params.keys())[2:]
    loss_cls.label_default_values = {}
    for label_name in loss_cls.label_names:
        if fwd_params[label_name].default is not inspect.Parameter.empty:
            loss_cls.label_default_values[label_name] = fwd_params[label_name].default
    loss_cls.forward = loss_fwd
    return loss_cls


def create_loss(loss_cls, config, extra_args, info_buffer, merge_labels=None):
    """
    loss_cls(paddle.nn.Layer): loss class
    config(PratrainedConfig): model config, to be consistent with loss defined
        in transformers
    extra_args(dict): create loss with more args not in config
    merge_labels: use a wrapped loss_cls whose label args are merged into one arg,
        this is useful to PipelineParallel and trainer.criterion since they only
        support loss format corresponding to this format.
    """
    # TODO(guosheng): merge_labels if loss_cls not
    ori_fwd = loss_cls.forward
    if merge_labels:
        fwd_params = inspect.signature(ori_fwd).parameters
        if len(fwd_params.keys()) > 3:  # merge_fwd_labels has not done
            loss_cls = merge_fwd_labels(loss_cls)
    # forward(self, predict, label1, label2, ...)
    loss_arg_names = list(inspect.signature(loss_cls.__init__).parameters.keys())[2:]
    if isinstance(extra_args, dict):
        loss_kwargs = {name: extra_args[name] for name in loss_arg_names if name in extra_args}
    else:
        # create from TrainingArguments
        loss_kwargs = {name: getattr(extra_args, name) for name in loss_arg_names if hasattr(extra_args, name)}
    if "info_buffer" in loss_arg_names:
        loss_kwargs["info_buffer"] = info_buffer
    loss = loss_cls(config, **loss_kwargs)
    return loss


def make_position_ids_from_input_ids(input_ids, pad_token_id=0):
    assert input_ids.ndim == 2, "input_ids's shape must be 2d"
    position_ids = paddle.zeros_like(input_ids)
    for index, row in enumerate(input_ids):
        non_zero_indices = paddle.nonzero(row != pad_token_id).flatten()
        start_index = non_zero_indices[0]
        position_ids[index, start_index + 1 :] = 1
    return position_ids.cumsum(-1)


# def make_position_ids_from_input_ids(input_ids, pad_token_id=0):
#     position_ids = (input_ids != pad_token_id).cast("int32").cumsum(-1) - 1
#     return position_ids.masked_fill(position_ids < 0, 0)


@paddle.no_grad()
def make_position_ids(attention_mask, source=None):
    """
    根据attention_mask生成位置id，如果source不为空则将源端padding部分设置为0。
    当attention_mask的形状是[B, L, H, W]时，表示causal mask，返回的position_ids是[B, H, W]；
    当attention_mask的形状是[B, L]时，表示padding mask，返回的position_ids是[B, L]。

    Args:
        attention_mask (Tensor, numpy.ndarray): 形状为[B, L, H, W]或者[B, L]的Tensor/numpy数组，其中L是序列长度，H是头数，W是宽度（可选）。
            每个元素为0表示该位置未被mask，非0表示该位置被mask。
        source (Tensor, numpy.ndarray, optional): 形状为[B, S]的Tensor/numpy数组，其中S是源端序列长度（可选）。默认值为None。

    Returns:
        Tensor: 形状为[B, H, W]或者[B, L]的Tensor，其中H是头数，W是宽度（可选）。每个元素为对应位置的位置id。
        如果source不为空，则在源端padding部分设置为0。
    """
    if len(attention_mask.shape) == 4:  # causal mask
        position_ids_p1 = attention_mask.cast(paddle.int64).sum(-1)
        position_ids = position_ids_p1 - 1
        position_ids = paddle.where(position_ids == -1, position_ids_p1, position_ids)
        return position_ids[:, 0, :]
    assert len(attention_mask.shape) == 2  # padding mask
    attention_mask_bool = attention_mask
    attention_mask = attention_mask.cast(paddle.int64)
    position_ids = attention_mask.cumsum(-1) - 1
    # Make padding positions in source be 0, since reward model use position_ids
    # plus with padding size (number of 0s) in source to calculate end offsets.
    # It does not matter when source is left padding and target is right padding
    # which is the output of non-FuseMT generation, while when using FuseMT whose
    # output is right padding source and right padding target, we have to set
    # padding positions in source be 0 to make compatible.
    if source is not None:
        src_len = position_ids[:, source.shape[-1] - 1].unsqueeze(-1)
        position_ids = paddle.where(
            paddle.logical_and(paddle.logical_not(attention_mask_bool), position_ids <= src_len),
            attention_mask,
            position_ids,
        )
        return position_ids
    position_ids = paddle.where(position_ids == -1, attention_mask, position_ids)
    return position_ids


@paddle.no_grad()
def make_attention_mask(
    input_ids,
    pad_id,
    eos_id=None,
    unk_id=None,
    past_key_values_length=0,
    causal_mask=True,
):
    """
    根据输入的`input_ids`，生成一个注意力掩码。如果`pad_id`不是`unk_id`和`eos_id`中的任何一个，则该位置将被忽略。
    如果`causal_mask`为`False`，则返回全部为`True`的注意力掩码。否则，返回一个三角形掩码，其中每个元素都小于或等于相应位置的元素。

    Args:
        input_ids (Tensor): 输入序列的ID，形状为（batch_size, seq_len）。
        pad_id (int): 用于padding的ID。
        eos_id (int, optional): 用于表示结束的ID，默认为None。如果设置了，则会从注意力掩码中删除对应位置。
        unk_id (int, optional): 用于表示未知的ID，默认为None。如果设置了，则会从注意力掩码中删除对应位置。
        past_key_values_length (int, optional): 预先存在的键值对的长度，默认为0。
        causal_mask (bool, optional): 是否使用因果掩码，默认为True。

    Returns:
        Tensor: 注意力掩码，形状为（batch_size, 1, seq_len, seq_len + past_len）。
    """
    unk_id = None

    attention_mask = input_ids != pad_id
    if unk_id is not None and pad_id != unk_id:
        if eos_id is not None and unk_id != eos_id:
            attention_mask = paddle.logical_and(attention_mask, input_ids != unk_id)
    if eos_id is not None and pad_id != eos_id:
        attention_mask = paddle.logical_and(attention_mask, input_ids != eos_id)
    if not causal_mask:
        return attention_mask

    batch_size, target_length = input_ids.shape  # target_length: seq_len
    mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))
    if past_key_values_length > 0:
        # [tgt_len, tgt_len + past_len]
        mask = paddle.concat(
            [
                paddle.ones([target_length, past_key_values_length], dtype="bool"),
                mask,
            ],
            axis=-1,
        )
    # [bs, 1, tgt_len, tgt_len + past_len]
    causal_mask = mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])

    attention_mask = attention_mask[:, None, None, :]
    expanded_attn_mask = attention_mask & causal_mask
    return expanded_attn_mask


def gather_log_probabilities(logits: paddle.Tensor, labels: paddle.Tensor) -> paddle.Tensor:
    """Gather log probabilities of the given labels from the logits."""
    # log_probs = F.log_softmax(logits, axis=-1)
    # log_probs_labels = paddle.take_along_axis(log_probs, axis=-1, indices=labels.unsqueeze(axis=-1))
    # return log_probs_labels.squeeze(axis=-1)
    token_loss = F.cross_entropy(
        logits.cast("float32"),
        labels,
        reduction="none",
    ).squeeze(axis=-1)
    return -token_loss.cast(logits.dtype)


def create_startend_row_indices(input_ids, pad_token_id=0):
    startend_row_indices = paddle.full(input_ids.shape, input_ids.shape[-1], dtype="int32")
    mask = (input_ids != pad_token_id).cast("int32").cumsum(-1) == 0
    return startend_row_indices.masked_fill(mask, 0)


class RLHFPPOLoss(nn.Layer):
    def __init__(self, config, clip_range_ratio=0.2, clip_range_ratio_low=None, clip_range_ratio_high=None):
        """
        Initialize the `ClipRewardRange` object.

        Args:
            config (dict): A dictionary containing environment configuration parameters.
                See :class:`~rllib.agents.Agent` for more information.
            clip_range_ratio (float, optional): The ratio of the range to which the reward is clipped.
                Defaults to 0.2.

        Raises:
            None.

        Returns:
            None.
        """
        super().__init__()
        self.clip_range_ratio = clip_range_ratio
        self.clip_range_ratio_low = clip_range_ratio_low
        self.clip_range_ratio_high = clip_range_ratio_high
        self.config = config

    def actor_loss_fn(
        self,
        log_probs: paddle.Tensor,
        old_log_probs: paddle.Tensor,
        advantages: paddle.Tensor,
        mask: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        计算演员的策略损失函数。该函数接受以下参数：
        Args:
            log_probs (paddle.Tensor): 当前状态下每个演员的对数产生概率，形状为[B, A]，其中B是批量大小，A是演员数量。
            old_log_probs (paddle.Tensor): 上一时间步骤的每个演员的对数产生概率，形状与log_probs相同。
            advantages (paddle.Tensor): 每个演员在当前状态下获得的价值函数估计值，形状为[B, A]。
            mask (paddle.Tensor): 用于过滤已完成或无效的轨迹，形状为[B, A]，其中B是批量大小，A是演员数量。
                如果轨迹已经完成（即reward不为None），则mask为1；否则为0。
        返回值 (paddle.Tensor):
            PG_loss (paddle.Tensor): 演员的策略损失，形状为[1]。
        """
        # policy gradient loss

        ratio = paddle.exp(log_probs - old_log_probs)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * paddle.clip(
            ratio,
            1.0 - self.clip_range_ratio_low,
            1.0 + self.clip_range_ratio_high,
        )
        return paddle.sum(paddle.maximum(pg_loss1, pg_loss2) * mask) / mask.sum()

    def forward(self, log_probs, old_log_probs, reward_advantages, sequence_mask):
        """
        Calculate the loss of the actor network.

        Args:
            logits (Tensor, shape [batch_size, seq_len, vocab_size]): The output logits of the model.
            input_ids (Tensor, shape [batch_size, seq_len]): The input ids of the batch.
            old_log_probs (Tensor, shape [batch_size, seq_len]): The previous log probabilities of the batch.
            reward_advantages (Tensor, shape [batch_size, seq_len]): The rewards or advantages of the batch.
            sequence_mask (Tensor, shape [batch_size, seq_len]): A mask indicating which elements are valid.
                Valid elements are those where sequence_mask is True.

        Returns:
            Tensor, shape [1], the loss of the actor network.

        Raises:
            None.
        """
        actor_loss = self.actor_loss_fn(
            log_probs,
            old_log_probs,
            reward_advantages,
            sequence_mask,
        )
        return actor_loss


class VocabParallelEntropy(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: paddle.Tensor, tensor_parallel_output=False) -> paddle.Tensor:
        try:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            tensor_parallel_degree = hcg.get_model_parallel_world_size()
        except:
            tensor_parallel_degree = 1
        logits_max = vocab_parallel_logits.max(axis=-1, keepdim=True)

        if tensor_parallel_degree > 1 and tensor_parallel_output:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=model_parallel_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp()
        normalized_sum_exp_logits = normalized_exp_logits.sum(axis=-1, keepdim=True)

        if tensor_parallel_degree > 1 and tensor_parallel_output:
            dist.all_reduce(normalized_sum_exp_logits, group=model_parallel_group)
        softmax_logits = normalized_exp_logits / normalized_sum_exp_logits
        sum_softmax_times_logits = (softmax_logits * vocab_parallel_logits).sum(axis=-1, keepdim=True)

        if tensor_parallel_degree > 1 and tensor_parallel_output:
            dist.all_reduce(sum_softmax_times_logits, group=model_parallel_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(softmax_logits * (sum_softmax_times_logits - vocab_parallel_logits))
        return entropy.squeeze(axis=-1)

    @staticmethod
    def backward(ctx, grad_output: paddle.Tensor) -> paddle.Tensor:
        return grad_output.unsqueeze(axis=-1) * ctx.saved_tensor()[0]


def entropy_from_logits(logits: paddle.Tensor, tensor_parallel_output=False):
    return VocabParallelEntropy.apply(logits, tensor_parallel_output)


@merge_fwd_labels
class RLHFPPOMixedLoss(nn.Layer):
    """provide two losses, one for PPO loss, the other for SFT loss."""

    def __init__(
        self,
        config,
        ptx_coeff=16,
        clip_range_ratio=0.2,
        clip_range_ratio_low=None,
        clip_range_ratio_high=None,
        kl_loss_coeff=0.001,
        clip_range_score=10,
        info_buffer=None,
        temperature=1.0,
        entropy_coeff=0.001,
        pg_loss_coeff=1.0,
        use_fp32_compute=False,
    ):
        """
        Args:
        config (Config): configuration object containing hyperparameters and options for the agent.
        ptx_coeff (int, optional): coefficient to use in the PTX loss calculation. Defaults to 16.
        clip_range_ratio (float, optional): ratio of clipped range to unclipped range. Defaults to 0.2.
        """
        super(RLHFPPOMixedLoss, self).__init__()
        self.config = config
        self.ptx_coeff = ptx_coeff
        # if self.config.use_fused_head_and_loss_fn:
        #     self.ppo_criterion = FusedPPOLoss(config, clip_range_ratio, clip_range_ratio_low, clip_range_ratio_high)
        # else:
        #     self.ppo_criterion = RLHFPPOLoss(config, clip_range_ratio, clip_range_ratio_low, clip_range_ratio_high)
        self.clip_range_ratio_low = clip_range_ratio_low if clip_range_ratio_low is not None else clip_range_ratio
        self.clip_range_ratio_high = clip_range_ratio_high if clip_range_ratio_high is not None else clip_range_ratio
        self.ppo_criterion = RLHFPPOLoss(
            config, clip_range_ratio, self.clip_range_ratio_low, self.clip_range_ratio_high
        )
        self.sft_criterion = PretrainingCriterion(config)
        self.kl_loss_coeff = kl_loss_coeff
        self.clip_range_score = clip_range_score
        self.info_buffer = info_buffer
        self.temperature = temperature
        self.clip_range_ratio = clip_range_ratio
        self.entropy_coeff = entropy_coeff
        self.pg_loss_coeff = pg_loss_coeff
        self.use_fp32_compute = use_fp32_compute

    def forward(
        self,
        logits,
        labels,
        input_ids,
        old_log_probs,
        reward_advantages,
        sequence_mask,
        ref_log_probs=None,
        response_start=0,
    ):
        """
        计算损失函数，包含两部分：soft target loss和PPO loss。
        如果labels不为None，则计算soft target loss；否则计算PPO loss。

        Args:
            logits (paddle.Tensor or List[paddle.Tensor]): 输入的预测结果，可以是单个tensor或list中的多个tensor。
                如果是单个tensor，表示对应的输出logits；如果是list，表示每个时间步的logits。
            labels (paddle.Tensor, optional): 真实标签，shape与logits相同。默认为None。
            input_ids (paddle.Tensor, optional): 输入序列的id，shape为(batch_size, max_len)。默认为None。
            old_log_probs (paddle.Tensor, optional): 上一个时间步的log probabilities，shape为(batch_size, max_len)。默认为None。
            reward_advantages (paddle.Tensor, optional): 回报优势，shape为(batch_size, max_len)。默认为None。
            sequence_mask (paddle.Tensor, optional): 序列掩码，shape为(batch_size, max_len)。默认为None。

        Returns:
            paddle.Tensor: 返回损失函数，如果labels不为None，则为soft target loss；否则为PPO loss。
        """

        if not self.config.use_fused_head_and_loss_fn:
            logits = logits if isinstance(logits, paddle.Tensor) else logits[0]
            if self.use_fp32_compute and logits.dtype != paddle.float32:
                logits = logits.cast(paddle.float32)
            logits = logits / self.temperature if self.temperature > 0.0 else logits
        else:
            hidden_states, weight, bias, transpose_y = logits
            if self.use_fp32_compute and hidden_states.dtype != paddle.float32:
                hidden_states = hidden_states.cast(paddle.float32)
                weight = weight.cast(paddle.float32)
                if bias is not None:
                    bias = bias.cast(paddle.float32)
            total_loss, pg_loss, entropy_loss, kl_loss = actor_fused_pg_entropy_kl_loss(
                hidden_states,
                weight,
                input_ids,
                old_log_probs,
                ref_log_probs,
                reward_advantages,
                sequence_mask,
                bias=bias,
                transpose_y=transpose_y,
                fused_linear=False,
                vocab_size=self.config.vocab_size,
                tensor_parallel_degree=self.config.tensor_parallel_degree,
                tensor_parallel_output=self.config.tensor_parallel_output,
                pg_loss_coeff=self.pg_loss_coeff,  # donot use this
                clip_range_ratio=self.clip_range_ratio,
                clip_range_ratio_low=self.clip_range_ratio_low,
                clip_range_ratio_high=self.clip_range_ratio_high,
                entropy_coeff=self.entropy_coeff,  # donot support this
                clip_range_score=self.clip_range_score,
                kl_loss_coeff=self.kl_loss_coeff,
                loop_chunk_size=1024,
                response_start=response_start,
                use_actor_fused_loss=True,  # TODO, currently only support kunbo's fused head loss
                temperature=self.temperature,
            )
            with paddle.no_grad():
                self.info_buffer["kl_loss"] = (
                    kl_loss.detach() / self.kl_loss_coeff if self.kl_loss_coeff > 0 else paddle.to_tensor([0.0])
                )
                self.info_buffer["entropy_loss"] = (
                    entropy_loss.detach() / self.entropy_coeff if self.entropy_coeff > 0 else paddle.to_tensor([0.0])
                )
                self.info_buffer["pure_policy_loss"] = (
                    pg_loss.detach() / self.pg_loss_coeff if self.pg_loss_coeff > 0 else paddle.to_tensor([0.0])
                )
            return total_loss
        loss = None
        # sft, pt loss
        if labels is not None:
            loss = self.ptx_coeff * self.sft_criterion(logits, labels)
        # ppo loss
        if reward_advantages is not None:
            if self.config.tensor_parallel_degree > 1 and self.config.tensor_parallel_output:
                log_probs = (
                    -ParallelCrossEntropy()(
                        logits[:, response_start:-1].astype("float32"), input_ids[:, response_start + 1 :]
                    )
                    .squeeze(axis=-1)
                    .astype(logits.dtype)
                )
            else:
                log_probs = gather_log_probabilities(logits[:, response_start:-1], input_ids[:, response_start + 1 :])
            if log_probs.shape[1] == old_log_probs.shape[1]:
                # labels (old_log_probs, reward_advantages, sequence_mask) has
                # src+tgt-1 length, valid length is determined by sequence_mask
                pass
            elif log_probs.shape[1] < old_log_probs.shape[1]:
                # labels (old_log_probs, reward_advantages, sequence_mask) has
                # src+tgt length and the last one is a padding to be consistent
                # with input_ids
                assert log_probs.shape[1] == old_log_probs.shape[1] - 1
                log_probs = paddle.concat(
                    [
                        log_probs,
                        paddle.zeros([log_probs.shape[0], 1], dtype=log_probs.dtype),
                    ],
                    -1,
                )
            else:
                # labels (old_log_probs, reward_advantages, sequence_mask) has tgt length
                log_probs = log_probs[:, -old_log_probs.shape[1] :]

            # TODO:support fused head and loss fn
            loss = self.ppo_criterion(log_probs, old_log_probs, reward_advantages, sequence_mask)
            self.info_buffer["pure_policy_loss"] = loss.detach()
            loss = self.pg_loss_coeff * loss

        if ref_log_probs is not None:
            kl_divergence_estimate = paddle.clip(
                paddle.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1,
                min=-self.clip_range_score,
                max=self.clip_range_score,
            )
            kl_loss = paddle.sum(kl_divergence_estimate * sequence_mask) / sequence_mask.sum()
            self.info_buffer["kl_loss"] = kl_loss.detach()
            loss += self.kl_loss_coeff * kl_loss

        if self.entropy_coeff > 0:
            entropy_loss_raw = entropy_from_logits(logits[:, response_start:-1], self.config.tensor_parallel_output)
            entropy_loss = paddle.sum(entropy_loss_raw * sequence_mask) / sequence_mask.sum()
            self.info_buffer["entropy_loss"] = entropy_loss.detach()
            loss -= self.entropy_coeff * entropy_loss
        else:
            self.info_buffer["entropy_loss"] = paddle.to_tensor([0.0])

        return loss


@merge_fwd_labels
class RLHFValueLoss(nn.Layer):
    def __init__(self, config, clip_range_value=5.0):
        """
        Initializes the `ClipRewardRange` object.

        Args:
            config (dict): The configuration dictionary for the environment.
                See :ref:`rllib-spaces` for more information.
            clip_range_value (Optional[float]): The value to which the rewards will be clipped. Defaults to 5.0.

        Raises:
            None.

        Returns:
            None.
        """
        super().__init__()
        self.clip_range_value = clip_range_value
        self.config = config

    def critic_loss_fn(
        self,
        values: paddle.Tensor,
        old_values: paddle.Tensor,
        returns: paddle.Tensor,
        mask: paddle.Tensor,
    ) -> paddle.Tensor:
        """Compute critic loss."""
        # TODO(guosheng): use paddle.clip when its min/max can support more than
        # 0D Tensor
        values_clipped = paddle.minimum(
            paddle.maximum(values, old_values - self.clip_range_value),
            old_values + self.clip_range_value,
        )
        vf_loss1 = paddle.square(values - returns)
        vf_loss2 = paddle.square(values_clipped - returns)
        return 0.5 * paddle.sum(paddle.maximum(vf_loss1, vf_loss2) * mask) / mask.sum()

    def forward(self, reward_values, old_reward_values, reward_returns, sequence_mask):
        """
        计算奖励值的损失函数。
        如果输入的奖励值和旧奖励值的长度相同，则使用给定的序列掩码来确定有效长度。
        如果输入的奖励值的长度比旧奖励值少一个，则将最后一个元素视为与输入IDs一致的填充，并删除它。
        否则，奖励值只有tgt长度。

        Args:
            reward_values (paddle.Tensor, list of paddle.Tensor or None, optional): 奖励值，可以是单个张量或列表中的多个张量。默认为None。
            old_reward_values (paddle.Tensor, optional): 旧奖励值。
            reward_returns (paddle.Tensor, optional): 奖励返回值。
            sequence_mask (paddle.Tensor, optional): 序列掩码。

        Returns:
            paddle.Tensor, float32: 奖励值的损失函数。

        Raises:
            ValueError: 当奖励值和旧奖励值的长度不匹配时引发。
        """
        reward_values = reward_values if isinstance(reward_values, paddle.Tensor) else reward_values[0]
        reward_values = reward_values.squeeze(axis=-1)[:, :-1]
        if reward_values.shape[1] == old_reward_values.shape[1]:
            # labels (old_reward_values, reward_returns, sequence_mask) has
            # src+tgt-1 length, valid length is determined by sequence_mask
            pass
        elif reward_values.shape[1] < old_reward_values.shape[1]:
            # labels (old_reward_values, reward_returns, sequence_mask) has
            # src+tgt length and the last one is a padding to be consistent
            # with input_ids
            assert reward_values.shape[1] == old_reward_values.shape[1] - 1
            reward_values = paddle.concat(
                [
                    reward_values,
                    paddle.zeros([reward_values.shape[0], 1], dtype=reward_values.dtype),
                ],
                -1,
            )
        else:
            # labels (old_reward_values, reward_returns, sequence_mask) has
            # tgt length
            reward_values = reward_values[:, -old_reward_values.shape[1] :]
        reward_critic_loss = self.critic_loss_fn(
            reward_values,
            old_reward_values,
            reward_returns,
            sequence_mask,
        )

        return reward_critic_loss


class ActorFusedLoss(paddle.autograd.PyLayer):
    """Fused Actor Loss"""

    @staticmethod
    def forward(
        ctx,
        hidden_states: paddle.Tensor,
        lm_head_weight: paddle.Tensor,
        lm_head_bias: paddle.Tensor,
        labels: paddle.Tensor,
        mask: paddle.Tensor,
        transpose_y: bool,
        num_embeddings: int,
        tensor_parallel_degree: int,
        tensor_parallel_output: bool,
        fused_linear: bool,
        loop_chunk_size: int,
        ignore_index: int,
        old_log_probs: paddle.Tensor,
        ref_log_probs: paddle.Tensor,
        advantages: paddle.Tensor,
        clip_range_ratio: float,
        clip_range_ratio_low: float,
        clip_range_ratio_high: float,
        clip_range_score: float,
        kl_loss_coeff: float,  # KL loss coefficient
        temperature: float,
    ):
        """
        forward function of ActorFusedLoss

        Args:
            ctx (paddle.autograd.PyLayerContext): context.
            hidden_states (paddle.Tensor): hidden_states, [batch_size, seq_len-1, hidden_size].
            lm_head_weight (paddle.Tensor): lm_head_weight, [hidden_size, vocab_size / tensor_parallel_degree].
            lm_head_bias (paddle.Tensor, optional): lm_head_bias, [vocab_size / tensor_parallel_degree].
            labels (paddle.Tensor): labels, [batch_size, seq_len-1].
            mask (paddle.Tensor): mask, [batch_size, seq_len-1].
            transpose_y (bool): whether to transpose lm_head_weight.
            num_embeddings (int): vocab_size.
            tensor_parallel_degree (int): tensor_parallel_degree.
            tensor_parallel_output (bool): tensor_parallel_output, set True in ppo_main.py.
            fused_linear (bool): Flag for using fused linear, always False.
            loop_chunk_size (int): chunk_size.
            ignore_index (int): not used now.
            old_log_probs (paddle.Tensor): old_log_probs, [batch_size, seq_len-1].
            advantages (paddle.Tensor): advantages, [batch_size, seq_len-1].
            clip_range_ratio (float): The clipping range for ratio.

        Returns:
            paddle.Tensor: loss

        """
        if fused_linear:
            # print("Cannot support fused_linear while using use_fused_head_and_loss_fn now!")
            fused_linear = False
        if tensor_parallel_degree > 1:
            assert tensor_parallel_output, (
                "When tensor_parallel_degree > 1 and use_fused_head_and_loss_fn, "
                "tensor_parallel_output needs to be set to True."
            )
        dtype = hidden_states.dtype
        # Parallel Configuration
        if tensor_parallel_degree > 1 and tensor_parallel_output:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            tensor_parallel_degree = hcg.get_model_parallel_world_size()

        # reshape
        original_shape = hidden_states.shape
        hidden_states_stop_grad = hidden_states.stop_gradient  # original stop_gradient
        hidden_states = hidden_states.reshape([-1, original_shape[-1]])
        labels = labels.reshape([-1])
        old_log_probs = old_log_probs.reshape([-1])
        if kl_loss_coeff > 0:
            ref_log_probs = ref_log_probs.reshape([-1])
        advantages = advantages.reshape([-1])
        loss_mask = mask.reshape([-1]).astype("float32")  # .astype(dtype)

        n_tokens = hidden_states.shape[0]
        n_classes = lm_head_weight.shape[0] if transpose_y else lm_head_weight.shape[1]

        # convert dtype of weights and biases of lm_head
        lm_head_weight_cast = lm_head_weight.astype(dtype)
        if lm_head_bias is not None:
            lm_head_bias_cast = lm_head_bias.astype(dtype)

        # use indices to distinguish the devices.
        if tensor_parallel_degree > 1 and tensor_parallel_output:
            rank = hcg.get_model_parallel_rank()
            per_part_size = num_embeddings // tensor_parallel_degree
            indices = paddle.arange(
                rank * per_part_size,
                rank * per_part_size + n_classes,
                dtype=labels.dtype,
            ).unsqueeze(0)
        else:
            indices = paddle.arange(num_embeddings, dtype=labels.dtype).unsqueeze(0)

        # initialize total_loss and divisor
        total_loss = paddle.zeros([1], dtype=dtype)
        total_kl_loss = paddle.zeros([1], dtype=dtype)
        total_entropy_loss = paddle.zeros([1], dtype=dtype)
        divisor = loss_mask.sum()

        # initialize grads
        if not lm_head_weight.stop_gradient:
            grad_lm_head_weight = paddle.zeros_like(lm_head_weight)
        else:
            grad_lm_head_weight = None
        if lm_head_bias is not None and not lm_head_bias.stop_gradient:
            grad_lm_head_bias = paddle.zeros_like(lm_head_bias)
        else:
            grad_lm_head_bias = None
        if not hidden_states_stop_grad:
            grad_hidden_states = paddle.zeros_like(hidden_states)
        else:
            grad_hidden_states = None

        for i in range(0, n_tokens, loop_chunk_size):
            token_start_idx = i
            token_end_idx = min(i + loop_chunk_size, n_tokens)
            hidden_states_chunk = hidden_states[token_start_idx:token_end_idx]
            labels_chunk = labels[token_start_idx:token_end_idx]
            old_log_probs_chunk = old_log_probs[token_start_idx:token_end_idx]
            if kl_loss_coeff > 0:
                ref_log_chunk = ref_log_probs[token_start_idx:token_end_idx]
            advantages_chunk = advantages[token_start_idx:token_end_idx]
            mask_chunk = loss_mask[token_start_idx:token_end_idx]

            # Calculate the current logits_chunk,  not fused linear
            logits_chunk_cast = paddle.matmul(hidden_states_chunk, lm_head_weight_cast, transpose_y=transpose_y)
            if lm_head_bias is not None:
                logits_chunk_cast += lm_head_bias_cast
            # logits_chunk_cast = paddle.nn.functional.linear(hidden_states_chunk, lm_head_weight_cast, lm_head_bias)

            logits_chunk = logits_chunk_cast.astype("float32")
            logits_chunk = logits_chunk / temperature

            labels_one_hot = labels_chunk.unsqueeze(1) == indices
            # rewritten as cross entropy
            if tensor_parallel_degree > 1 and tensor_parallel_output:
                token_loss_chunk, softmax_output_chunk = mp_ops._c_softmax_with_cross_entropy(
                    logits_chunk,
                    labels_chunk,
                    group=model_parallel_group,
                    return_softmax=True,
                )
            else:
                token_loss_chunk = F.cross_entropy(logits_chunk, labels_chunk, reduction="none")
                softmax_output_chunk = F.softmax(logits_chunk, axis=-1)

            log_probs_chunk = -token_loss_chunk.squeeze(axis=-1)
            # calculate gradient, note sign
            grad_logits_chunk = labels_one_hot.astype("float32") - softmax_output_chunk
            grad_logits_chunk = grad_logits_chunk.astype(dtype)

            # ratio
            ratio_chunk = paddle.exp(log_probs_chunk - old_log_probs_chunk)
            clipped_ratio_chunk = paddle.clip(
                ratio_chunk, min=1.0 - clip_range_ratio_low, max=1.0 + clip_range_ratio_high
            )

            # final loss
            pg_loss1_chunk = -advantages_chunk * ratio_chunk
            pg_loss2_chunk = -advantages_chunk * clipped_ratio_chunk
            pg_loss_chunk = paddle.maximum(pg_loss1_chunk, pg_loss2_chunk)

            # mask
            pg_loss_chunk = pg_loss_chunk * mask_chunk
            masked_loss_sum = paddle.sum(pg_loss_chunk)
            # add
            total_loss += masked_loss_sum

            # grads
            # direction
            I1_chunk = (pg_loss1_chunk >= pg_loss2_chunk).astype(dtype)
            I2_chunk = 1.0 - I1_chunk

            # clip
            clip_mask_chunk = (
                (ratio_chunk >= 1.0 - clip_range_ratio) & (ratio_chunk <= 1.0 + clip_range_ratio)
            ).astype(dtype)

            # ∂loss1/∂log_probs, ∂loss2/∂log_probs
            d_ratio_d_log_probs_chunk = ratio_chunk
            d_pg_loss1_d_log_probs_chunk = -advantages_chunk * d_ratio_d_log_probs_chunk
            d_pg_loss2_d_log_probs_chunk = -advantages_chunk * clip_mask_chunk * d_ratio_d_log_probs_chunk

            # ∂loss/∂log_probs
            d_loss_d_log_probs_chunk = (
                I1_chunk * d_pg_loss1_d_log_probs_chunk + I2_chunk * d_pg_loss2_d_log_probs_chunk
            )
            d_loss_d_log_probs_chunk = d_loss_d_log_probs_chunk * mask_chunk / divisor

            # ∂log_probs/∂logits, just take the previous one.
            d_log_probs_d_logits_chunk = grad_logits_chunk / temperature
            # ∂loss/∂logits
            d_loss_d_logits_chunk = d_loss_d_log_probs_chunk.unsqueeze(-1) * d_log_probs_d_logits_chunk

            if kl_loss_coeff > 0:
                # [3] kl loss
                delta_chunk = ref_log_chunk - log_probs_chunk
                exp_delta_chunk = paddle.exp(delta_chunk)
                kl_loss_estimate_chunk = exp_delta_chunk - delta_chunk - 1
                kl_loss_clipped_chunk = (
                    paddle.clip(
                        kl_loss_estimate_chunk,
                        min=-clip_range_score,
                        max=clip_range_score,
                    )
                    * mask_chunk
                )
                total_kl_loss += kl_loss_clipped_chunk.sum() * kl_loss_coeff
                # gradgradgradgrad kl loss
                kl_within_clip_chunk = (
                    (kl_loss_estimate_chunk >= -clip_range_score) & (kl_loss_estimate_chunk <= clip_range_score)
                ).astype(dtype)
                d_kl_log_probs_chunk = (
                    (1 - exp_delta_chunk) * kl_within_clip_chunk * mask_chunk * kl_loss_coeff / divisor
                )
                d_loss_d_logits_chunk += d_kl_log_probs_chunk.unsqueeze(-1) * d_log_probs_d_logits_chunk

            # grads
            if grad_hidden_states is not None:
                grad_hidden_states[token_start_idx:token_end_idx] = paddle.matmul(
                    d_loss_d_logits_chunk, lm_head_weight_cast, transpose_y=not transpose_y
                )
            if grad_lm_head_weight is not None:
                if transpose_y:
                    grad_lm_head_weight += paddle.matmul(d_loss_d_logits_chunk, hidden_states_chunk, transpose_x=True)
                else:
                    grad_lm_head_weight += paddle.matmul(hidden_states_chunk, d_loss_d_logits_chunk, transpose_x=True)
            if grad_lm_head_bias is not None:
                grad_lm_head_bias += d_loss_d_logits_chunk.astype("float32").sum(axis=0).astype(dtype)

        final_loss = (total_loss + total_kl_loss) / divisor
        ctx.hidden_states_has_grad = grad_hidden_states is not None
        ctx.lm_head_weight_has_grad = grad_lm_head_weight is not None
        ctx.lm_head_bias_has_grad = grad_lm_head_bias is not None

        grad_args = []
        if ctx.hidden_states_has_grad:
            if tensor_parallel_degree > 1:
                dist.all_reduce(grad_hidden_states, op=dist.ReduceOp.SUM, group=model_parallel_group)
            grad_args.append(grad_hidden_states.reshape(original_shape))
        if ctx.lm_head_weight_has_grad:
            grad_args.append(grad_lm_head_weight)
        if ctx.lm_head_bias_has_grad:
            grad_args.append(grad_lm_head_bias)

        ctx.save_for_backward(*grad_args)
        return (
            final_loss,
            (total_loss / divisor).detach(),
            total_entropy_loss.detach(),
            (total_kl_loss / divisor).detach(),
        )

    @staticmethod
    def backward(ctx, grad_output, *args):
        """
        backward function of ActorFusedLoss

        Args:
            ctx: Context.
            grad_output(paddle.Tensor): Gradient.
        Returns:
            tuple:
                - Gradient tensors for hidden_states, lm_head_weight, and lm_head_bias,
                  None values are used for inputs not requiring gradients.
        """
        grad_args = ctx.saved_tensor()
        idx = 0
        if ctx.hidden_states_has_grad:
            grad_hidden_states = grad_args[idx] * grad_output.astype(grad_args[idx].dtype)
            idx += 1
        else:
            grad_hidden_states = None

        if ctx.lm_head_weight_has_grad:
            grad_lm_head_weight = grad_args[idx] * grad_output.astype(grad_args[idx].dtype)
            idx += 1
        else:
            grad_lm_head_weight = None

        if ctx.lm_head_bias_has_grad:
            grad_lm_head_bias = grad_args[idx] * grad_output.astype(grad_args[idx].dtype)
            idx += 1
        else:
            grad_lm_head_bias = None
        return grad_hidden_states, grad_lm_head_weight, grad_lm_head_bias, None, None


class FusedPPOLoss(nn.Layer):
    """Fused PPOLoss"""

    def __init__(self, config, clip_range_ratio=0.2, clip_range_ratio_low=None, clip_range_ratio_high=None):
        """Initialize FusedPPOLoss class."""
        super().__init__()
        self.clip_range_ratio = clip_range_ratio
        self.clip_range_ratio_low = clip_range_ratio_low
        self.clip_range_ratio_high = clip_range_ratio_high
        self.config = config

    def forward(
        self,
        hidden_states: paddle.Tensor,
        lm_head_weight: paddle.Tensor,
        lm_head_bias: paddle.Tensor,
        input_ids: paddle.Tensor,
        old_log_probs: paddle.Tensor,
        reward_advantages: paddle.Tensor,
        sequence_mask: paddle.Tensor,
        transpose_y: bool,
    ):
        """
        forward function of FusedPPOLoss

        Args:
            hidden_states (paddle.Tensor): hidden_states, [batch_size, seq_len, hidden_size].
            lm_head_weight (paddle.Tensor): lm_head_weight, [hidden_size, vocab_size / tensor_parallel_degree].
            lm_head_bias (paddle.Tensor, optional): lm_head_bias, [vocab_size / tensor_parallel_degree].
            input_ids (paddle.Tensor): input_ids, [batch_size, seq_len].
            old_log_probs (paddle.Tensor): old_log_probs, [batch_size, seq_len-1].
            reward_advantages (paddle.Tensor): advantages, [batch_size, seq_len-1].
            sequence_mask (paddle.Tensor): mask, [batch_size, seq_len-1].
            transpose_y (bool): whether to transpose lm_head_weight.

        Returns:
            paddle.Tensor: loss

        """
        logits_next = hidden_states[:, :-1, :]
        labels_next = input_ids[:, 1:]

        if old_log_probs.shape[1] != labels_next.shape[1]:
            # labels（old_log_probs，reward_advantages，sequence_mask）的长度为 src + tgt - 1，实际长度由 sequence_mask 确定
            raise ValueError("old_log_probs and reward_advantages should have the same length")

        actor_loss = ActorFusedLoss.apply(
            hidden_states=logits_next,
            lm_head_weight=lm_head_weight,
            lm_head_bias=lm_head_bias,
            labels=labels_next,
            mask=sequence_mask,
            transpose_y=transpose_y,
            num_embeddings=self.config.vocab_size,
            tensor_parallel_degree=self.config.tensor_parallel_degree,
            tensor_parallel_output=self.config.tensor_parallel_output,
            fused_linear=False,
            loop_chunk_size=1024,  # 128,
            ignore_index=0,
            old_log_probs=old_log_probs,
            advantages=reward_advantages,
            clip_range_ratio=self.clip_range_ratio,
            clip_range_ratio_low=self.clip_range_ratio_low,
            clip_range_ratio_high=self.clip_range_ratio_high,
        )
        return actor_loss


class ActorFusedPGEntropyKLLoss(paddle.autograd.PyLayer):
    """ActorFusedPGEntropyKLLoss"""

    @staticmethod
    def forward(
        ctx,
        hidden_states: paddle.Tensor,
        weight: paddle.Tensor,
        bias: paddle.Tensor,
        sequence_mask: paddle.Tensor,
        labels: paddle.Tensor,
        old_log_probs: paddle.Tensor,
        advantages: paddle.Tensor,
        ref_log_probs: paddle.Tensor,  # 新增参考策略的log概率
        transpose_y: bool,
        vocab_size: int,
        tensor_parallel_degree: int,
        tensor_parallel_output: bool,
        pg_loss_coeff: float,
        clip_range_ratio: float,  # pg loss
        clip_range_ratio_low: float,
        clip_range_ratio_high: float,
        entropy_coeff: float,  # entropy loss
        clip_range_score: float,  # clip loss
        kl_loss_coeff: float,  # clip loss
        fused_linear: bool,
        loop_chunk_size: int,
        temperature: float,
    ):
        if ref_log_probs is None:
            kl_loss_coeff = 0.0
        if tensor_parallel_degree > 1:
            assert tensor_parallel_output, "tensor_parallel_output must be True when tensor_parallel_degree > 1."

        dtype = hidden_states.dtype

        if tensor_parallel_degree > 1 and tensor_parallel_output:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            tensor_parallel_degree = hcg.get_model_parallel_world_size()

        original_shape = hidden_states.shape
        hidden_states_stop_gradient = hidden_states.stop_gradient
        hidden_states = hidden_states.reshape([-1, original_shape[-1]])
        labels = labels.reshape([-1])
        old_log_probs = old_log_probs.reshape([-1])
        advantages = advantages.reshape([-1])
        if kl_loss_coeff > 0:
            ref_log_probs = ref_log_probs.reshape([-1])
        loss_mask = sequence_mask.reshape([-1]).astype("float32")
        divisor = loss_mask.sum()

        n_tokens = hidden_states.shape[0]
        n_classes = weight.shape[0] if transpose_y else weight.shape[1]

        lm_head_weight_cast = weight.cast(dtype)
        lm_head_bias_cast = bias.cast(dtype) if bias is not None else None

        def maybe_transpose(x):
            if transpose_y:
                return x.T
            return x

        # use indices to distinguish the devices.
        if tensor_parallel_degree > 1 and tensor_parallel_output:
            rank = hcg.get_model_parallel_rank()
            per_part_size = vocab_size // tensor_parallel_degree
            indices = paddle.arange(
                rank * per_part_size,
                rank * per_part_size + n_classes,
                dtype=labels.dtype,
            ).unsqueeze(0)
        else:
            indices = paddle.arange(vocab_size, dtype=labels.dtype).unsqueeze(0)

        final_loss = paddle.zeros([1], dtype="float32")
        total_pg_loss = paddle.zeros([1], dtype="float32")
        total_entropy_loss = paddle.zeros([1], dtype="float32")
        total_kl_loss = paddle.zeros([1], dtype="float32")

        grad_lm_head_weight = paddle.zeros_like(weight) if not weight.stop_gradient else None
        grad_lm_head_bias = paddle.zeros_like(bias) if bias is not None and not bias.stop_gradient else None
        grad_hidden_states = paddle.zeros_like(hidden_states) if not hidden_states_stop_gradient else None

        for i in range(0, n_tokens, loop_chunk_size):
            chunk_slice = slice(i, min(i + loop_chunk_size, n_tokens))
            hidden_chunk = hidden_states[chunk_slice]
            labels_chunk = labels[chunk_slice]
            old_log_prob_chunk = old_log_probs[chunk_slice]
            if kl_loss_coeff > 0:
                ref_log_chunk = ref_log_probs[chunk_slice]
            advantages_chunk = advantages[chunk_slice]
            mask_chunk = loss_mask[chunk_slice]

            if fused_linear:
                logits_chunk = PF.fused_linear(
                    hidden_chunk, maybe_transpose(lm_head_weight_cast), bias=lm_head_bias_cast
                )
            else:
                logits_chunk = F.linear(hidden_chunk, maybe_transpose(lm_head_weight_cast), bias=lm_head_bias_cast)
            logits_chunk = logits_chunk.astype("float32")
            logits_chunk = logits_chunk / temperature
            # 计算交叉熵和softmax
            if tensor_parallel_degree > 1 and tensor_parallel_output:
                ce_loss_chunk, softmax_out_chunk = mp_ops._c_softmax_with_cross_entropy(
                    logits_chunk, labels_chunk, group=model_parallel_group, return_softmax=True
                )
            else:
                ce_loss_chunk = F.cross_entropy(logits_chunk, labels_chunk, reduction="none")
                softmax_out_chunk = F.softmax(logits_chunk, axis=-1)

            log_probs_chunk = -ce_loss_chunk.squeeze(axis=-1)
            labels_one_hot = labels_chunk.unsqueeze(1) == indices
            grad_logits_chunk = labels_one_hot.astype("float32") - softmax_out_chunk
            grad_logits_chunk = grad_logits_chunk / temperature

            # [1] pg loss
            ratio_chunk = paddle.exp(log_probs_chunk - old_log_prob_chunk)
            clipped_ratio_chunk = paddle.clip(
                ratio_chunk,
                min=1.0 - clip_range_ratio_low,
                max=1.0 + clip_range_ratio_high,
            )

            pg_loss1_chunk = -advantages_chunk * ratio_chunk
            pg_loss2_chunk = -advantages_chunk * clipped_ratio_chunk

            pg_loss_chunk = paddle.maximum(pg_loss1_chunk, pg_loss2_chunk) * mask_chunk

            total_pg_loss += pg_loss_chunk.sum() * pg_loss_coeff / divisor

            # gradgradgradgrad pg loss
            pg_within_clip_chunk = (
                (ratio_chunk >= 1.0 - clip_range_ratio) & (ratio_chunk <= 1.0 + clip_range_ratio)
            ).astype(dtype)

            d_pg_log_probs_chunk = (
                paddle.where(
                    pg_loss1_chunk >= pg_loss2_chunk,
                    pg_loss1_chunk,
                    pg_loss2_chunk * pg_within_clip_chunk,
                )
                * mask_chunk
                * pg_loss_coeff
                / divisor
            )

            if entropy_coeff > 0:
                # [2] entropy loss
                log_prob_chunk = paddle.log(paddle.clip(softmax_out_chunk, min=1e-12))
                entropy_loss_chunk = -(softmax_out_chunk * log_prob_chunk).sum(axis=-1) * mask_chunk
                # entropy_loss_chunk shape is [bs, seqlen, vocab_size // tensor_parallel_degree], do all_reduce sum here
                if tensor_parallel_degree > 1 and tensor_parallel_output:
                    paddle.distributed.all_reduce(
                        entropy_loss_chunk, op=paddle.distributed.ReduceOp.SUM, group=model_parallel_group
                    )
                total_entropy_loss += entropy_loss_chunk.sum() * entropy_coeff / divisor

                # gradgradgradgrad entropy loss
                # grad_softmax_out_chunk = -(log_prob_chunk + 1) * mask_chunk.unsqueeze(-1) * entropy_coeff / divisor
                # sum_term = (softmax_out_chunk * grad_softmax_out_chunk).sum(axis=-1, keepdim=True)
                # if tensor_parallel_degree > 1 and tensor_parallel_output:
                #     paddle.distributed.all_reduce(
                #         sum_term, op=paddle.distributed.ReduceOp.SUM, group=model_parallel_group
                #     )
                # d_entropy_logits_chunk = softmax_out_chunk * (grad_softmax_out_chunk - sum_term)
                H = entropy_loss_chunk.unsqueeze(-1)
                d_entropy_logits_chunk = (
                    -softmax_out_chunk * (log_prob_chunk + H) * mask_chunk.unsqueeze(-1) * entropy_coeff / divisor
                ) / temperature

            if kl_loss_coeff > 0:
                # [3] kl loss
                delta_chunk = ref_log_chunk - log_probs_chunk
                exp_delta_chunk = paddle.exp(delta_chunk)
                kl_loss_estimate_chunk = exp_delta_chunk - delta_chunk - 1
                kl_loss_clipped_chunk = (
                    paddle.clip(
                        kl_loss_estimate_chunk,
                        min=-clip_range_score,
                        max=clip_range_score,
                    )
                    * mask_chunk
                )
                total_kl_loss += kl_loss_clipped_chunk.sum() * kl_loss_coeff / divisor
                # gradgradgradgrad kl loss
                kl_within_clip_chunk = (
                    (kl_loss_estimate_chunk >= -clip_range_score) & (kl_loss_estimate_chunk <= clip_range_score)
                ).astype(dtype)
                d_kl_log_probs_chunk = (
                    (1 - exp_delta_chunk) * kl_within_clip_chunk * mask_chunk * kl_loss_coeff / divisor
                )

            d_total_logits_chunk = d_pg_log_probs_chunk.unsqueeze(-1) * grad_logits_chunk
            if entropy_coeff > 0:
                d_total_logits_chunk -= d_entropy_logits_chunk
            if kl_loss_coeff > 0:
                d_total_logits_chunk += d_kl_log_probs_chunk.unsqueeze(-1) * grad_logits_chunk

            d_total_logits_chunk = d_total_logits_chunk.cast(dtype)

            if grad_hidden_states is not None:
                grad_hidden_states[chunk_slice] = paddle.matmul(
                    d_total_logits_chunk, lm_head_weight_cast, transpose_y=not transpose_y
                )
            if grad_lm_head_weight is not None:
                if transpose_y:
                    grad_lm_head_weight += paddle.matmul(d_total_logits_chunk, hidden_chunk, transpose_x=True)
                else:
                    grad_lm_head_weight += paddle.matmul(hidden_chunk, d_total_logits_chunk, transpose_x=True)
            if grad_lm_head_bias is not None:
                grad_lm_head_bias += d_total_logits_chunk.astype("float32").sum(axis=0).astype(dtype)

        final_loss += total_pg_loss
        if entropy_coeff > 0:
            final_loss -= total_entropy_loss
        if kl_loss_coeff > 0:
            final_loss += total_kl_loss

        ctx.hidden_states_has_grad = grad_hidden_states is not None
        ctx.lm_head_weight_has_grad = grad_lm_head_weight is not None
        ctx.lm_head_bias_has_grad = grad_lm_head_bias is not None

        if ctx.hidden_states_has_grad:
            if tensor_parallel_degree > 1:
                paddle.distributed.all_reduce(
                    grad_hidden_states, op=paddle.distributed.ReduceOp.SUM, group=model_parallel_group
                )
            grad_hidden_states = grad_hidden_states.reshape(original_shape)

        ctx.save_for_backward(
            *filter(lambda x: x is not None, [grad_hidden_states, grad_lm_head_weight, grad_lm_head_bias])
        )

        return final_loss, total_pg_loss.detach(), total_entropy_loss.detach(), total_kl_loss.detach()

    @staticmethod
    def backward(ctx, grad_output, *args):
        grad_args = ctx.saved_tensor()
        idx = 0
        if ctx.hidden_states_has_grad:
            grad_hidden = grad_args[idx] * grad_output.astype(grad_args[idx].dtype)
            idx += 1
        else:
            grad_hidden = None

        if ctx.lm_head_weight_has_grad:
            grad_lm_head_weight = grad_args[idx] * grad_output.astype(grad_args[idx].dtype)
            idx += 1
        else:
            grad_lm_head_weight = None

        if ctx.lm_head_bias_has_grad:
            grad_lm_head_bias = grad_args[idx] * grad_output.astype(grad_args[idx].dtype)
            idx += 1
        else:
            grad_lm_head_bias = None

        return grad_hidden, grad_lm_head_weight, grad_lm_head_bias


def actor_fused_pg_entropy_kl_loss(
    hidden_states: paddle.Tensor,
    weight: paddle.Tensor,
    input_ids: paddle.Tensor,
    old_log_probs: paddle.Tensor,
    ref_log_probs: paddle.Tensor,
    advantages: paddle.Tensor,
    sequence_mask: paddle.Tensor,
    bias: paddle.Tensor = None,
    transpose_y: bool = False,
    fused_linear: bool = False,
    vocab_size: int = 1024,
    tensor_parallel_degree: int = 1,
    tensor_parallel_output: bool = False,
    pg_loss_coeff: float = 1.0,
    clip_range_ratio: float = 0.2,
    clip_range_ratio_low: float = None,
    clip_range_ratio_high: float = None,
    entropy_coeff: float = 0.001,
    clip_range_score: float = 10.0,
    kl_loss_coeff: float = 0.001,
    response_start: int = 0,
    loop_chunk_size: int = 1024,
    use_actor_fused_loss: bool = True,
    temperature: float = 1.0,
):
    hidden_next = hidden_states[:, response_start:-1, :]
    labels_next = input_ids[:, response_start + 1 :]

    if ref_log_probs is None:
        kl_loss_coeff = 0.0

    if use_actor_fused_loss:
        return ActorFusedLoss.apply(
            hidden_states=hidden_next,
            lm_head_weight=weight,
            lm_head_bias=bias,
            labels=labels_next,
            mask=sequence_mask,
            transpose_y=transpose_y,
            num_embeddings=vocab_size,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_output=tensor_parallel_output,
            fused_linear=fused_linear,
            loop_chunk_size=loop_chunk_size,
            clip_range_ratio=clip_range_ratio,
            clip_range_ratio_low=clip_range_ratio_low,
            clip_range_ratio_high=clip_range_ratio_high,
            clip_range_score=clip_range_score,
            kl_loss_coeff=kl_loss_coeff,
            ignore_index=-100,
            temperature=temperature,
        )

    return ActorFusedPGEntropyKLLoss.apply(
        hidden_states=hidden_next,
        weight=weight,
        bias=bias,
        sequence_mask=sequence_mask,
        labels=labels_next,
        old_log_probs=old_log_probs,
        advantages=advantages,
        ref_log_probs=ref_log_probs,
        transpose_y=transpose_y,
        vocab_size=vocab_size,
        tensor_parallel_degree=tensor_parallel_degree,
        tensor_parallel_output=tensor_parallel_output,
        pg_loss_coeff=pg_loss_coeff,
        clip_range_ratio=clip_range_ratio,  # pg loss
        clip_range_ratio_low=clip_range_ratio_low,
        clip_range_ratio_high=clip_range_ratio_high,
        entropy_coeff=entropy_coeff,  # entropy loss
        clip_range_score=clip_range_score,  # clip loss
        kl_loss_coeff=kl_loss_coeff,  # clip loss
        fused_linear=fused_linear,
        loop_chunk_size=loop_chunk_size,
        temperature=temperature,
    )
