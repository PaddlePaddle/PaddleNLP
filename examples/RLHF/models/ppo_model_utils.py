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
import paddle.nn as nn
import paddle.nn.functional as F

# use LlamaPretrainingCriterion as common PretrainingCriterion
from paddlenlp.transformers import LlamaPretrainingCriterion as PretrainingCriterion
from paddlenlp.transformers.model_outputs import ModelOutput


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


def create_loss(loss_cls, config, extra_args):
    # forward(self, predict, label1, label2, ...)
    loss_arg_names = list(inspect.signature(loss_cls.__init__).parameters.keys())[2:]
    if isinstance(extra_args, dict):
        loss_kwargs = dict([(name, extra_args[name]) for name in loss_arg_names if name in extra_args])
    else:
        # create from TrainingArguments
        loss_kwargs = dict([(name, getattr(extra_args, name)) for name in loss_arg_names if hasattr(extra_args, name)])
    return loss_cls(config, **loss_kwargs)


def gather_log_probabilities(logits: paddle.Tensor, labels: paddle.Tensor) -> paddle.Tensor:
    """Gather log probabilities of the given labels from the logits."""
    log_probs = F.log_softmax(logits, axis=-1)
    log_probs_labels = paddle.take_along_axis(log_probs, axis=-1, indices=labels.unsqueeze(axis=-1))
    return log_probs_labels.squeeze(axis=-1)


class RLHFPPOLoss(nn.Layer):
    def __init__(self, config, clip_range_ratio=0.2):
        super().__init__()
        self.clip_range_ratio = clip_range_ratio
        self.config = config

    def actor_loss_fn(
        self, log_probs: paddle.Tensor, old_log_probs: paddle.Tensor, advantages: paddle.Tensor, mask: paddle.Tensor
    ) -> paddle.Tensor:
        # policy gradient loss
        ratio = paddle.exp(log_probs - old_log_probs)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * paddle.clip(
            ratio,
            1.0 - self.clip_range_ratio,
            1.0 + self.clip_range_ratio,
        )
        return paddle.sum(paddle.maximum(pg_loss1, pg_loss2) * mask) / mask.sum()

    def forward(self, logits, input_ids, old_log_probs, reward_advantages, sequence_mask, start=None):
        # tgt_mask or sequence_mask according to length

        # When used in pipe mode, batches among accumulation steps should be paded.
        # Hard to pad acorss batches, think in some cases one batch might have the
        # longest prompt+target length but the shortest target lengh, which might
        # cause mismatch between inputs with prompt+target length and labels with
        # target length. NOTE: Thus, we might make all fields be prompt+target
        # length rather rather than target and company an extra start input.
        # However trick can be used in pipe_model._prepare_pipeline_inputs_func,
        # label fields with target length such as old_log_probs/reward_advantages/sequence_mask
        # not need to join comm and thus there is no need to keep same shape among
        # batches of accumulation steps, they just need to pad as prompt+target
        # fields such as input_ids.
        log_probs = gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
        if start is not None:
            old_log_probs = old_log_probs[:, start:]
            sequence_mask = sequence_mask[:, start:]
        log_probs = log_probs[:, -old_log_probs.shape[1] :]
        actor_loss = self.actor_loss_fn(
            log_probs,
            old_log_probs,
            reward_advantages,
            sequence_mask,
        )
        return actor_loss


@merge_fwd_labels
class RLHFPPOMixedLoss(nn.Layer):
    """provide two losses, one for PPO loss, the other for SFT loss."""

    def __init__(self, config, ptx_coeff=16, clip_range_ratio=0.2):
        super(RLHFPPOMixedLoss, self).__init__()
        self.ptx_coeff = ptx_coeff
        self.ppo_criterion = RLHFPPOLoss(config, clip_range_ratio)
        self.sft_criterion = PretrainingCriterion(config)

    def forward(self, logits, labels, input_ids, old_log_probs, reward_advantages, sequence_mask):
        # def forward(self, logits, label_info):
        #     labels, input_ids, old_log_probs, reward_advantages, sequence_mask = label_info

        logits = logits if isinstance(logits, paddle.Tensor) else logits[0]
        loss = None
        # sft, pt loss
        if labels is not None:
            loss = self.ptx_coeff * self.sft_criterion(logits, labels)
        # ppo loss
        if reward_advantages is not None:
            loss = self.ppo_criterion(logits, input_ids, old_log_probs, reward_advantages, sequence_mask)

        return loss


@merge_fwd_labels
class RLHFValueLoss(nn.Layer):
    def __init__(self, config, clip_range_value=5.0):
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
            paddle.maximum(values, old_values - self.clip_range_value), old_values + self.clip_range_value
        )
        vf_loss1 = paddle.square(values - returns)
        vf_loss2 = paddle.square(values_clipped - returns)
        return 0.5 * paddle.sum(paddle.maximum(vf_loss1, vf_loss2) * mask) / mask.sum()

    def forward(
        self,
        reward_values,
        old_reward_values,
        reward_returns,
        sequence_mask,
        start=None,
        # label_info,
    ):
        # old_reward_values, reward_returns, sequence_mask = label_info
        reward_values = reward_values if isinstance(reward_values, paddle.Tensor) else reward_values[0]
        # if start is not None:
        #     old_reward_values = old_reward_values[:, start:]
        #     sequence_mask = sequence_mask[:, start:]
        reward_values = reward_values.squeeze(axis=-1)[:, :-1]
        reward_critic_loss = self.critic_loss_fn(
            reward_values[:, -old_reward_values.shape[1] :],
            old_reward_values,
            reward_returns,
            sequence_mask,
        )

        return reward_critic_loss
