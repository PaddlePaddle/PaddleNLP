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
    """
    Output type of the policy model.

    Attributes:
        loss (Optional[paddle.Tensor]): The loss computed by the model.
        logits (paddle.Tensor): The logits output by the model.
        past_key_values (Optional[Tuple[Tuple[paddle.Tensor]]]): The past key and values which can be used for fast
            auto-regressive decoding.
        hidden_states (Optional[Tuple[paddle.Tensor]]): Hidden states of the model at each layer.
        attentions (Optional[Tuple[paddle.Tensor]]): Attention weights at each layer.
        cross_attentions (Optional[Tuple[paddle.Tensor]]): Cross attention weights at each layer.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    # logits_entropy: Optional[paddle.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class ValueOutput(ModelOutput):
    """
    Output type of the value model.

    Attributes:
        loss (Optional[paddle.Tensor]): The loss computed by the model.
        value (paddle.Tensor): The value output by the model.
        reward (paddle.Tensor): The reward predicted by the model.
        past_key_values (Optional[Tuple[Tuple[paddle.Tensor]]]): The past key and values which can be used for fast
            auto-regressive decoding.
        hidden_states (Optional[Tuple[paddle.Tensor]]): Hidden states of the model at each layer.
        attentions (Optional[Tuple[paddle.Tensor]]): Attention weights at each layer.
        cross_attentions (Optional[Tuple[paddle.Tensor]]): Cross attention weights at each layer.
    """

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
    """
    Generate position IDs from input IDs.

    Args:
        input_ids (paddle.Tensor): Tensor of input IDs. Its shape must be 2D.
        pad_token_id (int, optional): The ID of the padding token. Defaults to 0.

    Returns:
        paddle.Tensor: Tensor of position IDs.
    """
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
    Generate position IDs based on the attention mask. If source is provided, set padding positions in the source to 0.

    When the shape of attention_mask is [B, L, H, W], it represents a causal mask, and the returned position_ids will
    be [B, H, W];
    When the shape of attention_mask is [B, L], it represents a padding mask, and the returned position_ids will be
    [B, L].

    Args:
        attention_mask (Tensor, numpy.ndarray): A Tensor/numpy array with shape [B, L, H, W] or [B, L], where L is the
        sequence length, H is the number of heads, and W is the width (optional). Each element is 0 if the position is
        not masked, otherwise non-zero.
        source (Tensor, numpy.ndarray, optional): A Tensor/numpy array with shape [B, S], where S is the source
        sequence length. Defaults to None.

    Returns:
        Tensor: A Tensor with shape [B, H, W] or [B, L], where H is the number of heads and W is the width (optional).
            Each element is the position ID of the corresponding position. If source is provided, padding positions in
            the source are set to 0.
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
    Generate an attention mask based on the input `input_ids`. Positions where `pad_id` is not `unk_id` or `eos_id`
    will be ignored. If `causal_mask` is `False`, return an attention mask filled with `True`. Otherwise, return a
    triangular mask where each element is less than or equal to the corresponding position element.

    Args:
        input_ids (paddle.Tensor): IDs of the input sequence with shape (batch_size, seq_len).
        pad_id (int): The ID used for padding.
        eos_id (int, optional): The ID used to indicate the end of the sequence. Defaults to None. If set, the
        corresponding positions will be removed from the attention mask.
        unk_id (int, optional): The ID used to indicate unknown tokens. Defaults to None. If set, the corresponding
        positions will be removed from the attention mask.
        past_key_values_length (int, optional): The length of pre-existing key-value pairs. Defaults to 0.
        causal_mask (bool, optional): Whether to use a causal mask. Defaults to True.

    Returns:
        paddle.Tensor: The attention mask with shape (batch_size, 1, seq_len, seq_len + past_len).
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
    """
    Create start and end row indices for each token in the input_ids.

    Args:
        input_ids (paddle.Tensor): Tensor of input IDs with shape (batch_size, seq_len).
        pad_token_id (int, optional): The ID of the padding token. Defaults to 0.

    Returns:
        paddle.Tensor: Tensor of start and end row indices with shape (batch_size, seq_len).
            For each token, the start row index is 0 and the end row index is the sequence length.
            Padding positions are set to 0.
    """
    startend_row_indices = paddle.full(input_ids.shape, input_ids.shape[-1], dtype="int32")
    mask = (input_ids != pad_token_id).cast("int32").cumsum(-1) == 0
    return startend_row_indices.masked_fill(mask, 0)


class RLHFPPOLoss(nn.Layer):
    """
    RLHFPPOLoss is a class that implements the actor loss function.
    """

    def __init__(
        self,
        config,
        clip_range_ratio=0.2,
        clip_range_ratio_low=None,
        clip_range_ratio_high=None,
    ):
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
        Compute the actor's policy loss function.

        Args:
            log_probs (paddle.Tensor): Log probabilities of each action taken by the actor in the current state,
                with shape [B, A], where B is the batch size and A is the number of actions.
            old_log_probs (paddle.Tensor): Log probabilities of each action taken by the actor in the previous time
            step, with the same shape as log_probs.
            advantages (paddle.Tensor): Advantage estimates for each action taken by the actor in the current state,
                with shape [B, A].
            mask (paddle.Tensor): A mask used to filter out completed or invalid trajectories, with shape [B, A].
                If a trajectory is completed (i.e., reward is not None), the mask is 1; otherwise, it is 0.

        Returns:
            paddle.Tensor: PG_loss (paddle.Tensor): The actor's policy loss, with shape [1].
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
    """
    VocabParallelEntropy is a PyLayer that computes the entropy of a vocabulary parallel logits tensor.
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits: paddle.Tensor, tensor_parallel_output=False) -> paddle.Tensor:
        """
        Forward pass of the VocabParallelEntropy layer.

        Args:
            ctx (paddle.autograd.PyLayerContext): The context object to save intermediate results for backward pass.
            vocab_parallel_logits (paddle.Tensor): The logits tensor with shape (batch_size, vocab_size).
            tensor_parallel_output (bool, optional): Whether to enable tensor parallel output. Defaults to False.

        Returns:
            paddle.Tensor: The entropy tensor with shape (batch_size,).
        """
        try:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            tensor_parallel_degree = hcg.get_model_parallel_world_size()
        except Exception:
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
        """
        Backward pass of the VocabParallelEntropy layer.

        Args:
            ctx (paddle.autograd.PyLayerContext): The context object containing saved intermediate results from
            forward pass.
            grad_output (paddle.Tensor): The gradient tensor with shape (batch_size,).

        Returns:
            paddle.Tensor: The gradient tensor with respect to the input logits, with shape (batch_size, vocab_size).
        """
        return grad_output.unsqueeze(axis=-1) * ctx.saved_tensor()[0]


def entropy_from_logits(logits: paddle.Tensor, tensor_parallel_output=False):
    """
    Compute the entropy from logits using the VocabParallelEntropy layer.

    Args:
        logits (paddle.Tensor): The input logits tensor with shape (batch_size, vocab_size).
        tensor_parallel_output (bool, optional): Whether to enable tensor parallel output. Defaults to False.

    Returns:
        paddle.Tensor: The computed entropy tensor with shape (batch_size,).
    """
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
            config,
            clip_range_ratio,
            self.clip_range_ratio_low,
            self.clip_range_ratio_high,
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
        # for varlen flashmask
        pad_size=0,
        raw_input_ids=None,
        indices=None,
        raw_input_shape=None,
        input_ids_rmpad_rolled=None,
    ):
        """
        Compute the loss function, which consists of two parts: soft target loss and PPO loss.
        If labels are provided, the soft target loss is computed; otherwise, the PPO loss is computed.

        Args:
            logits (paddle.Tensor or List[paddle.Tensor]): The predicted logits, which can be a single tensor or a
                                                        list of tensors. If it is a single tensor, it represents the
                                                        corresponding output logits; if it is a list,
                                                        it represents the logits at each time step.
            labels (paddle.Tensor, optional): The ground truth labels, with the same shape as logits. Defaults to None.
            input_ids (paddle.Tensor, optional): The IDs of the input sequence, with shape (batch_size, max_len).
                                                Defaults to None.
            reward_advantages (paddle.Tensor, optional): The reward advantages, with shape (batch_size, max_len).
                                                        Defaults to None.
            sequence_mask (paddle.Tensor, optional): The sequence mask, with shape (batch_size, max_len). Defaults to
                                                    None.
            ref_log_probs (paddle.Tensor, optional): The reference log probabilities, used for calculating the KL
                                                    divergence. Defaults to None.
            response_start (int, optional): The starting index of the response sequence. Defaults to 0.

        Returns:
            paddle.Tensor: The computed loss, which is the soft target loss if labels are provided; otherwise, it is
            the PPO loss.
        """
        use_remove_padding = indices is not None
        if not self.config.use_fused_head_and_loss_fn:
            logits = logits if isinstance(logits, paddle.Tensor) else logits[0]
            if self.use_fp32_compute and logits.dtype != paddle.float32:
                logits = logits.cast(paddle.float32)

            if self.temperature > 0.0:
                # use inplace method to save gpu memory
                logits.scale_(1.0 / self.temperature)

        else:
            hidden_states, weight, bias, transpose_y = logits
            if use_remove_padding:
                input_ids = raw_input_ids
                if pad_size > 0:
                    hidden_states = hidden_states[:, :-pad_size]

                from ..utils.bert_padding import pad_input

                hidden_states = pad_input(
                    hidden_states.squeeze(0), indices, batch=raw_input_shape[0], seqlen=raw_input_shape[1]
                ).contiguous()

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
                use_actor_fused_loss=self.entropy_coeff <= 0,  # currently only support kunbo's fused head loss
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
            if use_remove_padding:
                from ..utils.bert_padding import pad_input

                if self.config.tensor_parallel_degree > 1 and self.config.tensor_parallel_output:
                    log_probs = (
                        -ParallelCrossEntropy()(logits.astype("float32"), input_ids_rmpad_rolled)
                        .squeeze(axis=-1)
                        .astype(logits.dtype)
                    )
                else:
                    log_probs = gather_log_probabilities(logits, input_ids_rmpad_rolled)

                if pad_size > 0:
                    log_probs = log_probs[:, :-pad_size]
                log_probs = pad_input(
                    log_probs.squeeze(0).unsqueeze(-1), indices, batch=raw_input_shape[0], seqlen=raw_input_shape[1]
                ).squeeze(-1)
                log_probs = log_probs[:, response_start:-1].contiguous()
            else:
                if self.config.tensor_parallel_degree > 1 and self.config.tensor_parallel_output:
                    log_probs = (
                        -ParallelCrossEntropy()(
                            logits[:, response_start:-1].astype("float32"), input_ids[:, response_start + 1 :]
                        )
                        .squeeze(axis=-1)
                        .astype(logits.dtype)
                    )
                else:
                    log_probs = gather_log_probabilities(
                        logits[:, response_start:-1], input_ids[:, response_start + 1 :]
                    )

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
            if use_remove_padding:
                entropy_loss_rmpad = entropy_from_logits(
                    logits.cast("float32"), self.config.tensor_parallel_output
                ).cast(logits.dtype)
                if pad_size > 0:
                    entropy_loss_rmpad = entropy_loss_rmpad[:, :-pad_size]
                entropy_loss = pad_input(
                    entropy_loss_rmpad.squeeze(0).unsqueeze(-1),
                    indices,
                    batch=raw_input_shape[0],
                    seqlen=raw_input_shape[1],
                ).squeeze(-1)
                entropy_loss_raw = entropy_loss[:, response_start:-1].contiguous()
            else:
                entropy_loss_raw = entropy_from_logits(
                    logits[:, response_start:-1], self.config.tensor_parallel_output
                )
            entropy_loss = paddle.sum(entropy_loss_raw * sequence_mask) / sequence_mask.sum()
            self.info_buffer["entropy_loss"] = entropy_loss.detach()
            loss -= self.entropy_coeff * entropy_loss
        else:
            self.info_buffer["entropy_loss"] = paddle.to_tensor([0.0])

        return loss


@merge_fwd_labels
class RLHFValueLoss(nn.Layer):
    """
    RLHFValueLoss is a loss function for value estimation in reinforcement learning.
    """

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
        Compute the loss function for reward values.

        If the length of the input reward values is the same as that of the old reward values,
        the given sequence mask is used to determine the valid length.
        If the length of the input reward values is one less than that of the old reward values,
        the last element is considered as padding consistent with the input IDs and is removed.
        Otherwise, the reward values only have the tgt length.

        Args:
            reward_values (paddle.Tensor, list of paddle.Tensor or None, optional):
                Reward values, which can be a single tensor or multiple tensors in a list. Defaults to None.
            old_reward_values (paddle.Tensor, optional): Old reward values.
            reward_returns (paddle.Tensor, optional): Reward returns.
            sequence_mask (paddle.Tensor, optional): Sequence mask.

        Returns:
            paddle.Tensor, float32: The loss function for reward values.

        Raises:
            ValueError: Raised when the lengths of reward values and old reward values do not match.
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
                ratio_chunk,
                min=1.0 - clip_range_ratio_low,
                max=1.0 + clip_range_ratio_high,
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
                    d_loss_d_logits_chunk,
                    lm_head_weight_cast,
                    transpose_y=not transpose_y,
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

    def __init__(
        self,
        config,
        clip_range_ratio=0.2,
        clip_range_ratio_low=None,
        clip_range_ratio_high=None,
    ):
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
        """
        Forward pass of the ActorFusedPGEntropyKLLoss layer.

        Args:
            ctx (paddle.autograd.PyLayerContext): The context object to save intermediate results for backward pass.
            hidden_states (paddle.Tensor): The hidden states of the model.
            weight (paddle.Tensor): The weights of the linear layer.
            bias (paddle.Tensor): The biases of the linear layer.
            sequence_mask (paddle.Tensor): The sequence mask to filter out padding tokens.
            labels (paddle.Tensor): The ground truth labels.
            old_log_probs (paddle.Tensor): The old log probabilities from the previous time step.
            advantages (paddle.Tensor): The advantage estimates.
            ref_log_probs (paddle.Tensor): The reference log probabilities for KL divergence calculation.
            transpose_y (bool): Whether to transpose the weights.
            vocab_size (int): The size of the vocabulary.
            tensor_parallel_degree (int): The degree of tensor parallelism.
            tensor_parallel_output (bool): Whether to enable tensor parallel output.
            pg_loss_coeff (float): The coefficient for policy gradient loss.
            clip_range_ratio (float): The clipping range for policy gradient loss.
            clip_range_ratio_low (float): The lower bound for clipping ratio.
            clip_range_ratio_high (float): The upper bound for clipping ratio.
            entropy_coeff (float): The coefficient for entropy loss.
            clip_range_score (float): The clipping range for KL divergence loss.
            kl_loss_coeff (float): The coefficient for KL divergence loss.
            fused_linear (bool): Whether to use fused linear operation.
            loop_chunk_size (int): The chunk size for loop processing.
            temperature (float): The temperature scaling factor.

        Returns:
            tuple: A tuple containing the final loss, total policy gradient loss, total entropy loss, and total KL
            divergence loss.
        """
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
                    hidden_chunk,
                    maybe_transpose(lm_head_weight_cast),
                    bias=lm_head_bias_cast,
                )
            else:
                logits_chunk = F.linear(
                    hidden_chunk,
                    maybe_transpose(lm_head_weight_cast),
                    bias=lm_head_bias_cast,
                )
            logits_chunk = logits_chunk.astype("float32")
            logits_chunk = logits_chunk / temperature
            # 计算交叉熵和softmax
            if tensor_parallel_degree > 1 and tensor_parallel_output:
                ce_loss_chunk, softmax_out_chunk = mp_ops._c_softmax_with_cross_entropy(
                    logits_chunk,
                    labels_chunk,
                    group=model_parallel_group,
                    return_softmax=True,
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
                # entropy_loss_chunk shape is [bs, seqlen, vocab_size // tensor_parallel_degree], do all_reduce sum
                # here
                if tensor_parallel_degree > 1 and tensor_parallel_output:
                    paddle.distributed.all_reduce(
                        entropy_loss_chunk,
                        op=paddle.distributed.ReduceOp.SUM,
                        group=model_parallel_group,
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
                    d_total_logits_chunk,
                    lm_head_weight_cast,
                    transpose_y=not transpose_y,
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
                    grad_hidden_states,
                    op=paddle.distributed.ReduceOp.SUM,
                    group=model_parallel_group,
                )
            grad_hidden_states = grad_hidden_states.reshape(original_shape)

        ctx.save_for_backward(
            *filter(
                lambda x: x is not None,
                [grad_hidden_states, grad_lm_head_weight, grad_lm_head_bias],
            )
        )

        return (
            final_loss,
            total_pg_loss.detach(),
            total_entropy_loss.detach(),
            total_kl_loss.detach(),
        )

    @staticmethod
    def backward(ctx, grad_output, *args):
        """
        Backward pass of the ActorFusedPGEntropyKLLoss layer.

        Args:
            ctx (paddle.autograd.PyLayerContext): The context object containing saved intermediate results from the
                                                    forward pass.
            grad_output (paddle.Tensor): The gradient of the output loss with respect to the input.
            *args: Additional arguments passed from the forward pass.

        Returns:
            tuple: A tuple containing the gradients with respect to the hidden states, lm_head_weight, and
            lm_head_bias.
        """
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
    clip_range_ratio_low: Optional[float] = None,
    clip_range_ratio_high: Optional[float] = None,
    entropy_coeff: float = 0.001,
    clip_range_score: float = 10.0,
    kl_loss_coeff: float = 0.001,
    response_start: int = 0,
    loop_chunk_size: int = 1024,
    use_actor_fused_loss: bool = True,
    temperature: float = 1.0,
):
    """
    Compute the combined loss of policy gradient, entropy, and KL divergence for the actor network.

    Args:
        hidden_states (paddle.Tensor): The hidden states of the model.
        weight (paddle.Tensor): The weights of the linear layer.
        input_ids (paddle.Tensor): The IDs of the input sequence.
        old_log_probs (paddle.Tensor): The old log probabilities from the previous time step.
        ref_log_probs (paddle.Tensor): The reference log probabilities for KL divergence calculation.
        advantages (paddle.Tensor): The advantage estimates.
        sequence_mask (paddle.Tensor): The sequence mask to filter out padding tokens.
        bias (paddle.Tensor, optional): The biases of the linear layer. Defaults to None.
        transpose_y (bool, optional): Whether to transpose the weights. Defaults to False.
        fused_linear (bool, optional): Whether to use fused linear operation. Defaults to False.
        vocab_size (int, optional): The size of the vocabulary. Defaults to 1024.
        tensor_parallel_degree (int, optional): The degree of tensor parallelism. Defaults to 1.
        tensor_parallel_output (bool, optional): Whether to enable tensor parallel output. Defaults to False.
        pg_loss_coeff (float, optional): The coefficient for policy gradient loss. Defaults to 1.0.
        clip_range_ratio (float, optional): The clipping range for policy gradient loss. Defaults to 0.2.
        clip_range_ratio_low (float, optional): The lower bound for clipping ratio. Defaults to None.
        clip_range_ratio_high (float, optional): The upper bound for clipping ratio. Defaults to None.
        entropy_coeff (float, optional): The coefficient for entropy loss. Defaults to 0.001.
        clip_range_score (float, optional): The clipping range for KL divergence loss. Defaults to 10.0.
        kl_loss_coeff (float, optional): The coefficient for KL divergence loss. Defaults to 0.001.
        response_start (int, optional): The starting index of the response sequence. Defaults to 0.
        loop_chunk_size (int, optional): The chunk size for loop processing. Defaults to 1024.
        use_actor_fused_loss (bool, optional): Whether to use the fused loss for the actor network. Defaults to True.
        temperature (float, optional): The temperature scaling factor. Defaults to 1.0.

    Returns:
        tuple: A tuple containing the computed losses.
    """
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
