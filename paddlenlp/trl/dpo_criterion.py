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

import copy

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.meta_parallel import ParallelCrossEntropy

from paddlenlp.transformers import (
    AllGatherVarlenOp,
    fused_head_and_loss_fn,
    parallel_linear,
    parallel_matmul,
    sequence_parallel_sparse_mask_labels,
)
from paddlenlp.transformers.model_outputs import CausalLMOutputWithPast
from paddlenlp.utils import infohub
from paddlenlp.utils.tools import get_env_device


class DPOCriterion(nn.Layer):
    """DPO Criterion"""

    def __init__(self, config, dpo_config=None, use_infohub=False, ignore_eos_token=False):
        super(DPOCriterion, self).__init__()
        self.config = config
        if dpo_config is None:
            if getattr(self.config, "dpo_config", None) is None:
                raise ValueError("DPO Criterion requires model_config.dpo_config.")
            self.dpo_config = copy.deepcopy(config.dpo_config)
        else:
            self.dpo_config = dpo_config
        if self.config.tensor_parallel_output and self.config.tensor_parallel_degree > 1:
            self.logprobs = ParallelCrossEntropy()
        else:
            self.logprobs = nn.CrossEntropyLoss(reduction="none")
        self.use_infohub = use_infohub
        self.ignore_eos_token = ignore_eos_token

    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps):
        """DPO Loss"""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.dpo_config.loss_type == "sigmoid":
            loss = (
                -F.log_sigmoid(self.dpo_config.beta * logits) * (1 - self.dpo_config.label_smoothing)
                - F.log_sigmoid(-self.dpo_config.beta * logits) * self.dpo_config.label_smoothing
            )
        elif self.dpo_config.loss_type == "hinge":
            loss = F.relu(1 - self.dpo_config.beta * logits)
        elif self.dpo_config.loss_type == "simpo":
            gamma_logratios = self.dpo_config.simpo_gamma / self.dpo_config.beta
            logits -= gamma_logratios
            loss = (
                -F.log_sigmoid(self.dpo_config.beta * logits) * (1 - self.dpo_config.label_smoothing)
                - F.log_sigmoid(-self.dpo_config.beta * logits) * self.dpo_config.label_smoothing
            )
        elif self.dpo_config.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter
            # for the IPO loss, denoted by tau in the paper.
            loss = (logits - 1 / (2 * self.dpo_config.beta)) ** 2
        elif self.dpo_config.loss_type == "dpop":
            loss = -F.log_sigmoid(self.dpo_config.beta * logits)
            positive_reg = reference_chosen_logps - policy_chosen_logps
            loss += self.dpo_config.dpop_lambda * paddle.clip(positive_reg, min=0)
        elif self.dpo_config.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clip(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clip(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is
            # estimated using the rejected (chosen) half.
            loss = paddle.concat(
                (
                    1 - F.sigmoid(self.dpo_config.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.dpo_config.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        elif self.dpo_config.loss_type == "sppo_hard":
            # In the paper (https://arxiv.org/pdf/2405.00675), SPPO employs a soft probability approach,
            # estimated using the PairRM score. The probability calculation is conducted outside of
            # the trainer class. The version described here is the hard probability version, where P
            # in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            loss = (a - 0.5 / self.dpo_config.beta) ** 2 + (b + 0.5 / self.dpo_config.beta) ** 2
        elif self.dpo_config.loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.dpo_config.beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.dpo_config.beta
            loss = (
                -F.log_sigmoid(chosen_rewards)
                - 0.5 * F.log_sigmoid(-chosen_rewards)
                - 0.5 * F.log_sigmoid(-rejected_rewards)
            )
        elif self.dpo_config.loss_type == "or":
            # Derived from Eqs. (4) and (7) from https://arxiv.org/abs/2403.07691 by using
            # log identities and exp(log(P(y|x)) = P(y|x)
            log_odds = (policy_chosen_logps - policy_rejected_logps) - (
                paddle.log1p(-paddle.exp(policy_chosen_logps)) - paddle.log1p(-paddle.exp(policy_rejected_logps))
            )
            loss = -F.log_sigmoid(log_odds)
        else:
            raise ValueError(
                f"Unknown loss type: {self.dpo_config.loss_type}. "
                "Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair',"
                "'sppo_hard', 'nca_pair', 'dpop', 'or', 'simpo']"
            )
        return loss.mean() * self.dpo_config.pref_loss_ratio

    def dpo_logps(
        self,
        logits,
        chosen_labels,
        rejected_labels,
        response_indexs,
        average_log_prob=False,
    ):
        """DPO logprobs"""
        use_fused_head_and_loss_fn = getattr(self.config, "use_fused_head_and_loss_fn", False)
        use_sparse_head_and_loss_fn = getattr(self.config, "use_sparse_head_and_loss_fn", False)
        chunk_size = getattr(self.config, "chunk_size", 1024)
        labels = chosen_labels + rejected_labels
        if use_fused_head_and_loss_fn:
            hidden_states, weight, bias, transpose_y = logits
        elif use_sparse_head_and_loss_fn:
            hidden_states, weight, bias = logits

        if use_sparse_head_and_loss_fn:
            if self.config.tensor_parallel_degree > 1 and self.config.sequence_parallel:
                labels, sparse_tgt_idx = sequence_parallel_sparse_mask_labels(labels, 0)

                hidden_states = paddle.take_along_axis(hidden_states, sparse_tgt_idx, axis=0)
                hidden_states = AllGatherVarlenOp.apply(hidden_states)
            else:
                labels = labels.flatten()
                sparse_tgt_idx = paddle.nonzero(labels != 0).flatten()
                labels = paddle.take_along_axis(labels, sparse_tgt_idx, axis=0)

                hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])
                hidden_states = paddle.take_along_axis(hidden_states, sparse_tgt_idx.unsqueeze(-1), axis=0)

        if use_fused_head_and_loss_fn:
            per_token_logps = -fused_head_and_loss_fn(
                hidden_states,
                weight,
                bias,
                labels,
                None,
                transpose_y,
                self.config.vocab_size,
                self.config.tensor_parallel_degree,
                self.config.tensor_parallel_output,
                False,  # fused_linear
                chunk_size,
                return_token_loss=True,
                ignore_index=0,
            )
        elif use_sparse_head_and_loss_fn:
            if bias is None:
                logits = parallel_matmul(hidden_states, weight, self.config.tensor_parallel_output)
            else:
                logits = parallel_linear(hidden_states, weight, bias, self.config.tensor_parallel_output)
            logits = logits.astype("float32")
            per_token_logps = -self.logprobs(logits, labels)
        else:
            if isinstance(logits, tuple):
                logits = logits[0]
            elif isinstance(logits, CausalLMOutputWithPast):
                logits = logits.logits
            logits = logits.astype("float32")
            if logits.shape[:-1] != labels.shape:
                raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")
            # bs, seq
            per_token_logps = -self.logprobs(logits, labels.unsqueeze(2)).squeeze(2)

        if len(response_indexs.shape) == 3:
            response_indexs = response_indexs[0]
        if use_sparse_head_and_loss_fn:
            chosen_logps = paddle.stack(
                [(per_token_logps[response_index[1] : response_index[2]]).sum() for response_index in response_indexs],
                axis=0,
            )
            rejected_logps = paddle.stack(
                [(per_token_logps[response_index[2] : response_index[3]]).sum() for response_index in response_indexs],
                axis=0,
            )
        else:
            if get_env_device() == "npu":
                chosen_list = []
                for response_index in response_indexs:
                    begin = response_index[1]
                    end = response_index[2]
                    one_data = paddle.ones_like(per_token_logps[0])
                    mask_data = paddle.zeros_like(per_token_logps[0])
                    paddle.assign(one_data._slice(begin, end), mask_data._slice(begin, end))
                    chosen_list.append((per_token_logps[0] * mask_data).sum())
                chosen_logps = paddle.stack(chosen_list, axis=0)
                rejected_list = []
                for response_index in response_indexs:
                    begin = response_index[2]
                    if self.ignore_eos_token:
                        begin += 1
                    end = response_index[3]
                    one_data = paddle.ones_like(per_token_logps[0])
                    mask_data = paddle.zeros_like(per_token_logps[0])
                    paddle.assign(one_data._slice(begin, end), mask_data._slice(begin, end))
                    rejected_list.append((per_token_logps[0] * mask_data).sum())
                rejected_logps = paddle.stack(rejected_list, axis=0)
            else:
                chosen_logps = paddle.stack(
                    [
                        (per_token_logps[response_index[0]][response_index[1] : response_index[2]]).sum()
                        for response_index in response_indexs
                    ],
                    axis=0,
                )
                if self.ignore_eos_token:
                    rejected_logps = paddle.stack(
                        [
                            (per_token_logps[response_index[0]][response_index[2] + 1 : response_index[3]]).sum()
                            for response_index in response_indexs
                        ],
                        axis=0,
                    )
                else:
                    rejected_logps = paddle.stack(
                        [
                            (per_token_logps[response_index[0]][response_index[2] : response_index[3]]).sum()
                            for response_index in response_indexs
                        ],
                        axis=0,
                    )
        sft_loss = -chosen_logps.sum() / (chosen_labels != 0).sum()
        if average_log_prob:
            chosen_response_length = response_indexs[:, 2] - response_indexs[:, 1]
            rejected_response_length = response_indexs[:, 3] - response_indexs[:, 2]
            chosen_logps /= chosen_response_length.astype("float32")
            rejected_logps /= rejected_response_length.astype("float32")
        return chosen_logps, rejected_logps, sft_loss * self.dpo_config.sft_loss_ratio

    def forward(
        self,
        logits,
        labels,
    ):
        """Forward"""
        chosen_labels, rejected_labels, response_indexs, reference_chosen_logps, reference_rejected_logps = labels
        if self.dpo_config.loss_type in ["ipo", "or", "simpo"]:
            average_log_prob = True
        else:
            average_log_prob = False
        if reference_chosen_logps is None or reference_rejected_logps is None:
            reference_chosen_logps, reference_rejected_logps, sft_loss = self.dpo_logps(
                logits, chosen_labels, rejected_labels, response_indexs, average_log_prob
            )
            if self.use_infohub:
                infohub.reference_chosen_logps.append(reference_chosen_logps)
                infohub.reference_rejected_logps.append(reference_rejected_logps)
                # pipeline mode requires return loss when self._compute_loss is True
                return paddle.zeros([1])
            else:
                return reference_chosen_logps, reference_rejected_logps
        policy_chosen_logps, policy_rejected_logps, sft_loss = self.dpo_logps(
            logits, chosen_labels, rejected_labels, response_indexs, average_log_prob
        )
        dpo_loss = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
        )
        loss = dpo_loss + sft_loss
        if self.use_infohub:
            infohub.policy_chosen_logps.append(policy_chosen_logps)
            infohub.policy_rejected_logps.append(policy_rejected_logps)
            infohub.sft_loss.append(sft_loss)
            infohub.dpo_loss.append(dpo_loss)
            return loss
        else:
            return policy_chosen_logps, policy_rejected_logps, sft_loss, dpo_loss, loss
