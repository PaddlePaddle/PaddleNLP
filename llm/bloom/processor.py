# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
from abc import ABC

import paddle


class LogitsProcessorList(list):
    def __call__(self, input_ids, logits, **kwargs):
        for processor in self:
            processor_args = inspect.signature(processor.__call__).parameters
            if len(processor_args) > 2:
                assert all(
                    arg in kwargs for arg in list(processor_args.keys())[2:]
                ), f"The parameters don't match for {processor.__class__}"
                logits = processor(input_ids, logits, **kwargs)
            else:
                logits = processor(input_ids, logits)
        return logits


class LogitsProcessor(ABC):
    """
    Abstract base class for all logit processors that can be applied during
    generation.
    """

    def __call__(self, input_ids, logits):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. " "Only classes inheriting this class can be called."
        )


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    Enforcing a min-length by setting EOS probability to 0.
    Args:
        min_length (int): The minimum length of generation sequence.
        eos_token_id (int): The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length, eos_token_id):
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, logits):
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            logits[:, self.eos_token_id] = -float("inf")
        return logits


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    Enforcing an exponential penalty on repeated sequences.
    Args:
        repetition_penalty (float):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids, logits):
        score = paddle.index_sample(logits, input_ids)
        score = paddle.where(score < 0, score * self.penalty, score / self.penalty)
        input_ids = input_ids + paddle.arange(logits.shape[0]).unsqueeze(-1) * logits.shape[-1]
        outputs = paddle.scatter(logits.flatten(), input_ids.flatten(), score.flatten()).reshape(logits.shape)
        return outputs


class HammingDiversityLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces diverse beam search. Note that this logits
    processor is only effective for `group_beam_search`. See
    `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
    Args:
        diversity_rate (float): This value is subtracted from a beam's score if
            it generates a token same as any beam from other group at a particular
            time.
        num_beams (int): Number of beams used for group beam search.
        num_beam_groups (int): Number of groups to divide `num_beams` into in order
            to ensure diversity among different groups of beams.
    """

    def __init__(self, diversity_rate, num_beams, num_beam_groups):
        if not isinstance(diversity_rate, float) or (not diversity_rate > 0.0):
            raise ValueError("`diversity_rate` should be a float strictly larger than 0.")
        self._diversity_rate = diversity_rate
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(self, input_ids, scores, current_tokens, beam_group_idx):
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]

        if group_start_idx == 0:
            return scores

        for batch_idx in range(batch_size):
            previous_group_tokens = current_tokens[
                batch_idx * self._num_beams : batch_idx * self._num_beams + group_start_idx
            ]
            token_frequency = paddle.bincount(previous_group_tokens, minlength=vocab_size)
            scores[batch_idx * group_size : (batch_idx + 1) * group_size] -= self._diversity_rate * token_frequency

        return scores


class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces the first generated token to be the selected `forced_bos_token`.
    Args:
        forced_bos_token_id (:obj:`int`):
            The id of the token to be generated as the first token.
    """

    def __init__(self, forced_bos_token_id):
        self.forced_bos_token_id = forced_bos_token_id

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len == 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.forced_bos_token_id]] = -float("inf")
            scores[:, self.forced_bos_token_id] = 0
        return scores


class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces the last generated token to be the selected `forced_eos_token`.
    Args:
        max_length (int): The maximum length of the sequence to be generated.
        forced_eos_token_id (int): The id of the token to be generated as the last token.
    """

    def __init__(self, max_length, forced_eos_token_id):
        self.max_length = max_length
        self.forced_eos_token_id = forced_eos_token_id

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len == self.max_length - 1:
            num_tokens = scores.shape[1]
            scores[
                :, [i for i in range(num_tokens) if i != self.forced_eos_token_id]
            ] = -1e9  # TODO change back to -inf after paddle.topk is fixed
            scores[:, self.forced_eos_token_id] = 0
        return scores
