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

import inspect
from abc import ABC
from collections import OrderedDict
from typing import List, Union

import paddle
from paddle.nn.layer.layers import in_declarative_mode


class LogitsProcessor(ABC):
    """
    Abstract base class for all logit processors that can be applied during
    generation.
    """

    def __call__(self, input_ids: paddle.Tensor, logits: paddle.Tensor):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. " "Only classes inheriting this class can be called."
        )


class LogitsProcessorList:
    """use ordered dict to store processors"""

    def __init__(self, processors: List[LogitsProcessor] = None) -> None:
        self._processors = OrderedDict()
        processors = processors or []
        for processor in processors:
            self.append(processor)

    def __call__(self, input_ids: paddle.Tensor, logits: paddle.Tensor, **kwargs):
        for processor in self._processors.values():
            processor_args = inspect.signature(processor.__call__).parameters
            if len(processor_args) > 2:
                assert all(
                    arg in kwargs for arg in list(processor_args.keys())[2:]
                ), f"The parameters don't match for {processor.__class__}"
                logits = processor(input_ids, logits, **kwargs)
            else:
                logits = processor(input_ids, logits)
        return logits

    def append(self, processor: LogitsProcessor):
        self._processors[len(self._processors)] = processor


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    Enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (int): The minimum length of generation sequence.
        eos_token_id (int): The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length: int, eos_token_id: Union[int, List[int]]):
        if min_length < 0 and not in_declarative_mode():
            raise ValueError("`min_length` should be a positive integer, but get {}".format(min_length))

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError("`eos_token_id` should be a positive integer, but get {}".format(eos_token_id))

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: paddle.Tensor, logits: paddle.Tensor):
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            logits[:, self.eos_token_id] = paddle.finfo(logits.dtype).min
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
        if not (penalty > 0) and not in_declarative_mode():
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids: paddle.Tensor, logits: paddle.Tensor):
        score = paddle.index_sample(logits, input_ids)
        score = paddle.where(score < 0, score * self.penalty, score / self.penalty)
        input_ids = input_ids + paddle.arange(logits.shape[0], dtype="int64").unsqueeze(-1) * logits.shape[-1]
        outputs = paddle.scatter(logits.flatten(), input_ids.flatten(), score.flatten()).reshape(logits.shape)
        return outputs


def _get_ngrams(ngram_size: int, prev_input_ids: paddle.Tensor, num_hypos: int):
    """
    Assume ngram_size=2 and prev_input_ids=tensor([[40, 2883, 2712, 4346]]). The output of generated ngrams look like
    this {(40,): [2883], (2883,): [2712], (2712,): [4346]}.

    Args:
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        prev_input_ids (`paddle.Tensor`):
           Generated token ids for the current hypothesis.
        num_hypos (`int`):
            The number of hypotheses for which n-grams need to be generated.

    Returns:
        generated_ngrams (`dict`):
            Dictionary of generated ngrams.
    """
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    """
    Determines the banned tokens for the current hypothesis based on previously generated n-grams.

    Args:
        banned_ngrams (`dict`):
            A dictionary containing previously generated n-grams for each hypothesis.
        prev_input_ids (`paddle.Tensor`):
            Generated token ids for the current hypothesis.
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        cur_len (`int`):
            The current length of the token sequences for which the n-grams are being checked.

    Returns:
        List of tokens that are banned.
    """
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(ngram_size: int, prev_input_ids: paddle.Tensor, num_hypos: int, cur_len: int):
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]

    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)

    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).
    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor):
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            if len(banned_tokens) == 0:
                continue
            scores[i, banned_tokens] = paddle.finfo(scores.dtype).min

        return scores


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

    def __init__(self, diversity_rate: float, num_beams: int, num_beam_groups: int):
        if not isinstance(diversity_rate, float) or (not diversity_rate > 0.0):
            raise ValueError("`diversity_rate` should be a float strictly larger than 0.")
        self._diversity_rate = diversity_rate
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(
        self, input_ids: paddle.Tensor, scores: paddle.Tensor, current_tokens: paddle.Tensor, beam_group_idx: int
    ):
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

    def __init__(self, forced_bos_token_id: int):
        self.forced_bos_token_id = forced_bos_token_id

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor):
        cur_len = input_ids.shape[-1]
        if cur_len == 1:
            scores[:] = paddle.finfo(scores.dtype).min
            scores[:, self.forced_bos_token_id] = 0
        return scores


class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces the last generated token to be the selected `forced_eos_token`.

    Args:
        max_length (int): The maximum length of the sequence to be generated.
        forced_eos_token_id (int): The id of the token to be generated as the last token.
    """

    def __init__(self, max_length: int, forced_eos_token_id: Union[int, List[int]]):
        self.max_length = max_length
        self.forced_eos_token_id = forced_eos_token_id

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len == self.max_length - 1:
            scores[:] = -1e9  # TODO change back to -inf after paddle.topk is fixed
            scores[:, self.forced_eos_token_id] = 0
        return scores


def TopKProcess(probs: paddle.Tensor, top_k: int, min_tokens_to_keep: int):
    top_k = min(max(top_k, min_tokens_to_keep), probs.shape[-1])
    # Remove all tokens with a probability less than the last token of the top-k
    # cast to float16 to support generation & d2s
    if probs.dtype == paddle.bfloat16:
        probs = paddle.cast(probs, paddle.float32)
        topk_probs, _ = paddle.topk(probs, k=top_k)
        topk_probs = paddle.cast(topk_probs, paddle.bfloat16)
    else:
        topk_probs, _ = paddle.topk(probs, k=top_k)

    probs = paddle.where(probs >= topk_probs[:, -1:], probs, paddle.full_like(probs, 0.0))
    return probs


def TopPProcess(probs: paddle.Tensor, top_p: float, min_tokens_to_keep: int):
    sorted_indices = paddle.argsort(probs, descending=True)
    if isinstance(sorted_indices, tuple):
        sorted_probs, sorted_indices = sorted_indices
    else:
        sorted_probs = paddle.sort(probs, descending=True)

    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

    # Remove tokens with cumulative probs above the top_p, But keep at
    # least min_tokens_to_keep tokens
    sorted_indices_to_remove = cumulative_probs > top_p
    if min_tokens_to_keep > 1:
        # Set 'min_tokens_to_keep - 1' because the first token is kept
        sorted_indices_to_remove[:, : min_tokens_to_keep - 1] = 0
    # Keep the first token
    sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove, dtype="int64")
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    # Scatter sorted tensors to original indexing
    sorted_indices = sorted_indices + paddle.arange(probs.shape[0], dtype="int64").unsqueeze(-1) * probs.shape[-1]
    condition = paddle.scatter(
        sorted_indices_to_remove.flatten(), sorted_indices.flatten(), sorted_indices_to_remove.flatten()
    )
    condition = paddle.cast(condition, "bool").reshape(probs.shape)
    probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
    return probs


class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class TemperatureLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] for temperature (exponential scaling output probability distribution).
    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor):
        scores = scores / self.temperature
        return scores
