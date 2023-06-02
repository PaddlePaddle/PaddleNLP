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
from typing import List

import paddle


class LogitsProcessorList(List):
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
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError("`min_length` should be a positive integer, but get {}".format(min_length))

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError("`eos_token_id` should be a positive integer, but get {}".format(eos_token_id))

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


def _get_ngrams(ngram_size, prev_input_ids, num_hypos):
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(ngram_size, prev_input_ids, num_hypos, cur_len):
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

    def __init__(self, ngram_size):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def __call__(self, input_ids, scores):
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

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
            The id of the token to to be generated as the first token.
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
        forced_eos_token_id (int): The id of the token to to be generated as the last token.
    """

    def __init__(self, max_length, forced_eos_token_id):
        self.max_length = max_length
        self.forced_eos_token_id = forced_eos_token_id

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len == self.max_length - 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.forced_eos_token_id]] = -float("inf")
            scores[:, self.forced_eos_token_id] = 0
        return scores


class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    def __call__(self, input_ids, scores):
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


class TopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -float("inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, input_ids, probs):
        top_k = min(self.top_k, probs.shape[-1])  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        topk_probs, _ = paddle.topk(probs, k=top_k)

        # NOTE: probs need to be float32, otherwise, paddle.full_like will do truncation
        probs = probs.astype("float32")
        probs = paddle.where(
            probs < topk_probs[:, -1:], paddle.full_like(probs, self.filter_value, dtype="float32"), probs
        )
        return probs


class TopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float("inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids, probs):
        sorted_logits = paddle.sort(probs, descending=False)
        sorted_indices = paddle.argsort(probs, descending=False)
        cumulative_probs = paddle.nn.functional.softmax(sorted_logits, axis=-1).cumsum(axis=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove, dtype="int64")
        sorted_indices = sorted_indices + paddle.arange(probs.shape[0]).unsqueeze(-1) * probs.shape[-1]
        condition = paddle.scatter(
            sorted_indices_to_remove.flatten(), sorted_indices.flatten(), sorted_indices_to_remove.flatten()
        )
        condition = paddle.cast(condition, "bool").reshape(probs.shape)

        # NOTE: probs need to be float32, otherwise, paddle.full_like will do truncation
        probs = probs.astype("float32")
        probs = paddle.where(condition, paddle.full_like(probs, self.filter_value, dtype="float32"), probs)
        return probs
