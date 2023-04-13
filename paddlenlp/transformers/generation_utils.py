# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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


from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.common_ops_import import convert_dtype

try:
    from paddle.utils import map_structure
except ImportError:
    from paddle.fluid.layers.utils import map_structure

from paddle.fluid.dygraph.base import in_declarative_mode

from paddlenlp.utils.log import logger

from .model_outputs import CausalLMOutputWithPast, ModelOutput, Seq2SeqLMOutput

__all__ = ["GenerationMixin"]

from ..generation.logits_process import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from ..generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList


class BeamHypotheses:
    def __init__(self, num_beams, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs, origin_len=0):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (((hyp.shape[-1] - origin_len + 5) / 6) ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len, origin_len=0):
        """
        If there are enough hypotheses and that none of the hypotheses being
        generated can become better than the worst one in the heap, then we
        are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / ((cur_len - origin_len + 5) / 6) ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class BeamSearchScorer(object):
    """
    implementing standard beam search decoding.
    """

    def __init__(
        self,
        batch_size,
        max_length,
        num_beams,
        length_penalty=1.0,
        do_early_stopping=False,
        num_beam_hyps_to_keep=1,
        num_beam_groups=1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams, length_penalty=self.length_penalty, early_stopping=self.do_early_stopping
            )
            for _ in range(batch_size)
        ]
        self._done = paddle.to_tensor([0 for _ in range(batch_size)], dtype="int64")

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                "`num_beams` has to be an integer strictly greater than 1, but "
                "received {}. For `num_beams` == 1, one should make use of "
                "`greedy_search` instead.".format(num_beams)
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than "
                "`num_beams` and `num_beams` has to be divisible by "
                "`num_beam_groups`, but received num_beam_groups={}, num_beams="
                "{}.".format(num_beam_groups, num_beams)
            )

    @property
    def is_done(self):
        return paddle.min(self._done) == 1

    def process(
        self, input_ids, next_scores, next_tokens, next_indices, origin_len=0, pad_token_id=None, eos_token_id=None
    ):
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.group_size)

        next_beam_scores = paddle.zeros([batch_size, self.group_size], dtype=next_scores.dtype)
        next_beam_tokens = paddle.zeros([batch_size, self.group_size], dtype=next_tokens.dtype)
        next_beam_indices = paddle.zeros([batch_size, self.group_size], dtype=next_indices.dtype)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx] == 1:
                assert (
                    len(beam_hyp) >= self.num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(self.num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.numpy().item() == eos_token_id):
                    # If beam_token does not belong to top num_beams tokens,
                    # it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx.numpy().item()].clone(), next_score.numpy().item(), origin_len
                    )

                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token.numpy().item()
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx.numpy().item()
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    "At most {} tokens in `next_tokens[batch_idx]` can be equal "
                    "to `eos_token_id: {}`. Make sure `next_tokens[batch_idx]` "
                    "are corrected.".format(self.group_size, eos_token_id)
                )

            # Check if we are done so that we can save a pad step if all(done)
            if beam_hyp.is_done(next_scores[batch_idx].max().numpy().item(), cur_len, origin_len):
                self._done[batch_idx] = 1

        return {
            "next_beam_scores": next_beam_scores.reshape([-1]),
            "next_beam_tokens": next_beam_tokens.reshape([-1]),
            "next_beam_indices": next_beam_indices.reshape([-1]),
        }

    def finalize(
        self,
        input_ids,
        final_beam_scores,
        final_beam_tokens,
        final_beam_indices,
        origin_len=0,
        pad_token_id=None,
        eos_token_id=None,
    ):
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx] == 1:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].numpy().item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score, origin_len=origin_len)

        # select the best hypotheses
        sent_lengths = paddle.zeros([batch_size * self.num_beam_hyps_to_keep], dtype=input_ids.dtype)
        best = []

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_score, best_hyp = sorted_hyps.pop()
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                best.append([best_hyp, best_score])

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().numpy().item() + 1, self.max_length)
        decoded = paddle.zeros([batch_size * self.num_beam_hyps_to_keep, sent_max_len], dtype=input_ids.dtype)
        # shorter batches are padded if needed
        if sent_lengths.min().numpy().item() != sent_lengths.max().numpy().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded[:, :] = pad_token_id
        decoded_score = paddle.zeros([batch_size * self.num_beam_hyps_to_keep, 1])

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, score) in enumerate(best):
            decoded[i, : sent_lengths[i].numpy().item()] = hypo.numpy()
            decoded_score[i] = score
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i].numpy().item()] = eos_token_id
        return decoded, decoded_score


class GenerationMixin(object):
    r"""
    This class implements the interface for generation task.

    It's used as the base class of `paddlenlp.transformers.PretrainedModel
    <https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.model_utils.html>`__.
    """

    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
        return paddle.ones([batch_size, 1], dtype="int64") * bos_token_id

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id
        ).numpy().item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids == pad_token_id).astype(paddle.get_default_dtype()) * -1e9
        else:
            attention_mask = paddle.zeros_like(input_ids, dtype=paddle.get_default_dtype())
        return paddle.unsqueeze(attention_mask, axis=[1, 2])

    @staticmethod
    def prepare_seq_len_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id
        ).numpy().item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            seq_len = paddle.sum(input_ids != pad_token_id, axis=1).unsqueeze(-1)
        else:
            seq_len = paddle.full((input_ids.shape[0], 1), input_ids.shape[1], dtype="int64")
        return seq_len

    @staticmethod
    def get_logits_processor(
        min_length=None,
        max_length=None,
        eos_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        num_beams=1,
        num_beam_groups=1,
        diversity_rate=0.0,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        logits_processors=None,
    ):
        processors = LogitsProcessorList()

        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if num_beam_groups > 1 and diversity_rate > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_rate=diversity_rate, num_beams=num_beams, num_beam_groups=num_beam_groups
                )
            )
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
        # TODO
        # Add more pre_processing for distribution

        if logits_processors is not None:
            custom_processors = LogitsProcessorList()
            custom_processors_type = [type(lp) for lp in logits_processors]

            for processor in processors:
                if type(processor) not in custom_processors_type:
                    custom_processors.append(processor)
            custom_processors.extend(logits_processors)

            return custom_processors
        else:
            return processors

    @staticmethod
    def get_logits_warper(
        temperature=None,
        top_k=None,
        top_p=None,
        num_beams=1,
    ):
        # instantiate warpers list
        warpers = LogitsProcessorList()
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))

        min_tokens_to_keep = 2 if num_beams > 1 else 1
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=min_tokens_to_keep))
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep))
        # TODO
        # Add more pre_processing for distribution during generation

        return warpers

    @staticmethod
    def get_stopping_criteria(max_length: Optional[int], stopping_criteria: Optional[StoppingCriteriaList]):
        criteria = StoppingCriteriaList()
        if max_length is not None:
            criteria.append(MaxLengthCriteria(max_length=max_length))

        if stopping_criteria is not None:
            custom_criteria = StoppingCriteriaList()
            custom_criteria_type = [type(cr) for cr in stopping_criteria]

            for criterium in criteria:
                if type(criterium) not in custom_criteria_type:
                    custom_criteria.append(criterium)
            custom_criteria.extend(stopping_criteria)

            return custom_criteria
        else:
            return criteria

    @staticmethod
    def expand_inputs_for_generation(
        input_ids: Optional[paddle.Tensor] = None,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        **model_kwargs,
    ):
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], paddle.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, axis=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, axis=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # Update the model inputs during generation.
        # Note that If `token_type_ids` and `attention_mask` in `model_kwargs`
        # and they contain pad value, the result vectors updated by this method
        # may be different from expected. In this case, you need to rewrite the
        # method.

        # 拴Q了
        # def _convert_to_standard_cache(
        #     past_key_value, batch_size
        # ):
        #     """
        #     Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        #     num_heads, ...]))
        #     """
        #     batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        #     num_heads = batch_size_times_num_heads // batch_size
        #     # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        #     # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        #     return tuple(
        #         (
        #             layer_past[0].reshape([batch_size, num_heads, head_dim, seq_length]),
        #             layer_past[1].reshape([batch_size, num_heads, seq_length, head_dim]),
        #         )
        #         for layer_past in past_key_value
        #     )

        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["cache"] = outputs[1]

        if isinstance(outputs, ModelOutput) and "past_key_values" in outputs:
            model_kwargs["cache"] = outputs.past_key_values
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat([token_type_ids, token_type_ids[:, -1:]], axis=-1)

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat([position_ids, position_ids[..., -1:] + 1], axis=-1)

        # update attention_mask
        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            # nn.Pad2D don't support the data type `bool`
            if convert_dtype(attention_mask.dtype) == "bool":
                attention_mask = paddle.cast(attention_mask, "int64")
            if len(attention_mask.shape) == 4:
                cur_device = paddle.get_device()
                if cur_device.split(":")[0] == "npu":
                    attention_mask = nn.Pad2D([0, 0, 0, 1], mode="constant")(attention_mask)
                    attention_mask = nn.Pad2D([0, 1, 0, 0], value=0)(attention_mask)
                else:
                    attention_mask = nn.Pad2D([0, 0, 0, 1], mode="replicate")(attention_mask)
                    attention_mask = nn.Pad2D([0, 1, 0, 0], value=-1e4)(attention_mask)

                dtype = convert_dtype(attention_mask.dtype)
                if "int" in dtype:
                    attention_mask[:, :, -1, -1] = 1
                elif "float" in dtype:
                    attention_mask[:, :, -1, -1] = 0.0
                else:
                    raise ValueError("The data type of input `attention_mask` must " "be bool, int or float")
            else:
                attention_mask = paddle.concat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype="int64")], axis=-1
                )
            model_kwargs["attention_mask"] = attention_mask

        # update role_ids
        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.concat([role_ids, role_ids[:, -1:]], axis=-1)

        return model_kwargs

    @staticmethod
    def update_scores_for_generation(scores, next_scores, length, unfinished_flag):
        # update scores

        unfinished_scores = (scores * length + next_scores) / (length + 1)
        scores = paddle.where(unfinished_flag, unfinished_scores, scores)
        return scores

    def prepare_encoder_decoder_kwargs_for_generation(self, input_ids, model_kwargs):
        if "encoder_output" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (
                    argument.startswith("decoder_") or argument.startswith("cross_attn") or argument == "use_cache"
                )
            }
            # Use inputs_embeds as the priority if inputs_embeds exists
            if "inputs_embeds" in encoder_kwargs:
                model_kwargs["encoder_output"] = encoder(**encoder_kwargs)
            else:
                model_kwargs["encoder_output"] = encoder(input_ids=input_ids, **encoder_kwargs)
        return model_kwargs

    def prepare_decoder_input_ids_for_generation(self, input_ids, decoder_start_token_id=None, bos_token_id=None):
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )
        decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else bos_token_id

        decoder_input_ids = paddle.ones([input_ids.shape[0], 1], dtype="int64") * decoder_start_token_id

        return decoder_input_ids

    def get_decoder_start_token_id(self, decoder_start_token_id=None, bos_token_id=None):
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif self.config.decoder_start_token_id is not None:
            return self.config.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        elif self.config.bos_token_id is not None:
            return self.config.bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Implement in subclasses for custom behavior to prepare inputs in the
        # generate method.
        return {"input_ids": input_ids}

    def adjust_logits_during_generation(self, logits):
        # Implement in subclasses for custom behavior to adjust the logits in
        # the generate method.

        return logits

    def prepare_fast_entry(self, kwargs):
        return False

    def _convert_to_fast(self, kwargs):
        # try general convert
        pass

    def _build_fast(self, kwargs):
        self._fast_entry = False
        if kwargs["num_beam_groups"] != 1:
            # not support for group_beam_search yet in the fast version
            raise AttributeError("'num_beam_groups != 1' is not supported yet in the fast version")
        if paddle.get_default_dtype() == "float16" and kwargs["use_fp16_decoding"] is False:
            logger.info(
                "Since the default dtype is float16, float16 would be used " "though 'use_fp16_decoding=False'."
            )
            kwargs["use_fp16_decoding"] = True
        self.prepare_fast_entry(kwargs)

    @paddle.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        max_length=20,
        min_length=0,
        decode_strategy="greedy_search",
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        penalty_alpha=0.0,
        num_beams=1,
        num_beam_groups=1,
        length_penalty=0.0,
        early_stopping=False,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        decoder_start_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        no_repeat_ngram_size=None,
        num_return_sequences=1,
        diversity_rate=0.0,
        use_cache=True,
        use_fast=False,
        use_fp16_decoding=False,
        **model_kwargs
    ):
        r"""
        The interface for generation task. This method can generate sequences
        by using decoding strategy. Currently, there are three decoding
        strategies supported: "greedy_search", "sampling" and "beam_search".

        Args:
            input_ids (Tensor, optional): The input sequence ids for the
                generation. It is a Tensor with shape [batch_size, sequence_length].
                The data type should be int32 or int64. Default to None, which
                we will initialize it as a Tensor with shape [1, 1], filled
                with the value `bos_token_id`.
            max_length (int, optional): The maximum length of the sequence to
                be generated. Default to 20.
            min_length (int, optional): The minimum length of the sequence to
                be generated. Default to 0.
            decode_strategy (str, optional): The decoding strategy in generation.
                Currently, there are three decoding strategies supported:
                "greedy_search", "sampling" and "beam_search". Default to
                "greedy_search".
            temperature (float, optional): The value used to module the next
                token probabilities in the "sampling" strategy. Default to 1.0,
                which means no effect.
            top_k (int, optional): The number of highest probability tokens to
                keep for top-k-filtering in the "sampling" strategy. Default to
                0, which means no effect.
            top_p (float, optional): The cumulative probability for
                top-p-filtering in the "sampling" strategy. The value should
                satisfy :math:`0 <= top\_p < 1`. Default to 1.0, which means no
                effect.
            repetition_penalty (float, optional):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details. Defaults to 1.0.
            num_beams (int, optional): The number of beams in the "beam_search"
                strategy. Default to 1.
            num_beam_groups (int, optional):
                Number of groups to divide `num_beams` into in order to use DIVERSE
                BEAM SEARCH. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__
                for more details. Default to 1.
            length_penalty (float, optional): The exponential penalty to the
                sequence length in the "beam_search" strategy. The larger this
                param is, the more that the model would generate shorter
                sequences. Default to 0.0, which means no penalty.
            early_stopping (bool, optional): Whether to stop searching in the
                "beam_search" strategy when at least `num_beams` sentences are
                finished per batch or not. Default to False.
            bos_token_id (int, optional): The id of the `bos_token`. Default to
                None.
            eos_token_id (int, optional): The id of the `eos_token`. Default to
                None.
            pad_token_id (int, optional): The id of the `pad_token`. Default to
                None.
            decoder_start_token_id (int, optional): The start token id for
                encoder-decoder models. Default to None.
            forced_bos_token_id (int, optional): The id of the token to force as
                the first generated token. Usually use for multilingual models.
                Default to None.
            forced_eos_token_id (int, optional): The id of the token to force as
                the last generated token. Default to None.
            num_return_sequences (int, optional): The number of returned
                sequences for each sequence in the batch. Default to 1.
            diversity_rate (float, optional): If num_beam_groups is 1, this is the
                diversity_rate for Diverse Siblings Search. See
                `this paper https://arxiv.org/abs/1611.08562`__ for more details.
                If not, this is the diversity_rate for DIVERSE BEAM SEARCH.
            use_cache: (bool, optional): Whether to use the model cache to
                speed up decoding. Default to True.
            use_fast: (bool, optional): Whether to use fast entry of model
                for FastGeneration. Default to False.
            use_fp16_decoding: (bool, optional): Whether to use fp16 for decoding.
                Only works when fast entry is avalible. Default to False.
            model_kwargs (dict): It can be used to specify additional kwargs
                passed to the model.

        Returns:
            tuple[Tensor]: It is a tuple contains two elements: ids and scores.
            Each element is a Tensor.

            With the fields:

            - ids (Tensor):
                The ids of the generated sequences. It is a Tensor with shape
                [batch_size * num_return_sequences, sequence_length]. The data
                type is same as the input `input_ids`.
            - scores (Tensor):
                The scores of the generated sequences. It is a Tensor with shape
                [batch_size * num_return_sequences, 1]. The data type is float32
                or float64, which is the same as the parameters in the model.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import (
                    UnifiedTransformerLMHeadModel,
                    UnifiedTransformerTokenizer
                )

                paddle.seed(2)

                # Initialize the model and tokenizer
                model_name_or_path = 'unified_transformer-12L-cn-luge'
                model = UnifiedTransformerLMHeadModel.from_pretrained(model_name_or_path)
                tokenizer = UnifiedTransformerTokenizer.from_pretrained(model_name_or_path)

                # Prepare the model inputs.
                history = "早上好，今天空气质量不错。"
                inputs = tokenizer.dialogue_encode(history, task_type='chitchat',
                    add_start_token_as_response=True, return_tensors=True)

            .. code-block::

                # Generate the sequence by using "greedy_search" strategy
                ids, scores = model.generate(
                    input_ids=inputs['input_ids'],
                    token_type_ids=inputs['token_type_ids'],
                    position_ids=inputs['position_ids'],
                    attention_mask=inputs['attention_mask'],
                    decode_strategy="greedy_search")
                print(ids.shape, scores.shape)
                # [1, 3] [1, 1]
                sequence_ids = ids.numpy().tolist()[0]
                sequence_ids = sequence_ids[:sequence_ids.index(tokenizer.sep_token_id)]
                response = tokenizer.convert_ids_to_string(sequence_ids, keep_space=False)
                print(response)
                # 是的

            .. code-block::

                # Generate 2 sequences by using "sampling" strategy (top_k=5)
                ids, scores = model.generate(
                    input_ids=inputs['input_ids'],
                    token_type_ids=inputs['token_type_ids'],
                    position_ids=inputs['position_ids'],
                    attention_mask=inputs['attention_mask'],
                    decode_strategy="sampling",
                    top_k=5,
                    num_return_sequences=2)
                print(ids.shape, scores.shape)
                # [2, 7] [2, 1]
                response = []
                for sequence_ids in ids.numpy().tolist():
                    sequence_ids = sequence_ids[:sequence_ids.index(tokenizer.sep_token_id)]
                    text = tokenizer.convert_ids_to_string(sequence_ids, keep_space=False)
                    response.append(text)
                print(response)
                # ['天气好,心情也好', '你也是']

            .. code-block::

                # Generate 2 sequences by using "beam_search" strategy (num_beams=5)
                ids, scores = model.generate(
                    input_ids=inputs['input_ids'],
                    token_type_ids=inputs['token_type_ids'],
                    position_ids=inputs['position_ids'],
                    attention_mask=inputs['attention_mask'],
                    decode_strategy="beam_search",
                    num_beams=5,
                    num_return_sequences=2)
                print(ids.shape, scores.shape)
                # [2, 3] [2, 1]
                response = []
                for sequence_ids in ids.numpy().tolist():
                    sequence_ids = sequence_ids[:sequence_ids.index(tokenizer.sep_token_id)]
                    text = tokenizer.convert_ids_to_string(sequence_ids, keep_space=False)
                    response.append(text)
                print(response)
                # ['是的', '嗯嗯']
        """

        assert decode_strategy in [
            "greedy_search",
            "sampling",
            "beam_search",
            "contrastive_search",
        ], "`decode_strategy` must be one of 'greedy_search', 'sampling' or 'beam_search' but received {}.".format(
            decode_strategy
        )

        # Whether to dynamic to static
        is_tracing = False
        if in_declarative_mode():
            is_tracing = True

        if is_tracing:
            assert decode_strategy in [
                "sampling",
            ], "`generate()` only supports 'sampling' temporarily but received {}.".format(decode_strategy)

        if getattr(self, "deprecated_warnings", None) is None:
            self.deprecated_warnings = {}

        if "use_faster" in model_kwargs:
            use_fast = model_kwargs.pop("use_faster")
            if not self.deprecated_warnings.get("use_faster", False):
                logger.warning("`use_faster` will be deprecated in near future. Please use `use_fast` instead. ")
                self.deprecated_warnings["use_faster"] = True

        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else self.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else self.config.forced_eos_token_id
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )

        if is_tracing:
            self._fast_entry = None

        if getattr(self, "_fast_entry", None) is not False and use_fast:
            args = locals()
            args.pop("self")
            args.pop("__class__", None)
            model_kwargs = args.pop("model_kwargs")
            args.update(model_kwargs)
            try:
                if getattr(self, "_fast_entry", None) is None:
                    self._build_fast(args)
                if self._fast_entry:
                    output = self._fast_entry(**args)
                    if isinstance(output, tuple):
                        output_ids, dummy_srore = output
                    else:
                        output_ids = output
                        # make result and fast result oneconsistent
                        dummy_srore = None
                    if decode_strategy == "beam_search":
                        output_ids = output_ids.transpose([1, 2, 0])
                        output_ids = output_ids[:, :num_return_sequences, :].reshape([-1, output_ids.shape[-1]])
                        if dummy_srore is not None:
                            dummy_srore = dummy_srore[:, :num_return_sequences].flatten()
                    else:
                        output_ids = output_ids.transpose([1, 0])
                    return output_ids, dummy_srore

            except Exception as e:
                args["model_kwargs"] = model_kwargs
                # TODO
                # Prevent self._convert_to_fast to throw Exception
                self._convert_to_fast(args)
                logger.warning(e)
                logger.warning("FastGeneration is not available, " "and the original version would be used instead.")

        # params check
        if input_ids is None:
            # Init `input_ids` with bos_token_id
            input_ids = self.prepare_input_ids_for_generation(bos_token_id)

        # Add to model_kwargs
        model_kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            model_kwargs["position_ids"] = position_ids

        if model_kwargs.get("attention_mask", None) is None:
            # TODO
            # Init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self.prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )
        self.is_encoder_decoder = (
            getattr(self, "encoder", None) is not None and getattr(self, "decoder", None) is not None
        )
        if self.is_encoder_decoder:
            model_kwargs = self.prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)
            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self.prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id, bos_token_id
                )
        if pad_token_id is None and eos_token_id is not None:
            print("Setting `pad_token_id` to `eos_token_id`:{} for " "open-end generation.".format(eos_token_id))
            pad_token_id = eos_token_id

        model_kwargs["use_cache"] = use_cache

        # max_len includes prefix length
        if is_tracing:
            min_len = input_ids.shape[-1]
            max_len = input_ids.shape[-1]
            paddle.increment(min_len, min_length)
            paddle.increment(max_len, max_length)
        else:
            input_len = input_ids.shape[-1]
            min_len = input_len + min_length
            max_len = input_len + max_length

        # prepare distribution pre_processing samplers
        logits_processors = self.get_logits_processor(
            min_length=min_len if min_length > 0 else None,
            max_length=max_len,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_rate=diversity_rate,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            logits_processors=model_kwargs["logits_processors"]
            if "logits_processors" in model_kwargs
            and isinstance(model_kwargs["logits_processors"], LogitsProcessorList)
            else LogitsProcessorList(),
        )
        if "logits_processors" in model_kwargs:
            model_kwargs.pop("logits_processors")

        # prepare stopping criteria
        stopping_criteria = self.get_stopping_criteria(
            max_length=max_len,
            stopping_criteria=model_kwargs["stopping_criteria"]
            if "stopping_criteria" in model_kwargs
            and isinstance(model_kwargs["stopping_criteria"], StoppingCriteriaList)
            else StoppingCriteriaList(),
        )
        if "stopping_criteria" in model_kwargs:
            model_kwargs.pop("stopping_criteria")

        # go into different generation modes
        if decode_strategy == "greedy_search":

            logits_warper = self.get_logits_warper(temperature=temperature)
            if num_return_sequences > 1:
                raise ValueError(
                    "`num_return_sequences` has to be 1, but is {} "
                    "when doing greedy search.".format(num_return_sequences)
                )

            return self.greedy_search(
                input_ids,
                logits_processors,
                logits_warper,
                stopping_criteria,
                pad_token_id,
                eos_token_id,
                **model_kwargs,
            )

        elif decode_strategy == "contrastive_search":
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing"
                    " contrastive search."
                )
            logits_warper = self.get_logits_warper(temperature=temperature)
            return self.contrastive_search(
                input_ids,
                top_k,
                penalty_alpha,
                logits_processors,
                logits_warper,
                stopping_criteria,
                pad_token_id,
                eos_token_id,
                **model_kwargs,
            )

        elif decode_strategy == "sampling":
            if num_return_sequences > 1:
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_return_sequences, **model_kwargs
                )

            # top_k and top_p can combine to use
            logits_warper = self.get_logits_warper(temperature=temperature, top_k=top_k, top_p=top_p)

            if is_tracing:
                return self.sample_d2s(
                    input_ids,
                    logits_processors,
                    logits_warper,
                    stopping_criteria,
                    pad_token_id,
                    eos_token_id,
                    **model_kwargs,
                )
            else:
                return self.sample(
                    input_ids,
                    logits_processors,
                    logits_warper,
                    stopping_criteria,
                    pad_token_id,
                    eos_token_id,
                    **model_kwargs,
                )

        elif decode_strategy == "beam_search":
            batch_size = input_ids.shape[0]
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to "
                    "`num_beams`. But received `num_return_sequences` is {}, "
                    "`num_beams` is {}".format(num_return_sequences, num_beams)
                )
            if num_beams <= 1:
                raise ValueError(
                    "`num_beams` has to be bigger than 1. But received "
                    "`num_beams` is {}. If `num_beams` is 1, `decode_strategy` "
                    "should be 'greedy_search'".format(num_beams)
                )
            if num_beam_groups > 1:
                diverse_beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=max_len,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    do_early_stopping=early_stopping,
                    num_beam_hyps_to_keep=num_return_sequences,
                    num_beam_groups=num_beam_groups,
                )

                # interleave with `num_beams`
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_beams, **model_kwargs
                )

                return self.group_beam_search(
                    input_ids,
                    diverse_beam_scorer,
                    logits_processors,
                    max_len,
                    pad_token_id,
                    eos_token_id,
                    **model_kwargs,
                )
            else:
                beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=max_len,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    do_early_stopping=early_stopping,
                    num_beam_hyps_to_keep=num_return_sequences,
                )

                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_beams, **model_kwargs
                )

                return self.beam_search(
                    input_ids,
                    beam_scorer,
                    logits_processors,
                    max_len,
                    diversity_rate,
                    pad_token_id,
                    eos_token_id,
                    **model_kwargs,
                )

    def greedy_search(
        self,
        input_ids: paddle.Tensor,
        logits_processors: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs
    ):
        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())
        while True:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs

            # [batch_size, vocab_size]
            next_token_logits = logits[:, -1, :]

            # pre-process distribution
            next_token_logits = self.adjust_logits_during_generation(next_token_logits)
            next_tokens_scores = logits_processors(input_ids, next_token_logits)

            # record score before warper
            origin_probs = F.softmax(next_tokens_scores)
            origin_probs = paddle.log(origin_probs)

            next_tokens_scores = logits_warper(input_ids, next_tokens_scores)

            # greedy
            probs = F.softmax(next_tokens_scores)
            probs = paddle.log(probs)

            next_tokens = paddle.argmax(probs, axis=-1).unsqueeze(-1)
            next_scores = paddle.index_sample(origin_probs.astype("float32"), next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(unfinished_flag, next_tokens != eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag) or stopping_criteria(input_ids, scores):
                break

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )

        return input_ids[:, origin_len:], scores

    def contrastive_search(
        self,
        input_ids: paddle.Tensor,
        top_k: Optional[int] = 1,
        penalty_alpha: Optional[float] = 0,
        logits_processors: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs
    ):
        # init values
        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        batch_size, cur_len = input_ids.shape
        # record average score
        origin_len = cur_len
        # keep track of which sequences are already finished
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        # If the first step in the loop, encode all the prefix and obtain (1) past_key_values
        # (2) last_hidden_states (3) logit_for_next_step (4) update model kwargs for the next step
        while True:
            if model_kwargs.get("past_key_values") is None:

                model_kwargs["use_cache"] = True

                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                outputs = self(**model_inputs, return_dict=True, output_hidden_states=True, output_attentions=True)

                # last decoder hidden states will be used to compute the degeneration penalty (cosine simlarity with
                # pervious tokens)
                if self.config.is_encoder_decoder:
                    last_hidden_states = outputs.decoder_hidden_states[-1]
                else:
                    last_hidden_states = outputs.hidden_states[-1]

                # next logit for contrastive search to select top-k candidate tokens
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif isinstance(outputs, ModelOutput):
                    logits = outputs.logits
                else:
                    logits = outputs

                logit_for_next_step = logits[:, -1, :]

                model_kwargs = self.update_model_kwargs_for_generation(
                    outputs=outputs, model_kwargs=model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )

                # exapnds model inputs top_k times, for batched forward passes (akin to beam search)
                _, model_kwargs = self.expand_inputs_for_generation(
                    input_ids=input_ids, expand_size=top_k, **model_kwargs
                )

                past_key_values = model_kwargs.get("past_key_values")

                print(type(past_key_values))
                if past_key_values is None:
                    raise ValueError(
                        f"{self.__class__.__name__} does not support caching and therefore **can't** be used "
                        "for contrastive search."
                    )
                elif (
                    not isinstance(past_key_values[0], (tuple, paddle.Tensor))
                    or past_key_values[0][0].shape[0] != batch_size
                ):
                    raise ValueError(
                        f"{self.__class__.__name__} does not have a standard cache format and therefore **can't** be "
                        "used for contrastive search without further modifications."
                    )

            # contrastive search main logic starts:
            # contrastive search decoding consists of two steps: (1) candidate tokens recall (2) candidate rerank by degeneration penalty
            logit_for_next_step = self.adjust_logits_during_generation(logit_for_next_step)
            logit_for_next_step = logits_processors(input_ids, logit_for_next_step)

            origin_probs = F.softmax(logit_for_next_step)
            origin_probs = paddle.log(origin_probs)

            logit_for_next_step = logits_warper(input_ids, logit_for_next_step)

            next_probs = F.softmax(logit_for_next_step, axis=-1)

            top_k_probs, top_k_ids = paddle.topk(next_probs, k=top_k, axis=-1)

            # Replicates the new past_key_valuse to match the  `top_k` candidates
            new_key_values = []
            for layer in model_kwargs["past_key_values"]:
                items = []
                # item is either the key or the value matrix
                for item in layer:
                    items.append(item.repeat_interleave(top_k, axis=0))
                new_key_values.append(items)
            model_kwargs["past_key_values"] = new_key_values

            # compute the candidate tokens by the langugae model and collects their hidden states

            tmp = paddle.reshape(top_k_ids, (-1, 1))

            # need to clear cache
            model_kwargs["cache"] = None
            next_model_inputs = self.prepare_inputs_for_generation(tmp, **model_kwargs)

            outputs = self(**next_model_inputs, return_dict=True, output_hidden_states=True)

            # NOTE: must be ModelOutput
            next_past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            # name is different for encoder-decoder and decoder-only models
            if self.config.is_encoder_decoder:
                next_hidden = outputs.decoder_hidden_states[-1][:, -1, :].unsqueeze(1)
                full_hidden_states = outputs.decoder_hidden_states
            else:
                next_hidden = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                full_hidden_states = outputs.hidden_states

            context_hidden = last_hidden_states.repeat_interleave(top_k, axis=0)

            # compute the degeneration penalty and re-rank the candidates based on the degeneration penalty and the
            # model confidence
            selected_idx = _ranking_fast(context_hidden, next_hidden, top_k_probs, penalty_alpha, top_k)

            # prepare for the next step: (1) next_token_id (2) past_key_values (3) last_hidden_states for computing
            # the degeneration penalty (4) logits for selecting next top-k candidates (5) selected tokens socres
            # (model confidence minus degeneration penalty) (6) decoder hidden state

            next_tokens = top_k_ids[0 : len(top_k_ids), selected_idx.item()].unsqueeze(1)

            next_scores = paddle.index_sample(origin_probs, next_tokens)

            next_hidden = paddle.stack(paddle.split(next_hidden.squeeze(axis=1), top_k), axis=1)
            next_hidden = next_hidden[0:batch_size, selected_idx.item(), :]
            last_hidden_states = paddle.concat([last_hidden_states, next_hidden.unsqueeze(1)], axis=1)

            next_decoder_hidden_states = ()
            for layer in full_hidden_states:
                layer = paddle.stack(paddle.split(layer, top_k), axis=1)[0:batch_size, selected_idx.item(), :]
                next_decoder_hidden_states += (layer,)

            # select the past_key_value
            new_key_values = ()
            for layer in next_past_key_values:
                items = ()
                # itemi is either the key or the value matrix
                for item in layer:
                    item = paddle.stack(paddle.split(item, top_k, axis=0), axis=1)
                    item = item[0:batch_size, selected_idx.item(), ...]
                    items += (item,)
                new_key_values += (items,)
            next_past_key_values = new_key_values

            logit_for_next_step = paddle.stack(paddle.split(logits, top_k), axis=1)[
                0:batch_size, selected_idx.item(), :
            ]

            # Rebuilds the relevant parts of the model output for the selected token, for use in the next iteration.
            if self.config.is_encoder_decoder:
                outputs = Seq2SeqLMOutput(
                    past_key_values=next_past_key_values,
                    decoder_hidden_states=next_decoder_hidden_states,
                )
            else:
                outputs = CausalLMOutputWithPast(
                    past_key_values=next_past_key_values,
                    hidden_states=next_decoder_hidden_states,
                )

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            input_ids = paddle.concat([input_ids, next_tokens], axis=1)
            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)
            cur_len += 1
            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(unfinished_flag, next_tokens != eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag) or stopping_criteria(input_ids, scores):
                break

        return input_ids[:, origin_len:], scores

    def sample(
        self,
        input_ids: paddle.Tensor,
        logits_processors: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs
    ):
        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        batch_size, cur_len = input_ids.shape
        # record average score
        origin_len = cur_len
        # keep track of which sequences are already finished
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        while True:
            # prepare model inputs and get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs

            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # record score
            origin_probs = F.softmax(logits)
            origin_probs = paddle.log(origin_probs)

            # adjust distribution
            logits = logits_warper(input_ids, logits)

            # sample
            probs = F.softmax(logits)
            # multinomial not support fp16 and bf16 currently, issue: https://github.com/PaddlePaddle/Paddle/issues/51852
            if paddle.get_default_dtype() not in ["float32", "float64"]:
                probs = probs.astype("float32")

            next_tokens = paddle.multinomial(probs)
            next_scores = paddle.index_sample(origin_probs, next_tokens)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            input_ids = paddle.concat([input_ids, next_tokens], axis=1)
            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)
            cur_len += 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(unfinished_flag, next_tokens != eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag) or stopping_criteria(input_ids, scores):
                break

        return input_ids[:, origin_len:], scores

    def sample_d2s(
        self,
        input_ids: paddle.Tensor,
        logits_processors: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs
    ):

        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        batch_size, cur_len = paddle.shape(input_ids)
        # used for compute on gpu, avoid memcpy D2H
        cur_len_gpu = paddle.full([1], cur_len, dtype="int64")

        origin_len = paddle.shape(input_ids)[1]
        # used for compute on gpu, avoid memcpy D2H
        origin_len_gpu = paddle.full([1], origin_len, dtype="int64")

        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        # use_cache is immutable, we split it off other mutable kwargs.
        assert "use_cache" in model_kwargs
        immutable = {"use_cache": model_kwargs["use_cache"]}
        del model_kwargs["use_cache"]

        def _forward_(**args):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **args, **immutable)
            assert "use_cache" in model_inputs
            del model_inputs["use_cache"]
            return self(**model_inputs, **immutable)

        def _post_process_(outputs, input_ids, cur_len, origin_len, scores, unfinished_flag, model_kwargs):
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs

            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # sample
            origin_probs = F.softmax(logits)
            origin_probs = paddle.log(origin_probs)

            logits = logits_warper(input_ids, logits)
            probs = F.softmax(logits)
            # multinomial not support fp16 and bf16 currently, issue: https://github.com/PaddlePaddle/Paddle/issues/51852
            if paddle.get_default_dtype() not in ["float32", "float64"]:
                probs = probs.astype("float32")
            next_tokens = paddle.multinomial(probs)
            next_scores = paddle.index_sample(origin_probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)

            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(unfinished_flag, next_tokens != eos_token_id)

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )
            return input_ids, scores, unfinished_flag, model_kwargs

        outputs = _forward_(**model_kwargs)
        input_ids, scores, unfinished_flag, model_kwargs = _post_process_(
            outputs, input_ids, cur_len_gpu, origin_len_gpu, scores, unfinished_flag, model_kwargs
        )

        paddle.increment(cur_len)
        paddle.increment(cur_len_gpu)

        attn_mask = model_kwargs["attention_mask"]
        # make the shape of attention_mask = (-1, -1, -1, -1) in dy2static.
        model_kwargs["attention_mask"] = paddle.reshape(attn_mask, paddle.shape(attn_mask))
        model_kwargs["cache"] = outputs[1] if isinstance(outputs, tuple) else None

        # max_length = paddle.full([1], max_length, dtype="int64")

        while True:
            input_ids, scores, unfinished_flag, model_kwargs = _post_process_(
                _forward_(**model_kwargs),
                input_ids,
                cur_len_gpu,
                origin_len_gpu,
                scores,
                unfinished_flag,
                model_kwargs,
            )
            paddle.increment(cur_len)
            paddle.increment(cur_len_gpu)

            if not paddle.any(unfinished_flag) or stopping_criteria(input_ids, scores):
                break

        return input_ids[:, origin_len:], scores

    def beam_search(
        self,
        input_ids,
        beam_scorer,
        logits_processors,
        max_length,
        diversity_rate,
        pad_token_id,
        eos_token_id,
        **model_kwargs
    ):
        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        batch_beam_size, cur_len = input_ids.shape
        origin_len = cur_len

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {}, but received {}.".format(
            num_beams * batch_size, batch_beam_size
        )

        beam_scores = paddle.zeros((batch_size, num_beams), dtype=paddle.get_default_dtype())
        beam_scores[:, 1:] = -1e9
        beam_scores = paddle.reshape(beam_scores, [-1])

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs

            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            # beam search
            # [batch_size * num_beams, vocab_size]
            next_scores = F.softmax(logits)
            next_scores = paddle.log(next_scores)
            next_scores = logits_processors(input_ids, next_scores)
            next_scores = next_scores + beam_scores.unsqueeze(-1)

            vocab_size = next_scores.shape[-1]
            if diversity_rate == 0.0:
                # reshape for beam search
                next_scores = next_scores.reshape([batch_size, num_beams * vocab_size])

                next_scores, next_tokens = paddle.topk(next_scores, 2 * num_beams, axis=1)

                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens % vocab_size

            else:
                next_scores, next_tokens = paddle.topk(next_scores, 2 * num_beams, axis=1)

                sibling_score = paddle.arange(1, 2 * num_beams + 1).unsqueeze(0) * diversity_rate

                diversed_score = next_scores - sibling_score

                next_scores = next_scores.reshape([batch_size, 2 * num_beams * num_beams])
                next_tokens = next_tokens.reshape([batch_size, 2 * num_beams * num_beams])

                diversed_score = diversed_score.reshape([batch_size, 2 * num_beams * num_beams])
                diversed_score, diversed_tokens = paddle.topk(diversed_score, 2 * num_beams, axis=1)

                # TODO
                # Use gather_nd() to select origan token and score
                next_scores = paddle.stack(
                    [paddle.index_select(next_scores[i], diversed_tokens[i]) for i in range(next_scores.shape[0])]
                )
                next_tokens = paddle.stack(
                    [paddle.index_select(next_tokens[i], diversed_tokens[i]) for i in range(next_tokens.shape[0])]
                )

                next_indices = diversed_tokens // (2 * num_beams)

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
                origin_len=origin_len,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            cur_len += 1
            input_ids = paddle.concat(
                [paddle.index_select(input_ids, beam_idx), beam_next_tokens.unsqueeze(-1)], axis=-1
            )

            if beam_scorer.is_done:
                break
            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )
            if "cache" in model_kwargs and model_kwargs["cache"] is not None:
                # reorder the cache
                model_kwargs["cache"] = map_structure(
                    lambda x: paddle.index_select(x, beam_idx), model_kwargs["cache"]
                )

        pred_ids, scores = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            origin_len=origin_len,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        return pred_ids[:, origin_len:], scores

    def group_beam_search(
        self, input_ids, beam_scorer, logits_processors, max_length, pad_token_id, eos_token_id, **model_kwargs
    ):
        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups

        batch_beam_size, cur_len = input_ids.shape
        origin_len = cur_len

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {}, but received {}.".format(
            num_beams * batch_size, batch_beam_size
        )

        beam_scores = paddle.full((batch_size, num_beams), -1e9, dtype="float32")
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = paddle.reshape(beam_scores, [-1])

        while cur_len < max_length:
            # predicted tokens in cur_len step
            current_tokens = paddle.zeros(shape=[batch_size * num_beams], dtype=input_ids.dtype)

            # indices which will form the beams in the next time step
            reordering_indices = paddle.zeros(shape=[batch_size * num_beams], dtype="int64")
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs)

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )

                group_input_ids = input_ids[batch_group_indices]

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif isinstance(outputs, ModelOutput):
                    logits = outputs.logits
                else:
                    logits = outputs

                logits = logits[:, -1, :]
                logits = paddle.index_select(logits, paddle.to_tensor(batch_group_indices))
                logits = self.adjust_logits_during_generation(logits)

                next_scores = F.softmax(logits)
                next_scores = paddle.log(next_scores)
                vocab_size = next_scores.shape[-1]

                next_scores = logits_processors(
                    group_input_ids, next_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )

                next_scores = next_scores + beam_scores[batch_group_indices].unsqueeze(-1)

                # reshape for beam search
                next_scores = next_scores.reshape([batch_size, group_size * vocab_size])

                next_scores, next_tokens = paddle.topk(next_scores, 2 * group_size, axis=1)

                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens % vocab_size

                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_scores,
                    next_tokens,
                    next_indices,
                    origin_len=origin_len,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )

                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = paddle.concat(
                    [paddle.index_select(group_input_ids, index=beam_idx), beam_next_tokens.unsqueeze(-1)], axis=-1
                )
                current_tokens[batch_group_indices] = beam_next_tokens

                reordering_indices[batch_group_indices] = (
                    num_beams * (beam_idx // group_size) + group_start_idx + (beam_idx % group_size)
                )

            input_ids = paddle.concat([input_ids, current_tokens.unsqueeze(-1)], axis=-1)

            cur_len += 1
            if beam_scorer.is_done:
                break
            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )
            if "cache" in model_kwargs and model_kwargs["cache"] is not None:
                # reorder the cache
                model_kwargs["cache"] = map_structure(
                    lambda x: paddle.index_select(x, reordering_indices), model_kwargs["cache"]
                )

        pred_ids, scores = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            origin_len=origin_len,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        return pred_ids[:, origin_len:], scores


def _ranking_fast(
    context_hidden: paddle.Tensor,
    next_hidden: paddle.Tensor,
    next_top_k_probs: paddle.Tensor,
    alpha: float,
    beam_width: int,
):
    """
    Rerank the top_k candidates based on a degeneration penalty (cosine similarity with previous tokens).
    Rerank the index of the best candidate for each row in the batch.
    """
    norm_context_hidden = context_hidden / paddle.linalg.norm(context_hidden, axis=2, keepdim=True)
    norm_next_hidden = next_hidden / paddle.linalg.norm(next_hidden, axis=2, keepdim=True)
    consine_matrix = paddle.matmul(norm_context_hidden, norm_next_hidden.transpose([0, 2, 1])).squeeze(-1)
    degeneration_penalty = paddle.max(consine_matrix, axis=-1)
    next_top_k_probs = paddle.reshape(next_top_k_probs, (-1,))
    contrastive_score = (1.0 - alpha) * next_top_k_probs - alpha * degeneration_penalty
    contrastive_score = paddle.stack(paddle.split(contrastive_score, beam_width), axis=-1)
    selected_idx = paddle.argmax(contrastive_score, axis=-1)
    return selected_idx
