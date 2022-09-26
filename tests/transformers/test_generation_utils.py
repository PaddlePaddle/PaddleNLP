# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021, The HuggingFace Inc. team. All rights reserved.
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
import unittest
import numpy as np
import random

from tests.testing_utils import slow

from .test_modeling_common import floats_tensor, ids_tensor

import paddle
from paddlenlp.transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
)
from paddlenlp.transformers.generation_utils import (
    BeamSearchScorer, MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor, HammingDiversityLogitsProcessor,
    ForcedBOSTokenLogitsProcessor, ForcedEOSTokenLogitsProcessor, TopKProcess,
    TopPProcess, LogitsProcessorList)


def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=1.0,
    min_tokens_to_keep=1,
):
    if top_k > 0:
        logits = TopKProcess(logits, top_k, min_tokens_to_keep)

    if 0 <= top_p <= 1.0:
        logits = TopPProcess(logits, top_p, min_tokens_to_keep)

    return logits


class GenerationTesterMixin:
    model_tester = None
    # all_pretrained_model = []
    # all_pretrained_model_name = []
    all_generative_model_classes = {}
    input_name = "input_ids"
    is_encoder_decoder = False

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common(
        )

        input_ids = inputs_dict[self.input_name]
        attention_mask = paddle.zeros_like(input_ids, dtype=paddle.int64)

        max_batch_size = 2
        sequence_length = input_ids.shape[-1] // 2
        input_ids = input_ids[:max_batch_size, :sequence_length]
        attention_mask = attention_mask[:max_batch_size, :
                                        sequence_length].unsqueeze([1, 2])

        # generate max 3 tokens
        max_length = 3

        if config.get(
                "eos_token_id",
                None) is not None and config.get("pad_token_id", None) is None:
            # hack to allow generate for models such as GPT2 as is done in `generate()`
            config["pad_token_id"] = config["eos_token_id"]

        return config, input_ids, attention_mask, max_length

    @staticmethod
    def _get_logits_processor_and_kwargs(
        eos_token_id,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        max_length=None,
        diversity_rate=None,
    ):
        process_kwargs = {
            "min_length": 1 if max_length is None else max_length - 1,
            "repetition_penalty": 1.2,
        }

        if diversity_rate is not None:
            process_kwargs["diversity_rate"] = diversity_rate
        logits_processor = LogitsProcessorList(([
            HammingDiversityLogitsProcessor(
                diversity_rate, num_beams=2, num_beam_groups=2),
        ] if diversity_rate is not None else []) + ([
            MinLengthLogitsProcessor(process_kwargs["min_length"], eos_token_id
                                     ),
        ] if eos_token_id is not None else []) + ([
            ForcedBOSTokenLogitsProcessor(forced_bos_token_id),
        ] if forced_bos_token_id is not None else []) + (
            [ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id)]
            if forced_eos_token_id is not None else []) + [
                RepetitionPenaltyLogitsProcessor(
                    process_kwargs["repetition_penalty"]),
            ])
        return process_kwargs, logits_processor

    @staticmethod
    def _get_warper_and_kwargs():
        warp_kwargs = {"top_k": 10, "top_p": 0.7, "temperature": 0.7}
        return warp_kwargs

    @staticmethod
    def _get_beam_scorer_and_kwargs(batch_size,
                                    max_length,
                                    num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
        }
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=beam_kwargs["num_beams"],
            length_penalty=beam_kwargs["length_penalty"],
            do_early_stopping=beam_kwargs["early_stopping"],
            num_beam_hyps_to_keep=num_return_sequences,
        )
        return beam_kwargs, beam_scorer

    @staticmethod
    def _get_diverse_beam_scorer_and_kwargs(batch_size,
                                            max_length,
                                            num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
            "num_beam_groups": 2,  # one beam per group
            "diversity_rate": 2.0,
        }
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=beam_kwargs["num_beams"],
            length_penalty=beam_kwargs["length_penalty"],
            do_early_stopping=beam_kwargs["early_stopping"],
            num_beam_hyps_to_keep=num_return_sequences,
            num_beam_groups=beam_kwargs["num_beam_groups"],
        )
        return beam_kwargs, beam_scorer

    @staticmethod
    def _get_encoder_outputs(
        model,
        input_ids,
        attention_mask,
        output_attentions=None,
        output_hidden_states=None,
        num_interleave=1,
    ):
        model.eval()
        encoder = model.get_encoder()
        encoder_outputs = encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        if isinstance(encoder_outputs, (list, tuple)):
            encoder_outputs = encoder_outputs[0]

        encoder_outputs = encoder_outputs.repeat_interleave(num_interleave,
                                                            axis=0)

        input_ids = paddle.zeros_like(
            input_ids[:, :1],
            dtype="int64") + model.get_decoder_start_token_id()
        # attention_mask = None
        return encoder_outputs, input_ids, attention_mask

    def _greedy_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
    ):
        if self.is_encoder_decoder:
            max_length = 4
        logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
            eos_token_id=getattr(
                model, model.base_model_prefix).config["eos_token_id"],
            forced_bos_token_id=getattr(
                getattr(model, model.base_model_prefix).config,
                "forced_bos_token_id", None),
            forced_eos_token_id=getattr(
                getattr(model, model.base_model_prefix).config,
                "forced_eos_token_id", None),
            max_length=max_length,
        )

        kwargs = {}

        with paddle.no_grad():
            output_generate = model.generate(
                input_ids,
                max_length=max_length,
                decode_strategy='greedy_search',
                attention_mask=attention_mask,
                **logits_process_kwargs,
            )

        if self.is_encoder_decoder:
            encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
            )
            kwargs["encoder_output"] = encoder_outputs

        with paddle.no_grad():
            output_greedy = model.greedy_search(
                input_ids,
                max_length=max_length +
                1 if self.is_encoder_decoder else max_length +
                input_ids.shape[-1],
                attention_mask=attention_mask,
                logits_processors=logits_processor,
                pad_token_id=getattr(
                    model, model.base_model_prefix).config["pad_token_id"],
                eos_token_id=getattr(
                    model, model.base_model_prefix).config["eos_token_id"],
                **kwargs,
            )
        return output_greedy, output_generate

    def _sample_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
        num_return_sequences,
        logits_processors,
        logits_warper,
        process_kwargs,
    ):
        with paddle.no_grad():
            output_generate = model.generate(
                input_ids,
                max_length=max_length,
                decode_strategy='sampling',
                num_return_sequences=num_return_sequences,
                attention_mask=attention_mask,
                top_k=1,
                **process_kwargs,
            )

        kwargs = {}
        if self.is_encoder_decoder:
            encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                num_interleave=num_return_sequences,
            )
            kwargs["encoder_output"] = encoder_outputs
            input_ids_clone = input_ids_clone.repeat_interleave(
                num_return_sequences, axis=0)
            attention_mask_clone = attention_mask_clone.repeat_interleave(
                num_return_sequences, axis=0)
        else:
            attention_mask_clone = attention_mask.repeat_interleave(
                num_return_sequences, axis=0)
            input_ids_clone = input_ids.repeat_interleave(num_return_sequences,
                                                          axis=0)

        with paddle.no_grad():
            output_sample = model.sample(
                input_ids_clone,
                attention_mask=attention_mask_clone,
                max_length=max_length +
                1 if self.is_encoder_decoder else max_length +
                input_ids.shape[-1],
                logits_processors=logits_processors,
                pad_token_id=getattr(
                    model, model.base_model_prefix).config["pad_token_id"],
                eos_token_id=getattr(
                    model, model.base_model_prefix).config["eos_token_id"],
                top_k=1,
                **process_kwargs,
                **kwargs,
            )
        return output_sample, output_generate

    def _beam_search_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
        beam_scorer,
        beam_kwargs,
        logits_processor,
        logits_process_kwargs,
    ):
        with paddle.no_grad():
            output_generate = model.generate(
                input_ids,
                decode_strategy='beam_search',
                attention_mask=attention_mask,
                max_length=max_length,
                **beam_kwargs,
                **logits_process_kwargs,
            )

        # beam_search does not automatically interleave `batch_size` dim for `num_beams`
        kwargs = {}
        if self.is_encoder_decoder:
            encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                num_interleave=beam_scorer.num_beams,
            )
            kwargs["encoder_output"] = encoder_outputs
            input_ids_clone = input_ids_clone.repeat_interleave(
                beam_scorer.num_beams, axis=0)
            attention_mask_clone = attention_mask_clone.repeat_interleave(
                beam_scorer.num_beams, axis=0)
        else:
            attention_mask_clone = attention_mask.repeat_interleave(
                beam_scorer.num_beams, axis=0)
            input_ids_clone = input_ids.repeat_interleave(beam_scorer.num_beams,
                                                          axis=0)

        kwargs["use_cache"] = True

        with paddle.no_grad():
            output_beam_search = model.beam_search(
                input_ids_clone,
                beam_scorer,
                max_length=max_length +
                1 if self.is_encoder_decoder else max_length +
                input_ids.shape[-1],
                attention_mask=attention_mask_clone,
                logits_processors=logits_processor,
                diversity_rate=getattr(logits_process_kwargs, "diversity_rate",
                                       0.0),
                pad_token_id=getattr(
                    model, model.base_model_prefix).config["pad_token_id"],
                eos_token_id=getattr(
                    model, model.base_model_prefix).config["eos_token_id"],
                **kwargs,
            )
        return output_generate, output_beam_search

    def _group_beam_search_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
        beam_scorer,
        beam_kwargs,
        logits_processor,
        logits_process_kwargs,
    ):
        beam_kwargs.pop("diversity_rate")
        model.eval()
        with paddle.no_grad():
            output_generate = model.generate(
                input_ids,
                decode_strategy='beam_search',
                attention_mask=attention_mask,
                max_length=max_length,
                **beam_kwargs,
                **logits_process_kwargs,
            )

        # group_beam_search does not automatically interleave `batch_size` dim for `num_beams`
        kwargs = {}
        if self.is_encoder_decoder:
            encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                num_interleave=beam_scorer.num_beams,
            )
            kwargs["encoder_output"] = encoder_outputs
            input_ids_clone = input_ids_clone.repeat_interleave(
                beam_scorer.num_beams, axis=0)
            attention_mask_clone = attention_mask_clone.repeat_interleave(
                beam_scorer.num_beams, axis=0)
        else:
            attention_mask_clone = attention_mask.repeat_interleave(
                beam_scorer.num_beams, axis=0)
            input_ids_clone = input_ids.repeat_interleave(beam_scorer.num_beams,
                                                          axis=0)

        kwargs["use_cache"] = True

        with paddle.no_grad():
            output_group_beam_search = model.group_beam_search(
                input_ids_clone,
                beam_scorer,
                max_length=max_length +
                1 if self.is_encoder_decoder else max_length +
                input_ids.shape[-1],
                attention_mask=attention_mask_clone,
                logits_processors=logits_processor,
                pad_token_id=getattr(
                    model, model.base_model_prefix).config["pad_token_id"],
                eos_token_id=getattr(
                    model, model.base_model_prefix).config["eos_token_id"],
                **kwargs,
            )
        return output_generate, output_group_beam_search

    def test_greedy_generate(self):
        # check `generate()` and `greedy_search()` are equal
        for model_class in self.all_generative_model_classes.keys():
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config(
            )
            pretrained_model = self.all_generative_model_classes[model_class][
                0](**config)
            model = model_class(pretrained_model)
            model.eval()

            output_greedy, output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length)

            self.assertListEqual(output_greedy[0].tolist(),
                                 output_generate[0].tolist())

    def test_sample_generate(self):
        random.seed(128)
        np.random.seed(128)
        paddle.seed(128)

        for model_class in self.all_generative_model_classes.keys():
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config(
            )
            pretrained_model = self.all_generative_model_classes[model_class][
                0](**config)
            model = model_class(pretrained_model)
            model.eval()

            if self.is_encoder_decoder:
                max_length = 4

            process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                getattr(model, model.base_model_prefix).config["eos_token_id"],
                forced_bos_token_id=getattr(
                    getattr(model, model.base_model_prefix).config,
                    "forced_bos_token_id", None),
                forced_eos_token_id=getattr(
                    getattr(model, model.base_model_prefix).config,
                    "forced_eos_token_id", None),
                max_length=max_length,
            )
            logits_warper = self._get_warper_and_kwargs()

            # check `generate()` and `sample()` are equal
            output_sample, output_generate = self._sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                logits_processors=logits_processor,
                logits_warper=logits_warper,
                process_kwargs=process_kwargs,
            )
            self.assertListEqual(output_sample[0].tolist(),
                                 output_generate[0].tolist())

            # check `generate()` and `sample()` yield equal results for `num_return_sequences`
            output_sample, output_generate = self._sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=3,
                logits_processors=logits_processor,
                logits_warper=logits_warper,
                process_kwargs=process_kwargs,
            )
            self.assertListEqual(output_sample[0].tolist(),
                                 output_generate[0].tolist())

    def test_beam_search_generate(self):
        paddle.seed(100)
        for model_class in self.all_generative_model_classes.keys():
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config(
            )

            pretrained_model = self.all_generative_model_classes[model_class][
                0](**config)
            model = model_class(pretrained_model)
            model.eval()

            if self.is_encoder_decoder:
                max_length = 4

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                config["eos_token_id"],
                getattr(config, "forced_bos_token_id", None),
                getattr(config, "forced_eos_token_id", None),
                max_length,
            )
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(
                input_ids.shape[0],
                max_length + 1 if self.is_encoder_decoder else max_length +
                input_ids.shape[-1])

            # check `generate()` and `beam_search()` are equal
            output_generate, output_beam_search = self._beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
                logits_processor=logits_processor,
            )

            self.assertListEqual(output_generate[0].tolist(),
                                 output_beam_search[0].tolist())

            # check `generate()` and `beam_search()` are equal for `num_return_sequences`
            num_return_sequences = 2
            if self.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(
                input_ids.shape[0],
                max_length + 1 if self.is_encoder_decoder else max_length +
                input_ids.shape[-1],
                num_return_sequences=num_return_sequences)

            output_generate, output_beam_search = self._beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
                logits_processor=logits_processor,
            )
            self.assertListEqual(output_generate[0].tolist(),
                                 output_beam_search[0].tolist())

    def test_generate_without_input_ids(self):
        config, _, _, max_length = self._get_input_ids_and_config()

        # if no bos token id => cannot generate from None
        if config["bos_token_id"] is None:
            return

        for model_class in self.all_generative_model_classes.keys():
            pretrained_model = self.all_generative_model_classes[model_class][
                0](**config)
            model = model_class(pretrained_model)
            model.eval()

            output_ids_generate = model.generate(
                decode_strategy='greedy_search',
                max_length=max_length,
            )

            self.assertIsNotNone(output_ids_generate)

    def test_group_beam_search_generate(self):
        for model_class in self.all_generative_model_classes.keys():
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config(
            )

            pretrained_model = self.all_generative_model_classes[model_class][
                0](**config)
            model = model_class(pretrained_model)
            model.eval()

            if self.is_encoder_decoder:
                max_length = 4

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                config["eos_token_id"],
                getattr(config, "forced_bos_token_id", None),
                getattr(config, "forced_eos_token_id", None),
                max_length,
                diversity_rate=2.0,
            )

            # check `generate()` and `group_beam_search()` are equal
            beam_kwargs, beam_scorer = self._get_diverse_beam_scorer_and_kwargs(
                input_ids.shape[0],
                max_length + 1 if self.is_encoder_decoder else max_length +
                input_ids.shape[-1])
            output_generate, output_group_beam_search = self._group_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_processor=logits_processor,
                logits_process_kwargs=logits_process_kwargs,
            )
            self.assertListEqual(output_generate[0].tolist(),
                                 output_group_beam_search[0].tolist())

            # check `generate()` and `group_beam_search()` are equal for `num_return_sequences`
            num_return_sequences = 2
            if self.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_diverse_beam_scorer_and_kwargs(
                input_ids.shape[0],
                max_length + 1 if self.is_encoder_decoder else max_length +
                input_ids.shape[-1],
                num_return_sequences=num_return_sequences)
            output_generate, output_group_beam_search = self._group_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_processor=logits_processor,
                logits_process_kwargs=logits_process_kwargs,
            )
            self.assertListEqual(output_generate[0].tolist(),
                                 output_group_beam_search[0].tolist())

    def _check_sequence_inside_sequence(self, tensor_1, tensor_2):
        # check if tensor_1 inside tensor_2 or tensor_2 inside tensor_1.
        # set to same device. we don't care what device.

        if not isinstance(tensor_1, list):
            tensor_1 = tensor_1.cpu().tolist()
        if not isinstance(tensor_2, list):
            tensor_2 = tensor_2.cpu().tolist()

        in_order = len(tensor_1) <= len(tensor_2)
        longer = tensor_2 if in_order else tensor_1
        shorter = tensor_1 if in_order else tensor_2

        flag = False
        chunk_size = len(shorter)
        for chunk_idx in range(len(longer) - chunk_size + 1):
            subseq = longer[chunk_idx:chunk_idx + chunk_size]
            if subseq == shorter:
                flag = True
                break

        self.assertTrue(flag)


class UtilsFunctionsTest(unittest.TestCase):

    # tests whether the top_k_top_p function behaves as expected
    def test_top_k_top_p_filtering(self):
        logits = paddle.to_tensor(
            [
                [
                    8.2220991,  # 3rd highest value; idx. 0
                    -0.5620044,
                    5.23229752,
                    4.0386393,
                    -6.8798378,
                    -0.54785802,
                    -3.2012153,
                    2.92777176,
                    1.88171953,
                    7.35341276,
                    8.43207833,  # 2nd highest value; idx. 10
                    -9.85711836,
                    -5.96209236,
                    -1.13039161,
                    -7.1115294,
                    -0.8369633,
                    -5.3186408,
                    7.06427407,
                    0.81369344,
                    -0.82023817,
                    -5.9179796,
                    0.58813443,
                    -6.99778438,
                    4.71551189,
                    -0.18771637,
                    7.44020759,  # 4th highest value; idx. 25
                    9.38450987,  # 1st highest value; idx. 26
                    2.12662941,
                    -9.32562038,
                    2.35652522,
                ],  # cummulative prob of 4 highest values <= 0.6
                [
                    0.58425518,
                    4.53139238,
                    -5.57510464,
                    -6.28030699,
                    -7.19529503,
                    -4.02122551,
                    1.39337037,
                    -6.06707057,
                    1.59480517,
                    -9.643119,
                    0.03907799,
                    0.67231762,
                    -8.88206726,
                    6.27115922,  # 4th highest value; idx. 13
                    2.28520723,
                    4.82767506,
                    4.30421368,
                    8.8275313,  # 2nd highest value; idx. 17
                    5.44029958,
                    -4.4735794,
                    7.38579536,  # 3rd highest value; idx. 20
                    -2.91051663,
                    2.61946077,
                    -2.5674762,
                    -9.48959302,
                    -4.02922645,
                    -1.35416918,
                    9.67702323,  # 1st highest value; idx. 27
                    -5.89478553,
                    1.85370467,
                ],  # cummulative prob of 4 highest values <= 0.6
            ],
            dtype="float32",
        )

        non_inf_expected_idx = paddle.to_tensor(
            [[0, 0], [0, 10], [0, 25], [0, 26], [1, 13], [1, 17], [1, 20],
             [1, 27]],
            dtype="int64",
        )  # expected non filtered idx as noted above

        non_inf_expected_output = paddle.to_tensor(
            [
                8.2221,
                8.4321,
                7.4402,
                9.3845,
                6.2712,
                8.8275,
                7.3858,
                9.6770,
            ],  # expected non filtered values as noted above
            dtype="float32",
        )

        output = top_k_top_p_filtering(logits,
                                       top_k=10,
                                       top_p=0.6,
                                       min_tokens_to_keep=4)
        non_inf_output = output[output >= -10000]
        non_inf_idx = (output >= -10000).nonzero()

        self.assertTrue(
            paddle.allclose(non_inf_expected_output, non_inf_output,
                            atol=1e-12))
        self.assertTrue(paddle.all(paddle.eq(non_inf_expected_idx,
                                             non_inf_idx)))


class GenerationIntegrationTests(unittest.TestCase):

    @slow
    def test_diverse_beam_search(self):
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood.
        The celebrity couple announced the arrival of their son, Silas Randall Timberlake, in statements to People.
        "Silas was the middle name of Timberlake's maternal grandfather Bill Bomar, who died in 2012, while Randall is the musician's own middle name, as well as his father's first," People reports.
        The couple announced the pregnancy in January, with an Instagram post. It is the first baby for both."""

        bart_tokenizer = BartTokenizer.from_pretrained("bart-base")
        bart_model = BartForConditionalGeneration.from_pretrained("bart-base")
        input_ids = paddle.to_tensor(
            bart_tokenizer(article)["input_ids"]).unsqueeze([0])

        bart_model.eval()

        outputs = bart_model.generate(
            input_ids,
            decode_strategy="beam_search",
            num_beams=4,
            num_return_sequences=3,
            num_beam_groups=4,
            diversity_rate=2.0,
        )

        generated_text = bart_tokenizer.batch_decode(outputs,
                                                     skip_special_tokens=True)

    def test_max_length_backward_compat_greedy(self):
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""

        bart_tokenizer = BartTokenizer.from_pretrained("bart-base")
        bart_model = BartForConditionalGeneration.from_pretrained("bart-base")
        input_ids = paddle.to_tensor(
            bart_tokenizer(article)["input_ids"]).unsqueeze([0])

        bart_model.eval()

        max_length = 5
        input_ids = paddle.tile(input_ids, [2, 1])

        bos_token_id = getattr(bart_model, 'bos_token_id', None)
        eos_token_id = getattr(bart_model, 'eos_token_id', None)
        pad_token_id = getattr(bart_model, 'pad_token_id', None)
        decoder_start_token_id = getattr(bart_model, 'decoder_start_token_id',
                                         None)

        model_kwargs = {}

        model_kwargs[
            "attention_mask"] = bart_model.prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id)

        bart_model.is_encoder_decoder = hasattr(
            bart_model, 'encoder') and hasattr(bart_model, 'decoder')

        model_kwargs = bart_model.prepare_encoder_decoder_kwargs_for_generation(
            input_ids, model_kwargs)

        if "decoder_input_ids" in model_kwargs:
            input_ids = model_kwargs.pop("decoder_input_ids")
        else:
            input_ids = bart_model.prepare_decoder_input_ids_for_generation(
                input_ids, decoder_start_token_id, bos_token_id)

        model_kwargs["use_cache"] = True
        max_length += input_ids.shape[-1]

        bart_model.greedy_search(
            input_ids,
            max_length=max_length,
            pad_token_id=bart_model.bart.config["pad_token_id"],
            eos_token_id=bart_model.bart.config["eos_token_id"],
            logits_processors=None,
            **model_kwargs,
        )

    def test_max_length_backward_compat_sample(self):
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""

        bart_tokenizer = BartTokenizer.from_pretrained("bart-base")
        bart_model = BartForConditionalGeneration.from_pretrained("bart-base")
        input_ids = paddle.to_tensor(
            bart_tokenizer(article)["input_ids"]).unsqueeze([0])

        bart_model.eval()

        max_length = 5
        input_ids = paddle.tile(input_ids, [2, 1])

        bos_token_id = getattr(bart_model, 'bos_token_id', None)
        eos_token_id = getattr(bart_model, 'eos_token_id', None)
        pad_token_id = getattr(bart_model, 'pad_token_id', None)
        decoder_start_token_id = getattr(bart_model, 'decoder_start_token_id',
                                         None)

        model_kwargs = {}

        model_kwargs[
            "attention_mask"] = bart_model.prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id)

        bart_model.is_encoder_decoder = hasattr(
            bart_model, 'encoder') and hasattr(bart_model, 'decoder')

        model_kwargs = bart_model.prepare_encoder_decoder_kwargs_for_generation(
            input_ids, model_kwargs)

        if "decoder_input_ids" in model_kwargs:
            input_ids = model_kwargs.pop("decoder_input_ids")
        else:
            input_ids = bart_model.prepare_decoder_input_ids_for_generation(
                input_ids, decoder_start_token_id, bos_token_id)

        model_kwargs["use_cache"] = True
        max_length += input_ids.shape[-1]

        bart_model.sample(
            input_ids,
            max_length=max_length,
            pad_token_id=bart_model.bart.config["pad_token_id"],
            eos_token_id=bart_model.bart.config["eos_token_id"],
            logits_processors=None,
            top_k=4,
            **model_kwargs,
        )

    def test_max_length_backward_compat_beam_search(self):
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""

        bart_tokenizer = BartTokenizer.from_pretrained("bart-base")
        bart_model = BartForConditionalGeneration.from_pretrained("bart-base")
        input_ids = paddle.to_tensor(
            bart_tokenizer(article)["input_ids"]).unsqueeze([0])

        bart_model.eval()

        max_length = 5
        input_ids = paddle.tile(input_ids, [2, 1])

        bos_token_id = getattr(bart_model, 'bos_token_id', None)
        eos_token_id = getattr(bart_model, 'eos_token_id', None)
        pad_token_id = getattr(bart_model, 'pad_token_id', None)
        decoder_start_token_id = getattr(bart_model, 'decoder_start_token_id',
                                         None)

        model_kwargs = {}

        model_kwargs[
            "attention_mask"] = bart_model.prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id)

        bart_model.is_encoder_decoder = hasattr(
            bart_model, 'encoder') and hasattr(bart_model, 'decoder')

        model_kwargs = bart_model.prepare_encoder_decoder_kwargs_for_generation(
            input_ids, model_kwargs)

        if "decoder_input_ids" in model_kwargs:
            input_ids = model_kwargs.pop("decoder_input_ids")
        else:
            input_ids = bart_model.prepare_decoder_input_ids_for_generation(
                input_ids, decoder_start_token_id, bos_token_id)

        model_kwargs["use_cache"] = True
        max_length += input_ids.shape[-1]

        beam_scorer = BeamSearchScorer(batch_size=2,
                                       max_length=max_length,
                                       num_beams=2)

        input_ids, model_kwargs = bart_model.expand_inputs_for_generation(
            input_ids, expand_size=2, **model_kwargs)

        bart_model.beam_search(
            input_ids,
            num_beams=2,
            max_length=max_length,
            beam_scorer=beam_scorer,
            logits_processors=None,
            diversity_rate=0.0,
            pad_token_id=bart_model.bart.config["pad_token_id"],
            eos_token_id=bart_model.bart.config["eos_token_id"],
            **model_kwargs)

    def test_max_length_backward_compat_group_beam_search(self):
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""

        bart_tokenizer = BartTokenizer.from_pretrained("bart-base")
        bart_model = BartForConditionalGeneration.from_pretrained("bart-base")
        input_ids = paddle.to_tensor(
            bart_tokenizer(article)["input_ids"]).unsqueeze([0])

        bart_model.eval()

        max_length = 5
        input_ids = paddle.tile(input_ids, [2, 1])

        bos_token_id = getattr(bart_model, 'bos_token_id', None)
        eos_token_id = getattr(bart_model, 'eos_token_id', None)
        pad_token_id = getattr(bart_model, 'pad_token_id', None)
        decoder_start_token_id = getattr(bart_model, 'decoder_start_token_id',
                                         None)

        model_kwargs = {}

        model_kwargs[
            "attention_mask"] = bart_model.prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id)

        bart_model.is_encoder_decoder = hasattr(
            bart_model, 'encoder') and hasattr(bart_model, 'decoder')

        model_kwargs = bart_model.prepare_encoder_decoder_kwargs_for_generation(
            input_ids, model_kwargs)

        if "decoder_input_ids" in model_kwargs:
            input_ids = model_kwargs.pop("decoder_input_ids")
        else:
            input_ids = bart_model.prepare_decoder_input_ids_for_generation(
                input_ids, decoder_start_token_id, bos_token_id)

        model_kwargs["use_cache"] = True
        max_length += input_ids.shape[-1]

        diverse_beam_scorer = BeamSearchScorer(batch_size=2,
                                               max_length=max_length,
                                               num_beams=2,
                                               num_beam_groups=2)

        input_ids, model_kwargs = bart_model.expand_inputs_for_generation(
            input_ids, expand_size=2, **model_kwargs)

        bart_model.group_beam_search(
            input_ids,
            num_beams=2,
            max_length=max_length,
            beam_scorer=diverse_beam_scorer,
            logits_processors=None,
            pad_token_id=bart_model.bart.config["pad_token_id"],
            eos_token_id=bart_model.bart.config["eos_token_id"],
            **model_kwargs)

    def test_custom_logits_processor(self):
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""

        bart_tokenizer = BartTokenizer.from_pretrained("bart-base")
        bart_model = BartForConditionalGeneration.from_pretrained("bart-base")
        input_ids = paddle.to_tensor(
            bart_tokenizer(article)["input_ids"]).unsqueeze([0])

        bart_model.eval()

        logits_processor = LogitsProcessorList()
        # 1 means decoder_start_token.
        logits_processor.append(
            MinLengthLogitsProcessor(
                min_length=25 + 1,
                eos_token_id=bart_model.bart.config["forced_eos_token_id"]))

        bart_model.generate(input_ids,
                            decode_strategy="sampling",
                            top_k=1,
                            max_length=30,
                            logits_processors=logits_processor)

        bart_model.generate(input_ids,
                            decode_strategy="sampling",
                            top_k=1,
                            max_length=30,
                            min_length=25)

    # BART supports inputs_embeds
    # def test_encoder_decoder_generate_with_inputs_embeds(self):
    #     article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
    #     bart_tokenizer = BartTokenizer.from_pretrained("bart-base")
    #     bart_model = BartForConditionalGeneration.from_pretrained("bart-base")
    #     bart_model.eval()

    #     bart_model.bart.config["eos_token_id"] = None
    #     input_ids = paddle.to_tensor(bart_tokenizer(articles[0])["input_ids"]).unsqueeze([0])
    #     inputs_embeds = bart_model.get_input_embeddings()(input_ids)

    #     output_sequences = bart_model.generate(inputs_embeds=inputs_embeds)

    #     self.assertEqual(output_sequences.shape, (1, 5))

    def test_encoder_decoder_generate_attention_mask(self):
        articles = [
            "Timberlake",
            "Jessica Biel, welcome to parenthood among other things"
        ]
        bart_tokenizer = BartTokenizer.from_pretrained("bart-base")
        bart_model = BartForConditionalGeneration.from_pretrained("bart-base")
        bart_model.eval()

        input_ids = paddle.to_tensor(bart_tokenizer(
            articles[0])["input_ids"]).unsqueeze([0])
        input_ids_batched = paddle.to_tensor(
            bart_tokenizer(articles, padding=True)["input_ids"])

        output_sequences_batched = bart_model.generate(
            input_ids=input_ids_batched, decode_strategy="greedy_search")
        output_sequences = bart_model.generate(input_ids=input_ids,
                                               decode_strategy="greedy_search")

        batched_out = output_sequences_batched[1]
        out = output_sequences[1]

        diff = (batched_out - out).abs()

        self.assertTrue(diff.numpy() < 1e-6)
