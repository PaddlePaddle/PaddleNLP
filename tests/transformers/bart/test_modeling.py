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

import copy
import tempfile
import unittest
import numpy as np
import random
from parameterized import parameterized_class

from tests.testing_utils import slow

from ..test_generation_utils import GenerationTesterMixin
from ..test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from paddlenlp.transformers.tokenizer_utils_base import PaddingStrategy, TruncationStrategy

import paddle

from paddlenlp.transformers import (
    AutoModelForSequenceClassification,
    BartForConditionalGeneration,
    BartForQuestionAnswering,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
)
from paddlenlp.transformers.bart.modeling import BartDecoder, BartEncoder, shift_tokens_right


def prepare_bart_inputs_dict(
    config,
    input_ids,
    decoder_input_ids=None,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = paddle.cast(
            input_ids == config["pad_token_id"],
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
    if decoder_attention_mask is None:
        decoder_attention_mask = paddle.cast(
            decoder_input_ids == config["pad_token_id"],
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": attention_mask,
    }


class BartModelTester:

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

        # forcing a certain token to be generated, sets all other tokens to -inf
        # if however the token to be generated is already at -inf then it can lead token
        # `nan` values and thus break generation
        self.forced_eos_token_id = None

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length],
                               self.vocab_size,
                               dtype="int64")
        input_ids = paddle.clip(
            ids_tensor([self.batch_size, self.seq_length],
                       self.vocab_size,
                       dtype="int64"), 3)
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length],
                                       self.vocab_size,
                                       dtype="int64")

        config = self.get_config()
        inputs_dict = prepare_bart_inputs_dict(config, input_ids,
                                               decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.hidden_size,
            "num_encoder_layers": self.num_hidden_layers,
            "num_decoder_layers": self.num_hidden_layers,
            "encoder_attention_heads": self.num_attention_heads,
            "decoder_attention_heads": self.num_attention_heads,
            "encoder_ffn_dim": self.intermediate_size,
            "decoder_ffn_dim": self.intermediate_size,
            "dropout": self.hidden_dropout_prob,
            "attention_dropout": self.attention_probs_dropout_prob,
            "max_position_embeddings": self.max_position_embeddings,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
            "forced_eos_token_id": self.forced_eos_token_id,
        }

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(
            self, config, inputs_dict):
        encoder = BartModel(**config).get_encoder()
        decoder = BartModel(**config).get_decoder()

        encoder.eval()
        decoder.eval()

        input_ids = inputs_dict["input_ids"]
        decoder_input_ids = paddle.zeros_like(
            input_ids[:, :1],
            dtype="int64") + BartModel(**config).decoder_start_token_id

        attention_mask = inputs_dict["attention_mask"]
        decoder_attention_mask = paddle.zeros([input_ids.shape[0], 1, 1, 1],
                                              dtype=paddle.get_default_dtype())

        encoder_output = encoder(input_ids,
                                 attention_mask,
                                 return_dict=self.parent.return_dict)
        origin_cache = decoder.decoder.gen_cache(encoder_output)
        outputs = decoder(decoder_input_ids,
                          decoder_attention_mask,
                          encoder_output,
                          attention_mask,
                          cache=origin_cache,
                          return_dict=self.parent.return_dict)

        output, cache = outputs[:2]

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3),
                                 config["vocab_size"],
                                 dtype="int64")
        next_attn_mask = paddle.zeros([self.batch_size, 1, 1, 3],
                                      dtype=paddle.get_default_dtype())

        # append to next input_ids and
        next_input_ids = paddle.concat([decoder_input_ids, next_tokens],
                                       axis=-1)
        next_attention_mask = paddle.concat(
            [decoder_attention_mask, next_attn_mask], axis=-1)

        output_from_no_past = decoder(next_input_ids,
                                      next_attention_mask,
                                      encoder_output,
                                      attention_mask,
                                      return_dict=self.parent.return_dict)
        if self.parent.return_dict:
            output_from_no_past = output_from_no_past[0]
        output_from_past, _ = decoder(next_tokens,
                                      next_attention_mask,
                                      encoder_output,
                                      attention_mask,
                                      cache=cache,
                                      return_dict=self.parent.return_dict)[:2]

        # select random slice
        random_slice_idx = ids_tensor((1, ),
                                      output_from_past.shape[-1],
                                      dtype="int64").item()
        output_from_no_past_slice = output_from_no_past[:, -3:,
                                                        random_slice_idx].detach(
                                                        )
        output_from_past_slice = output_from_past[:, :,
                                                  random_slice_idx].detach()

        self.parent.assertTrue(
            output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(
            paddle.allclose(output_from_past_slice,
                            output_from_no_past_slice,
                            atol=1e-3))


@parameterized_class(("return_dict", "use_labels"), [
    [False, False],
    [False, True],
    [True, False],
    [True, True],
])
class BartHeadTests(unittest.TestCase):
    vocab_size = 99
    use_labels = False
    return_dict = False

    def _get_config_and_data(self):
        input_ids = paddle.to_tensor(
            [
                [71, 82, 18, 33, 46, 91, 2],
                [68, 34, 26, 58, 30, 82, 2],
                [5, 97, 17, 39, 94, 40, 2],
                [76, 83, 94, 25, 70, 78, 2],
                [87, 59, 41, 35, 48, 66, 2],
                [55, 13, 16, 58, 5, 2, 1],  # note padding
                [64, 27, 31, 51, 12, 75, 2],
                [52, 64, 86, 17, 83, 39, 2],
                [48, 61, 9, 24, 71, 82, 2],
                [26, 1, 60, 48, 22, 13, 2],
                [21, 5, 62, 28, 14, 76, 2],
                [45, 98, 37, 86, 59, 48, 2],
                [70, 70, 50, 9, 28, 0, 2],
            ],
            dtype="int64",
        )

        batch_size = input_ids.shape[0]
        config = {
            "vocab_size": self.vocab_size,
            "d_model": 24,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "encoder_attention_heads": 2,
            "decoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "decoder_ffn_dim": 32,
            "max_position_embeddings": 48,
            "eos_token_id": 2,
            "pad_token_id": 1,
            "bos_token_id": 0,
        }
        return config, input_ids, batch_size

    def test_sequence_classification_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        bart_model = BartModel(**config)
        num_labels = 2
        labels = _long_tensor([1] * batch_size) if self.use_labels else None
        model = BartForSequenceClassification(bart_model, num_labels=num_labels)
        outputs = model(input_ids=input_ids,
                        decoder_input_ids=input_ids,
                        labels=labels,
                        return_dict=self.return_dict)
        expected_shape = [batch_size, num_labels]
        if self.use_labels:
            self.assertIsInstance(outputs[0].item(), float)  # test loss
            self.assertEqual(outputs[1].shape, expected_shape)  # test logits
        elif isinstance(outputs, paddle.Tensor):
            self.assertEqual(outputs.shape, expected_shape)
        else:
            self.assertEqual(outputs[0].shape, expected_shape)

    def test_question_answering_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        sequence_labels = ids_tensor([batch_size],
                                     2) if self.use_labels else None
        bart_model = BartModel(**config)
        model = BartForQuestionAnswering(bart_model)
        outputs = model(input_ids=input_ids,
                        start_positions=sequence_labels,
                        end_positions=sequence_labels,
                        return_dict=self.return_dict)

        if self.use_labels:
            loss, start_logits, end_logits = outputs[:3]
            self.assertIsInstance(loss.item(), float)
        else:
            start_logits, end_logits = outputs[:2]
        self.assertEqual(start_logits.shape, input_ids.shape)
        self.assertEqual(end_logits.shape, input_ids.shape)

    def test_lm_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        bart_model = BartModel(**config)
        lm_labels = ids_tensor([batch_size, input_ids.shape[1]],
                               self.vocab_size) if self.use_labels else None
        lm_model = BartForConditionalGeneration(bart_model)
        outputs = lm_model(input_ids=input_ids,
                           labels=lm_labels,
                           return_dict=self.return_dict)
        expected_shape = [batch_size, input_ids.shape[1], config["vocab_size"]]
        if self.use_labels:
            self.assertIsInstance(outputs[0].item(), float)
            self.assertEqual(outputs[1].shape, expected_shape)
        elif isinstance(outputs, paddle.Tensor):
            self.assertEqual(outputs.shape, expected_shape)
        else:
            self.assertEqual(outputs[0].shape, expected_shape)

    def test_lm_uneven_forward(self):
        config = {
            "vocab_size": self.vocab_size,
            "d_model": 14,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "encoder_attention_heads": 2,
            "decoder_attention_heads": 2,
            "encoder_ffn_dim": 8,
            "decoder_ffn_dim": 8,
            "max_position_embeddings": 48,
        }
        bart_model = BartModel(**config)
        lm_model = BartForConditionalGeneration(bart_model)
        context = paddle.to_tensor(
            [[71, 82, 18, 33, 46, 91, 2], [68, 34, 26, 58, 30, 2, 1]],
            dtype="int64")
        summary = paddle.to_tensor([[82, 71, 82, 18, 2], [58, 68, 2, 1, 1]],
                                   dtype="int64")
        outputs = lm_model(input_ids=context,
                           decoder_input_ids=summary,
                           labels=summary if self.use_labels else None,
                           return_dict=self.return_dict)
        expected_shape = summary.shape
        expected_shape.append(config["vocab_size"])
        if self.use_labels:
            self.assertIsInstance(outputs[0].item(), float)
        elif isinstance(outputs, paddle.Tensor):
            self.assertEqual(outputs.shape, expected_shape)
        else:
            self.assertEqual(outputs[0].shape, expected_shape)

    def test_generate_beam_search(self):
        input_ids = paddle.to_tensor([[71, 82, 2], [68, 34, 2]], dtype="int64")
        config = {
            "vocab_size": self.vocab_size,
            "d_model": 24,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "encoder_attention_heads": 2,
            "decoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "decoder_ffn_dim": 32,
            "max_position_embeddings": 48,
            "eos_token_id": 2,
            "pad_token_id": 1,
            "bos_token_id": 0,
        }
        bart_model = BartModel(**config)
        lm_model = BartForConditionalGeneration(bart_model)
        lm_model.eval()

        max_length = 5
        generated_ids = lm_model.generate(
            input_ids,
            decode_strategy="sampling",
            num_return_sequences=1,
            max_length=max_length,
            top_k=4,
        )[0]
        self.assertEqual(generated_ids.shape, [input_ids.shape[0], max_length])

    def test_shift_tokens_right(self):
        input_ids = paddle.to_tensor(
            [[71, 82, 18, 33, 2, 1, 1], [68, 34, 26, 58, 30, 82, 2]],
            dtype="int64")
        shifted = shift_tokens_right(input_ids, 2)
        n_pad_before = paddle.equal(input_ids, 1).sum().numpy()
        n_pad_after = paddle.equal(shifted, 1).sum().numpy()
        self.assertEqual(shifted.shape, input_ids.shape)
        self.assertEqual(n_pad_after, n_pad_before - 1)
        self.assertTrue(paddle.equal(shifted[:, 0], 2).all())

    @slow
    def test_tokenization(self):
        tokenizer = BartTokenizer.from_pretrained("bart-large")
        examples = [" Hello world",
                    " DomDramg"]  # need leading spaces for equality
        fairseq_results = [
            paddle.to_tensor([0, 20920, 232, 2]),
            paddle.to_tensor([0, 11349, 495, 4040, 571, 2]),
        ]
        for ex, desired_result in zip(examples, fairseq_results):
            bart_toks = tokenizer.encode(
                ex, return_tensors="pd")["input_ids"].squeeze()
            assert_tensors_close(desired_result, bart_toks, prefix=ex)


class BartModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = BartModel

    all_model_classes = (BartModel, BartForConditionalGeneration,
                         BartForSequenceClassification,
                         BartForQuestionAnswering)

    all_generative_model_classes = {
        BartForConditionalGeneration: (BartModel, "bart")
    }
    is_encoder_decoder = True
    fx_compatible = True
    test_pruning = False
    test_missing_keys = False
    use_labels = False
    return_dict = False

    def setUp(self):
        self.model_tester = BartModelTester(self)
        random.seed(128)
        np.random.seed(128)
        paddle.seed(128)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(
            *config_and_inputs)


def assert_tensors_close(a, b, atol=1e-12, prefix=""):
    """If tensors have different shapes, different values or a and b are not both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if paddle.allclose(a.astype("float32"), b.astype("float32"), atol=atol):
            return True
        raise
    except Exception:
        pct_different = ((a - b).abs() > atol).astype("float").mean().item()
        if a.numel() > 100:
            msg = f"tensor values are {pct_different:.1%} percent different."
        else:
            msg = f"{a} != {b}"
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


def _long_tensor(tok_lst):
    return paddle.to_tensor(tok_lst, dtype="int64")


@slow
class FastIntegrationTests(unittest.TestCase):
    """These tests are useful for debugging since they operate on a model with 1 encoder layer and 1 decoder layer."""

    def tok(self):
        return BartTokenizer.from_pretrained("bart-large")

    def bart_base(self):
        return BartForConditionalGeneration.from_pretrained("bart-base")

    def test_bart_base_generation(self):
        model = self.bart_base()
        model.eval()
        tok = self.tok()
        ARTICLE = (
            "The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
            " Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The"
            " formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based."
            " The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its"
            ' jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East'
            ' Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the'
            " situation in Palestinian territories, paving the way for possible war crimes investigations against"
            " Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and"
            " the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the"
            " body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a"
            ' move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the'
            ' world is also a step closer to ending a long era of impunity and injustice," he said, according to an'
            ' ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge'
            " Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the"
            ' Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine'
            " acquires all the rights as well as responsibilities that come with being a State Party to the Statute."
            ' These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights'
            ' Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should'
            " immediately end their pressure, and countries that support universal acceptance of the court's treaty"
            ' should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the'
            " group. \"What's objectionable is the attempts to undermine international justice, not Palestine's"
            ' decision to join a treaty to which over 100 countries around the world are members." In January, when'
            " the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an"
            ' outrage, saying the court was overstepping its boundaries. The United States also said it "strongly"'
            " disagreed with the court's decision. \"As we have said repeatedly, we do not believe that Palestine is a"
            ' state and therefore we do not believe that it is eligible to join the ICC," the State Department said in'
            ' a statement. It urged the warring sides to resolve their differences through direct negotiations. "We'
            ' will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace,"'
            " it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the"
            ' territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the'
            " court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou"
            ' Bensouda said her office would "conduct its analysis in full independence and impartiality." The war'
            " between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry"
            " will include alleged war crimes committed since June. The International Criminal Court was set up in"
            " 2002 to prosecute genocide, crimes against humanity and war crimes."
        )
        EXPECTED = (
            'The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC\'s founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war crimes investigations against Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and the United States, neither of which is an ICC member, opposed the Palestinians\' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday\'s ceremony, said it was a move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the world is also a step closer to ending a long era of impunity and injustice," he said, according to an ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Rome Statute today, the world is also a step closer to ending a long era of impunity and injustice," he said, according to an ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Statute. These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should immediately end their pressure, and countries that support universal acceptance of the court\'s treaty should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the group. "What\'s objectionable is the attempts to undermine international justice, not Palestine\'s decision to join a treaty to which over 100 countries around the world are members." In January, when the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was overstepping its boundaries. The United States also said it "strongly" disagreed with the court\'s decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC," the State Department said in a statement. It urged the warring sides to resolve their differences through direct negotiations. "We will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace," it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou Bensouda said her office would "conduct its analysis in full independence and impartiality." The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry will include alleged war crimes committed since June. The International Criminal Court was set up in 2002 to prosecute genocide, crimes against humanity and war crimes.'
        )

        dct = tok(ARTICLE, return_tensors="pd")

        dct.pop("token_type_ids")
        generated_ids, _ = model.generate(**dct,
                                          num_beams=4,
                                          decode_strategy="beam_search",
                                          max_length=1024)
        result = tok.batch_decode(generated_ids, skip_special_tokens=True)[0]
        assert EXPECTED == result, f"{EXPECTED}\n{result}"

    def test_xsum_1_1_batch_generation(self):
        # test batch
        batch = self.tok()(
            [
                "The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
                " Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories."
                " The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is"
                " based. The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted"
                ' its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including'
                ' East Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination'
                " into the situation in Palestinian territories, paving the way for possible war crimes investigations"
                " against Israelis. As members of the court, Palestinians may be subject to counter-charges as well."
                " Israel and the United States, neither of which is an ICC member, opposed the Palestinians' efforts"
                " to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony,"
                ' said it was a move toward greater justice. "As Palestine formally becomes a State Party to the Rome'
                ' Statute today, the world is also a step closer to ending a long era of impunity and injustice," he'
                ' said, according to an ICC news release. "Indeed, today brings us closer to our shared goals of'
                ' justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was'
                ' just the first step for the Palestinians. "As the Rome Statute today enters into force for the State'
                " of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a"
                ' State Party to the Statute. These are substantive commitments, which cannot be taken lightly," she'
                ' said. Rights group Human Rights Watch welcomed the development. "Governments seeking to penalize'
                " Palestine for joining the ICC should immediately end their pressure, and countries that support"
                " universal acceptance of the court's treaty should speak out to welcome its membership,\" said"
                " Balkees Jarrah, international justice counsel for the group. \"What's objectionable is the attempts"
                " to undermine international justice, not Palestine's decision to join a treaty to which over 100"
                ' countries around the world are members." In January, when the preliminary ICC examination was'
                " opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was"
                ' overstepping its boundaries. The United States also said it "strongly" disagreed with the court\'s'
                ' decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we'
                ' do not believe that it is eligible to join the ICC," the State Department said in a statement. It'
                ' urged the warring sides to resolve their differences through direct negotiations. "We will continue'
                ' to oppose actions against Israel at the ICC as counterproductive to the cause of peace," it said.'
                " But the ICC begs to differ with the definition of a state for its purposes and refers to the"
                ' territories as "Palestine." While a preliminary examination is not a formal investigation, it allows'
                " the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor"
                ' Fatou Bensouda said her office would "conduct its analysis in full independence and impartiality."'
                " The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The"
                " inquiry will include alleged war crimes committed since June. The International Criminal Court was"
                " set up in 2002 to prosecute genocide, crimes against humanity and war crimes.",
                "The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted"
                " Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor"
                ' Brice Robin told CNN that "so far no videos were used in the crash investigation." He added, "A'
                " person who has such a video needs to immediately give it to the investigators.\" Robin's comments"
                " follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video"
                " showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the"
                " French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was"
                " recovered from a phone at the wreckage site. The two publications described the supposed video, but"
                " did not post it on their websites. The publications said that they watched the video, which was"
                " found by a source close to the investigation. \"One can hear cries of 'My God' in several"
                ' languages," Paris Match reported. "Metallic banging can also be heard more than three times, perhaps'
                " of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy"
                ' shake, stronger than the others, the screaming intensifies. Then nothing." "It is a very disturbing'
                " scene,\" said Julian Reichelt, editor-in-chief of Bild online. An official with France's accident"
                " investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col. Jean-Marc"
                " Menichini, a French Gendarmerie spokesman in charge of communications on rescue efforts around the"
                ' Germanwings crash site, told CNN that the reports were "completely wrong" and "unwarranted." Cell'
                ' phones have been collected at the site, he said, but that they "hadn\'t been exploited yet."'
                " Menichini said he believed the cell phones would need to be sent to the Criminal Research Institute"
                " in Rosny sous-Bois, near Paris, in order to be analyzed by specialized technicians working"
                " hand-in-hand with investigators. But none of the cell phones found so far have been sent to the"
                " institute, Menichini said. Asked whether staff involved in the search could have leaked a memory"
                ' card to the media, Menichini answered with a categorical "no." Reichelt told "Erin Burnett:'
                ' Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match are'
                ' "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered'
                ' cell phones from the crash site after Bild and Paris Match published their reports. "That is'
                " something we did not know before. ... Overall we can say many things of the investigation weren't"
                ' revealed by the investigation at the beginning," he said. What was mental state of Germanwings'
                " co-pilot? German airline Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled"
                " depression years before he took the controls of Germanwings Flight 9525, which he's accused of"
                " deliberately crashing last week in the French Alps. Lubitz told his Lufthansa flight training school"
                ' in 2009 that he had a "previous episode of severe depression," the airline said Tuesday. Email'
                " correspondence between Lubitz and the school discovered in an internal investigation, Lufthansa"
                " said, included medical documents he submitted in connection with resuming his flight training. The"
                " announcement indicates that Lufthansa, the parent company of Germanwings, knew of Lubitz's battle"
                " with depression, allowed him to continue training and ultimately put him in the cockpit. Lufthansa,"
                " whose CEO Carsten Spohr previously said Lubitz was 100% fit to fly, described its statement Tuesday"
                ' as a "swift and seamless clarification" and said it was sharing the information and documents --'
                " including training and medical records -- with public prosecutors. Spohr traveled to the crash site"
                " Wednesday, where recovery teams have been working for the past week to recover human remains and"
                " plane debris scattered across a steep mountainside. He saw the crisis center set up in"
                " Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash site, where grieving"
                " families have left flowers at a simple stone memorial. Menichini told CNN late Tuesday that no"
                " visible human remains were left at the site but recovery teams would keep searching. French"
                " President Francois Hollande, speaking Tuesday, said that it should be possible to identify all the"
                " victims using DNA analysis by the end of the week, sooner than authorities had previously suggested."
                " In the meantime, the recovery of the victims' personal belongings will start Wednesday, Menichini"
                " said. Among those personal belongings could be more cell phones belonging to the 144 passengers and"
                " six crew on board. Check out the latest from our correspondents . The details about Lubitz's"
                " correspondence with the flight school during his training were among several developments as"
                " investigators continued to delve into what caused the crash and Lubitz's possible motive for"
                " downing the jet. A Lufthansa spokesperson told CNN on Tuesday that Lubitz had a valid medical"
                ' certificate, had passed all his examinations and "held all the licenses required." Earlier, a'
                " spokesman for the prosecutor's office in Dusseldorf, Christoph Kumpa, said medical records reveal"
                " Lubitz suffered from suicidal tendencies at some point before his aviation career and underwent"
                " psychotherapy before he got his pilot's license. Kumpa emphasized there's no evidence suggesting"
                " Lubitz was suicidal or acting aggressively before the crash. Investigators are looking into whether"
                " Lubitz feared his medical condition would cause him to lose his pilot's license, a European"
                ' government official briefed on the investigation told CNN on Tuesday. While flying was "a big part'
                " of his life,\" the source said, it's only one theory being considered. Another source, a law"
                " enforcement official briefed on the investigation, also told CNN that authorities believe the"
                " primary motive for Lubitz to bring down the plane was that he feared he would not be allowed to fly"
                " because of his medical problems. Lubitz's girlfriend told investigators he had seen an eye doctor"
                " and a neuropsychologist, both of whom deemed him unfit to work recently and concluded he had"
                " psychological issues, the European government official said. But no matter what details emerge about"
                " his previous mental health struggles, there's more to the story, said Brian Russell, a forensic"
                ' psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the'
                " fact that maybe they weren't going to keep doing their job and they're upset about that and so"
                ' they\'re suicidal," he said. "But there is no mental illness that explains why somebody then feels'
                " entitled to also take that rage and turn it outward on 149 other people who had nothing to do with"
                " the person's problems.\" Germanwings crash compensation: What we know . Who was the captain of"
                " Germanwings Flight 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from"
                " Dusseldorf, while Laura Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff,"
                " Antonia Mortensen, Sandrine Amiel and Anna-Maja Rappard contributed to this report.",
            ],
            return_tensors="pd",
            padding="longest",
            truncation=True,
        )
        model = self.bart_base()
        model.eval()

        generated_ids, _ = model.generate(**batch,
                                          num_beams=4,
                                          decode_strategy="beam_search")
        result = self.tok().batch_decode(generated_ids,
                                         skip_special_tokens=True)
        assert (
            result[0] ==
            "The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a"
        )
        assert (
            result[1] ==
            "The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted Wednesday that"
        )


class BartModelIntegrationTests(unittest.TestCase):

    def default_tokenizer(self):
        return BartTokenizer.from_pretrained("bart-large")

    @slow
    def test_inference_no_head(self):
        model = BartModel.from_pretrained("bart-large")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]],
            dtype="int64")

        attention_mask = paddle.cast(
            input_ids == model.config["pad_token_id"],
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
        with paddle.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        expected_shape = [1, 11, 1024]
        self.assertEqual(output[0].shape, expected_shape)

    @slow
    def test_cnn_summarization_same_as_fairseq(self):

        model = BartForConditionalGeneration.from_pretrained("bart-large")
        model.eval()
        tok = BartTokenizer.from_pretrained("bart-large")

        FRANCE_ARTICLE = (  # @noq
            " Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings"
            " Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane."
            ' Marseille prosecutor Brice Robin told CNN that "so far no videos were used in the crash investigation."'
            ' He added, "A person who has such a video needs to immediately give it to the investigators." Robin\'s'
            " comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video"
            " showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French"
            " Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a"
            " phone at the wreckage site. The two publications described the supposed video, but did not post it on"
            " their websites. The publications said that they watched the video, which was found by a source close to"
            " the investigation. \"One can hear cries of 'My God' in several languages,\" Paris Match reported."
            ' "Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the'
            " cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the"
            ' screaming intensifies. Then nothing." "It is a very disturbing scene," said Julian Reichelt,'
            " editor-in-chief of Bild online. An official with France's accident investigation agency, the BEA, said"
            " the agency is not aware of any such video. Lt. Col. Jean-Marc Menichini, a French Gendarmerie spokesman"
            " in charge of communications on rescue efforts around the Germanwings crash site, told CNN that the"
            ' reports were "completely wrong" and "unwarranted." Cell phones have been collected at the site, he said,'
            ' but that they "hadn\'t been exploited yet." Menichini said he believed the cell phones would need to be'
            " sent to the Criminal Research Institute in Rosny sous-Bois, near Paris, in order to be analyzed by"
            " specialized technicians working hand-in-hand with investigators. But none of the cell phones found so"
            " far have been sent to the institute, Menichini said. Asked whether staff involved in the search could"
            ' have leaked a memory card to the media, Menichini answered with a categorical "no." Reichelt told "Erin'
            ' Burnett: Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match'
            ' are "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered'
            ' cell phones from the crash site after Bild and Paris Match published their reports. "That is something'
            " we did not know before. ... Overall we can say many things of the investigation weren't revealed by the"
            ' investigation at the beginning," he said. What was mental state of Germanwings co-pilot? German airline'
            " Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled depression years before he took the"
            " controls of Germanwings Flight 9525, which he's accused of deliberately crashing last week in the"
            ' French Alps. Lubitz told his Lufthansa flight training school in 2009 that he had a "previous episode of'
            ' severe depression," the airline said Tuesday. Email correspondence between Lubitz and the school'
            " discovered in an internal investigation, Lufthansa said, included medical documents he submitted in"
            " connection with resuming his flight training. The announcement indicates that Lufthansa, the parent"
            " company of Germanwings, knew of Lubitz's battle with depression, allowed him to continue training and"
            " ultimately put him in the cockpit. Lufthansa, whose CEO Carsten Spohr previously said Lubitz was 100%"
            ' fit to fly, described its statement Tuesday as a "swift and seamless clarification" and said it was'
            " sharing the information and documents -- including training and medical records -- with public"
            " prosecutors. Spohr traveled to the crash site Wednesday, where recovery teams have been working for the"
            " past week to recover human remains and plane debris scattered across a steep mountainside. He saw the"
            " crisis center set up in Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash"
            " site, where grieving families have left flowers at a simple stone memorial. Menichini told CNN late"
            " Tuesday that no visible human remains were left at the site but recovery teams would keep searching."
            " French President Francois Hollande, speaking Tuesday, said that it should be possible to identify all"
            " the victims using DNA analysis by the end of the week, sooner than authorities had previously suggested."
            " In the meantime, the recovery of the victims' personal belongings will start Wednesday, Menichini said."
            " Among those personal belongings could be more cell phones belonging to the 144 passengers and six crew"
            " on board. Check out the latest from our correspondents . The details about Lubitz's correspondence with"
            " the flight school during his training were among several developments as investigators continued to"
            " delve into what caused the crash and Lubitz's possible motive for downing the jet. A Lufthansa"
            " spokesperson told CNN on Tuesday that Lubitz had a valid medical certificate, had passed all his"
            ' examinations and "held all the licenses required." Earlier, a spokesman for the prosecutor\'s office in'
            " Dusseldorf, Christoph Kumpa, said medical records reveal Lubitz suffered from suicidal tendencies at"
            " some point before his aviation career and underwent psychotherapy before he got his pilot's license."
            " Kumpa emphasized there's no evidence suggesting Lubitz was suicidal or acting aggressively before the"
            " crash. Investigators are looking into whether Lubitz feared his medical condition would cause him to"
            " lose his pilot's license, a European government official briefed on the investigation told CNN on"
            ' Tuesday. While flying was "a big part of his life," the source said, it\'s only one theory being'
            " considered. Another source, a law enforcement official briefed on the investigation, also told CNN that"
            " authorities believe the primary motive for Lubitz to bring down the plane was that he feared he would"
            " not be allowed to fly because of his medical problems. Lubitz's girlfriend told investigators he had"
            " seen an eye doctor and a neuropsychologist, both of whom deemed him unfit to work recently and concluded"
            " he had psychological issues, the European government official said. But no matter what details emerge"
            " about his previous mental health struggles, there's more to the story, said Brian Russell, a forensic"
            ' psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the fact'
            " that maybe they weren't going to keep doing their job and they're upset about that and so they're"
            ' suicidal," he said. "But there is no mental illness that explains why somebody then feels entitled to'
            " also take that rage and turn it outward on 149 other people who had nothing to do with the person's"
            ' problems." Germanwings crash compensation: What we know . Who was the captain of Germanwings Flight'
            " 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from Dusseldorf, while Laura"
            " Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff, Antonia Mortensen, Sandrine"
            " Amiel and Anna-Maja Rappard contributed to this report.")

        SHORTER_ARTICLE = (
            " (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
            " Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The"
            " formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based."
            " The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its"
            ' jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East'
            ' Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the'
            " situation in Palestinian territories, paving the way for possible war crimes investigations against"
            " Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and"
            " the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the"
            " body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a"
            ' move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the'
            ' world is also a step closer to ending a long era of impunity and injustice," he said, according to an'
            ' ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge'
            " Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the"
            ' Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine'
            " acquires all the rights as well as responsibilities that come with being a State Party to the Statute."
            ' These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights'
            ' Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should'
            " immediately end their pressure, and countries that support universal acceptance of the court's treaty"
            ' should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the'
            " group. \"What's objectionable is the attempts to undermine international justice, not Palestine's"
            ' decision to join a treaty to which over 100 countries around the world are members." In January, when'
            " the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an"
            ' outrage, saying the court was overstepping its boundaries. The United States also said it "strongly"'
            " disagreed with the court's decision. \"As we have said repeatedly, we do not believe that Palestine is a"
            ' state and therefore we do not believe that it is eligible to join the ICC," the State Department said in'
            ' a statement. It urged the warring sides to resolve their differences through direct negotiations. "We'
            ' will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace,"'
            " it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the"
            ' territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the'
            " court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou"
            ' Bensouda said her office would "conduct its analysis in full independence and impartiality." The war'
            " between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry"
            " will include alleged war crimes committed since June. The International Criminal Court was set up in"
            " 2002 to prosecute genocide, crimes against humanity and war crimes. CNN's Vasco Cotovio, Kareem Khadder"
            " and Faith Karimi contributed to this report.")

        # The below article tests that we don't add any hypotheses outside of the top n_beams
        IRAN_ARTICLE = (
            " (CNN)The United States and its negotiating partners reached a very strong framework agreement with Iran"
            " in Lausanne, Switzerland, on Thursday that limits Iran's nuclear program in such a way as to effectively"
            " block it from building a nuclear weapon. Expect pushback anyway, if the recent past is any harbinger."
            " Just last month, in an attempt to head off such an agreement, House Speaker John Boehner invited Israeli"
            " Prime Minister Benjamin Netanyahu to preemptively blast it before Congress, and 47 senators sent a"
            " letter to the Iranian leadership warning them away from a deal. The debate that has already begun since"
            " the announcement of the new framework will likely result in more heat than light. It will not be helped"
            " by the gathering swirl of dubious assumptions and doubtful assertions. Let us address some of these: ."
            " The most misleading assertion, despite universal rejection by experts, is that the negotiations'"
            " objective at the outset was the total elimination of any nuclear program in Iran. That is the position"
            " of Netanyahu and his acolytes in the U.S. Congress. But that is not and never was the objective. If it"
            " had been, there would have been no Iranian team at the negotiating table. Rather, the objective has"
            " always been to structure an agreement or series of agreements so that Iran could not covertly develop a"
            " nuclear arsenal before the United States and its allies could respond. The new framework has exceeded"
            " expectations in achieving that goal. It would reduce Iran's low-enriched uranium stockpile, cut by"
            " two-thirds its number of installed centrifuges and implement a rigorous inspection regime. Another"
            " dubious assumption of opponents is that the Iranian nuclear program is a covert weapons program. Despite"
            " sharp accusations by some in the United States and its allies, Iran denies having such a program, and"
            " U.S. intelligence contends that Iran has not yet made the decision to build a nuclear weapon. Iran's"
            " continued cooperation with International Atomic Energy Agency inspections is further evidence on this"
            " point, and we'll know even more about Iran's program in the coming months and years because of the deal."
            " In fact, the inspections provisions that are part of this agreement are designed to protect against any"
            " covert action by the Iranians. What's more, the rhetoric of some members of Congress has implied that"
            " the negotiations have been between only the United States and Iran (i.e., the 47 senators' letter"
            " warning that a deal might be killed by Congress or a future president). This of course is not the case."
            " The talks were between Iran and the five permanent members of the U.N. Security Council (United States,"
            " United Kingdom, France, China and Russia) plus Germany, dubbed the P5+1. While the United States has"
            " played a leading role in the effort, it negotiated the terms alongside its partners. If the agreement"
            " reached by the P5+1 is rejected by Congress, it could result in an unraveling of the sanctions on Iran"
            " and threaten NATO cohesion in other areas. Another questionable assertion is that this agreement"
            " contains a sunset clause, after which Iran will be free to do as it pleases. Again, this is not the"
            " case. Some of the restrictions on Iran's nuclear activities, such as uranium enrichment, will be eased"
            " or eliminated over time, as long as 15 years. But most importantly, the framework agreement includes"
            " Iran's ratification of the Additional Protocol, which allows IAEA inspectors expanded access to nuclear"
            " sites both declared and nondeclared. This provision will be permanent. It does not sunset. Thus, going"
            " forward, if Iran decides to enrich uranium to weapons-grade levels, monitors will be able to detect such"
            " a move in a matter of days and alert the U.N. Security Council. Many in Congress have said that the"
            ' agreement should be a formal treaty requiring the Senate to "advise and consent." But the issue is not'
            " suited for a treaty. Treaties impose equivalent obligations on all signatories. For example, the New"
            " START treaty limits Russia and the United States to 1,550 deployed strategic warheads. But any agreement"
            " with Iran will not be so balanced.  The restrictions and obligations in the final framework agreement"
            " will be imposed almost exclusively on Iran. The P5+1 are obligated only to ease and eventually remove"
            " most but not all economic sanctions, which were imposed as leverage to gain this final deal. Finally"
            " some insist that any agreement must address Iranian missile programs, human rights violations or support"
            " for Hamas or Hezbollah.  As important as these issues are, and they must indeed be addressed, they are"
            " unrelated to the most important aim of a nuclear deal: preventing a nuclear Iran.  To include them in"
            " the negotiations would be a poison pill. This agreement should be judged on its merits and on how it"
            " affects the security of our negotiating partners and allies, including Israel. Those judgments should be"
            " fact-based, not based on questionable assertions or dubious assumptions."
        )

        ARTICLE_SUBWAY = (
            " New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A"
            " year later, she got married again in Westchester County, but to a different man and without divorcing"
            " her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos"
            ' declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married'
            " once more, this time in the Bronx. In an application for a marriage license, she stated it was her"
            ' "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false'
            ' instrument for filing in the first degree," referring to her false statements on the 2010 marriage'
            " license application, according to court documents. Prosecutors said the marriages were part of an"
            " immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to"
            " her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was"
            " arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New"
            " York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total,"
            " Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All"
            " occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be"
            " married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors"
            " said the immigration scam involved some of her husbands, who filed for permanent residence status"
            " shortly after the marriages.  Any divorces happened only after such filings were approved. It was"
            " unclear whether any of the men will be prosecuted. The case was referred to the Bronx District"
            " Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's"
            ' Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt,'
            " Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his"
            " native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces"
            " up to four years in prison.  Her next court appearance is scheduled for May 18."
        )

        dct = tok._batch_encode_plus(
            [FRANCE_ARTICLE, SHORTER_ARTICLE, IRAN_ARTICLE, ARTICLE_SUBWAY],
            max_length=1024,
            padding_strategy=PaddingStrategy("max_length"),
            truncation_strategy=TruncationStrategy("only_first"),
            return_tensors="pd",
            return_attention_mask=True,
        )

        self.assertEqual(1024, dct["input_ids"].shape[1])
        hypotheses_batch, _ = model.generate(
            input_ids=dct["input_ids"],
            attention_mask=dct["attention_mask"],
            num_beams=2,
            decode_strategy="beam_search",
            max_length=1024,
        )

        EXPECTED = [
            "A French prosecutor says he is not aware of any video footage from on board the plane. Two German "
            "magazines claim to have found a cell phone video showing the crash. The publications say they watched "
            "the video, which was found by a source close to the investigation. All 150 on board Germanwings Flight "
            "9525 were killed.",
            "Palestinian Authority becomes 123rd member of the International Criminal Court. The move gives the court "
            "jurisdiction over alleged crimes in Palestinian territories. Israel and the United States opposed the "
            "Palestinians' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki said it was a "
            "move toward greater justice.",
            "U.S. and its negotiating partners reached a strong framework agreement with Iran. Peter Bergen: The "
            "debate that has already begun will likely result in more heat than light. He says critics have made "
            "dubious assumptions and doubtful assertions. Bergen says the goal was to block Iran from building a "
            "nuclear weapon.",
            "Liana Barrientos, 39, has been married 10 times, sometimes within two weeks of each other. Prosecutors "
            "say the marriages were part of an immigration scam. She pleaded not guilty at State Supreme Court in the "
            "Bronx on Friday. If convicted, she faces up to four years in prison.",
        ]

        generated_summaries = tok.batch_decode(
            hypotheses_batch.tolist(),
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True)
