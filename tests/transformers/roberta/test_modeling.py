# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import unittest
import paddle

from paddlenlp.transformers import (
    RobertaPretrainedModel,
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)

from ...transformers.test_modeling_common import ids_tensor, floats_tensor, random_attention_mask, ModelTesterMixin
from ...testing_utils import slow

ROBERTA_TINY = "sshleifer/tiny-distilroberta-base"


class RobertaModelTester:

    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_mask = True
        self.use_token_type_ids = True
        self.use_labels = True
        self.vocab_size = 99
        self.hidden_size = 32
        self.num_hidden_layers = 5
        self.num_attention_heads = 4
        self.intermediate_size = 37
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 16
        self.type_sequence_label_size = 2
        self.initializer_range = 0.02
        self.pad_token_id = 0,
        self.layer_norm_eps = 1e-12,
        self.cls_token_id = 101
        self.num_labels = 3
        self.num_choices = 4
        self.scope = None

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length],
                               self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask(
                [self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length],
                                        self.type_vocab_size)

        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "pad_token_id": 0,
            "layer_norm_eps": 1e-12,
            "cls_token_id": 101,
        }

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
        ) = self.prepare_config_and_inputs()

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
        )

    def create_and_check_model(self, config, input_ids, token_type_ids,
                               input_mask):
        model = RobertaModel(**config)
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids, return_dict=True)

        self.parent.assertEqual(
            result.last_hidden_state.shape,
            [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result.pooler_output.shape,
                                [self.batch_size, self.hidden_size])

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = RobertaForCausalLM(RobertaModel(**config))
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       return_dict=True)
        self.parent.assertEqual(
            result.logits.shape,
            [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_masked_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = RobertaForMaskedLM(RobertaModel(**config))
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       return_dict=True)
        self.parent.assertEqual(
            result.logits.shape,
            [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_token_classification(self, config, input_ids,
                                                  token_type_ids, input_mask):
        model = RobertaForTokenClassification(RobertaModel(**config),
                                              num_classes=self.num_labels,
                                              dropout=None)
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       return_dict=True)
        self.parent.assertEqual(
            result.logits.shape,
            [self.batch_size, self.seq_length, self.num_labels])

    def create_and_check_for_multiple_choice(self, config, input_ids,
                                             token_type_ids, input_mask):
        model = RobertaForMultipleChoice(RobertaModel(**config))
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(
            [-1, self.num_choices, -1])
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(
            [-1, self.num_choices, -1])
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(
            [-1, self.num_choices, -1])
        result = model(multiple_choice_inputs_ids,
                       attention_mask=multiple_choice_input_mask,
                       token_type_ids=multiple_choice_token_type_ids,
                       return_dict=True)
        self.parent.assertEqual(result.logits.shape,
                                [self.batch_size, self.num_choices])

    def create_and_check_for_question_answering(self, config, input_ids,
                                                token_type_ids, input_mask):
        model = RobertaForQuestionAnswering(RobertaModel(**config))
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       return_dict=True)
        self.parent.assertEqual(result.start_logits.shape,
                                [self.batch_size, self.seq_length])
        self.parent.assertEqual(result.end_logits.shape,
                                [self.batch_size, self.seq_length])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask
        }
        return config, inputs_dict


class RobertaModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = RobertaModel

    all_model_classes = (
        RobertaForCausalLM,
        RobertaForMaskedLM,
        RobertaModel,
        RobertaForSequenceClassification,
        RobertaForTokenClassification,
        RobertaForMultipleChoice,
        RobertaForQuestionAnswering,
    )
    all_generative_model_classes = (RobertaForCausalLM, )

    def setUp(self):
        self.model_tester = RobertaModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder(
        )
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(
            *config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(
            *config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(
            *config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(RobertaPretrainedModel.
                               pretrained_init_configuration.keys())[:1]:
            model = RobertaModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class RobertaModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_masked_lm(self):
        model = RobertaForMaskedLM.from_pretrained("roberta-base")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)
        expected_shape = [1, 11, 50265]
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = paddle.to_tensor([[[33.8802, -4.3103, 22.7761],
                                            [4.6539, -2.8098, 13.6253],
                                            [1.8228, -3.6898, 8.8600]]])
        self.assertTrue(
            paddle.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    @slow
    def test_inference_no_head(self):
        model = RobertaModel.from_pretrained("roberta-base")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        # compare the actual values for a slice.
        expected_slice = paddle.to_tensor([[[-0.0231, 0.0782, 0.0074],
                                            [-0.1854, 0.0540, -0.0175],
                                            [0.0548, 0.0799, 0.1687]]])
        self.assertTrue(
            paddle.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
