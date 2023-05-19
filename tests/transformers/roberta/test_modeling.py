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
from __future__ import annotations

import tempfile
import unittest

import numpy as np
import paddle
from parameterized import parameterized, parameterized_class

from paddlenlp.transformers import (
    RobertaConfig,
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
    RobertaPretrainedModel,
)

from ...testing_utils import require_package, slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask

ROBERTA_TINY = "sshleifer/tiny-distilroberta-base"


class RobertaModelTester:
    def __init__(self, parent: RobertaModelTest):
        self.parent: RobertaModelTest = parent
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
        self.pad_token_id = (0,)
        self.layer_norm_eps = (1e-12,)
        self.cls_token_id = 101
        self.num_labels = 3
        self.num_choices = 4
        self.dropout = 0.56
        self.scope = None

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return RobertaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            pad_token_id=0,
            layer_norm_eps=1e-12,
            cls_token_id=101,
            num_labels=self.num_labels,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):

        model = RobertaModel(config)
        model.eval()

        result = model(
            input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, return_dict=self.parent.return_dict
        )
        result = model(input_ids, token_type_ids=token_type_ids, return_dict=self.parent.return_dict)
        result = model(input_ids, return_dict=self.parent.return_dict)

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.hidden_size])

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RobertaForCausalLM(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
            return_dict=self.parent.return_dict,
        )
        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_masked_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RobertaForMaskedLM(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
            return_dict=self.parent.return_dict,
        )

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):

        model = RobertaForTokenClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            return_dict=self.parent.return_dict,
            labels=token_labels,
        )

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.num_labels])

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RobertaForSequenceClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
            return_dict=self.parent.return_dict,
        )

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_labels])

    def create_and_check_for_multiple_choice(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):

        model = RobertaForMultipleChoice(config)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand([-1, self.num_choices, -1])
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand([-1, self.num_choices, -1])
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand([-1, self.num_choices, -1])
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            token_type_ids=multiple_choice_token_type_ids,
            return_dict=self.parent.return_dict,
            labels=choice_labels,
        )

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_choices])

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):

        model = RobertaForQuestionAnswering(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            return_dict=self.parent.return_dict,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )

        if sequence_labels is not None:
            start_logits, end_logits = result[1], result[2]
        else:
            start_logits, end_logits = result[0], result[1]

        self.parent.assertEqual(start_logits.shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(end_logits.shape, [self.batch_size, self.seq_length])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@parameterized_class(
    ("return_dict", "use_labels"),
    [
        [False, False],
        [False, True],
        [True, False],
        [True, True],
    ],
)
class RobertaModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = RobertaModel
    use_test_inputs_embeds: bool = True
    return_dict: bool = False
    use_labels: bool = False
    test_tie_weights = True

    all_model_classes = (
        RobertaForCausalLM,
        RobertaForMaskedLM,
        RobertaModel,
        RobertaForSequenceClassification,
        RobertaForTokenClassification,
        RobertaForMultipleChoice,
        RobertaForQuestionAnswering,
    )
    all_generative_model_classes = (RobertaForCausalLM,)

    def setUp(self):
        self.model_tester = RobertaModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(RobertaPretrainedModel.pretrained_init_configuration.keys())[:1]:
            model = RobertaModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class RobertaCompatibilityTest(unittest.TestCase):
    test_model_id = "hf-internal-testing/tiny-random-RobertaModel"

    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import RobertaModel

        cls.torch_model_path = tempfile.TemporaryDirectory().name
        model = RobertaModel.from_pretrained(cls.test_model_id)
        model.save_pretrained(cls.torch_model_path)

    @require_package("transformers", "torch")
    def test_roberta_model_converter(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the paddle model
            from paddlenlp.transformers import RobertaModel

            paddle_model = RobertaModel.from_pretrained(self.test_model_id, from_hf_hub=True, cache_dir=tempdir)
            paddle_model.eval()
            paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

            # 3. forward the torch  model
            import torch
            from transformers import RobertaModel

            torch_model = RobertaModel.from_pretrained(self.torch_model_path)
            torch_model.eval()
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    rtol=1e-4,
                )
            )

    @require_package("transformers", "torch")
    def test_roberta_converter_from_local_dir_with_enable_torch(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 2. forward the torch  model
            from transformers import RobertaModel

            torch_model = RobertaModel.from_pretrained(self.torch_model_path)
            torch_model.save_pretrained(tempdir)

            # 2. forward the paddle model
            from paddlenlp.transformers import RobertaModel, model_utils

            model_utils.ENABLE_TORCH_CHECKPOINT = False

            with self.assertRaises(ValueError) as error:
                RobertaModel.from_pretrained(tempdir)
                self.assertIn("conversion is been disabled" in str(error.exception))
            model_utils.ENABLE_TORCH_CHECKPOINT = True

    @parameterized.expand(
        [
            # ("RobertaForCausalLM",),  TODO: need to tie weights
            # ("RobertaForMaskedLM",),  TODO: need to tie weights
            ("RobertaModel",),
            ("RobertaForSequenceClassification",),
            ("RobertaForTokenClassification",),
            ("RobertaForQuestionAnswering",),
        ]
    )
    @require_package("transformers", "torch")
    def test_roberta_classes_from_local_dir(self, class_name):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch model
            import torch
            import transformers

            torch_model_class = getattr(transformers, class_name)
            torch_model = torch_model_class.from_pretrained(self.torch_model_path)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

            # 2. forward the paddle model
            from paddlenlp import transformers

            paddle_model_class = getattr(transformers, class_name)
            paddle_model = paddle_model_class.from_pretrained(tempdir)
            paddle_model.eval()
            paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]
            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    atol=1e-3,
                )
            )


class RobertaModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = RobertaForMaskedLM.from_pretrained("roberta-base")
        model.eval()
        input_ids = paddle.to_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)
        expected_shape = [1, 11, 50265]
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = paddle.to_tensor(
            [[[33.8802, -4.3103, 22.7761], [4.6539, -2.8098, 13.6253], [1.8228, -3.6898, 8.8600]]]
        )
        self.assertTrue(paddle.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    @slow
    def test_inference_no_head(self):
        model = RobertaModel.from_pretrained("roberta-base")
        model.eval()
        input_ids = paddle.to_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        # compare the actual values for a slice.
        expected_slice = paddle.to_tensor(
            [[[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]]
        )
        self.assertTrue(paddle.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
