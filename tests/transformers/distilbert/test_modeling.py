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
from parameterized import parameterized

from paddlenlp.transformers import (
    DistilBertForMaskedLM,
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertModel,
)
from paddlenlp.transformers.distilbert.configuration import DistilBertConfig

from ...testing_utils import require_package, slow
from ..test_configuration_common import ConfigTester
from ..test_modeling_common import (
    ModelTesterMixin,
    ModelTesterPretrainedMixin,
    ids_tensor,
    random_attention_mask,
)


class DistilBertModelTester:
    def __init__(
        self,
        parent: DistilBertModelTest,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        pad_token_id=0,
        pool_act="tanh",
        fuse=False,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        dropout=0.56,
        return_dict=False,
    ):
        self.parent: DistilBertModelTest = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.pool_act = pool_act
        self.fuse = fuse
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.dropout = dropout
        self.return_dict = return_dict

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self) -> DistilBertConfig:
        return DistilBertConfig(
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
            pad_token_id=self.pad_token_id,
            pool_act=self.pool_act,
            fuse=self.fuse,
            num_labels=self.num_labels,
            num_choices=self.num_choices,
        )

    def create_and_check_model(
        self, config: DistilBertConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = DistilBertModel(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.hidden_size])
        result = model(input_ids)
        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_masked_lm(
        self,
        config: DistilBertConfig,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = DistilBertForMaskedLM(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = DistilBertForQuestionAnswering(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
        )

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length])

    def create_and_check_for_sequence_classification(
        self,
        config: DistilBertConfig,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):

        model = DistilBertForSequenceClassification(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.shape, [self.batch_size, self.num_labels])

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = DistilBertForTokenClassification(config)

        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.num_labels])

    def test_addition_params(self, config: DistilBertConfig, *args, **kwargs):
        config.num_labels = 7
        config.classifier_dropout = 0.98

        model = DistilBertForTokenClassification(config)
        model.eval()

        self.parent.assertEqual(model.classifier.weight.shape, [config.hidden_size, 7])
        self.parent.assertEqual(model.dropout.p, 0.98)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class DistilBertModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = DistilBertModel
    return_dict = False
    use_labels = False
    test_resize_embeddings = False

    all_model_classes = (
        DistilBertModel,
        DistilBertForMaskedLM,
        DistilBertForQuestionAnswering,
        DistilBertForSequenceClassification,
        DistilBertForTokenClassification,
    )

    def setUp(self):
        super().setUp()

        self.model_tester = DistilBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DistilBertConfig, vocab_size=256, hidden_size=24)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_custom_params(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.test_addition_params(*config_and_inputs)

    def test_model_name_list(self):
        config = self.model_tester.get_config()
        model = self.base_model_class(config)
        self.assertTrue(len(model.model_name_list) != 0)

    @slow
    def test_params_compatibility_of_init_method(self):
        """test initing model with different params"""
        model: DistilBertForTokenClassification = DistilBertForTokenClassification.from_pretrained(
            "distilbert-base-uncased", num_classes=4, dropout=0.3
        )
        assert model.num_labels == 4
        assert model.dropout.p == 0.3


class DistilBertModelCompatibilityTest(unittest.TestCase):
    model_id = "hf-internal-testing/tiny-random-DistilBertModel"

    @require_package("transformers", "torch")
    def test_distilBert_converter(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # 1. create input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the paddle model
            from paddlenlp.transformers import DistilBertModel

            paddle_model = DistilBertModel.from_pretrained(self.model_id, from_hf_hub=True, cache_dir=tempdir)
            paddle_model.eval()
            paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

            # 3. forward the torch model
            import torch
            from transformers import DistilBertModel

            torch_model = DistilBertModel.from_pretrained(self.model_id, cache_dir=tempdir)
            torch_model.eval()
            torch_logit = torch_model(torch.tensor(input_ids))[0]

            # 4. compare results
            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    rtol=1e-4,
                )
            )

    @require_package("transformers", "torch")
    def test_distilBert_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch  model
            import torch
            from transformers import DistilBertModel

            torch_model = DistilBertModel.from_pretrained(self.model_id)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(torch.tensor(input_ids))[0]

            # 2. forward the paddle model
            from paddlenlp.transformers import DistilBertModel

            paddle_model = DistilBertModel.from_pretrained(tempdir, convert_from_torch=True)
            paddle_model.eval()
            paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    rtol=1e-4,
                )
            )

    @parameterized.expand(
        [
            ("DistilBertModel",),
            ("DistilBertForQuestionAnswering",),
            ("DistilBertForSequenceClassification",),
            ("DistilBertForTokenClassification",),
        ]
    )
    @require_package("transformers", "torch")
    def test_distilBert_classes_from_local_dir(self, class_name, pytorch_class_name=None):
        pytorch_class_name = pytorch_class_name or class_name
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch model
            import torch
            import transformers

            torch_model_class = getattr(transformers, pytorch_class_name)
            torch_model = torch_model_class.from_pretrained(self.model_id)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(torch.tensor(input_ids))[0]

            # 3. forward the paddle model
            from paddlenlp import transformers

            paddle_model_class = getattr(transformers, class_name)
            paddle_model = paddle_model_class.from_pretrained(tempdir, convert_from_torch=True)
            paddle_model.eval()

            paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    atol=1e-3,
                )
            )


class DistilBertModelIntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):
    base_model_class = DistilBertModel

    @slow
    def test_inference_no_attention(self):
        model = DistilBertModel.from_pretrained("__internal_testing__/tiny-random-distilbert")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
        expected_shape = [1, 11, 8]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [0.50366199, -1.33068442, -1.73558784],
                    [1.72435653, 1.08600891, -0.28388503],
                    [-0.19172087, -0.56781638, 0.51192915],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @slow
    def test_inference_with_attention(self):
        model = DistilBertModel.from_pretrained("__internal_testing__/tiny-random-distilbert")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
        expected_shape = [1, 11, 8]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [0.50366199, -1.33068442, -1.73558784],
                    [1.72435653, 1.08600891, -0.28388503],
                    [-0.19172087, -0.56781638, 0.51192915],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
