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
from typing import Optional, Tuple, Dict, Any
import paddle
from paddle import Tensor

from dataclasses import dataclass, asdict, fields, Field
from paddlenlp.transformers import (
    ErnieDocModel,
    ErnieDocPretrainedModel,
    ErnieDocForSequenceClassification,
    ErnieDocForTokenClassification,
    ErnieDocForQuestionAnswering,
)
from tests.transformers.test_modeling_common import (ids_tensor, floats_tensor,
                                                     random_attention_mask,
                                                     ModelTesterMixin)
from tests.testing_utils import slow


@dataclass
class ErnieDocTestModelConfig:
    """ernie-doc model config which keep consist with pretrained_init_configuration sub fields
    """
    attention_dropout_prob: float = 0.0
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    relu_dropout: float = 0.0
    hidden_size: int = 64
    initializer_range: float = 0.02
    max_position_embeddings: int = 20
    num_attention_heads: int = 8
    num_hidden_layers: int = 8
    task_type_vocab_size: int = 3
    vocab_size: int = 100
    memory_len: int = 50
    epsilon: float = 1e-12
    pad_token_id: int = 1

    @property
    def model_kwargs(self) -> dict:
        """get the model kwargs configuration to init the model"""
        model_config_fields: Tuple[Field, ...] = fields(ErnieDocTestModelConfig)
        return {
            field.name: getattr(self, field.name)
            for field in model_config_fields
        }


@dataclass
class ErnieDocTestConfig(ErnieDocTestModelConfig):
    """all of ErnieDoc Test configuration
    """
    batch_size: int = 2
    is_training: bool = False
    use_input_mask: bool = False
    use_token_type_ids: bool = False

    # used for sequence classification
    num_classes: int = 3


class ErnieDocModelTester:
    """Base ErnieDoc Model tester which can test:
    """

    def __init__(self, parent, config: Optional[ErnieDocTestConfig] = None):
        self.parent = parent
        self.config = config or ErnieDocTestConfig()

        self.is_training = self.config.is_training

    def prepare_config_and_inputs(
            self) -> Tuple[Dict[str, Any], Tensor, Tensor, Tensor]:
        config = self.config
        input_ids = ids_tensor([config.batch_size, config.memory_len],
                               config.vocab_size)

        input_mask = None
        if config.use_input_mask:
            input_mask = random_attention_mask(
                [config.batch_size, config.memory_len])

        token_type_ids = None
        if config.use_token_type_ids:
            token_type_ids = ids_tensor([config.batch_size, config.memory_len],
                                        config.type_vocab_size)

        return config.model_kwargs, input_ids, token_type_ids, input_mask

    def create_and_check_model(
        self,
        config,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attn_mask: Tensor,
    ):
        model = ErnieDocModel(**config)
        model.eval()

        result = model(input_ids,
                       attn_mask=attn_mask,
                       token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [
            self.config.batch_size, self.config.memory_len,
            self.config.hidden_size
        ])
        self.parent.assertEqual(
            result[1].shape, [self.config.batch_size, self.config.hidden_size])

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids: Tensor,
        token_type_ids: Tensor,
        input_mask: Tensor,
    ):
        model = ErnieDocForSequenceClassification(
            ErnieDocModel(**config), num_classes=self.config.num_classes)
        model.eval()
        result, _ = model(
            input_ids,
            attn_mask=input_mask,
            token_type_ids=token_type_ids,
        )
        self.parent.assertEqual(
            result.shape, [self.config.batch_size, self.config.num_classes])

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = ErnieDocForTokenClassification(
            ErnieDocModel(**config), num_classes=self.config.num_classes)
        model.eval()
        result, _ = model(input_ids,
                          attn_mask=input_mask,
                          token_type_ids=token_type_ids)
        self.parent.assertEqual(result.shape, [
            self.config.batch_size, self.config.memory_len,
            self.config.num_classes
        ])

    def create_and_check_for_question_answering(self, config, input_ids,
                                                token_type_ids, input_mask):
        model = ErnieDocForQuestionAnswering(ErnieDocModel(**config))
        model.eval()

        result = model(
            input_ids,
            position_ids=None,
            attn_mask=input_mask,
            token_type_ids=token_type_ids,
        )
        self.parent.assertEqual(
            result[0].shape, [self.config.batch_size, self.config.memory_len])
        self.parent.assertEqual(
            result[1].shape, [self.config.batch_size, self.config.memory_len])

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, token_type_ids, input_mask = self.prepare_config_and_inputs(
        )
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attn_mask": input_mask,
        }
        return config, inputs_dict

    def get_config(self) -> dict:
        """get the base model kwargs

        Returns:
            dict: the values of kwargs
        """
        return self.config.model_kwargs


class ErnieDocModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = ErnieDocModel

    all_model_classes = (
        ErnieDocModel,
        ErnieDocForSequenceClassification,
        ErnieDocForTokenClassification,
        ErnieDocForQuestionAnswering,
    )

    def setUp(self):
        self.model_tester = ErnieDocModelTester(self)

    def get_config():
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(
            *config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(
            *config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(
            *config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(
                ErnieDocPretrainedModel.pretrained_init_configuration)[:1]:
            model = ErnieDocModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_save_load(self):
        # TODO(wj-Mcat): should be removed later
        pass


class ErnieDocModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_no_attention(self):
        model = ErnieDocModel.from_pretrained("ernie-doc-base-en")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        expected_shape = [1, 11, 1024]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[0.31737554, 0.58842468, 0.43969756],
              [0.20048048, 0.04142965, -0.2655520],
              [0.49883127, -0.15263288, 0.46780178]]])
        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-5))

    @slow
    def test_inference_with_attention(self):
        model = ErnieDocModel.from_pretrained("ernie-doc-base-en")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attn_mask = paddle.ones(shape=[1, model.memory_len, 1])

        with paddle.no_grad():
            output = model(input_ids, attn_mask=attn_mask)[0]
        expected_shape = [1, 11, 1024]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[0.31737554, 0.58842468, 0.43969756],
              [0.20048048, 0.04142965, -0.2655520],
              [0.49883127, -0.15263288, 0.46780178]]])
        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
