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
    SkepPretrainedModel,
    SkepModel,
    SkepForSequenceClassification,
    SkepForTokenClassification,
    SkepCrfForTokenClassification,
)
from tests.transformers.test_modeling_common import (ids_tensor, floats_tensor,
                                                     random_attention_mask,
                                                     ModelTesterMixin)
from tests.testing_utils import slow


@dataclass
class SkepTestModelConfig:
    """skep model config which keep consist with pretrained_init_configuration sub fields
    """
    attention_probs_dropout_prob: float = 0.1
    hidden_act: str = "relu"
    hidden_dropout_prob: float = 0.1
    hidden_size: int = 1024
    initializer_range: float = 0.02
    intermediate_size: int = 4096
    max_position_embeddings: int = 512
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    type_vocab_size: int = 4
    vocab_size: int = 12800
    pad_token_id: int = 0

    @property
    def model_kwargs(self) -> dict:
        """get the model kwargs configuration to init the model"""
        model_config_fields: Tuple[Field, ...] = fields(SkepTestModelConfig)
        return {
            field.name: getattr(self, field.name)
            for field in model_config_fields
        }


@dataclass
class SkepTestTrainConfig:
    """train config under unittest code"""
    batch_size: int = 2
    seq_length: int = 7
    is_training: bool = False
    use_input_mask: bool = False
    use_token_type_ids: bool = False

    # used for sequence classification
    num_classes: int = 3


@dataclass
class SkepTestConfig(SkepTestModelConfig, SkepTestTrainConfig):
    """all of Skep Test configuration
    
    TODO(wj-Mcat): can be intialized with `from_pretrained` style, and it's fixed at current
    """
    pass


class SkepModelTester:
    """Base Skep Model tester which can test:
    """

    def __init__(self, parent, config: Optional[SkepTestConfig] = None):
        self.parent = parent
        self.config = config or SkepTestConfig()

        self.is_training = self.config.is_training

    def prepare_config_and_inputs(
            self) -> Tuple[Dict[str, Any], Tensor, Tensor, Tensor]:
        config = self.config
        input_ids = ids_tensor([config.batch_size, config.seq_length],
                               config.vocab_size)

        input_mask = None
        if config.use_input_mask:
            input_mask = random_attention_mask(
                [config.batch_size, config.seq_length])

        token_type_ids = None
        if config.use_token_type_ids:
            token_type_ids = ids_tensor([config.batch_size, config.seq_length],
                                        config.type_vocab_size)

        return config.model_kwargs, input_ids, token_type_ids, input_mask

    def create_and_check_model(
        self,
        config,
        input_ids: Tensor,
        token_type_ids: Tensor,
        input_mask: Tensor,
    ):
        model = SkepModel(**config)
        model.eval()

        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [
            self.config.batch_size, self.config.seq_length,
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
        model = SkepForSequenceClassification(
            SkepModel(**config), num_classes=self.config.num_classes)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
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
        model = SkepForTokenClassification(SkepModel(**config),
                                           num_classes=self.config.num_classes)
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids)
        self.parent.assertEqual(result.shape, [
            self.config.batch_size, self.config.seq_length,
            self.config.num_classes
        ])

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, token_type_ids, input_mask = self.prepare_config_and_inputs(
        )
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict

    # def _prepare_for_class(self, inputs_dict, model_class):
    #     inputs_dict = copy.deepcopy(inputs_dict)
    #     if model_class.__name__.endswith("ForMultipleChoice"):
    #         inputs_dict = {
    #             k: v.unsqueeze(1).expand(
    #                 shape=[-1, self.model_tester.num_choices, -1])
    #             if isinstance(v, paddle.Tensor) and v.ndim > 1 else v
    #             for k, v in inputs_dict.items()
    #         }
    #     return inputs_dict

    def get_config(self) -> dict:
        """get the base model kwargs

        Returns:
            dict: the values of kwargs
        """
        return self.config.model_kwargs


class SkepModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = SkepModel

    all_model_classes = (
        SkepModel,
        SkepCrfForTokenClassification,
        SkepForSequenceClassification,
        SkepForTokenClassification,
    )

    def setUp(self):
        self.model_tester = SkepModelTester(self)

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

    def test_model_from_pretrained(self):
        for model_name in list(
                SkepPretrainedModel.pretrained_init_configuration)[:1]:
            model = SkepModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class SkepModelIntegrationTest(unittest.TestCase):

    def test_inference_no_attention(self):
        model = SkepModel.from_pretrained("skep_ernie_1.0_large_ch")
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

    def test_inference_with_attention(self):
        model = SkepModel.from_pretrained("skep_ernie_1.0_large_ch")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
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
