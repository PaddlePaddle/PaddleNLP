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

import unittest

import paddle
from paddle import Tensor

from paddlenlp.transformers import (
    LayoutLMv2Config,
    LayoutLMv2ForTokenClassification,
    LayoutLMv2Model,
    LayoutLMv2PretrainedModel,
)

from ...testing_utils import slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


class LayoutLMv2ModelTester:
    """Base LayoutLMv2 Model tester which can test:"""

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_position_ids=True,
        vocab_size=103,
        hidden_size=24,
        coordinate_size=4,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        pad_token_id=0,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        num_classes=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_position_ids = use_position_ids
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.coordinate_size = coordinate_size
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
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.num_classes = num_classes
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        attention_mask = None
        if self.use_input_mask:
            attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        position_ids = None
        if self.use_position_ids:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones

        bbox = paddle.expand(paddle.to_tensor([0, 0, 0, 0]), [self.batch_size, self.seq_length, 4])
        image = paddle.zeros([self.batch_size, 3, 224, 224])

        sequence_labels = None
        token_labels = None

        if self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_classes)

        config = self.get_config()
        return config, input_ids, position_ids, attention_mask, bbox, image, sequence_labels, token_labels

    def get_config(self):
        return LayoutLMv2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            coordinate_size=self.coordinate_size,
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
            num_class=self.num_classes,
            num_labels=self.num_labels,
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, position_ids, attention_mask, bbox, image, _, _ = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "bbox": bbox,
            "image": image,
        }
        return config, inputs_dict

    def create_and_check_model(
        self,
        config: LayoutLMv2Config,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        bbox: Tensor,
        image: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
    ):
        model = LayoutLMv2Model(config)
        model.eval()

        result = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, bbox=bbox, image=image)

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length + 49, self.hidden_size])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.hidden_size])

    def create_and_check_for_token_classification(
        self,
        config: LayoutLMv2Config,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        bbox: Tensor,
        image: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
    ):
        model = LayoutLMv2ForTokenClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            bbox=bbox,
            image=image,
            labels=token_labels,
        )

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.num_classes])


class LayoutLMv2ModelModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = LayoutLMv2Model
    use_labels = False
    return_dict = False

    all_model_classes = (
        LayoutLMv2Model,
        LayoutLMv2ForTokenClassification,
    )

    def setUp(self):
        self.model_tester = LayoutLMv2ModelTester(self)

        # set attribute in setUp to overwrite the static attribute
        self.test_resize_embeddings = False

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(LayoutLMv2PretrainedModel.pretrained_init_configuration)[:1]:
            model = LayoutLMv2Model.from_pretrained(model_name)
            self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
