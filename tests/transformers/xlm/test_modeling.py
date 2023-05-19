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

import random
import unittest

import paddle
from parameterized import parameterized_class

from paddlenlp.transformers import (
    XLMConfig,
    XLMForMultipleChoice,
    XLMForQuestionAnsweringSimple,
    XLMForSequenceClassification,
    XLMForTokenClassification,
    XLMModel,
    XLMPretrainedModel,
    XLMWithLMHeadModel,
)

from ...testing_utils import slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


class XLMModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_encoder=True,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        hidden_act="gelu",
        n_langs=1,
        lang_id=0,
        lang2id=None,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=2,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        num_classes=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_encoder = is_encoder
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.n_langs = n_langs
        self.lang_id = lang_id
        self.lang2id = lang2id
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.pad_token_id = pad_token_id
        self.type_sequence_label_size = type_sequence_label_size
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.seed = 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None

        if self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def set_seed(self):
        random.seed(self.seed)
        paddle.seed(self.seed)

    def get_config(self):
        return XLMConfig(
            vocab_size=self.vocab_size,
            emb_dim=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_act=self.hidden_act,
            n_langs=self.n_langs,
            lang_id=self.lang_id,
            lang2id=self.lang2id,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            pad_token_id=self.pad_token_id,
            num_class=self.num_classes,
            num_labels=self.num_labels,
            num_choices=self.num_choices,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = XLMModel(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, output_hidden_states=self.parent.return_dict)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_multiple_choice(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = XLMForMultipleChoice(config)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand([-1, self.num_choices, -1])
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand([-1, self.num_choices, -1])
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
        )

        self.parent.assertEqual(result.shape, [self.batch_size, self.num_choices])

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = XLMForQuestionAnsweringSimple(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
        )

        start_logits, end_logits = result[0], result[1]

        self.parent.assertEqual(start_logits.shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(end_logits.shape, [self.batch_size, self.seq_length])

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = XLMForSequenceClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
        )

        self.parent.assertEqual(result.shape, [self.batch_size, self.num_classes])

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = XLMForTokenClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
        )

        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.num_classes])

    def create_and_check_for_lm_head(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = XLMWithLMHeadModel(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            labels=token_labels,
        )

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

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
        inputs_dict = {"input_ids": input_ids}
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
class XLMModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = XLMModel
    return_dict: bool = False
    use_labels: bool = True

    all_model_classes = (
        XLMModel,
        XLMForMultipleChoice,
        XLMForQuestionAnsweringSimple,
        XLMForSequenceClassification,
        XLMForTokenClassification,
        XLMWithLMHeadModel,
    )

    def setUp(self):
        self.model_tester = XLMModelTester(self)

    def test_model(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_multiple_choice(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_ml_head(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_lm_head(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(XLMPretrainedModel.pretrained_init_configuration)[:1]:
            model = XLMModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
