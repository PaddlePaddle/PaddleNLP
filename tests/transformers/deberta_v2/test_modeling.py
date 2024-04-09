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
from __future__ import annotations

import tempfile
import unittest
from typing import List

import numpy as np
import paddle

from paddlenlp.transformers import (
    DebertaV2Config,
    DebertaV2ForMultipleChoice,
    DebertaV2ForQuestionAnswering,
    DebertaV2ForSequenceClassification,
    DebertaV2ForTokenClassification,
    DebertaV2Model,
)
from paddlenlp.transformers.model_utils import PretrainedModel

from ...testing_utils import require_package
from ..test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


class DebertaV2CompatibilityTest(unittest.TestCase):
    test_model_id = "hf-tiny-model-private/tiny-random-DebertaV2Model"

    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import DebertaV2Model

        # when python application is done, `TemporaryDirectory` will be free
        cls.torch_model_path = tempfile.TemporaryDirectory().name
        model = DebertaV2Model.from_pretrained(cls.test_model_id)
        model.save_pretrained(cls.torch_model_path)

    def test_model_config_mapping(self):
        config = DebertaV2Config(num_labels=22, hidden_dropout_prob=0.99)
        self.assertEqual(config.hidden_dropout_prob, 0.99)
        self.assertEqual(config.num_labels, 22)

    def setUp(self) -> None:
        self.tempdirs: List[tempfile.TemporaryDirectory] = []

    def tearDown(self) -> None:
        for tempdir in self.tempdirs:
            tempdir.cleanup()

    def get_tempdir(self) -> str:
        tempdir = tempfile.TemporaryDirectory()
        self.tempdirs.append(tempdir)
        return tempdir.name

    def compare_two_model(self, first_model: PretrainedModel, second_model: PretrainedModel):

        first_weight_name = "encoder.layer.3.attention.self.in_proj.weight"

        second_weight_name = "encoder.layer.3.attention.self.in_proj.weight"

        first_tensor = first_model.state_dict()[first_weight_name]
        second_tensor = second_model.state_dict()[second_weight_name]
        self.compare_two_weight(first_tensor, second_tensor)

    def compare_two_weight(self, first_tensor, second_tensor):
        diff = paddle.sum(first_tensor - second_tensor).numpy().item()
        self.assertEqual(diff, 0.0)

    @require_package("transformers", "torch")
    def test_deberta_v2_converter(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the paddle model
            from paddlenlp.transformers.deberta_v2.modeling import DebertaV2Model

            paddle_model = DebertaV2Model.from_pretrained(
                "hf-internal-testing/tiny-random-DebertaV2Model", from_hf_hub=True, cache_dir=tempdir
            )
            paddle_model.eval()
            paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

            # 3. forward the torch  model
            import torch
            from transformers import DebertaV2Model

            torch_model = DebertaV2Model.from_pretrained(
                "hf-internal-testing/tiny-random-DebertaV2Model", cache_dir=tempdir
            )
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
    def test_deberta_v2_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch  model
            import torch
            from transformers import DebertaV2Model

            torch_model = DebertaV2Model.from_pretrained("hf-internal-testing/tiny-random-DebertaV2Model")
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

            # 2. forward the paddle model
            from paddlenlp.transformers.deberta_v2.modeling import DebertaV2Model

            paddle_model = DebertaV2Model.from_pretrained(tempdir, convert_from_torch=True)
            paddle_model.eval()
            paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    rtol=1e-4,
                )
            )


class DebertaV2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=0,
        initializer_range=0.02,
        pad_token_id=0,
        type_sequence_label_size=2,
        use_relative_position=True,
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
        self.type_sequence_label_size = type_sequence_label_size
        self.use_relative_position = use_relative_position
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

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
        return DebertaV2Config(
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
            use_relative_position=self.use_relative_position,
            num_class=self.num_classes,
            num_labels=self.num_labels,
            num_choices=self.num_choices,
            pooler_hidden_size=self.hidden_size,
            pooler_dropout=self.hidden_dropout_prob,
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
        model = DebertaV2Model(config)
        model.eval()
        result = model(
            input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, return_dict=self.parent.return_dict
        )
        result = model(input_ids, return_dict=self.parent.return_dict)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

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
        model = DebertaV2ForMultipleChoice(config)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand([-1, self.num_choices, -1])
        result = model(
            multiple_choice_inputs_ids,
            labels=choice_labels,
            return_dict=self.parent.return_dict,
        )
        if choice_labels is not None:
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
        model = DebertaV2ForQuestionAnswering(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            return_dict=self.parent.return_dict,
        )
        if sequence_labels is not None:
            start_logits, end_logits = result[1], result[2]
        else:
            start_logits, end_logits = result[0], result[1]

        self.parent.assertEqual(start_logits.shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(end_logits.shape, [self.batch_size, self.seq_length])

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
        model = DebertaV2ForSequenceClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
            return_dict=self.parent.return_dict,
        )
        if sequence_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_classes])

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
        model = DebertaV2ForTokenClassification(config)
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

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.num_classes])

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


class DebertaModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = DebertaV2Model
    return_dict: bool = False
    use_labels: bool = False
    use_test_inputs_embeds: bool = False

    all_model_classes = (
        DebertaV2Model,
        DebertaV2ForMultipleChoice,
        DebertaV2ForQuestionAnswering,
        DebertaV2ForSequenceClassification,
        DebertaV2ForTokenClassification,
    )

    def setUp(self):
        self.model_tester = DebertaV2ModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)


if __name__ == "__main__":
    unittest.main()
