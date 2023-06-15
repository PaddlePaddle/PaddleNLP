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

import copy
import inspect
import unittest

import paddle

from paddlenlp.transformers import (
    LukeConfig,
    LukeForEntityClassification,
    LukeForEntityPairClassification,
    LukeForEntitySpanClassification,
    LukeForMaskedLM,
    LukeForQuestionAnswering,
    LukeModel,
    LukePretrainedModel,
)

from ...testing_utils import slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor


class LukeModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=2,
        entity_vocab_size=32,
        entity_emb_size=16,
        initializer_range=0.02,
        pad_token_id=1,
        cls_token_id=2,
        entity_pad_token_id=0,
        num_labels=2,
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
        self.cls_token_id = cls_token_id
        self.entity_vocab_size = entity_vocab_size
        self.entity_emb_size = entity_emb_size
        self.entity_pad_token_id = entity_pad_token_id
        self.num_labels = num_labels

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = paddle.ones([self.batch_size, self.seq_length], dtype="int32")

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        entity_ids = paddle.randint(0, self.entity_vocab_size, [self.batch_size, 2])
        entity_position_ids = paddle.randint(0, self.max_position_embeddings, [self.batch_size, 2, self.seq_length])
        config = self.get_config()

        entity_start_positions = paddle.ones([self.batch_size, 2], dtype="int32")
        entity_end_positions = paddle.ones([self.batch_size, 2], dtype="int32")
        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            entity_ids,
            entity_position_ids,
            entity_start_positions,
            entity_end_positions,
        )

    def get_config(self):
        return LukeConfig(
            vocab_size=self.vocab_size,
            entity_vocab_size=self.entity_vocab_size,
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
            entity_emb_size=self.entity_emb_size,
            entity_pad_token_id=self.entity_pad_token_id,
            num_labels=self.num_labels,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            entity_ids,
            entity_position_ids,
            entity_start_positions,
            entity_end_positions,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
            "entity_ids": entity_ids,
            "entity_position_ids": entity_position_ids,
            "entity_start_positions": entity_start_positions,
            "entity_end_positions": entity_end_positions,
        }
        return config, inputs_dict

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        entity_ids,
        entity_position_ids,
        entity_start_positions,
        entity_end_positions,
    ):
        model = LukeModel(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result[2].shape, [self.batch_size, self.hidden_size])

    def create_and_check_masked_lm_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        entity_ids,
        entity_position_ids,
        entity_start_positions,
        entity_end_positions,
    ):
        model = LukeForMaskedLM(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_question_answering_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        entity_ids,
        entity_position_ids,
        entity_start_positions,
        entity_end_positions,
    ):
        model = LukeForQuestionAnswering(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length])

    def create_and_check_entity_classification_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        entity_ids,
        entity_position_ids,
        entity_start_positions,
        entity_end_positions,
    ):
        model = LukeForEntityClassification(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
        )
        self.parent.assertEqual(result.shape, [self.batch_size, self.num_labels])

    def create_and_check_entity_span_classification_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        entity_ids,
        entity_position_ids,
        entity_start_positions,
        entity_end_positions,
    ):
        model = LukeForEntitySpanClassification(config)
        model.eval()
        result = model(
            entity_start_positions=entity_start_positions,
            entity_end_positions=entity_end_positions,
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
        )
        self.parent.assertEqual(result.shape, [self.batch_size, 2, self.num_labels])

    def create_and_check_entity_pair_classification_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        entity_ids,
        entity_position_ids,
        entity_start_positions,
        entity_end_positions,
    ):
        model = LukeForEntityPairClassification(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
        )
        self.parent.assertEqual(result.shape, [self.batch_size, self.num_labels])


class LukeModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = LukeModel
    return_dict: bool = False
    use_labels: bool = False
    use_test_inputs_embeds: bool = False

    all_model_classes = (
        LukeModel,
        LukeForEntitySpanClassification,
        LukeForEntityPairClassification,
        LukeForEntityClassification,
        LukeForMaskedLM,
        LukeForQuestionAnswering,
    )

    def setUp(self):
        self.model_tester = LukeModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_masked_lm_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_masked_lm_model(*config_and_inputs)

    def test_question_answering_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_question_answering_model(*config_and_inputs)

    def test_Entity_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_entity_classification_model(*config_and_inputs)

    def test_entity_pair_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_entity_pair_classification_model(*config_and_inputs)

    def test_entity_span_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_entity_span_classification_model(*config_and_inputs)

    def _prepare_for_class(self, inputs_dict, model_class):
        inputs_dict = copy.deepcopy(inputs_dict)
        if model_class.__name__.endswith("SpanClassification"):
            return inputs_dict
        else:
            del inputs_dict["entity_start_positions"]
            del inputs_dict["entity_end_positions"]
            return inputs_dict

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ["input_ids"]
            if not model_class.__name__.endswith("SpanClassification"):
                self.assertListEqual(arg_names[:1], expected_arg_names)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(LukePretrainedModel.pretrained_init_configuration)[:1]:
            model = LukeModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
