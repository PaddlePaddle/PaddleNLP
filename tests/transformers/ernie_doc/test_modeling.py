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

import numpy as np
import paddle

from paddlenlp.transformers import (
    ErnieDocConfig,
    ErnieDocForQuestionAnswering,
    ErnieDocForSequenceClassification,
    ErnieDocForTokenClassification,
    ErnieDocModel,
    ErnieDocPretrainedModel,
)

from ...testing_utils import slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor


class ErnieDocModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        num_hidden_layers=5,
        num_attention_heads=4,
        hidden_size=32,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        relu_dropout=0.0,
        hidden_act="gelu",
        memory_len=7,
        vocab_size=99,
        type_vocab_size=2,
        max_position_embeddings=256,
        task_type_vocab_size=3,
        normalize_before=False,
        epsilon=1e-5,
        rel_pos_params_sharing=False,
        initializer_range=0.02,
        pad_token_id=0,
        cls_token_idx=-1,
        type_sequence_label_size=2,
        num_classes=2,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.relu_dropout = relu_dropout
        self.hidden_act = hidden_act
        self.memory_len = memory_len
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.task_type_vocab_size = task_type_vocab_size
        self.type_vocab_size = type_vocab_size
        self.normalize_before = normalize_before
        self.epsilon = epsilon
        self.rel_pos_params_sharing = rel_pos_params_sharing
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.cls_token_idx = cls_token_idx
        self.num_classes = num_classes
        self.type_sequence_label_size = type_sequence_label_size
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length, 1], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = paddle.ones([self.batch_size, self.seq_length, 1])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length, 1], self.type_vocab_size, dtype="int64")

        position_ids = None
        token_labels = None

        def get_related_pos(insts, seq_len, memory_len=128):
            beg = seq_len + seq_len + memory_len
            r_position = [list(range(beg - 1, seq_len - 1, -1)) + list(range(0, seq_len)) for i in range(len(insts))]
            return np.array(r_position).astype("int64").reshape([len(insts), beg, 1])

        position_ids = paddle.to_tensor(get_related_pos(input_ids, self.seq_length, self.memory_len))

        tensor = paddle.zeros([self.batch_size, self.seq_length, self.hidden_size], dtype="float32")
        memories = [tensor for i in range(self.num_hidden_layers)]

        if self.parent.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_classes)

        config = self.get_config()
        return config, input_ids, memories, token_type_ids, input_mask, position_ids, token_labels

    def get_config(self):
        return ErnieDocConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            attention_dropout_prob=self.attention_dropout_prob,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            relu_dropout=self.relu_dropout,
            memory_len=self.memory_len,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            num_class=self.num_classes,
            task_type_vocab_size=self.task_type_vocab_size,
            normalize_before=self.normalize_before,
            epsilon=self.epsilon,
            rel_pos_params_sharing=self.rel_pos_params_sharing,
            cls_token_idx=self.cls_token_idx,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, memories, token_type_ids, input_mask, position_ids, token_labels) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attn_mask": input_mask,
            "memories": memories,
            "position_ids": position_ids,
        }
        return config, inputs_dict

    def create_and_check_model(
        self,
        config,
        input_ids,
        memories,
        token_type_ids,
        input_mask,
        position_ids,
        token_labels,
    ):
        model = ErnieDocModel(config)
        model.eval()
        result = model(
            input_ids,
            memories=memories,
            attn_mask=input_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.hidden_size])

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        memories,
        token_type_ids,
        input_mask,
        position_ids,
        token_labels,
    ):
        model = ErnieDocForQuestionAnswering(config)
        model.eval()
        result = model(
            input_ids,
            memories=memories,
            attn_mask=input_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        start_logits, end_logits = result[0], result[1]

        self.parent.assertEqual(start_logits.shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(end_logits.shape, [self.batch_size, self.seq_length])

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        memories,
        token_type_ids,
        input_mask,
        position_ids,
        token_labels,
    ):
        model = ErnieDocForSequenceClassification(config)
        model.eval()
        result = model(
            input_ids,
            memories=memories,
            attn_mask=input_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        if position_ids is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0][1].shape, [self.batch_size, self.memory_len, self.hidden_size])

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        memories,
        token_type_ids,
        input_mask,
        position_ids,
        token_labels,
    ):
        model = ErnieDocForTokenClassification(config)
        model.eval()
        result = model(
            input_ids,
            memories=memories,
            attn_mask=input_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.num_classes])


class ErnieDocModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = ErnieDocModel
    return_dict: bool = False
    use_labels: bool = False
    use_test_inputs_embeds: bool = True

    all_model_classes = (
        ErnieDocModel,
        ErnieDocForSequenceClassification,
        ErnieDocForTokenClassification,
        ErnieDocForQuestionAnswering,
    )

    def setUp(self):
        self.model_tester = ErnieDocModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_inputs_embeds(self):
        # Direct input embedding tokens is currently not supported
        self.skipTest("Direct input embedding tokens is currently not supported")

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(ErnieDocPretrainedModel.pretrained_init_configuration)[:1]:
            model = ErnieDocModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
