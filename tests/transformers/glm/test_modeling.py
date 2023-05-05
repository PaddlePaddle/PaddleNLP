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

import random

import numpy as np
import paddle

from paddlenlp.transformers import (
    GLMConfig,
    GLMForConditionalGeneration,
    GLMForMultipleChoice,
    GLMModel,
    GLMTokenizer,
)
from tests.testing_utils import PaddleNLPModelTest, slow
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)

GLM_PRETRAINED_MODEL_ARCHIVE_LIST = ["THUDM/glm-515m", "THUDM/glm-2b", "THUDM/glm-large-chinese"]


class GLMModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_attention_mask=True,
        use_position_ids=True,
        num_layers=5,
        vocab_size=99,
        hidden_size=32,
        num_attention_heads=4,
        embedding_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        output_dropout_prob=0.1,
        max_sequence_length=512,
        checkpoint_activations=False,
        checkpoint_num_layers=1,
        parallel_output=True,
        relative_encoding=False,
        block_position_encoding=True,
        output_predict=False,
        spell_length=None,
        spell_func="lstm",
        attention_scale=1.0,
        initializer_range=0.02,
        type_vocab_size=16,
        type_sequence_label_size=2,
        pool_token="cls",
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_position_ids = use_position_ids
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.max_sequence_length = max_sequence_length
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.parallel_output = parallel_output
        self.relative_encoding = relative_encoding
        self.block_position_encoding = block_position_encoding
        self.output_predict = output_predict
        self.spell_length = spell_length
        self.spell_func = spell_func
        self.attention_scale = attention_scale
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.pool_token = pool_token
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = None

    def prepare_config_and_inputs(self, model_class):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64")

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = random_attention_mask(
                [self.batch_size, 1, self.seq_length, self.seq_length], dtype="int64"
            )

        position_ids = None
        if self.use_position_ids:
            position_ids = paddle.arange(0, self.seq_length, dtype="int64").unsqueeze(0).unsqueeze(1)
            position_ids = paddle.expand(position_ids, shape=[self.batch_size, 2, -1])

        sequence_labels = None
        choice_labels = None
        if self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size, dtype="int64")
            choice_labels = ids_tensor([self.batch_size], self.num_choices, dtype="int64")

        return (
            config,
            input_ids,
            position_ids,
            attention_mask,
            sequence_labels,
            choice_labels,
        )

    def get_config(self):
        return GLMConfig(
            num_layers=self.num_layers,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            embedding_dropout_prob=self.embedding_dropout_prob,
            attention_dropout_prob=self.attention_dropout_prob,
            output_dropout_prob=self.output_dropout_prob,
            max_sequence_length=self.max_sequence_length,
            checkpoint_activations=self.checkpoint_activations,
            checkpoint_num_layers=self.checkpoint_num_layers,
            parallel_output=self.parallel_output,
            relative_encoding=self.relative_encoding,
            block_position_encoding=self.block_position_encoding,
            output_predict=self.output_predict,
            spell_length=self.spell_length,
            spell_func=self.spell_func,
            attention_scale=self.attention_scale,
            initializer_range=self.initializer_range,
            pool_token=self.pool_token,
            use_scaled_init_for_output_weights=True,
            layernorm_epsilon=1e-5,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        position_ids,
        attention_mask,
        sequence_labels,
        choice_labels,
    ):
        model = GLMModel(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(len(result.past_key_values), config["num_layers"] + 1)

    def create_and_check_for_multiple_choice(
        self,
        config,
        input_ids,
        position_ids,
        attention_mask,
        sequence_labels,
        choice_labels,
    ):
        self.parent.assertEqual(position_ids.shape, [self.batch_size, 2, self.seq_length])
        config.output_predict = True
        model = GLMForMultipleChoice(config=config)
        model.eval()
        choice_labels = ids_tensor([self.batch_size, self.num_choices], self.num_choices, dtype="int64")
        choice_indices = paddle.to_tensor([[x for x in batch] for batch in choice_labels], dtype="int64")
        choice_ids = paddle.to_tensor(
            [[x for x in batch] for batch in ids_tensor(choice_labels.shape, vocab_size=self.vocab_size)],
            dtype="int64",
        )

        result = model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            choice_ids=choice_ids,
            choice_indices=choice_indices,
            return_dict=True,
        )

        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.num_choices])

    def create_and_check_for_conditional_generation(
        self,
        config,
        input_ids,
        position_ids,
        attention_mask,
        sequence_labels,
        choice_labels,
    ):
        model = GLMForConditionalGeneration(config=config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        self.parent.assertEqual(len(result.past_key_values), self.num_layers + 1)
        self.parent.assertEqual(result.past_key_values[0].shape, [self.seq_length, self.seq_length, self.hidden_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs("GLMModel")
        (
            config,
            input_ids,
            position_ids,
            attention_mask,
            sequence_labels,
            choice_labels,
        ) = config_and_inputs

        input_dict = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

        return config, input_dict


class GLMModelTest(ModelTesterMixin, GenerationTesterMixin, PaddleNLPModelTest):
    base_model_class = GLMModel
    use_labels = False
    return_dict = False

    all_model_classes = (GLMModel,)
    all_generative_model_classes = {}
    test_missing_keys = False
    test_model_parallel = True
    use_test_input_embeds = False

    def setUp(self):
        self.model_tester = GLMModelTester(self)
        random.seed(128)
        np.random.seed(128)
        paddle.seed(128)

    def test_glm_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs("GLMModel")
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs("GLMForMultipleChoice")
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_conditional_generation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs("GLMForConditionalGeneration")
        self.model_tester.create_and_check_for_conditional_generation(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in GLM_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = GLMModel.from_pretrained(model_name)
            tokenizer = GLMTokenizer.from_pretrained(model_name)
            tokens = tokenizer._encode("hello world [MASK]")
            input_ids, _, _, position_ids, attention_mask, _, _ = tokenizer.build_input_from_ids(
                text_a_ids=tokens, tokenizer=tokenizer
            )
            input_ids = paddle.to_tensor([input_ids])
            position_ids = paddle.to_tensor([position_ids])
            attention_mask = paddle.to_tensor([attention_mask])
            model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
            self.assertIsNotNone(model)
