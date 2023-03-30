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

import paddle

from paddlenlp.transformers import (
    BlenderbotSmallConfig,
    BlenderbotSmallForCausalLM,
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallModel,
    BlenderbotSmallPretrainedModel,
)

from ...testing_utils import slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor


class BlenderbotSmallModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        bos_token_id=1,
        pad_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=1,
        d_model=32,
        num_encoder_layers=2,
        num_decoder_layers=4,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        dropout=0.1,
        activation_function="gelu",
        attention_dropout=0.0,
        activation_dropout=0.0,
        max_position_embeddings=128,
        init_std=0.02,
        scale_embedding=True,
        normalize_before=True,
        scope=None,
    ):
        self.parent = parent
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.num_decoder_layers = num_decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.encoder_ffn_dim = encoder_ffn_dim
        self.normalize_before = normalize_before
        self.decoder_ffn_dim = decoder_ffn_dim
        self.dropout = dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.init_std = init_std
        self.scale_embedding = scale_embedding
        self.pad_token_id = pad_token_id
        self.attention_dropout = attention_dropout
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = paddle.ones([self.batch_size, self.seq_length], dtype="float32")

        config = self.get_config()
        return config, input_ids, input_mask

    def get_config(self):
        return BlenderbotSmallConfig(
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            d_model=self.d_model,
            num_encoder_layers=self.num_encoder_layers,
            encoder_attention_heads=self.encoder_attention_heads,
            num_decoder_layers=self.num_decoder_layers,
            decoder_attention_heads=self.decoder_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            encoder_ffn_dim=self.encoder_ffn_dim,
            normalize_before=self.normalize_before,
            decoder_ffn_dim=self.decoder_ffn_dim,
            dropout=self.dropout,
            activation_function=self.activation_function,
            activation_dropout=self.activation_dropout,
            init_std=self.init_std,
            scale_embedding=self.scale_embedding,
            pad_token_id=self.pad_token_id,
            attention_dropout=self.attention_dropout,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict

    def create_and_check_model(
        self,
        config,
        input_ids,
        input_mask,
    ):
        model = BlenderbotSmallModel(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
        )
        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.d_model])

    def create_and_check_conditiona_generation_model(
        self,
        config,
        input_ids,
        input_mask,
    ):
        model = BlenderbotSmallForConditionalGeneration(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
        )
        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_causal_lm_model(
        self,
        config,
        input_ids,
        input_mask,
    ):
        model = BlenderbotSmallForCausalLM(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
        )
        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.vocab_size])


class BlenderbotSmallModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = BlenderbotSmallModel
    return_dict: bool = False
    use_labels: bool = False
    use_test_inputs_embeds: bool = False

    all_model_classes = (
        BlenderbotSmallModel,
        BlenderbotSmallForConditionalGeneration,
        BlenderbotSmallForCausalLM,
    )

    def setUp(self):
        self.model_tester = BlenderbotSmallModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_conditiona_generation_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_conditiona_generation_model(*config_and_inputs)

    def test_causal_lm_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(BlenderbotSmallPretrainedModel.pretrained_init_configuration)[:1]:
            model = BlenderbotSmallModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
