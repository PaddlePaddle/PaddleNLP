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
    ProphetNetConfig,
    ProphetNetForConditionalGeneration,
    ProphetNetModel,
    ProphetNetPretrainedModel,
)

from ...testing_utils import slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor


class ProphetNetModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        seq_length=7,
        tgt_seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        bos_token_id=1,
        pad_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=1,
        hidden_size=32,
        relative_max_distance=32,
        ngram=2,
        num_buckets=8,
        num_encoder_layers=2,
        num_decoder_layers=4,
        num_encoder_attention_heads=4,
        num_decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        dropout=0.1,
        activation_function="gelu",
        attention_dropout=0.1,
        activation_dropout=0.1,
        max_position_embeddings=128,
        init_std=0.02,
        eps=0.1,
        add_cross_attention=True,
        disable_ngram_loss=False,
    ):
        self.parent = parent
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.tgt_seq_length = tgt_seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_encoder_attention_heads = num_encoder_attention_heads
        self.num_decoder_layers = num_decoder_layers
        self.num_decoder_attention_heads = num_decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.encoder_ffn_dim = encoder_ffn_dim
        self.add_cross_attention = add_cross_attention
        self.decoder_ffn_dim = decoder_ffn_dim
        self.dropout = dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.init_std = init_std
        self.disable_ngram_loss = disable_ngram_loss
        self.pad_token_id = pad_token_id
        self.attention_dropout = attention_dropout
        self.eps = eps
        self.ngram = ngram
        self.relative_max_distance = relative_max_distance
        self.num_buckets = num_buckets

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        decoder_input_ids = ids_tensor([self.batch_size, self.tgt_seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = paddle.ones([self.batch_size, self.seq_length], dtype="float32")
        decoder_input_mask = None
        if self.use_input_mask:
            decoder_input_mask = paddle.ones([self.batch_size, self.tgt_seq_length], dtype="float32")

        config = self.get_config()
        return config, input_ids, input_mask, decoder_input_ids, decoder_input_mask

    def get_config(self):
        return ProphetNetConfig(
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            hidden_size=self.hidden_size,
            decoder_start_token_id=self.decoder_start_token_id,
            max_position_embeddings=self.max_position_embeddings,
            activation_function=self.activation_function,
            activation_dropout=self.activation_dropout,
            dropout=self.dropout,
            relative_max_distance=self.relative_max_distance,
            ngram=self.ngram,
            num_buckets=self.num_buckets,
            encoder_ffn_dim=self.encoder_ffn_dim,
            num_encoder_attention_heads=self.num_encoder_attention_heads,
            num_encoder_layers=self.num_encoder_layers,
            decoder_ffn_dim=self.decoder_ffn_dim,
            num_decoder_attention_heads=self.num_decoder_attention_heads,
            num_decoder_layers=self.num_decoder_layers,
            attention_dropout=self.attention_dropout,
            init_std=self.init_std,
            eps=self.eps,
            add_cross_attention=self.add_cross_attention,
            disable_ngram_loss=self.disable_ngram_loss,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask, decoder_input_ids, decoder_input_mask) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_input_mask,
        }
        return config, inputs_dict

    def create_and_check_model(self, config, input_ids, input_mask, decoder_input_ids, decoder_attention_mask):
        model = ProphetNetModel(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_conditional_generation_model(
        self, config, input_ids, input_mask, decoder_input_ids, decoder_attention_mask
    ):
        model = ProphetNetForConditionalGeneration(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])


class ProphetNetModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = ProphetNetModel
    return_dict: bool = False
    use_labels: bool = False
    use_test_inputs_embeds: bool = False

    all_model_classes = (
        ProphetNetModel,
        ProphetNetForConditionalGeneration,
    )

    def setUp(self):
        self.model_tester = ProphetNetModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_conditional_generation_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_conditional_generation_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(ProphetNetPretrainedModel.pretrained_init_configuration)[:1]:
            model = ProphetNetModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
