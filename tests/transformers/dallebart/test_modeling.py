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

from paddlenlp.transformers import (
    DalleBartConfig,
    DalleBartForConditionalGeneration,
    DalleBartModel,
)

from ..test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


class DalleBartModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=False,
        text_vocab_size=99,
        image_vocab_size=1024,
        max_text_length=12,
        max_image_length=32,
        bos_token_id=1024,
        pad_token_id=1024,
        eos_token_id=1024,
        decoder_start_token_id=1024,
        d_model=32,
        num_encoder_layers=4,
        num_decoder_layers=4,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        dropout=0.0,
        activation_function="gelu",
        attention_dropout=0.0,
        activation_dropout=0.0,
        use_bias=False,
        init_std=0.02,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.text_vocab_size = text_vocab_size
        self.image_vocab_size = image_vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_text_length = max_text_length
        self.max_image_length = max_image_length
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.dropout = dropout
        self.activation_function = activation_function
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.use_bias = use_bias
        self.init_std = init_std
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.text_vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()
        return config, input_ids, input_mask

    def get_config(self):
        return DalleBartConfig(
            text_vocab_size=self.text_vocab_size,
            image_vocab_size=self.image_vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            max_text_length=self.max_text_length,
            max_image_length=self.max_image_length,
            d_model=self.d_model,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            encoder_attention_heads=self.encoder_attention_heads,
            decoder_attention_heads=self.decoder_attention_heads,
            encoder_ffn_dim=self.encoder_ffn_dim,
            decoder_ffn_dim=self.decoder_ffn_dim,
            dropout=self.dropout,
            activation_function=self.activation_function,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            use_bias=self.use_bias,
            init_std=self.init_std,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = DalleBartModel(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)

        self.parent.assertEqual(result[0].shape, [self.seq_length, self.d_model])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict

    def create_and_check_conditional_generation(self, config, input_ids, input_mask):
        model = DalleBartForConditionalGeneration(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)

        self.parent.assertEqual(result[0].shape, [self.seq_length, self.image_vocab_size + 1])


class DalleBartModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = DalleBartModel
    return_dict: bool = False
    use_labels: bool = False
    use_test_inputs_embeds: bool = True

    all_model_classes = (DalleBartForConditionalGeneration, DalleBartModel)

    def setUp(self):
        self.model_tester = DalleBartModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_conditional_generation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_conditional_generation(*config_and_inputs)

    def test_inputs_embeds(self):
        # Direct input embedding tokens is currently not supported
        self.skipTest("Direct input embedding tokens is currently not supported")
