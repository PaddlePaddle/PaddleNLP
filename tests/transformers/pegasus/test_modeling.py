# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021, The HuggingFace Inc. team. All rights reserved.
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

import tempfile
import unittest

import paddle

from paddlenlp.transformers import (
    PegasusConfig,
    PegasusDecoder,
    PegasusEncoder,
    PegasusForConditionalGeneration,
    PegasusModel,
)

from ..test_configuration_common import ConfigTester
from ..test_generation_utils import GenerationTesterMixin
from ..test_modeling_common import ModelTesterMixin, ids_tensor


class PegasusModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

        # forcing a certain token to be generated, sets all other tokens to -inf
        # if however the token to be generated is already at -inf then it can lead token
        # `nan` values and thus break generation
        self.forced_bos_token_id = None
        self.forced_eos_token_id = None

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64")
        input_ids = paddle.clip(ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64"), 3)
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64")

        config = self.get_config()
        attention_mask = (
            paddle.cast(input_ids == config.pad_token_id, dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
        )
        decoder_attention_mask = (
            paddle.cast(decoder_input_ids == config.pad_token_id, dtype=paddle.get_default_dtype()).unsqueeze([1, 2])
            * -1e4
        )
        inputs_dict = {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return config, inputs_dict

    def get_config(self):
        return PegasusConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            forced_bos_token_id=self.forced_bos_token_id,
            forced_eos_token_id=self.forced_eos_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = PegasusModel(config=config)
        encoder = model.get_encoder()
        decoder = model.get_decoder()
        encoder.eval()
        decoder.eval()

        input_ids = inputs_dict["input_ids"]
        decoder_input_ids = (
            paddle.zeros_like(input_ids[:, :1], dtype="int64") + PegasusModel(config).decoder_start_token_id
        )

        attention_mask = inputs_dict["attention_mask"]
        decoder_attention_mask = paddle.zeros([input_ids.shape[0], 1, 1, 1], dtype=paddle.get_default_dtype())

        encoder_output = encoder(input_ids, attention_mask)
        origin_cache = decoder.decoder.gen_cache(encoder_output)
        outputs = decoder(
            decoder_input_ids,
            decoder_attention_mask,
            encoder_output,
            attention_mask,
            cache=origin_cache,
        )

        output, cache = outputs

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size, dtype="int64")
        next_attn_mask = paddle.zeros([self.batch_size, 1, 1, 3], dtype=paddle.get_default_dtype())

        # append to next input_ids and
        next_input_ids = paddle.concat([decoder_input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.concat([decoder_attention_mask, next_attn_mask], axis=-1)

        output_from_no_past, _ = decoder(next_input_ids, next_attention_mask, encoder_output, attention_mask)
        output_from_past, _ = decoder(
            next_tokens,
            next_attention_mask,
            encoder_output,
            attention_mask,
            cache=cache,
        )

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1], dtype="int64").item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-2))

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = PegasusModel(config=config)
        model.eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs[2]
        last_hidden_state = outputs[0]

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = PegasusEncoder.from_pretrained(tmpdirname)
            encoder.eval()
        encoder_last_hidden_state_2 = encoder(inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"])

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = PegasusDecoder.from_pretrained(tmpdirname)
            decoder.eval()

        last_hidden_state_2 = decoder(
            decoder_input_ids=inputs_dict["decoder_input_ids"],
            decoder_attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_output=encoder_last_hidden_state,
            memory_mask=inputs_dict["attention_mask"],
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


class PegasusModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = PegasusModel
    all_model_classes = (PegasusModel, PegasusForConditionalGeneration)
    all_generative_model_classes = {PegasusForConditionalGeneration: (PegasusModel, "pegasus")}

    is_encoder_decoder = True
    fx_compatible = True
    test_resize_position_embeddings = False
    test_pruning = False
    test_missing_keys = False
    use_labels = False
    use_test_model_name_list = False
    use_test_inputs_embeds = False
    return_dict = False

    def setUp(self):
        self.model_tester = PegasusModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PegasusConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2 = model_class.from_pretrained(tmpdirname)

            missing_keys = []
            for k in model2.state_dict().keys():
                if k not in model.state_dict().keys():
                    missing_keys.append(k)

            self.assertEqual(missing_keys, [])

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_encoder_decoder_model_standalone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_encoder_decoder_model_standalone(*config_and_inputs)

    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_ids = input_dict["input_ids"]
        attention_mask = paddle.cast(input_ids == 1, dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
        model = PegasusForConditionalGeneration(config=config)
        model.eval()
        with paddle.amp.auto_cast():
            model.generate(input_ids, attention_mask=attention_mask)
            model.generate(
                decode_strategy="beam_search",
                num_beams=4,
                do_sample=True,
                early_stopping=False,
                num_return_sequences=3,
            )
