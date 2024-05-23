# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Google MT5 Authors and HuggingFace Inc. team.
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
import tempfile
import unittest

import numpy as np
import paddle
from parameterized import parameterized_class

from paddlenlp.transformers import (
    MT5_PRETRAINED_MODEL_ARCHIVE_LIST,
    MT5Config,
    MT5EncoderModel,
    MT5ForConditionalGeneration,
    MT5Model,
    T5Tokenizer,
)
from tests.testing_utils import require_package, slow
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import ModelTesterMixin, ids_tensor


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def make_model_instance(config: MT5Config, model_class, base_model_class):
    if model_class == base_model_class:
        return model_class(config)
    else:
        return model_class(base_model_class(config))


class MT5ModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        decoder_seq_length=9,
        # For common tests
        is_training=True,
        use_attention_mask=True,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        dropout_rate=0.1,
        initializer_factor=0.002,
        eos_token_id=1,
        pad_token_id=0,
        scope=None,
        decoder_layers=None,
    ):

        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = pad_token_id
        self.scope = None
        self.decoder_layers = decoder_layers

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)
        decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        attention_mask = None
        decoder_attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)
            decoder_attention_mask = ids_tensor([self.batch_size, self.decoder_seq_length], vocab_size=2)

        lm_labels = None
        if self.parent.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        config = self.get_config()

        return (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    def get_pipeline_config(self) -> MT5Config:
        return MT5Config(
            vocab_size=66,  # mt5 forces 0 extra tokens
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
        )

    def get_config(self) -> MT5Config:
        return MT5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
        )

    def check_prepare_lm_labels_via_shift_left(
        self,
        config: MT5Config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        if not self.parent.use_labels:
            return
        model = MT5Model(config)
        model.eval()

        # make sure that lm_labels are correctly padded from the right
        lm_labels = masked_fill(lm_labels, (lm_labels == self.decoder_start_token_id), self.eos_token_id)

        # add casaul pad token mask
        triangular_mask = paddle.tril(paddle.ones(lm_labels.shape)).logical_not()
        lm_labels = masked_fill(lm_labels, triangular_mask, self.pad_token_id)
        decoder_input_ids = model._shift_right(lm_labels)

        for i, (decoder_input_ids_slice, lm_labels_slice) in enumerate(zip(decoder_input_ids, lm_labels)):
            # first item
            self.parent.assertEqual(decoder_input_ids_slice[0].item(), self.decoder_start_token_id)
            if i < decoder_input_ids_slice.shape[-1]:
                if i < decoder_input_ids.shape[-1] - 1:
                    # items before diagonal
                    self.parent.assertListEqual(
                        decoder_input_ids_slice[1 : i + 1].tolist(), lm_labels_slice[:i].tolist()
                    )
                # pad items after diagonal
                if i < decoder_input_ids.shape[-1] - 2:
                    self.parent.assertListEqual(
                        decoder_input_ids_slice[i + 2 :].tolist(), lm_labels_slice[i + 1 : -1].tolist()
                    )
            else:
                # all items after square
                self.parent.assertListEqual(decoder_input_ids_slice[1:].tolist(), lm_labels_slice[:-1].tolist())

    def create_and_check_model(
        self,
        config: MT5Config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = MT5Model(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=self.parent.return_dict,
        )
        result = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=self.parent.return_dict)
        decoder_output = result[0]
        decoder_past = result[1]
        encoder_output = result[2]

        self.parent.assertEqual(encoder_output.shape, [self.batch_size, self.encoder_seq_length, self.hidden_size])
        self.parent.assertEqual(decoder_output.shape, [self.batch_size, self.decoder_seq_length, self.hidden_size])
        # There should be `num_layers` key value embeddings stored in decoder_past
        self.parent.assertEqual(len(decoder_past), config["num_layers"])
        # There should be a self attn key, a self attn value, a cross attn key and a cross attn value stored in each decoder_past tuple
        self.parent.assertEqual(len(decoder_past[0]), 4)

    def create_and_check_with_lm_head(
        self,
        config: MT5Config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = MT5ForConditionalGeneration(config)
        model.eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
            return_dict=self.parent.return_dict,
        )
        self.parent.assertEqual(len(outputs), 4 if self.parent.use_labels else 3)
        if self.parent.use_labels:
            self.parent.assertEqual(outputs[1].shape, [self.batch_size, self.decoder_seq_length, self.vocab_size])
            self.parent.assertIsInstance(outputs[0].item(), float)
        else:
            self.parent.assertEqual(outputs[0].shape, [self.batch_size, self.decoder_seq_length, self.vocab_size])

    def create_and_check_decoder_model_past(
        self,
        config: MT5Config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = MT5Model(config).get_decoder()
        model.eval()
        # first forward pass
        outputs = model(input_ids, use_cache=True, return_dict=self.parent.return_dict)
        outputs_use_cache_conf = model(input_ids, return_dict=self.parent.return_dict)
        outputs_no_past = model(input_ids, use_cache=False, return_dict=self.parent.return_dict)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf) + 1)
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        output, past_key_values = outputs[:2]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor([self.batch_size, 1], config["vocab_size"])

        # append to next input_ids and
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)

        output_from_no_past = model(next_input_ids, return_dict=self.parent.return_dict)[0]
        output_from_past = model(next_tokens, cache=past_key_values, return_dict=self.parent.return_dict)[0]

        # select random slice
        random_slice_idx = ids_tensor(
            [
                1,
            ],
            output_from_past.shape[-1],
        ).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_decoder_model_attention_mask_past(
        self,
        config: MT5Config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = MT5Model(config).get_decoder()
        model.eval()

        # create attention mask
        attn_mask = paddle.ones(input_ids.shape, dtype="int64")

        half_seq_length = input_ids.shape[-1] // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        output, past_key_values = model(
            input_ids, attention_mask=attn_mask, use_cache=True, return_dict=self.parent.return_dict
        )[:2]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor([self.batch_size, 1], config["vocab_size"])

        # change a random masked slice from input_ids
        random_seq_idx_to_change = (
            ids_tensor(
                [
                    1,
                ],
                half_seq_length,
            ).item()
            + 1
        )
        random_other_next_tokens = ids_tensor([self.batch_size, 1], config["vocab_size"]).squeeze(-1)
        input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens

        # append to next input_ids and attn_mask
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        attn_mask = paddle.concat(
            [attn_mask, paddle.ones((attn_mask.shape[0], 1), dtype="int64")],
            axis=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask, return_dict=self.parent.return_dict)[0]
        output_from_past = model(
            next_tokens,
            cache=past_key_values,
            attention_mask=paddle.ones((attn_mask.shape[0], 1), dtype="int64"),
            return_dict=self.parent.return_dict,
        )[0]

        # select random slice
        random_slice_idx = ids_tensor(
            [
                1,
            ],
            output_from_past.shape[-1],
        ).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config: MT5Config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = MT5Model(config).get_decoder()
        model.eval()
        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True, return_dict=self.parent.return_dict)

        output, past_key_values = outputs[:2]

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor([self.batch_size, 3], config["vocab_size"])
        next_mask = ids_tensor([self.batch_size, 3], vocab_size=2)

        # append to next input_ids and
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.concat([attention_mask, next_mask], axis=-1)

        output_from_no_past = model(
            next_input_ids, attention_mask=next_attention_mask, return_dict=self.parent.return_dict
        )[0]
        output_from_past = model(
            next_tokens, attention_mask=next_attention_mask, cache=past_key_values, return_dict=self.parent.return_dict
        )[0]

        # select random slice
        random_slice_idx = ids_tensor(
            [
                1,
            ],
            output_from_past.shape[-1],
        ).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_generate_with_past_key_values(
        self,
        config: MT5Config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        paddle.seed(0)
        np.random.seed(0)
        random.seed(0)

        model = MT5ForConditionalGeneration(config)
        model.eval()

        output_without_past_cache, _ = model.generate(
            input_ids[:1], top_k=1, max_length=5, decode_strategy="sampling", use_cache=False
        )

        paddle.seed(0)
        np.random.seed(0)
        random.seed(0)

        output_with_past_cache, _ = model.generate(input_ids[:1], top_k=1, max_length=5, decode_strategy="sampling")

        self.parent.assertTrue(paddle.all(output_with_past_cache == output_without_past_cache))

    def check_resize_embeddings_mt5_v1_1(
        self,
        config: MT5Config,
    ):
        prev_vocab_size = config["vocab_size"]

        model = MT5ForConditionalGeneration(config)
        model.eval()
        model.resize_token_embeddings(prev_vocab_size - 10)

        self.parent.assertEqual(model.get_input_embeddings().weight.shape[0], prev_vocab_size - 10)
        self.parent.assertEqual(model.get_output_embeddings().weight.shape[0], prev_vocab_size - 10)
        self.parent.assertEqual(model.mt5.config["vocab_size"], prev_vocab_size - 10)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "use_cache": False,
        }
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
class MT5ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = MT5Model
    return_dict: bool = False
    use_labels: bool = False

    all_model_classes = (MT5Model, MT5ForConditionalGeneration)
    all_generative_model_classes = {MT5ForConditionalGeneration: (MT5Model, "mt5")}
    all_parallelizable_model_classes = (MT5Model, MT5ForConditionalGeneration)
    fx_compatible = True
    test_pruning = False
    test_resize_embeddings = True
    test_model_parallel = True
    use_test_inputs_embeds = True
    is_encoder_decoder = True
    use_test_model_name_list = False
    # The small MT5 model needs higher percentages for CPU/MP tests
    model_split_percents = [0.8, 0.9]

    def setUp(self):
        self.model_tester = MT5ModelTester(self)

    def test_shift_right(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_prepare_lm_labels_via_shift_left(*config_and_inputs)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_v1_1(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        # check that gated gelu feed forward and different word embeddings work
        config = config_and_inputs[0]
        config["feed_forward_proj"] = "gated-gelu"
        self.model_tester.create_and_check_model(config, *config_and_inputs[1:])

    def test_config_and_model_silu_gated(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]
        config["feed_forward_proj"] = "gated-silu"
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_with_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_lm_head(*config_and_inputs)

    def test_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past(*config_and_inputs)

    def test_decoder_model_past_with_attn_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_attention_mask_past(*config_and_inputs)

    def test_decoder_model_past_with_3d_attn_mask(self):
        (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = self.model_tester.prepare_config_and_inputs()

        attention_mask = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.encoder_seq_length, self.model_tester.encoder_seq_length],
            vocab_size=2,
        )
        decoder_attention_mask = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.decoder_seq_length, self.model_tester.decoder_seq_length],
            vocab_size=2,
        )

        self.model_tester.create_and_check_decoder_model_attention_mask_past(
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_generate_with_past_key_values(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_generate_with_past_key_values(*config_and_inputs)

    def test_v1_1_resize_embeddings(self):
        config = self.model_tester.prepare_config_and_inputs()[0]
        self.model_tester.check_resize_embeddings_mt5_v1_1(config)

    @slow
    def test_model_from_pretrained(self):
        for model_name in MT5_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = MT5Model.from_pretrained(model_name)
            self.assertIsNotNone(model)


class MT5EncoderOnlyModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        # For common tests
        use_attention_mask=True,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        is_training=False,
        dropout_rate=0.1,
        initializer_factor=0.002,
        is_encoder_decoder=False,
        eos_token_id=1,
        pad_token_id=0,
        scope=None,
    ):

        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        # For common tests
        self.seq_length = self.encoder_seq_length
        self.use_attention_mask = use_attention_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.scope = None
        self.is_training = is_training

    def get_config(self):
        config = MT5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )
        return config

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)

        config = self.get_config()
        return (
            config,
            input_ids,
            attention_mask,
        )

    def create_and_check_model(
        self,
        config: MT5Config,
        input_ids,
        attention_mask,
    ):
        model = MT5EncoderModel(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        result = model(input_ids=input_ids)
        encoder_output = result[0]

        self.parent.assertEqual(encoder_output.shape, [self.batch_size, self.encoder_seq_length, self.hidden_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


class MT5EncoderOnlyModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (MT5EncoderModel,)
    test_pruning = False
    test_resize_embeddings = False
    test_model_parallel = True
    all_parallelizable_model_classes = (MT5EncoderModel,)

    def _make_model_instance(self, config: MT5Config, model_class):
        return model_class(config)

    def setUp(self):
        self.model_tester = MT5EncoderOnlyModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_name_list(self):
        pass


class MT5CompatibilityTest(unittest.TestCase):
    @require_package("transformers", "torch")
    @slow
    def test_mt5_converter(self):
        with tempfile.TemporaryDirectory() as tempdir:
            model_id = "google/mt5-small"
            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the paddle model
            from paddlenlp.transformers import MT5Model

            paddle_model = MT5Model.from_pretrained(model_id, cache_dir=tempdir)
            paddle_model.eval()
            paddle_logit = paddle_model(
                input_ids=paddle.to_tensor(input_ids), decoder_input_ids=paddle.to_tensor(input_ids)
            )[0][0]

            # 3. forward the torch  model
            import torch
            from transformers import MT5Model

            torch_model = MT5Model.from_pretrained(model_id, cache_dir=tempdir)
            torch_model.eval()
            torch_logit = torch_model(
                input_ids=torch.tensor(input_ids), decoder_input_ids=torch.tensor(input_ids), return_dict=False
            )[0][0]
            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().numpy()[:4, :4], torch_logit.detach().cpu().numpy()[:4, :4], rtol=1e-3
                )
            )

    @require_package("transformers", "torch")
    @slow
    def test_mt5_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            model_id = "google/mt5-small"
            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch  model
            import torch
            from transformers import MT5Model

            torch_model = MT5Model.from_pretrained(model_id)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(
                input_ids=torch.tensor(input_ids), decoder_input_ids=torch.tensor(input_ids), return_dict=False
            )[0][0]

            # 2. forward the paddle model
            from paddlenlp.transformers import MT5Model

            paddle_model = MT5Model.from_pretrained(tempdir, convert_from_torch=True)
            paddle_model.eval()
            paddle_logit = paddle_model(
                input_ids=paddle.to_tensor(input_ids), decoder_input_ids=paddle.to_tensor(input_ids)
            )[0][0]

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    rtol=1e-3,
                )
            )

    @require_package("transformers", "torch")
    @slow
    def test_mt5_for_conditional_generation(self):
        with tempfile.TemporaryDirectory() as tempdir:
            model_id = "google/mt5-small"
            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch  model
            import torch
            from transformers import MT5ForConditionalGeneration

            torch_model = MT5ForConditionalGeneration.from_pretrained(model_id)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(
                input_ids=torch.tensor(input_ids), decoder_input_ids=torch.tensor(input_ids), return_dict=False
            )[0][0]

            # 2. forward the paddle model
            from paddlenlp.transformers import MT5ForConditionalGeneration

            paddle_model = MT5ForConditionalGeneration.from_pretrained(tempdir)
            paddle_model.eval()
            paddle_logit = paddle_model(
                input_ids=paddle.to_tensor(input_ids), decoder_input_ids=paddle.to_tensor(input_ids)
            )[0][0]

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    atol=1e-3,
                )
            )


class MT5ModelIntegrationTests(unittest.TestCase):
    @slow
    def test_small_integration_test(self):

        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

        model.eval()

        input_ids = tokenizer("Hello there", return_tensors="pd")["input_ids"]
        labels = tokenizer("Hi I am", return_tensors="pd")["input_ids"]

        loss = model(input_ids, labels=labels)[0]
        mtf_score = -(labels.shape[-1] * loss.item())

        EXPECTED_SCORE = -84.9127
        self.assertTrue(abs(mtf_score - EXPECTED_SCORE) < 1e-4)
