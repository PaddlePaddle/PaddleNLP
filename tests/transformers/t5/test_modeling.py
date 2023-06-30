# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Google T5 Authors and HuggingFace Inc. team.
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
    T5EncoderModel,
    T5ForConditionalGeneration,
    T5Model,
    T5Tokenizer,
)
from paddlenlp.transformers.t5.configuration import T5Config
from paddlenlp.transformers.t5.modeling import T5_PRETRAINED_MODEL_ARCHIVE_LIST
from tests.testing_utils import require_package, slow

from ..test_generation_utils import GenerationTesterMixin
from ..test_modeling_common import ModelTesterMixin, ids_tensor


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def make_model_instance(config: T5Config, model_class, base_model_class):
    if model_class == base_model_class:
        return model_class(config)
    else:
        return model_class(base_model_class(config))


class T5ModelTester:
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

    def get_pipeline_config(self) -> T5Config:
        return T5Config(
            vocab_size=166,  # t5 forces 100 extra tokens
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

    def get_config(self) -> T5Config:
        return T5Config(
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
        config: T5Config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        if not self.parent.use_labels:
            return
        model = T5Model(config)
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
        config: T5Config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = T5Model(config)
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
        config: T5Config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = T5ForConditionalGeneration(config)
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
        config: T5Config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = T5Model(config).get_decoder()
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
        config: T5Config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = T5Model(config).get_decoder()
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
        config: T5Config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = T5Model(config).get_decoder()
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
        config: T5Config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        paddle.seed(0)
        np.random.seed(0)
        random.seed(0)

        model = T5ForConditionalGeneration(config)
        model.eval()

        output_without_past_cache, _ = model.generate(
            input_ids[:1], top_k=1, max_length=5, decode_strategy="sampling", use_cache=False
        )

        paddle.seed(0)
        np.random.seed(0)
        random.seed(0)

        output_with_past_cache, _ = model.generate(input_ids[:1], top_k=1, max_length=5, decode_strategy="sampling")

        self.parent.assertTrue(paddle.all(output_with_past_cache == output_without_past_cache))

    def check_resize_embeddings_t5_v1_1(
        self,
        config: T5Config,
    ):
        prev_vocab_size = config["vocab_size"]

        model = T5ForConditionalGeneration(config)
        model.eval()
        model.resize_token_embeddings(prev_vocab_size - 10)

        self.parent.assertEqual(model.get_input_embeddings().weight.shape[0], prev_vocab_size - 10)
        self.parent.assertEqual(model.get_output_embeddings().weight.shape[0], prev_vocab_size - 10)
        self.parent.assertEqual(model.t5.config["vocab_size"], prev_vocab_size - 10)

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
class T5ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = T5Model
    return_dict: bool = False
    use_labels: bool = False

    all_model_classes = (T5Model, T5ForConditionalGeneration)
    all_generative_model_classes = {T5ForConditionalGeneration: (T5Model, "t5")}
    all_parallelizable_model_classes = (T5Model, T5ForConditionalGeneration)
    fx_compatible = True
    test_pruning = False
    test_resize_embeddings = True
    test_model_parallel = True
    use_test_inputs_embeds = True
    is_encoder_decoder = True
    # The small T5 model needs higher percentages for CPU/MP tests
    model_split_percents = [0.8, 0.9]

    def setUp(self):
        self.model_tester = T5ModelTester(self)

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
        self.model_tester.check_resize_embeddings_t5_v1_1(config)

    @slow
    def test_model_from_pretrained(self):
        for model_name in T5_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = T5Model.from_pretrained(model_name)
            self.assertIsNotNone(model)


class T5EncoderOnlyModelTester:
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
        config = T5Config(
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
        config: T5Config,
        input_ids,
        attention_mask,
    ):
        model = T5EncoderModel(config)
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


class T5EncoderOnlyModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (T5EncoderModel,)
    test_pruning = False
    test_resize_embeddings = False
    test_model_parallel = True
    all_parallelizable_model_classes = (T5EncoderModel,)

    def _make_model_instance(self, config: T5Config, model_class):
        return model_class(config)

    def setUp(self):
        self.model_tester = T5EncoderOnlyModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_name_list(self):
        pass


class T5CompatibilityTest(unittest.TestCase):
    @require_package("transformers", "torch")
    def test_t5_converter(self):
        with tempfile.TemporaryDirectory() as tempdir:
            model_id = "hf-internal-testing/tiny-random-T5Model"
            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the paddle model
            from paddlenlp.transformers import T5Model

            paddle_model = T5Model.from_pretrained(model_id, from_hf_hub=True, cache_dir=tempdir)
            paddle_model.eval()
            paddle_logit = paddle_model(
                input_ids=paddle.to_tensor(input_ids), decoder_input_ids=paddle.to_tensor(input_ids)
            )[0][0]

            # 3. forward the torch  model
            import torch
            from transformers import T5Model

            torch_model = T5Model.from_pretrained(model_id, cache_dir=tempdir)
            torch_model.eval()
            torch_logit = torch_model(
                input_ids=torch.tensor(input_ids), decoder_input_ids=torch.tensor(input_ids), return_dict=False
            )[0][0]
            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().numpy()[:4, :4], torch_logit.detach().cpu().numpy()[:4, :4], rtol=1e-4
                )
            )

    @require_package("transformers", "torch")
    def test_t5_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            model_id = "hf-internal-testing/tiny-random-T5Model"
            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch  model
            import torch
            from transformers import T5Model

            torch_model = T5Model.from_pretrained(model_id)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(
                input_ids=torch.tensor(input_ids), decoder_input_ids=torch.tensor(input_ids), return_dict=False
            )[0][0]

            # 2. forward the paddle model
            from paddlenlp.transformers import T5Model

            paddle_model = T5Model.from_pretrained(tempdir, convert_from_torch=True)
            paddle_model.eval()
            paddle_logit = paddle_model(
                input_ids=paddle.to_tensor(input_ids), decoder_input_ids=paddle.to_tensor(input_ids)
            )[0][0]

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    rtol=1e-4,
                )
            )

    @require_package("transformers", "torch")
    def test_t5_for_conditional_generation(self):
        with tempfile.TemporaryDirectory() as tempdir:
            model_id = "hf-internal-testing/tiny-random-T5Model"
            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch  model
            import torch
            from transformers import T5ForConditionalGeneration

            torch_model = T5ForConditionalGeneration.from_pretrained(model_id)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(
                input_ids=torch.tensor(input_ids), decoder_input_ids=torch.tensor(input_ids), return_dict=False
            )[0][0]

            # 2. forward the paddle model
            from paddlenlp.transformers import T5ForConditionalGeneration

            paddle_model = T5ForConditionalGeneration.from_pretrained(tempdir, convert_from_torch=True)
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


class T5ModelIntegrationTests(unittest.TestCase):
    def model(self):
        return T5ForConditionalGeneration.from_pretrained("t5-base")

    def tokenizer(self):
        return T5Tokenizer.from_pretrained("t5-base")

    @slow
    def test_small_generation(self):
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")

        model.eval()

        input_ids = tokenizer("summarize: Hello there", return_tensors="pd")["input_ids"]

        sequences = model.generate(input_ids, max_length=8, decode_strategy="greedy_search")[0]

        output_str = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
        self.assertTrue(output_str == "Hello there!")

    @slow
    def test_small_integration_test(self):

        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")

        model.eval()

        input_ids = tokenizer("Hello there", return_tensors="pd")["input_ids"]
        labels = tokenizer("Hi I am", return_tensors="pd")["input_ids"]

        loss = model(input_ids, labels=labels)[0]
        mtf_score = -(labels.shape[-1] * loss.item())

        EXPECTED_SCORE = -19.084566
        self.assertTrue(abs(mtf_score - EXPECTED_SCORE) < 1e-4)

    @slow
    def test_small_v1_1_integration_test(self):

        model = T5ForConditionalGeneration.from_pretrained("t5-v1_1-base")
        tokenizer = T5Tokenizer.from_pretrained("t5-v1_1-base")

        model.eval()

        input_ids = tokenizer("Hello there", return_tensors="pd")["input_ids"]
        labels = tokenizer("Hi I am", return_tensors="pd")["input_ids"]

        loss = model(input_ids, labels=labels)[0]
        mtf_score = -(labels.shape[-1] * loss.item())

        EXPECTED_SCORE = -56.207352
        self.assertTrue(abs(mtf_score - EXPECTED_SCORE) < 1e-4)

    @slow
    def test_summarization(self):
        model = self.model()
        model.eval()
        tok = self.tokenizer()

        FRANCE_ARTICLE = (  # @noqa
            "Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings"
            " Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane."
            ' Marseille prosecutor Brice Robin told CNN that "so far no videos were used in the crash investigation."'
            ' He added, "A person who has such a video needs to immediately give it to the investigators." Robin\'s'
            " comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video"
            " showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French"
            " Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a"
            " phone at the wreckage site. The two publications described the supposed video, but did not post it on"
            " their websites. The publications said that they watched the video, which was found by a source close to"
            " the investigation. \"One can hear cries of 'My God' in several languages,\" Paris Match reported."
            ' "Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the'
            " cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the"
            ' screaming intensifies. Then nothing." "It is a very disturbing scene," said Julian Reichelt,'
            " editor-in-chief of Bild online. An official with France's accident investigation agency, the BEA, said"
            " the agency is not aware of any such video. Lt. Col. Jean-Marc Menichini, a French Gendarmerie spokesman"
            " in charge of communications on rescue efforts around the Germanwings crash site, told CNN that the"
            ' reports were "completely wrong" and "unwarranted." Cell phones have been collected at the site, he said,'
            ' but that they "hadn\'t been exploited yet." Menichini said he believed the cell phones would need to be'
            " sent to the Criminal Research Institute in Rosny sous-Bois, near Paris, in order to be analyzed by"
            " specialized technicians working hand-in-hand with investigators. But none of the cell phones found so"
            " far have been sent to the institute, Menichini said. Asked whether staff involved in the search could"
            ' have leaked a memory card to the media, Menichini answered with a categorical "no." Reichelt told "Erin'
            ' Burnett: Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match'
            ' are "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered'
            ' cell phones from the crash site after Bild and Paris Match published their reports. "That is something'
            " we did not know before. ... Overall we can say many things of the investigation weren't revealed by the"
            ' investigation at the beginning," he said. What was mental state of Germanwings co-pilot? German airline'
            " Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled depression years before he took the"
            " controls of Germanwings Flight 9525, which he's accused of deliberately crashing last week in the"
            ' French Alps. Lubitz told his Lufthansa flight training school in 2009 that he had a "previous episode of'
            ' severe depression," the airline said Tuesday. Email correspondence between Lubitz and the school'
            " discovered in an internal investigation, Lufthansa said, included medical documents he submitted in"
            " connection with resuming his flight training. The announcement indicates that Lufthansa, the parent"
            " company of Germanwings, knew of Lubitz's battle with depression, allowed him to continue training and"
            " ultimately put him in the cockpit. Lufthansa, whose CEO Carsten Spohr previously said Lubitz was 100%"
            ' fit to fly, described its statement Tuesday as a "swift and seamless clarification" and said it was'
            " sharing the information and documents -- including training and medical records -- with public"
            " prosecutors. Spohr traveled to the crash site Wednesday, where recovery teams have been working for the"
            " past week to recover human remains and plane debris scattered across a steep mountainside. He saw the"
            " crisis center set up in Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash"
            " site, where grieving families have left flowers at a simple stone memorial. Menichini told CNN late"
            " Tuesday that no visible human remains were left at the site but recovery teams would keep searching."
            " French President Francois Hollande, speaking Tuesday, said that it should be possible to identify all"
            " the victims using DNA analysis by the end of the week, sooner than authorities had previously suggested."
            " In the meantime, the recovery of the victims' personal belongings will start Wednesday, Menichini said."
            " Among those personal belongings could be more cell phones belonging to the 144 passengers and six crew"
            " on board. Check out the latest from our correspondents . The details about Lubitz's correspondence with"
            " the flight school during his training were among several developments as investigators continued to"
            " delve into what caused the crash and Lubitz's possible motive for downing the jet. A Lufthansa"
            " spokesperson told CNN on Tuesday that Lubitz had a valid medical certificate, had passed all his"
            ' examinations and "held all the licenses required." Earlier, a spokesman for the prosecutor\'s office in'
            " Dusseldorf, Christoph Kumpa, said medical records reveal Lubitz suffered from suicidal tendencies at"
            " some point before his aviation career and underwent psychotherapy before he got his pilot's license."
            " Kumpa emphasized there's no evidence suggesting Lubitz was suicidal or acting aggressively before the"
            " crash. Investigators are looking into whether Lubitz feared his medical condition would cause him to"
            " lose his pilot's license, a European government official briefed on the investigation told CNN on"
            ' Tuesday. While flying was "a big part of his life," the source said, it\'s only one theory being'
            " considered. Another source, a law enforcement official briefed on the investigation, also told CNN that"
            " authorities believe the primary motive for Lubitz to bring down the plane was that he feared he would"
            " not be allowed to fly because of his medical problems. Lubitz's girlfriend told investigators he had"
            " seen an eye doctor and a neuropsychologist, both of whom deemed him unfit to work recently and concluded"
            " he had psychological issues, the European government official said. But no matter what details emerge"
            " about his previous mental health struggles, there's more to the story, said Brian Russell, a forensic"
            ' psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the fact'
            " that maybe they weren't going to keep doing their job and they're upset about that and so they're"
            ' suicidal," he said. "But there is no mental illness that explains why somebody then feels entitled to'
            " also take that rage and turn it outward on 149 other people who had nothing to do with the person's"
            ' problems." Germanwings crash compensation: What we know . Who was the captain of Germanwings Flight'
            " 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from Dusseldorf, while Laura"
            " Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff, Antonia Mortensen, Sandrine"
            " Amiel and Anna-Maja Rappard contributed to this report."
        )
        SHORTER_ARTICLE = (
            "(CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
            " Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The"
            " formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based."
            " The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its"
            ' jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East'
            ' Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the'
            " situation in Palestinian territories, paving the way for possible war crimes investigations against"
            " Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and"
            " the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the"
            " body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a"
            ' move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the'
            ' world is also a step closer to ending a long era of impunity and injustice," he said, according to an'
            ' ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge'
            " Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the"
            ' Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine'
            " acquires all the rights as well as responsibilities that come with being a State Party to the Statute."
            ' These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights'
            ' Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should'
            " immediately end their pressure, and countries that support universal acceptance of the court's treaty"
            ' should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the'
            " group. \"What's objectionable is the attempts to undermine international justice, not Palestine's"
            ' decision to join a treaty to which over 100 countries around the world are members." In January, when'
            " the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an"
            ' outrage, saying the court was overstepping its boundaries. The United States also said it "strongly"'
            " disagreed with the court's decision. \"As we have said repeatedly, we do not believe that Palestine is a"
            ' state and therefore we do not believe that it is eligible to join the ICC," the State Department said in'
            ' a statement. It urged the warring sides to resolve their differences through direct negotiations. "We'
            ' will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace,"'
            " it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the"
            ' territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the'
            " court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou"
            ' Bensouda said her office would "conduct its analysis in full independence and impartiality." The war'
            " between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry"
            " will include alleged war crimes committed since June. The International Criminal Court was set up in"
            " 2002 to prosecute genocide, crimes against humanity and war crimes. CNN's Vasco Cotovio, Kareem Khadder"
            " and Faith Karimi contributed to this report."
        )
        IRAN_ARTICLE = (
            "(CNN)The United States and its negotiating partners reached a very strong framework agreement with Iran"
            " in Lausanne, Switzerland, on Thursday that limits Iran's nuclear program in such a way as to effectively"
            " block it from building a nuclear weapon. Expect pushback anyway, if the recent past is any harbinger."
            " Just last month, in an attempt to head off such an agreement, House Speaker John Boehner invited Israeli"
            " Prime Minister Benjamin Netanyahu to preemptively blast it before Congress, and 47 senators sent a"
            " letter to the Iranian leadership warning them away from a deal. The debate that has already begun since"
            " the announcement of the new framework will likely result in more heat than light. It will not be helped"
            " by the gathering swirl of dubious assumptions and doubtful assertions. Let us address some of these: ."
            " The most misleading assertion, despite universal rejection by experts, is that the negotiations'"
            " objective at the outset was the total elimination of any nuclear program in Iran. That is the position"
            " of Netanyahu and his acolytes in the U.S. Congress. But that is not and never was the objective. If it"
            " had been, there would have been no Iranian team at the negotiating table. Rather, the objective has"
            " always been to structure an agreement or series of agreements so that Iran could not covertly develop a"
            " nuclear arsenal before the United States and its allies could respond. The new framework has exceeded"
            " expectations in achieving that goal. It would reduce Iran's low-enriched uranium stockpile, cut by"
            " two-thirds its number of installed centrifuges and implement a rigorous inspection regime. Another"
            " dubious assumption of opponents is that the Iranian nuclear program is a covert weapons program. Despite"
            " sharp accusations by some in the United States and its allies, Iran denies having such a program, and"
            " U.S. intelligence contends that Iran has not yet made the decision to build a nuclear weapon. Iran's"
            " continued cooperation with International Atomic Energy Agency inspections is further evidence on this"
            " point, and we'll know even more about Iran's program in the coming months and years because of the deal."
            " In fact, the inspections provisions that are part of this agreement are designed to protect against any"
            " covert action by the Iranians. What's more, the rhetoric of some members of Congress has implied that"
            " the negotiations have been between only the United States and Iran (i.e., the 47 senators' letter"
            " warning that a deal might be killed by Congress or a future president). This of course is not the case."
            " The talks were between Iran and the five permanent members of the U.N. Security Council (United States,"
            " United Kingdom, France, China and Russia) plus Germany, dubbed the P5+1. While the United States has"
            " played a leading role in the effort, it negotiated the terms alongside its partners. If the agreement"
            " reached by the P5+1 is rejected by Congress, it could result in an unraveling of the sanctions on Iran"
            " and threaten NATO cohesion in other areas. Another questionable assertion is that this agreement"
            " contains a sunset clause, after which Iran will be free to do as it pleases. Again, this is not the"
            " case. Some of the restrictions on Iran's nuclear activities, such as uranium enrichment, will be eased"
            " or eliminated over time, as long as 15 years. But most importantly, the framework agreement includes"
            " Iran's ratification of the Additional Protocol, which allows IAEA inspectors expanded access to nuclear"
            " sites both declared and nondeclared. This provision will be permanent. It does not sunset. Thus, going"
            " forward, if Iran decides to enrich uranium to weapons-grade levels, monitors will be able to detect such"
            " a move in a matter of days and alert the U.N. Security Council. Many in Congress have said that the"
            ' agreement should be a formal treaty requiring the Senate to "advise and consent." But the issue is not'
            " suited for a treaty. Treaties impose equivalent obligations on all signatories. For example, the New"
            " START treaty limits Russia and the United States to 1,550 deployed strategic warheads. But any agreement"
            " with Iran will not be so balanced.  The restrictions and obligations in the final framework agreement"
            " will be imposed almost exclusively on Iran. The P5+1 are obligated only to ease and eventually remove"
            " most but not all economic sanctions, which were imposed as leverage to gain this final deal. Finally"
            " some insist that any agreement must address Iranian missile programs, human rights violations or support"
            " for Hamas or Hezbollah.  As important as these issues are, and they must indeed be addressed, they are"
            " unrelated to the most important aim of a nuclear deal: preventing a nuclear Iran.  To include them in"
            " the negotiations would be a poison pill. This agreement should be judged on its merits and on how it"
            " affects the security of our negotiating partners and allies, including Israel. Those judgments should be"
            " fact-based, not based on questionable assertions or dubious assumptions."
        )
        ARTICLE_SUBWAY = (
            "New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A"
            " year later, she got married again in Westchester County, but to a different man and without divorcing"
            " her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos"
            ' declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married'
            " once more, this time in the Bronx. In an application for a marriage license, she stated it was her"
            ' "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false'
            ' instrument for filing in the first degree," referring to her false statements on the 2010 marriage'
            " license application, according to court documents. Prosecutors said the marriages were part of an"
            " immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to"
            " her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was"
            " arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New"
            " York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total,"
            " Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All"
            " occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be"
            " married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors"
            " said the immigration scam involved some of her husbands, who filed for permanent residence status"
            " shortly after the marriages.  Any divorces happened only after such filings were approved. It was"
            " unclear whether any of the men will be prosecuted. The case was referred to the Bronx District"
            " Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's"
            ' Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt,'
            " Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his"
            " native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces"
            " up to four years in prison.  Her next court appearance is scheduled for May 18."
        )

        expected_summaries = [
            'prosecutor: "so far no videos were used in the crash investigation" two magazines claim to have found a cell phone video of the final seconds . "one can hear cries of \'My God\' in several languages," one magazine says . all 150 on board were killed in the crash .',
            "the formal accession was marked by a ceremony at The Hague, in the Netherlands . the ICC opened a preliminary examination into the situation in the occupied Palestinian territory . as members of the court, Palestinians may be subject to counter-charges as well .",
            "the u.s. and its negotiating partners reached a very strong framework agreement with Iran . aaron miller: the debate that has already begun since the announcement of the new framework will likely result in more heat than light . he says the new framework would reduce Iran's low-enriched uranium stockpile and cut centrifuges . miller: if it had been, there would have been no Iranian team at the negotiating table .",
            'prosecutors say the marriages were part of an immigration scam . barrientos pleaded not guilty to two counts of "offering a false instrument for filing in the first degree" she has been married 10 times, with nine of her marriages occurring between 1999 and 2002 .',
        ]

        dct = tok(
            ["summarize: " + x for x in [FRANCE_ARTICLE, SHORTER_ARTICLE, IRAN_ARTICLE, ARTICLE_SUBWAY]],
            padding="max_length",
            truncation=True,
            return_tensors="pd",
        )
        self.assertEqual(512, dct["input_ids"].shape[1])

        hypotheses_batch = model.generate(
            **dct,
            num_beams=4,
            length_penalty=2.0,
            max_length=142,
            min_length=56,
            decode_strategy="beam_search",
            early_stopping=True,
        )

        decoded = tok.batch_decode(hypotheses_batch[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        self.assertListEqual(
            expected_summaries,
            decoded,
        )

    @slow
    def test_translation_en_to_de(self):
        model = self.model()
        model.eval()
        tok = self.tokenizer()

        en_text = '"Luigi often said to me that he never wanted the brothers to end up in court", she wrote.'
        expected_translation = '"Luigi sagte mir oft, er wollte nie, dass die Brüder am Gericht enden", schrieb sie.'

        input_ids = tok.encode("translate English to German: " + en_text, return_tensors="pd")["input_ids"]
        output = model.generate(input_ids, decode_strategy="greedy_search", max_length=100)
        translation = tok.decode(output[0][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.assertEqual(translation, expected_translation)

    @slow
    def test_translation_en_to_fr(self):
        model = self.model()  # t5-base
        model.eval()
        tok = self.tokenizer()

        en_text = (
            ' This image section from an infrared recording by the Spitzer telescope shows a "family portrait" of'
            " countless generations of stars: the oldest stars are seen as blue dots. "
        )

        input_ids = tok.encode("translate English to French: " + en_text, return_tensors="pd")["input_ids"]

        output = model.generate(
            input_ids=input_ids,
            num_beams=4,
            length_penalty=2.0,
            max_length=100,
            decode_strategy="beam_search",
            early_stopping=True,
        )
        translation = tok.decode(output[0][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        new_truncated_translation = [
            "Cette section d'images d'un enregistrement infrarouge du télescope Spitzer montre un « portrait familial » d'innombrables générations d'étoiles : les étoiles les plus anciennes sont visibles sous forme de points bleus."
        ]

        self.assertEqual(translation, new_truncated_translation[0])

    @slow
    def test_translation_en_to_ro(self):
        model = self.model()
        model.eval()
        tok = self.tokenizer()

        en_text = "Taco Bell said it plans to add 2,000 locations in the US by 2022."
        expected_translation = "Taco Bell a declarat că intenţionează să adauge 2 000 de locaţii în SUA până în 2022."

        input_ids = tok("translate English to Romanian: " + en_text, return_tensors="pd")["input_ids"]
        output = model.generate(input_ids, decode_strategy="greedy_search", max_length=100)
        translation = tok.decode(output[0][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.assertEqual(translation, expected_translation)


@parameterized_class(
    ("return_dict", "use_labels"),
    [
        [False, False],
        [False, True],
        [True, False],
        [True, True],
    ],
)
class TestAsymmetricT5(unittest.TestCase):
    return_dict = False
    use_labels = False

    def build_model_and_check_forward_pass(self, **kwargs):
        tester = T5ModelTester(self, **kwargs)
        config, *inputs = tester.prepare_config_and_inputs()
        (
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = inputs
        model = T5ForConditionalGeneration(config)
        model.eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
            return_dict=self.return_dict,
        )
        # outputs = model(*inputs)
        assert len(outputs) == (4 if self.use_labels else 3), f"{type(outputs)}, {type(lm_labels)}"

        if self.use_labels:
            assert outputs[1].shape == [tester.batch_size, tester.decoder_seq_length, tester.vocab_size]
            assert isinstance(outputs[0].item(), float)
        else:
            assert outputs[0].shape == [tester.batch_size, tester.decoder_seq_length, tester.vocab_size]
        return model

    def test_small_decoder(self):
        # num_hidden_layers is passed to T5Config as num_layers
        model = self.build_model_and_check_forward_pass(decoder_layers=1, num_hidden_layers=2)
        assert len(model.encoder.block) == 2
        assert len(model.decoder.block) == 1

    def test_defaulting_to_symmetry(self):
        # num_hidden_layers is passed to T5Config as num_layers
        model = self.build_model_and_check_forward_pass(num_hidden_layers=2)
        assert len(model.decoder.block) == len(model.encoder.block) == 2
