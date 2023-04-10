# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

""" Testing suite for the PyTorch BLIP-2 model. """


import copy
import inspect
import tempfile
import unittest

import numpy as np
import paddle
import paddle.nn as nn
import requests
from PIL import Image

from paddlenlp.transformers import (
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2Model,
    Blip2QFormerConfig,
    Blip2VisionConfig,
    Blip2VisionModel,
    OPTConfig,
    T5Config,
)
from paddlenlp.transformers.blip_2.modeling import BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST

from ...testing_utils import slow
from ..test_configuration_common import ConfigTester
from ..test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
    return configs_no_init


class Blip2VisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=1e-10,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return Blip2VisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values):
        model = Blip2VisionModel(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, num_patches + 1, self.hidden_size])
        self.parent.assertEqual(result.pooler_output.shape, [self.batch_size, self.hidden_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


class Blip2VisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as BLIP-2's vision encoder does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (Blip2VisionModel,)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = Blip2VisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Blip2VisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="BLIP-2's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Layer))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Blip2VisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Blip2VisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = Blip2VisionModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class Blip2QFormerModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        bos_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64")

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length], dtype="int64")

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return Blip2QFormerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            bos_token_id=self.bos_token_id,
        )


# this class is based on `OPTModelTester` found in tests/models/opt/test_modeling_opt.py
class Blip2TextModelDecoderOnlyTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        embed_dim=16,
        num_labels=3,
        word_embed_proj_dim=16,
        type_sequence_label_size=2,
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
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.type_sequence_label_size = type_sequence_label_size
        self.word_embed_proj_dim = word_embed_proj_dim
        self.is_encoder_decoder = False

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64").clip(
            3,
        )
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        attention_mask = input_ids.not_equal(paddle.to_tensor([self.pad_token_id], dtype="int64")).cast("int64")

        return config, input_ids, attention_mask

    def get_config(self):
        return OPTConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            embed_dim=self.embed_dim,
            is_encoder_decoder=False,
            word_embed_proj_dim=self.word_embed_proj_dim,
        )


# this model tester uses a decoder-only language model (OPT)
class Blip2ForConditionalGenerationDecoderOnlyModelTester:
    def __init__(
        self, parent, vision_kwargs=None, qformer_kwargs=None, text_kwargs=None, is_training=True, num_query_tokens=10
    ):
        if vision_kwargs is None:
            vision_kwargs = {}
        if qformer_kwargs is None:
            qformer_kwargs = {}
        if text_kwargs is None:
            text_kwargs = {}

        self.parent = parent
        self.vision_model_tester = Blip2VisionModelTester(parent, **vision_kwargs)
        self.qformer_model_tester = Blip2QFormerModelTester(parent, **qformer_kwargs)
        self.text_model_tester = Blip2TextModelDecoderOnlyTester(parent, **text_kwargs)
        self.is_training = is_training
        self.num_query_tokens = num_query_tokens

    def prepare_config_and_inputs(self):
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        _, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return Blip2Config.from_vision_qformer_text_configs(
            vision_config=self.vision_model_tester.get_config(),
            qformer_config=self.qformer_model_tester.get_config(),
            text_config=self.text_model_tester.get_config(),
            num_query_tokens=self.num_query_tokens,
        )

    def create_and_check_for_conditional_generation(self, config, input_ids, attention_mask, pixel_values):
        model = Blip2ForConditionalGeneration(config)
        model.eval()
        with paddle.no_grad():
            result = model(pixel_values, input_ids, attention_mask, return_dict=True)
        expected_seq_length = self.num_query_tokens + self.text_model_tester.seq_length
        self.parent.assertEqual(
            result.logits.shape,
            [self.vision_model_tester.batch_size, expected_seq_length, self.text_model_tester.vocab_size],
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
            "return_dict": True,
        }
        return config, inputs_dict


class Blip2ForConditionalGenerationDecoderOnlyTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Blip2ForConditionalGeneration,)
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = Blip2ForConditionalGenerationDecoderOnlyModelTester(self)

    def test_for_conditional_generation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_conditional_generation(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Blip2Model does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="There's no base Blip2Model")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="There's no base Blip2Model")
    def test_save_load_fast_init_to_base(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_load_vision_qformer_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save Blip2Config and check if we can load Blip2VisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = Blip2VisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save Blip2Config and check if we can load Blip2QFormerConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            qformer_config = Blip2QFormerConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.qformer_config.to_dict(), qformer_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        for model_name in BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST:
            model = Blip2ForConditionalGeneration.from_pretrained(model_name)
            self.assertIsNotNone(model)


# this class is based on `T5ModelTester` found in tests/models/t5/test_modeling_t5.py
class Blip2TextModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=12,
        encoder_seq_length=7,
        decoder_seq_length=9,
        # For common tests
        is_training=True,
        use_attention_mask=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        dropout_rate=0.1,
        initializer_factor=0.002,
        eos_token_id=1,
        pad_token_id=0,
        decoder_start_token_id=0,
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
        self.use_labels = use_labels
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
        self.decoder_start_token_id = decoder_start_token_id
        self.scope = None
        self.decoder_layers = decoder_layers

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)
        decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        attention_mask = None
        decoder_attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2, dtype="int64")
            decoder_attention_mask = ids_tensor(
                [self.batch_size, self.decoder_seq_length], vocab_size=2, dtype="int64"
            )

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size, dtype="int64")

        config = self.get_config()

        return (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    def get_config(self):
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
            decoder_start_token_id=self.decoder_start_token_id,
        )


# this model tester uses an encoder-decoder language model (T5)
class Blip2ForConditionalGenerationModelTester:
    def __init__(
        self, parent, vision_kwargs=None, qformer_kwargs=None, text_kwargs=None, is_training=True, num_query_tokens=10
    ):
        if vision_kwargs is None:
            vision_kwargs = {}
        if qformer_kwargs is None:
            qformer_kwargs = {}
        if text_kwargs is None:
            text_kwargs = {}

        self.parent = parent
        self.vision_model_tester = Blip2VisionModelTester(parent, **vision_kwargs)
        self.qformer_model_tester = Blip2QFormerModelTester(parent, **qformer_kwargs)
        self.text_model_tester = Blip2TextModelTester(parent, **text_kwargs)
        self.is_training = is_training
        self.num_query_tokens = num_query_tokens

    def prepare_config_and_inputs(self):
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        (
            _,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = self.text_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values, decoder_input_ids, decoder_attention_mask, lm_labels

    def get_config(self):
        return Blip2Config.from_vision_qformer_text_configs(
            vision_config=self.vision_model_tester.get_config(),
            qformer_config=self.qformer_model_tester.get_config(),
            text_config=self.text_model_tester.get_config(),
            num_query_tokens=self.num_query_tokens,
        )

    def create_and_check_for_conditional_generation(
        self, config, input_ids, attention_mask, pixel_values, decoder_input_ids, decoder_attention_mask, labels
    ):
        model = Blip2ForConditionalGeneration(config)
        model.eval()
        with paddle.no_grad():
            result = model(
                pixel_values, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, return_dict=True
            )

        self.parent.assertEqual(
            result.logits.shape,
            [
                self.vision_model_tester.batch_size,
                self.text_model_tester.seq_length,
                self.text_model_tester.vocab_size,
            ],
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            pixel_values,
            decoder_input_ids,
            decoder_attention_mask,
            labels,
        ) = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }
        return config, inputs_dict


# this model tester uses an encoder-decoder language model (T5)
class Blip2ModelTester:
    def __init__(
        self, parent, vision_kwargs=None, qformer_kwargs=None, text_kwargs=None, is_training=True, num_query_tokens=10
    ):
        if vision_kwargs is None:
            vision_kwargs = {}
        if qformer_kwargs is None:
            qformer_kwargs = {}
        if text_kwargs is None:
            text_kwargs = {}

        self.parent = parent
        self.vision_model_tester = Blip2VisionModelTester(parent, **vision_kwargs)
        self.qformer_model_tester = Blip2QFormerModelTester(parent, **qformer_kwargs)
        self.text_model_tester = Blip2TextModelTester(parent, **text_kwargs)
        self.is_training = is_training
        self.num_query_tokens = num_query_tokens

    def prepare_config_and_inputs(self):
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        (
            _,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = self.text_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values, decoder_input_ids, decoder_attention_mask, lm_labels

    def get_config(self):
        return Blip2Config.from_vision_qformer_text_configs(
            vision_config=self.vision_model_tester.get_config(),
            qformer_config=self.qformer_model_tester.get_config(),
            text_config=self.text_model_tester.get_config(),
            num_query_tokens=self.num_query_tokens,
        )

    def create_and_check_for_conditional_generation(
        self, config, input_ids, attention_mask, pixel_values, decoder_input_ids, decoder_attention_mask, labels
    ):
        model = Blip2ForConditionalGeneration(config)
        model.eval()
        with paddle.no_grad():
            result = model(
                pixel_values, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, return_dict=True
            )

        self.parent.assertEqual(
            result.logits.shape,
            [
                self.vision_model_tester.batch_size,
                self.text_model_tester.seq_length,
                self.text_model_tester.vocab_size,
            ],
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            pixel_values,
            decoder_input_ids,
            decoder_attention_mask,
            labels,
        ) = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }
        return config, inputs_dict


class Blip2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Blip2ForConditionalGeneration, Blip2Model)
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False
    use_test_inputs_embeds: bool = False

    def setUp(self):
        self.model_tester = Blip2ModelTester(self)

    def test_for_conditional_generation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_conditional_generation(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Blip2Model does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="There's no base Blip2Model")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="There's no base Blip2Model")
    def test_save_load_fast_init_to_base(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_load_vision_qformer_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save Blip2Config and check if we can load Blip2VisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = Blip2VisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save Blip2Config and check if we can load Blip2QFormerConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            qformer_config = Blip2QFormerConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.qformer_config.to_dict(), qformer_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        for model_name in BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST:
            model = Blip2ForConditionalGeneration.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_get_text_features(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        inputs_dict = {
            "input_ids": paddle.to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
            "attention_mask": paddle.to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
            "decoder_input_ids": paddle.to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
        }

        model = Blip2Model(config)
        model.eval()
        text_features = model.get_text_features(**inputs_dict)
        self.assertEqual(text_features[0].shape, [1, 10, config.text_config.vocab_size])

    def test_get_image_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        keys_to_pop = ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"]

        for key in keys_to_pop:
            inputs_dict.pop(key)

        model = Blip2Model(config)
        model.eval()
        image_features = model.get_image_features(**inputs_dict)
        self.assertEqual(
            image_features[0].shape,
            [
                self.model_tester.vision_model_tester.batch_size,
                self.model_tester.vision_model_tester.seq_length,
                config.vision_config.hidden_size,
            ],
        )

    def test_get_qformer_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        keys_to_pop = ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"]

        for key in keys_to_pop:
            inputs_dict.pop(key)

        model = Blip2Model(config)
        model.eval()
        qformer_features = model.get_qformer_features(**inputs_dict)
        self.assertEqual(
            qformer_features[0].shape,
            [self.model_tester.vision_model_tester.batch_size, 10, config.vision_config.hidden_size],
        )

    # override from common to deal with nested configurations (`vision_config`, `text_config` and `qformer_config`)
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for key in ["vision_config", "qformer_config", "text_config"]:
            setattr(configs_no_init, key, _config_zero_init(getattr(configs_no_init, key)))
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if not param.stop_gradient:
                    self.assertIn(
                        ((param.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )


# We will verify our results on an image of cute cats
def prepare_img():
    url = "https://huggingface.co/hf-internal-testing/blip-test-image/resolve/main/demo.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image
