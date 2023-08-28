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

import copy
import inspect
import tempfile
import unittest

import numpy as np
import paddle
from datasets import load_dataset
from paddle import nn

from paddlenlp.transformers import (
    ClapAudioConfig,
    ClapAudioModel,
    ClapAudioModelWithProjection,
    ClapConfig,
    ClapModel,
    ClapProcessor,
    ClapTextConfig,
    ClapTextModel,
    ClapTextModelWithProjection,
)
from paddlenlp.transformers.clap.modeling import CLAP_PRETRAINED_MODEL_ARCHIVE_LIST

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


class ClapAudioModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=60,
        num_mel_bins=16,
        window_size=4,
        spec_size=64,
        patch_size=2,
        patch_stride=2,
        seq_length=16,
        freq_ratio=2,
        num_channels=3,
        is_training=True,
        hidden_size=256,
        patch_embeds_hidden_size=32,
        projection_dim=32,
        num_hidden_layers=4,
        num_heads=[2, 2, 2, 2],
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_mel_bins = num_mel_bins
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_attention_heads = num_heads[0]
        self.seq_length = seq_length
        self.spec_size = spec_size
        self.freq_ratio = freq_ratio
        self.patch_stride = patch_stride
        self.patch_embeds_hidden_size = patch_embeds_hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, 1, self.hidden_size, self.num_mel_bins])
        config = self.get_config()

        return config, input_features

    def get_config(self):
        return ClapAudioConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_mel_bins=self.num_mel_bins,
            window_size=self.window_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            patch_stride=self.patch_stride,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
            spec_size=self.spec_size,
            freq_ratio=self.freq_ratio,
            patch_embeds_hidden_size=self.patch_embeds_hidden_size,
        )

    def create_and_check_model(self, config, input_features):
        model = ClapAudioModel(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(input_features, return_dict=True)
        self.parent.assertEqual(result.pooler_output.shape, [self.batch_size, self.hidden_size])

    def create_and_check_model_with_projection(self, config, input_features):
        model = ClapAudioModelWithProjection(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(input_features, return_dict=True)
        self.parent.assertEqual(result.audio_embeds.shape, [self.batch_size, self.projection_dim])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_features = config_and_inputs
        inputs_dict = {"input_features": input_features}
        return config, inputs_dict


class ClapAudioModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as CLAP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (ClapAudioModel, ClapAudioModelWithProjection)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = ClapAudioModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ClapAudioConfig, has_text_modality=False, hidden_size=37)

    # def test_config(self):
    #     self.config_tester.run_common_tests()

    @unittest.skip(reason="ClapAudioModel does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Layer))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.eval()

            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class), return_dict=True)

            hidden_states = outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.patch_embeds_hidden_size, self.model_tester.patch_embeds_hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason="ClapAudioModel does not output any loss term in the forward pass")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_features"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_projection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_projection(*config_and_inputs)

    @unittest.skip(reason="ClapAudioModel does not output any loss term in the forward pass")
    def test_training(self):
        pass

    @unittest.skip(reason="ClapAudioModel does not output any loss term in the forward pass")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="ClapAudioModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="ClapAudioModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLAP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ClapAudioModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    def test_model_with_projection_from_pretrained(self):
        for model_name in CLAP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ClapAudioModelWithProjection.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, "audio_projection"))


class ClapTextModelTester:
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
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        scope=None,
        projection_hidden_act="relu",
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
        self.projection_hidden_act = projection_hidden_act

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return ClapTextConfig(
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
            projection_hidden_act=self.projection_hidden_act,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = ClapTextModel(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(input_ids, attention_mask=input_mask, return_dict=True)
            result = model(input_ids, return_dict=True)
        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result.pooler_output.shape, [self.batch_size, self.hidden_size])

    def create_and_check_model_with_projection(self, config, input_ids, input_mask):
        model = ClapTextModelWithProjection(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(input_ids, attention_mask=input_mask, return_dict=True)
            result = model(input_ids, return_dict=True)
        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result.text_embeds.shape, [self.batch_size, self.projection_dim])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class ClapTextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ClapTextModel, ClapTextModelWithProjection)
    fx_compatible = False
    test_pruning = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = ClapTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ClapTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_projection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_projection(*config_and_inputs)

    @unittest.skip(reason="ClapTextModel does not output any loss term in the forward pass")
    def test_training(self):
        pass

    @unittest.skip(reason="ClapTextModel does not output any loss term in the forward pass")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="ClapTextModel does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="ClapTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="ClapTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLAP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ClapTextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    def test_model_with_projection_from_pretrained(self):
        for model_name in CLAP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ClapTextModelWithProjection.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, "text_projection"))


class ClapModelTester:
    def __init__(self, parent, text_kwargs=None, audio_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if audio_kwargs is None:
            audio_kwargs = {}

        self.parent = parent
        self.text_model_tester = ClapTextModelTester(parent, **text_kwargs)
        self.audio_model_tester = ClapAudioModelTester(parent, **audio_kwargs)
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        _, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        _, input_features = self.audio_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, input_features

    def get_config(self):
        return ClapConfig.from_text_audio_configs(
            self.text_model_tester.get_config(), self.audio_model_tester.get_config(), projection_dim=64
        )

    def create_and_check_model(self, config, input_ids, attention_mask, input_features):
        model = ClapModel(config)
        model.eval()
        with paddle.no_grad():
            result = model(input_ids, input_features, attention_mask, return_dict=True)
        self.parent.assertEqual(
            result.logits_per_audio.shape, [self.audio_model_tester.batch_size, self.text_model_tester.batch_size]
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, [self.text_model_tester.batch_size, self.audio_model_tester.batch_size]
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, input_features = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "return_loss": True,
        }
        return config, inputs_dict


class ClapModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ClapModel,)
    pipeline_model_mapping = {"feature-extraction": ClapModel}
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = ClapModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="ClapModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    # override as the `logit_scale` parameter initilization is different for CLAP
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.stop_gradient:
                    # check if `logit_scale` is initilized as per the original implementation
                    if name == "logit_scale":
                        self.assertAlmostEqual(
                            param.item(),
                            np.log(1 / 0.07),
                            delta=1e-3,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_load_audio_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save ClapConfig and check if we can load ClapAudioConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            audio_config = ClapAudioConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.audio_config.to_dict(), audio_config.to_dict())

        # Save ClapConfig and check if we can load ClapTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = ClapTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLAP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ClapModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@slow
class ClapModelIntegrationTest(unittest.TestCase):
    paddings = ["repeatpad", "repeat", "pad"]

    def test_integration_unfused(self):
        EXPECTED_MEANS_UNFUSED = {
            "repeatpad": 0.0024,
            "pad": 0.0020,
            "repeat": 0.0023,
        }

        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        audio_sample = librispeech_dummy[-1]

        model_id = "laion/clap-htsat-unfused"

        model = ClapModel.from_pretrained(model_id)
        model.eval()
        processor = ClapProcessor.from_pretrained(model_id)

        for padding in self.paddings:
            inputs = processor(audios=audio_sample["audio"]["array"], return_tensors="pd", padding=padding)

            audio_embed = model.get_audio_features(**inputs)
            expected_mean = EXPECTED_MEANS_UNFUSED[padding]

            self.assertTrue(
                paddle.allclose(audio_embed.cpu().mean(), paddle.to_tensor([expected_mean]), atol=1e-3, rtol=1e-3)
            )

    def test_integration_fused(self):
        EXPECTED_MEANS_FUSED = {
            "repeatpad": 0.00069,
            "repeat": 0.00196,
            "pad": -0.000379,
        }

        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        audio_sample = librispeech_dummy[-1]

        model_id = "laion/clap-htsat-fused"

        model = ClapModel.from_pretrained(model_id)
        model.eval()
        processor = ClapProcessor.from_pretrained(model_id)

        for padding in self.paddings:
            inputs = processor(
                audios=audio_sample["audio"]["array"], return_tensors="pd", padding=padding, truncation="fusion"
            )

            audio_embed = model.get_audio_features(**inputs)
            expected_mean = EXPECTED_MEANS_FUSED[padding]

            self.assertTrue(
                paddle.allclose(audio_embed.cpu().mean(), paddle.to_tensor([expected_mean]), atol=1e-3, rtol=1e-3)
            )

    def test_batched_fused(self):
        EXPECTED_MEANS_FUSED = {
            "repeatpad": 0.0010,
            "repeat": 0.0020,
            "pad": 0.0006,
        }

        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        audio_samples = [sample["array"] for sample in librispeech_dummy[0:4]["audio"]]

        model_id = "laion/clap-htsat-fused"

        model = ClapModel.from_pretrained(model_id)
        model.eval()
        processor = ClapProcessor.from_pretrained(model_id)

        for padding in self.paddings:
            inputs = processor(audios=audio_samples, return_tensors="pd", padding=padding, truncation="fusion")

            audio_embed = model.get_audio_features(**inputs)
            expected_mean = EXPECTED_MEANS_FUSED[padding]
            self.assertTrue(
                paddle.allclose(audio_embed.cpu().mean(), paddle.to_tensor([expected_mean]), atol=1e-3, rtol=1e-3)
            )

    def test_batched_unfused(self):
        EXPECTED_MEANS_FUSED = {
            "repeatpad": 0.0016,
            "repeat": 0.0019,
            "pad": 0.0019,
        }

        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        audio_samples = [sample["array"] for sample in librispeech_dummy[0:4]["audio"]]

        model_id = "laion/clap-htsat-unfused"

        model = ClapModel.from_pretrained(model_id)
        model.eval()
        processor = ClapProcessor.from_pretrained(model_id)

        for padding in self.paddings:
            inputs = processor(audios=audio_samples, return_tensors="pd", padding=padding)

            audio_embed = model.get_audio_features(**inputs)
            expected_mean = EXPECTED_MEANS_FUSED[padding]

            self.assertTrue(
                paddle.allclose(audio_embed.cpu().mean(), paddle.to_tensor([expected_mean]), atol=1e-3, rtol=1e-3)
            )
