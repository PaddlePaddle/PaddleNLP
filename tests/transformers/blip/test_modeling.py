# coding=utf-8
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the Paddle Blip model. """


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
    BlipConfig,
    BlipForConditionalGeneration,
    BlipForImageTextRetrieval,
    BlipForQuestionAnswering,
    BlipModel,
    BlipProcessor,
    BlipTextConfig,
    BlipTextModel,
    BlipVisionConfig,
    BlipVisionModel,
)
from paddlenlp.transformers.blip.modeling import BLIP_PRETRAINED_MODEL_ARCHIVE_LIST

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


class BlipVisionModelTester:
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
        return BlipVisionConfig(
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
        model = BlipVisionModel(config=config)
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


class BlipVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Blip does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (BlipVisionModel,)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = BlipVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BlipVisionConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Blip does not use inputs_embeds")
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

    @unittest.skip(reason="BlipVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="BlipVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in BLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = BlipVisionModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class BlipTextModelTester:
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
        return BlipTextConfig(
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

    def create_and_check_model(self, config, input_ids, input_mask):
        model = BlipTextModel(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result.pooler_output.shape, [self.batch_size, self.hidden_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class BlipTextModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (BlipTextModel,)
    fx_compatible = False
    test_pruning = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = BlipTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BlipTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Blip does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="BlipTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="BlipTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in BLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = BlipTextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class BlipModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):

        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = BlipTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = BlipVisionModelTester(parent, **vision_kwargs)
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return BlipConfig.from_text_vision_configs(
            self.text_model_tester.get_config(), self.vision_model_tester.get_config(), projection_dim=64
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = BlipModel(config)
        model.eval()
        with paddle.no_grad():
            result = model(input_ids, pixel_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.logits_per_image.shape, [self.vision_model_tester.batch_size, self.text_model_tester.batch_size]
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, [self.text_model_tester.batch_size, self.vision_model_tester.batch_size]
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "return_loss": True,
        }
        return config, inputs_dict


class BlipModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (BlipModel,)
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = BlipModelTester(self)

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

    @unittest.skip(reason="BlipModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    # override as the `logit_scale` parameter initilization is different for Blip
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if not param.stop_gradient:
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

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save BlipConfig and check if we can load BlipVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = BlipVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save BlipConfig and check if we can load BlipTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = BlipTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        for model_name in BLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = BlipModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class BlipTextImageModelsModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):

        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = BlipTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = BlipVisionModelTester(parent, **vision_kwargs)
        self.is_training = is_training
        self.num_choices = 4

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return BlipConfig.from_text_vision_configs(
            self.text_model_tester.get_config(), self.vision_model_tester.get_config(), projection_dim=64
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = BlipModel(config)
        model.eval()
        with paddle.no_grad():
            result = model(input_ids, pixel_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.logits_per_image.shape, [self.vision_model_tester.batch_size, self.text_model_tester.batch_size]
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, [self.text_model_tester.batch_size, self.vision_model_tester.batch_size]
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        return config, inputs_dict


class BlipTextImageModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        BlipForConditionalGeneration,
        BlipForQuestionAnswering,
        BlipForImageTextRetrieval,
    )
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)
        if model_class.__name__.endswith("ForMultipleChoice"):
            inputs_dict = {
                k: v.unsqueeze(1).expand(shape=[-1, self.model_tester.num_choices, -1])
                if isinstance(v, paddle.Tensor) and v.ndim > 1
                else v
                for k, v in inputs_dict.items()
            }

        if return_labels:
            if model_class.__name__.endswith("ForMultipleChoice"):
                inputs_dict["labels"] = paddle.ones(
                    (self.model_tester.text_model_tester.batch_size,), dtype=paddle.int64
                )
            elif model_class.__name__.endswith("ForQuestionAnswering"):
                inputs_dict["decoder_input_ids"] = paddle.zeros(
                    (self.model_tester.text_model_tester.batch_size, 4), dtype=paddle.int64
                )
            elif model_class.__name__.endswith("ImageTextRetrieval"):
                inputs_dict["labels"] = paddle.zeros(
                    (self.model_tester.text_model_tester.batch_size,), dtype=paddle.int64
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = BlipTextImageModelsModelTester(self)

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

    @unittest.skip(reason="BlipModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_ids"] if model_class != BlipForConditionalGeneration else ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_training(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes[:-1]:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            model = model_class(config)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    @slow
    def test_training_gradient_checkpointing(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes[:-1]:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True

            model = model_class(config)
            model.gradient_checkpointing_enable()
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    # override as the `logit_scale` parameter initilization is different for Blip
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if not param.stop_gradient:
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

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save BlipConfig and check if we can load BlipVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = BlipVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save BlipConfig and check if we can load BlipTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = BlipTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        for model_name in BLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = BlipModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of dog and girl
def prepare_img():
    url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@slow
class BlipModelIntegrationTest(unittest.TestCase):
    def test_inference_image_captioning(self):
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.eval()
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        image = prepare_img()
        assert model.config.text_config.num_attention_heads == 12
        assert model.config.vision_config.layer_norm_eps == 1e-6
        # image only
        inputs = processor(images=image, return_tensors="pd")

        predictions = model.generate(**inputs)[0]

        # Test output
        self.assertEqual(predictions[0].tolist(), [1037, 2450, 3564, 2006, 1996, 3509, 2007, 2014, 3899, 102])

        # image and context
        context = ["a picture of"]
        inputs = processor(images=image, text=context, return_tensors="pd")

        predictions = model.generate(**inputs)[0]

        # Test output
        self.assertEqual(
            predictions[0].tolist(),
            [1037, 2450, 1998, 2014, 3899, 2006, 1996, 3509, 102],
        )

    def test_inference_vqa(self):
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        model.eval()
        assert model.config.text_config.num_attention_heads == 12
        assert model.config.vision_config.layer_norm_eps == 1e-6
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

        image = prepare_img()
        text = "how many dogs are in the picture?"

        inputs = processor(images=image, text=text, return_tensors="pd")
        out = model.generate(**inputs)[0]

        # Test output
        self.assertEqual(out[0].tolist(), [1015, 102])

    def test_inference_itm(self):
        model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        model.eval()
        assert model.config.text_config.num_attention_heads == 12
        assert model.config.vision_config.layer_norm_eps == 1e-6
        processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

        image = prepare_img()
        text = "A woman and her dog sitting in a beach"

        inputs = processor(images=image, text=text, return_tensors="pd")
        with paddle.no_grad():
            out_itm = model(**inputs)
            out = model(**inputs, use_itm_head=False)

        expected_scores = paddle.to_tensor([[0.00289215, 0.99710792]])

        self.assertTrue(paddle.allclose(nn.functional.softmax(out_itm[0]), expected_scores, rtol=1e-3, atol=1e-3))
        self.assertTrue(paddle.allclose(out[0], paddle.to_tensor([[0.51626438]]), rtol=1e-3, atol=1e-3))
