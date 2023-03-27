# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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
import requests
from paddle import nn
from PIL import Image

from paddlenlp.transformers import (
    CLIPSegConfig,
    CLIPSegForImageSegmentation,
    CLIPSegModel,
    CLIPSegProcessor,
    CLIPSegTextConfig,
    CLIPSegTextModel,
    CLIPSegVisionConfig,
    CLIPSegVisionModel,
)
from paddlenlp.transformers.clipseg.modeling import (
    CLIPSEG_PRETRAINED_MODEL_ARCHIVE_LIST,
)

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


class CLIPSegVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
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
        return CLIPSegVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values):
        model = CLIPSegVisionModel(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(pixel_values, return_dict=True)
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


class CLIPSegVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as CLIPSeg does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (CLIPSegVisionModel,)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = CLIPSegVisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=CLIPSegVisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="CLIPSeg does not use inputs_embeds")
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

    @unittest.skip(reason="CLIPSegVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="CLIPSegVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLIPSEG_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLIPSegVisionModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class CLIPSegTextModelTester:
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
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
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
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope

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
        return CLIPSegTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = CLIPSegTextModel(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(input_ids, attention_mask=input_mask, return_dict=True)
            result = model(input_ids, return_dict=True)
        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result.pooler_output.shape, [self.batch_size, self.hidden_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class CLIPSegTextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (CLIPSegTextModel,)
    fx_compatible = False
    test_pruning = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = CLIPSegTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLIPSegTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="CLIPSeg does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="CLIPSegTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="CLIPSegTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLIPSEG_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLIPSegTextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class CLIPSegModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = CLIPSegTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = CLIPSegVisionModelTester(parent, **vision_kwargs)
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return CLIPSegConfig.from_text_vision_configs(
            self.text_model_tester.get_config(),
            self.vision_model_tester.get_config(),
            projection_dim=64,
            reduce_dim=32,
            extract_layers=[1, 2, 3],
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = CLIPSegModel(config)
        model.eval()
        with paddle.no_grad():
            result = model(input_ids, pixel_values, attention_mask, return_dict=True)
        self.parent.assertEqual(
            result.logits_per_image.shape, [self.vision_model_tester.batch_size, self.text_model_tester.batch_size]
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, [self.text_model_tester.batch_size, self.vision_model_tester.batch_size]
        )

    def create_and_check_model_for_image_segmentation(self, config, input_ids, attention_maks, pixel_values):
        model = CLIPSegForImageSegmentation(config)
        model.eval()
        with paddle.no_grad():
            result = model(input_ids, pixel_values, return_dict=True)
        self.parent.assertEqual(
            result.logits.shape,
            [
                self.vision_model_tester.batch_size,
                self.vision_model_tester.image_size,
                self.vision_model_tester.image_size,
            ],
        )
        self.parent.assertEqual(
            result.conditional_embeddings.shape, [self.text_model_tester.batch_size, config.projection_dim]
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


class CLIPSegModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (CLIPSegModel, CLIPSegForImageSegmentation)
    pipeline_model_mapping = {"feature-extraction": CLIPSegModel}
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # CLIPSegForImageSegmentation requires special treatment
        if return_labels:
            if model_class.__name__ == "CLIPSegForImageSegmentation":
                batch_size, _, height, width = inputs_dict["pixel_values"].shape
                inputs_dict["labels"] = paddle.zeros([batch_size, height, width], dtype="float32")

        return inputs_dict

    def setUp(self):
        self.model_tester = CLIPSegModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_for_image_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_for_image_segmentation(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="CLIPSegModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    # override as the some parameters require custom initialization
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.stop_gradient is False:
                    # check if `logit_scale` is initilized as per the original implementation]
                    if "logit_scale" in name:
                        self.assertAlmostEqual(
                            param.item(),
                            np.log(1 / 0.07),
                            delta=1e-3,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    elif "film" in name or "transposed_conv" in name or "reduce" in name:
                        # those parameters use PyTorch' default nn.Linear initialization scheme
                        pass
                    else:
                        self.assertIn(
                            ((param.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save CLIPSegConfig and check if we can load CLIPSegVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = CLIPSegVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save CLIPSegConfig and check if we can load CLIPSegTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = CLIPSegTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLIPSEG_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLIPSegModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


class CLIPSegModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_image_segmentation(self):
        model_name = "CIDAS/clipseg-rd64-refined"
        processor = CLIPSegProcessor.from_pretrained(model_name)
        model = CLIPSegForImageSegmentation.from_pretrained(model_name)

        image = prepare_img()
        texts = ["a cat", "a remote", "a blanket"]
        inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

        # forward pass
        with paddle.no_grad():
            outputs = model(**inputs, return_dict=True)

        # verify the predicted masks
        self.assertEqual(
            outputs.logits.shape,
            [3, 352, 352],
        )
        expected_masks_slice = paddle.to_tensor(
            [[-7.4613, -7.4785, -7.3628], [-7.3268, -7.0899, -7.1333], [-6.9838, -6.7900, -6.8913]]
        )
        self.assertTrue(paddle.allclose(outputs.logits[0, :3, :3], expected_masks_slice, atol=1e-3))

        # verify conditional and pooled output
        expected_conditional = paddle.to_tensor([0.5601, -0.0314, 0.1980])
        expected_pooled_output = paddle.to_tensor([0.5036, -0.2681, -0.2644])
        self.assertTrue(paddle.allclose(outputs.conditional_embeddings[0, :3], expected_conditional, atol=1e-3))
        self.assertTrue(paddle.allclose(outputs.pooled_output[0, :3], expected_pooled_output, atol=1e-3))
