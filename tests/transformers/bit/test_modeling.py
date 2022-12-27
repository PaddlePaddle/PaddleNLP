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


import inspect
import os
import shutil
import tempfile
import unittest

import paddle
import paddle.nn as nn
from PIL import Image

from paddlenlp.transformers import (
    BitBackbone,
    BitConfig,
    BitForImageClassification,
    BitImageProcessor,
    BitModel,
)
from paddlenlp.utils.env import CONFIG_NAME, LEGACY_CONFIG_NAME

from ...testing_utils import get_tests_dir, slow
from ..test_configuration_common import ConfigTester
from ..test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor

BIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/bit-50",
    # See all BiT models at https://huggingface.co/models?filter=bit
]


class BitModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        image_size=32,
        num_channels=3,
        embeddings_size=10,
        hidden_sizes=[8, 16, 32, 64],
        depths=[1, 1, 2, 1],
        is_training=True,
        use_labels=True,
        hidden_act="relu",
        num_labels=3,
        scope=None,
        out_features=["stage2", "stage3", "stage4"],
        num_groups=1,
        return_dict=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.embeddings_size = embeddings_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_act = hidden_act
        self.num_labels = num_labels
        self.scope = scope
        self.num_stages = len(hidden_sizes)
        self.out_features = out_features
        self.num_groups = num_groups
        self.return_dict = return_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return BitConfig(
            num_channels=self.num_channels,
            embeddings_size=self.embeddings_size,
            hidden_sizes=self.hidden_sizes,
            depths=self.depths,
            hidden_act=self.hidden_act,
            num_labels=self.num_labels,
            out_features=self.out_features,
            num_groups=self.num_groups,
            return_dict=self.return_dict,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = BitModel(config=config)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            [self.batch_size, self.hidden_sizes[-1], self.image_size // 32, self.image_size // 32],
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = BitForImageClassification(config)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.num_labels])

    def create_and_check_backbone(self, config, pixel_values, labels):
        model = BitBackbone(config=config)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[1], 4, 4])

        # verify channels
        self.parent.assertEqual(len(model.channels), len(config.out_features))
        self.parent.assertListEqual(model.channels, config.hidden_sizes[1:])

        # verify backbone works with out_features=None
        config.out_features = None
        model = BitBackbone(config=config)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[-1], 1, 1])

        # verify channels
        self.parent.assertEqual(len(model.channels), 1)
        self.parent.assertListEqual(model.channels, [config.hidden_sizes[-1]])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


class BitModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = BitModel
    all_model_classes = (BitModel, BitForImageClassification, BitBackbone)
    test_resize_embeddings = False
    has_attentions = False

    def setUp(self):
        super().setUp()

        self.model_tester = BitModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BitConfig, has_text_modality=False)

    def test_config(self):
        self.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_classes()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def create_and_test_config_common_properties(self):
        return

    def test_pretrained_config_save_load(self):

        if self.base_model_class is None or not self.base_model_class.constructed_from_pretrained_config():
            return

        config_class = self.base_model_class.config_class
        with tempfile.TemporaryDirectory() as tempdir:
            config = config_class()

            config.save_pretrained(tempdir)

            # check the file exist
            self.assertFalse(os.path.exists(os.path.join(tempdir, LEGACY_CONFIG_NAME)))
            self.assertTrue(os.path.exists(os.path.join(tempdir, CONFIG_NAME)))

            # rename the CONFIG_NAME
            shutil.move(os.path.join(tempdir, CONFIG_NAME), os.path.join(tempdir, LEGACY_CONFIG_NAME))

            loaded_config = config.__class__.from_pretrained(tempdir)
            self.assertEqual(config.hidden_sizes, loaded_config.hidden_sizes)

    @unittest.skip(reason="Bit does not use model_name_list")
    def test_model_name_list(self):
        pass

    @unittest.skip(reason="Bit does not output attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="Bit does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Bit does not support input and output embeddings")
    def test_model_common_attributes(self):
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

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_backbone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            for name, module in model.named_sublayers():
                if isinstance(module, (nn.BatchNorm2D, nn.GroupNorm)):
                    self.assertTrue(
                        paddle.all(module.weight == 1),
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )
                    self.assertTrue(
                        paddle.all(module.bias == 0),
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.eval()

            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_stages = self.model_tester.num_stages
            self.assertEqual(len(hidden_states), expected_num_stages + 1)

            # Bit's feature maps are of shape (batch_size, num_channels, height, width)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.image_size // 4, self.model_tester.image_size // 4],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        layers_type = ["preactivation", "bottleneck"]
        for model_class in self.all_model_classes:
            for layer_type in layers_type:
                config.layer_type = layer_type
                inputs_dict["output_hidden_states"] = True
                check_hidden_states_output(inputs_dict, config, model_class)

                # check that output_hidden_states also work using config
                del inputs_dict["output_hidden_states"]
                config.output_hidden_states = True

                check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason="Bit does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in BIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = BitModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    CUTE_CATS = get_tests_dir("fixtures/tests_samples/COCO/000000039769.png")
    image = Image.open(CUTE_CATS)
    return image


class BitModelIntegrationTest(unittest.TestCase):
    def default_image_processor(self):
        return BitImageProcessor.from_pretrained(BIT_PRETRAINED_MODEL_ARCHIVE_LIST[0])

    @slow
    def test_inference_image_classification_head(self):
        model = BitForImageClassification.from_pretrained(BIT_PRETRAINED_MODEL_ARCHIVE_LIST[0])
        model.eval()
        image_processor = self.default_image_processor()
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pd")

        # forward pass
        with paddle.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = [1, 1000]
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = paddle.to_tensor([-0.65258133, -0.52634168, -1.43975902])

        self.assertTrue(paddle.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
