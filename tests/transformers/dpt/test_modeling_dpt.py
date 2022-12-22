# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
""" Testing suite for the Paddle DPT model. """


import inspect
import unittest

import paddle
import paddle.nn as nn
from PIL import Image

from paddlenlp.transformers import (
    DPTConfig,
    DPTForDepthEstimation,
    DPTForSemanticSegmentation,
    DPTImageProcessor,
    DPTModel,
)

from ...testing_utils import get_tests_dir, slow
from ..test_configuration_common import ConfigTester
from ..test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor

DPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Intel/dpt-large",
    "Intel/dpt-hybrid-midas",
    # See all DPT models at https://huggingface.co/models?filter=dpt
]


class DPTModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=32,
        patch_size=16,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=4,
        backbone_out_indices=[0, 1, 2, 3],
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        num_labels=3,
        is_hybrid=False,
        return_dict=True,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.backbone_out_indices = backbone_out_indices
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope
        self.is_hybrid = is_hybrid
        # sequence length of DPT = num_patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1
        self.return_dict = return_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return DPTConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            backbone_out_indices=self.backbone_out_indices,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            is_hybrid=self.is_hybrid,
            return_dict=self.return_dict,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = DPTModel(config=config)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_depth_estimation(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = DPTForDepthEstimation(config)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.predicted_depth.shape, [self.batch_size, self.image_size, self.image_size])

    def create_and_check_for_semantic_segmentation(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = DPTForSemanticSegmentation(config)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(
            result.logits.shape, [self.batch_size, self.num_labels, self.image_size, self.image_size]
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


class DPTModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as DPT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (DPTModel, DPTForDepthEstimation, DPTForSemanticSegmentation)

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = DPTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DPTConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="DPT does not use model_name_list")
    def test_model_name_list(self):
        pass

    @unittest.skip(reason="DPT does not use inputs_embeds")
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

    def test_for_depth_estimation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_depth_estimation(*config_and_inputs)

    def test_for_semantic_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_semantic_segmentation(*config_and_inputs)

    def test_training(self):
        for model_class in self.all_model_classes:
            if model_class.__name__ in ["DPTModel", "DPTForDepthEstimation"]:
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            model = model_class(config)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class)
            inputs["labels"] = paddle.zeros(
                (self.model_tester.batch_size, self.model_tester.image_size, self.model_tester.image_size),
                dtype=paddle.int64,
            )
            loss = model(**inputs).loss
            loss.backward()

    @slow
    def test_training_gradient_checkpointing(self):
        for model_class in self.all_model_classes:
            if model_class.__name__ in ["DPTModel", "DPTForDepthEstimation"]:
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True

            if not model_class.supports_gradient_checkpointing:
                continue
            model = model_class(config)
            model.gradient_checkpointing_enable()
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class)
            inputs["labels"] = paddle.zeros(
                (self.model_tester.batch_size, self.model_tester.image_size, self.model_tester.image_size),
                dtype=paddle.int64,
            )
            loss = model(**inputs).loss
            loss.backward()

    @slow
    def test_model_from_pretrained(self):
        for model_name in DPT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = DPTModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    CUTE_CATS = get_tests_dir("fixtures/tests_samples/COCO/000000039769.png")
    image = Image.open(CUTE_CATS)
    return image


@slow
class DPTModelIntegrationTest(unittest.TestCase):
    def test_inference_depth_estimation(self):
        image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        model.eval()
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pd")

        # forward pass
        with paddle.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # verify the predicted depth
        expected_shape = [1, 384, 384]
        self.assertEqual(predicted_depth.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[6.3199, 6.3629, 6.4148], [6.3850, 6.3615, 6.4166], [6.3519, 6.3176, 6.3575]]
        )

        self.assertTrue(paddle.allclose(outputs.predicted_depth[0, :3, :3], expected_slice, atol=1e-4))

    def test_inference_semantic_segmentation(self):
        image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large-ade")
        model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")
        model.eval()

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pd")

        # forward pass
        with paddle.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = [1, 150, 480, 480]
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[4.0480, 4.2420, 4.4360], [4.3124, 4.5693, 4.8261], [4.5768, 4.8965, 5.2163]]
        )

        self.assertTrue(paddle.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4))

    def test_post_processing_semantic_segmentation(self):
        image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large-ade")
        model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")
        model.eval()
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pd")

        # forward pass
        with paddle.no_grad():
            outputs = model(**inputs)

        outputs.logits = outputs.logits.detach().cpu()

        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs, target_sizes=[[500, 300]])
        expected_shape = [500, 300]
        self.assertEqual(segmentation[0].shape, expected_shape)

        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs)
        expected_shape = [480, 480]
        self.assertEqual(segmentation[0].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
