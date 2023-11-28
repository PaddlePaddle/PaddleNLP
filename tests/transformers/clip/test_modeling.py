# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the Paddle CLIP model. """


import copy
import inspect
import tempfile
import unittest

import numpy as np
import paddle
from paddle import nn
from PIL import Image

from paddlenlp.transformers import (
    CLIPConfig,
    CLIPModel,
    CLIPProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionConfig,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
    CLIPVisionTransformer,
)
from paddlenlp.transformers.clip.modeling import CLIP_PRETRAINED_MODEL_ARCHIVE_LIST

from ...testing_utils import get_tests_dir, require_package, slow
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


class CLIPVisionModelTester:
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
        return CLIPVisionConfig(
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
        model = CLIPVisionModel(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, num_patches + 1, self.hidden_size])
        self.parent.assertEqual(result.pooler_output.shape, [self.batch_size, self.hidden_size])

    def create_and_check_model_with_projection(self, config, pixel_values):
        model = CLIPVisionModelWithProjection(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, num_patches + 1, self.hidden_size])
        self.parent.assertEqual(result.image_embeds.shape, [self.batch_size, self.projection_dim])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


class CLIPVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as CLIP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (CLIPVisionModel, CLIPVisionModelWithProjection)
    test_resize_embeddings = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = CLIPVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLIPVisionConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="CLIP does not use inputs_embeds")
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

    def test_model_with_projection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_projection(*config_and_inputs)

    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="CLIPVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="CLIPVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLIPVisionModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    def test_model_with_projection_from_pretrained(self):
        for model_name in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLIPVisionModelWithProjection.from_pretrained(model_name)
            self.assertIsNotNone(model)
            if isinstance(model.vision_model, CLIPVisionTransformer):
                self.assertTrue(hasattr(model, "vision_projection"))


class CLIPTextModelTester:
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
        return CLIPTextConfig(
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
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = CLIPTextModel(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result.pooler_output.shape, [self.batch_size, self.hidden_size])

    def create_and_check_model_with_projection(self, config, input_ids, input_mask):
        model = CLIPTextModelWithProjection(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result.text_embeds.shape, [self.batch_size, self.projection_dim])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class CLIPTextModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (CLIPTextModel, CLIPTextModelWithProjection)
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = CLIPTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLIPTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_projection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_projection(*config_and_inputs)

    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="CLIP does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="CLIPTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="CLIPTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLIPTextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    def test_model_with_projection_from_pretrained(self):
        for model_name in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLIPTextModelWithProjection.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, "text_projection"))


class CLIPModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):

        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = CLIPTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = CLIPVisionModelTester(parent, **vision_kwargs)
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return CLIPConfig.from_text_vision_configs(
            self.text_model_tester.get_config(), self.vision_model_tester.get_config(), projection_dim=64
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = CLIPModel(config)
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


class CLIPModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (CLIPModel,)
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = CLIPModelTester(self)

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

    @unittest.skip(reason="CLIPModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    # override as the `logit_scale` parameter initilization is different for CLIP
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

        # Save CLIPConfig and check if we can load CLIPVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = CLIPVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save CLIPConfig and check if we can load CLIPTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = CLIPTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLIPModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    CUTE_CATS = get_tests_dir("fixtures/tests_samples/COCO/000000039769.png")
    image = Image.open(CUTE_CATS)
    return image


class CLIPModelCompatibilityTest(unittest.TestCase):
    model_id = "hf-internal-testing/tiny-random-CLIPModel"

    def setUp(self):
        # 1. create input
        processor = CLIPProcessor.from_pretrained(self.model_id, from_hf_hub=True)
        image = prepare_img()
        self.inputs = processor(
            text=["a photo of a cat", "a photo of a dog"], images=image, padding=True, return_tensors="np"
        )

    @unittest.skip("model diff exists, need to be fixed")
    @require_package("transformers", "torch")
    def test_clip_converter(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # 1. create input
            inputs = self.inputs

            # 2. forward the paddle model
            from paddlenlp.transformers import CLIPModel

            paddle_model = CLIPModel.from_pretrained(self.model_id, from_hf_hub=True, cache_dir=tempdir)
            paddle_model.eval()
            paddle_logit = paddle_model(
                input_ids=paddle.to_tensor(inputs["input_ids"]), pixel_values=paddle.to_tensor(inputs["pixel_values"])
            )

            # 3. forward the torch model
            import torch
            from transformers import CLIPModel

            torch_model = CLIPModel.from_pretrained(self.model_id, cache_dir=tempdir)
            torch_model.eval()
            torch_logit = torch_model(
                input_ids=torch.tensor(inputs["input_ids"]), pixel_values=torch.tensor(inputs["pixel_values"])
            )

            # 4. compare results
            self.assertTrue(
                np.allclose(
                    paddle_logit.logits_per_image.detach().cpu().numpy(),
                    torch_logit.logits_per_image.detach().cpu().numpy(),
                    rtol=1e-4,
                )
            )

            self.assertTrue(
                np.allclose(
                    paddle_logit.logits_per_text.detach().cpu().numpy(),
                    torch_logit.logits_per_text.detach().cpu().numpy(),
                    rtol=1e-4,
                )
            )

    @unittest.skip("model diff exists, need to be fixed")
    @require_package("transformers", "torch")
    def test_clip_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # 1. create input
            inputs = self.inputs

            # 2. forward the torch  model
            import torch
            from transformers import CLIPModel

            torch_model = CLIPModel.from_pretrained(self.model_id)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(
                input_ids=torch.tensor(inputs["input_ids"]), pixel_values=torch.tensor(inputs["pixel_values"])
            )

            # 3. forward the paddle model
            from paddlenlp.transformers import CLIPModel

            paddle_model = CLIPModel.from_pretrained(tempdir, convert_from_torch=True)
            paddle_model.eval()
            paddle_logit = paddle_model(
                input_ids=paddle.to_tensor(inputs["input_ids"]), pixel_values=paddle.to_tensor(inputs["pixel_values"])
            )

            # 4. compare results
            self.assertTrue(
                np.allclose(
                    paddle_logit.logits_per_image.detach().cpu().numpy(),
                    torch_logit.logits_per_image.detach().cpu().numpy(),
                    rtol=1e-4,
                )
            )

            self.assertTrue(
                np.allclose(
                    paddle_logit.logits_per_text.detach().cpu().numpy(),
                    torch_logit.logits_per_text.detach().cpu().numpy(),
                    rtol=1e-4,
                )
            )

    @require_package("transformers", "torch")
    def test_clip_text_converter(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # 1. create input
            inputs = self.inputs

            # 2. forward the paddle model
            from paddlenlp.transformers import CLIPTextModel

            paddle_model = CLIPTextModel.from_pretrained(self.model_id, from_hf_hub=True, cache_dir=tempdir)
            paddle_model.eval()
            paddle_logit = paddle_model(input_ids=paddle.to_tensor(inputs["input_ids"]))

            # 3. forward the torch model
            import torch
            from transformers import CLIPTextModel

            torch_model = CLIPTextModel.from_pretrained(self.model_id, cache_dir=tempdir)
            torch_model.eval()
            torch_logit = torch_model(input_ids=torch.tensor(inputs["input_ids"]))

            # 4. compare results
            np.testing.assert_equal(
                paddle_logit.last_hidden_state.shape,
                torch_logit.last_hidden_state.shape,
            )
            np.testing.assert_equal(
                paddle_logit.pooler_output.shape,
                torch_logit.pooler_output.shape,
            )

    @require_package("transformers", "torch")
    def test_clip_vision_converter(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # 1. create input
            inputs = self.inputs

            # 2. forward the paddle model
            from paddlenlp.transformers import CLIPVisionModel

            paddle_model = CLIPVisionModel.from_pretrained(self.model_id, from_hf_hub=True, cache_dir=tempdir)
            paddle_model.eval()
            paddle_logit = paddle_model(pixel_values=paddle.to_tensor(inputs["pixel_values"]))

            # 3. forward the torch model
            import torch
            from transformers import CLIPVisionModel

            torch_model = CLIPVisionModel.from_pretrained(self.model_id, cache_dir=tempdir)
            torch_model.eval()
            torch_logit = torch_model(pixel_values=torch.tensor(inputs["pixel_values"]))

            # 4. compare results
            np.testing.assert_equal(
                paddle_logit.last_hidden_state.shape,
                torch_logit.last_hidden_state.shape,
            )
            np.testing.assert_equal(
                paddle_logit.pooler_output.shape,
                torch_logit.pooler_output.shape,
            )

    @require_package("transformers", "torch")
    def test_clip_text_with_projection_converter(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # 1. create input
            inputs = self.inputs

            # 2. forward the paddle model
            from paddlenlp.transformers import CLIPTextModelWithProjection

            paddle_model = CLIPTextModelWithProjection.from_pretrained(
                self.model_id, from_hf_hub=True, cache_dir=tempdir
            )
            paddle_model.eval()
            paddle_logit = paddle_model(input_ids=paddle.to_tensor(inputs["input_ids"]))

            # 3. forward the torch model
            import torch
            from transformers import CLIPTextModelWithProjection

            torch_model = CLIPTextModelWithProjection.from_pretrained(
                self.model_id, cache_dir=tempdir, ignore_mismatched_sizes=True
            )
            torch_model.eval()
            torch_logit = torch_model(input_ids=torch.tensor(inputs["input_ids"]))

            # 4. compare results
            np.testing.assert_equal(
                paddle_logit.last_hidden_state.shape,
                torch_logit.last_hidden_state.shape,
            )

    @require_package("transformers", "torch")
    def test_clip_vision_with_projection_converter(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # 1. create input
            inputs = self.inputs

            # 2. forward the paddle model
            from paddlenlp.transformers import CLIPVisionModelWithProjection

            paddle_model = CLIPVisionModelWithProjection.from_pretrained(
                self.model_id, from_hf_hub=True, cache_dir=tempdir
            )
            paddle_model.eval()
            paddle_logit = paddle_model(pixel_values=paddle.to_tensor(inputs["pixel_values"]))

            # 3. forward the torch model
            import torch
            from transformers import CLIPVisionModelWithProjection

            torch_model = CLIPVisionModelWithProjection.from_pretrained(
                self.model_id, cache_dir=tempdir, ignore_mismatched_sizes=True
            )
            torch_model.eval()
            torch_logit = torch_model(pixel_values=torch.tensor(inputs["pixel_values"]))

            # 4. compare results
            np.testing.assert_equal(
                paddle_logit.last_hidden_state.shape,
                torch_logit.last_hidden_state.shape,
            )


class CLIPModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        model.eval()
        processor = CLIPProcessor.from_pretrained(model_name)

        image = prepare_img()
        inputs = processor(
            text=["a photo of a cat", "a photo of a dog"], images=image, padding=True, return_tensors="pd"
        )

        # forward pass
        with paddle.no_grad():
            outputs = model(**inputs)

        # verify the logits
        self.assertEqual(
            outputs.logits_per_image.shape,
            [inputs.pixel_values.shape[0], inputs.input_ids.shape[0]],
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            [inputs.input_ids.shape[0], inputs.pixel_values.shape[0]],
        )

        expected_logits = paddle.to_tensor([[24.5701, 19.3049]])

        self.assertTrue(paddle.allclose(outputs.logits_per_image, expected_logits, atol=1e-3))
