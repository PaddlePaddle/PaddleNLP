# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import gc
import random
import unittest

import numpy as np
import paddle
from PIL import Image
from ppdiffusers_test.pipeline_params import (
    IMAGE_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    IMAGE_GUIDED_IMAGE_INPAINTING_PARAMS,
)
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import CLIPImageProcessor, CLIPVisionConfig
from ppdiffusers import (
    AutoencoderKL,
    PaintByExamplePipeline,
    PNDMScheduler,
    UNet2DConditionModel,
)
from ppdiffusers.pipelines.paint_by_example import PaintByExampleImageEncoder
from ppdiffusers.utils import floats_tensor, load_image, slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class PaintByExamplePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = PaintByExamplePipeline
    params = IMAGE_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = IMAGE_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS

    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=9,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = PNDMScheduler(skip_prk_steps=True)
        paddle.seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        paddle.seed(0)
        config = CLIPVisionConfig(
            hidden_size=32,
            projection_dim=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            image_size=32,
            patch_size=4,
        )
        image_encoder = PaintByExampleImageEncoder(config, proj_size=32)
        feature_extractor = CLIPImageProcessor(crop_size=32, size=32)
        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "image_encoder": image_encoder,
            "safety_checker": None,
            "feature_extractor": feature_extractor,
        }
        return components

    def convert_to_pd(self, image):
        image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)
        image = paddle.to_tensor(data=image).cast("float32") / 127.5 - 1.0
        return image

    # TODO check this
    def test_save_load_float16(self):
        pass

    def get_dummy_inputs(self, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().transpose(perm=[0, 2, 3, 1])[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((64, 64))
        example_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((32, 32))
        generator = paddle.Generator().manual_seed(seed)

        inputs = {
            "example_image": example_image,
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_paint_by_example_inpaint(self):
        components = self.get_dummy_components()
        pipe = PaintByExamplePipeline(**components)
        pipe = pipe.to("cpu")
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.57881105, 0.5201367, 0.38523138, 0.45297998, 0.39643988, 0.48735288, 0.32640088, 0.23076776, 0.40672228]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_paint_by_example_image_tensor(self):
        inputs = self.get_dummy_inputs()
        inputs.pop("mask_image")
        image = self.convert_to_pd(inputs.pop("image"))
        mask_image = image.clip(min=0, max=1) / 2
        pipe = PaintByExamplePipeline(**self.get_dummy_components())
        pipe.set_progress_bar_config(disable=None)
        output = pipe(image=image, mask_image=mask_image[:, 0], **inputs)
        out_1 = output.images
        image = image.cpu().transpose(perm=[0, 2, 3, 1])[0]
        mask_image = mask_image.cpu().transpose(perm=[0, 2, 3, 1])[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(mask_image)).convert("RGB")
        output = pipe(**self.get_dummy_inputs())
        out_2 = output.images
        assert out_1.shape == (1, 64, 64, 3)
        assert np.abs(out_1.flatten() - out_2.flatten()).max() < 0.05

    def test_paint_by_example_inpaint_with_num_images_per_prompt(self):
        pipe = PaintByExamplePipeline(**self.get_dummy_components())
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        images = pipe(**inputs, num_images_per_prompt=2).images
        assert len(images) == 2


@slow
@require_paddle_gpu
class PaintByExamplePipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_paint_by_example(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/paint_by_example/dog_in_bucket.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/paint_by_example/mask.png"
        )
        example_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/paint_by_example/panda.jpg"
        )
        pipe = PaintByExamplePipeline.from_pretrained("Fantasy-Studio/Paint-by-Example")
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(seed=321)
        output = pipe(
            image=init_image,
            mask_image=mask_image,
            example_image=example_image,
            generator=generator,
            guidance_scale=5.0,
            num_inference_steps=50,
            output_type="np",
        )
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.4834, 0.4811, 0.4874, 0.5122, 0.5081, 0.5144, 0.5291, 0.529, 0.5374])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.02
