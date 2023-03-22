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
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_INPAINTING_PARAMS,
)
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    prepare_mask_and_masked_image,
)
from ppdiffusers.utils import floats_tensor, load_image, load_numpy, nightly, slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class StableDiffusionInpaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS

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
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config).eval()
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().transpose(perm=[0, 2, 3, 1])[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((64, 64))
        generator = paddle.Generator().manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_inpaint(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.54607296, 0.45091373, 0.3229987, 0.29081488, 0.27186918, 0.45776647, 0.22677511, 0.15349615, 0.34303916]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_inpaint_image_tensor(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs)
        out_pil = output.images
        inputs = self.get_dummy_inputs()
        inputs["image"] = (
            paddle.to_tensor(np.array(inputs["image"]) / 127.5 - 1).transpose(perm=[2, 0, 1]).unsqueeze(axis=0)
        )
        inputs["mask_image"] = (
            paddle.to_tensor(np.array(inputs["mask_image"]) / 255).transpose(perm=[2, 0, 1])[:1].unsqueeze(axis=0)
        )
        output = sd_pipe(**inputs)
        out_tensor = output.images
        assert out_pil.shape == (1, 64, 64, 3)
        assert np.abs(out_pil.flatten() - out_tensor.flatten()).max() < 0.05

    def test_stable_diffusion_inpaint_with_num_images_per_prompt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        images = sd_pipe(**inputs, num_images_per_prompt=2).images
        assert len(images) == 2


@slow
@require_paddle_gpu
class StableDiffusionInpaintPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype="float32", seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/input_bench_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/input_bench_mask.png"
        )
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_inpaint_ddim(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.05978, 0.10983, 0.10514, 0.07922, 0.08483, 0.08587, 0.05302, 0.03218, 0.01636])
        assert np.abs(expected_slice - image_slice).max() < 0.0001

    def test_stable_diffusion_inpaint_fp16(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", paddle_dtype=paddle.float16, safety_checker=None
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(dtype="float16")
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [
                0.06103516,
                0.11010742,
                0.10571289,
                0.07910156,
                0.08569336,
                0.08398438,
                0.05273438,
                0.03222656,
                0.01660156,
            ]
        )
        assert np.abs(expected_slice - image_slice).max() < 0.05

    def test_stable_diffusion_inpaint_pndm(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.06892, 0.06994, 0.07905, 0.05366, 0.04709, 0.04890, 0.04107, 0.05083, 0.04180])
        assert np.abs(expected_slice - image_slice).max() < 0.0001

    def test_stable_diffusion_inpaint_k_lms(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.23513, 0.22413, 0.29442, 0.24243, 0.26214, 0.30329, 0.26431, 0.25025, 0.25197])
        assert np.abs(expected_slice - image_slice).max() < 0.0001


@nightly
@require_paddle_gpu
class StableDiffusionInpaintPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype="float32", seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/input_bench_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/input_bench_mask.png"
        )
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_inpaint_ddim(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/stable_diffusion_inpaint_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_inpaint_pndm(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.scheduler = PNDMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/stable_diffusion_inpaint_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_inpaint_lms(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/stable_diffusion_inpaint_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_inpaint_dpm(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 30
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/stable_diffusion_inpaint_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001


class StableDiffusionInpaintingPrepareMaskAndMaskedImageTests(unittest.TestCase):
    def test_pil_inputs(self):
        im = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        im = Image.fromarray(im)
        mask = np.random.randint(0, 255, (32, 32), dtype=np.uint8) > 127.5
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        t_mask, t_masked = prepare_mask_and_masked_image(im, mask)
        self.assertTrue(isinstance(t_mask, paddle.Tensor))
        self.assertTrue(isinstance(t_masked, paddle.Tensor))
        self.assertEqual(t_mask.ndim, 4)
        self.assertEqual(t_masked.ndim, 4)
        self.assertEqual(t_mask.shape, [1, 1, 32, 32])
        self.assertEqual(t_masked.shape, [1, 3, 32, 32])
        self.assertTrue(t_mask.dtype == paddle.float32)
        self.assertTrue(t_masked.dtype == paddle.float32)
        self.assertTrue(t_mask.logsumexp() >= 0.0)
        self.assertTrue(t_mask.max() <= 1.0)
        self.assertTrue(t_masked.min() >= -1.0)
        self.assertTrue(t_masked.min() <= 1.0)

        self.assertTrue(t_mask.sum() > 0.0)

    def test_np_inputs(self):
        im_np = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        im_pil = Image.fromarray(im_np)
        mask_np = np.random.randint(0, 255, (32, 32), dtype=np.uint8) > 127.5
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        t_mask_np, t_masked_np = prepare_mask_and_masked_image(im_np, mask_np)
        t_mask_pil, t_masked_pil = prepare_mask_and_masked_image(im_pil, mask_pil)
        self.assertTrue((t_mask_np == t_mask_pil).all())
        self.assertTrue((t_masked_np == t_masked_pil).all())

    def test_paddle_3D_2D_inputs(self):
        im_tensor = paddle.randint(0, 255, (3, 32, 32)).cast("uint8")
        mask_tensor = paddle.randint(0, 255, (32, 32)).cast("uint8") > 127.5
        im_np = im_tensor.numpy().transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()

        t_mask_tensor, t_masked_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor.cast("int64")
        )
        t_mask_np, t_masked_np = prepare_mask_and_masked_image(im_np, mask_np)

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())

    def test_paddle_3D_3D_inputs(self):
        im_tensor = paddle.randint(0, 255, (3, 32, 32)).cast("uint8")
        mask_tensor = paddle.randint(0, 255, (1, 32, 32)).cast("uint8") > 127.5
        im_np = im_tensor.numpy().transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0]
        t_mask_tensor, t_masked_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor.cast("int64")
        )
        t_mask_np, t_masked_np = prepare_mask_and_masked_image(im_np, mask_np)
        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())

    def test_paddle_4D_2D_inputs(self):
        im_tensor = paddle.randint(0, 255, (1, 3, 32, 32)).cast("uint8")
        mask_tensor = paddle.randint(0, 255, (32, 32)).cast("uint8") > 127.5
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()
        t_mask_tensor, t_masked_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor.cast("int64")
        )
        t_mask_np, t_masked_np = prepare_mask_and_masked_image(im_np, mask_np)
        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())

    def test_paddle_4D_3D_inputs(self):
        im_tensor = paddle.randint(0, 255, (1, 3, 32, 32)).cast("uint8")
        mask_tensor = paddle.randint(0, 255, (1, 32, 32)).cast("uint8") > 127.5
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0]
        t_mask_tensor, t_masked_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor.cast("int64")
        )
        t_mask_np, t_masked_np = prepare_mask_and_masked_image(im_np, mask_np)
        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())

    def test_paddle_4D_4D_inputs(self):
        im_tensor = paddle.randint(0, 255, (1, 3, 32, 32)).cast("uint8")
        mask_tensor = paddle.randint(0, 255, (1, 1, 32, 32)).cast("uint8") > 127.5
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0][0]
        t_mask_tensor, t_masked_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor.cast("int64")
        )
        t_mask_np, t_masked_np = prepare_mask_and_masked_image(im_np, mask_np)
        self.assertTrue((t_mask_tensor == t_mask_np.cast("float64")).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())

    def test_paddle_batch_4D_3D(self):
        im_tensor = paddle.randint(0, 255, (2, 3, 32, 32)).cast("uint8")
        mask_tensor = paddle.randint(0, 255, (2, 32, 32)).cast("uint8") > 127.5
        im_nps = [im.numpy().transpose(1, 2, 0) for im in im_tensor]
        mask_nps = [mask.numpy() for mask in mask_tensor]
        t_mask_tensor, t_masked_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor.cast("int64")
        )
        nps = [prepare_mask_and_masked_image(i, m) for i, m in zip(im_nps, mask_nps)]
        t_mask_np = paddle.concat(x=[n[0] for n in nps])
        t_masked_np = paddle.concat(x=[n[1] for n in nps])
        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())

    def test_paddle_batch_4D_4D(self):
        im_tensor = paddle.randint(0, 255, (2, 3, 32, 32)).cast("uint8")
        mask_tensor = paddle.randint(0, 255, (2, 32, 32)).cast("uint8") > 127.5

        im_nps = [im.numpy().transpose(1, 2, 0) for im in im_tensor]
        mask_nps = [mask.numpy() for mask in mask_tensor]
        t_mask_tensor, t_masked_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor.cast("int64")
        )
        nps = [prepare_mask_and_masked_image(i, m) for i, m in zip(im_nps, mask_nps)]
        t_mask_np = paddle.concat(x=[n[0] for n in nps])
        t_masked_np = paddle.concat(x=[n[1] for n in nps])
        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())

    def test_shape_mismatch(self):
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(paddle.randn(shape=[3, 32, 32]), paddle.randn(shape=[64, 64]))
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(paddle.randn(shape=[2, 3, 32, 32]), paddle.randn(shape=[4, 64, 64]))
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(paddle.randn(shape=[2, 3, 32, 32]), paddle.randn(shape=[4, 1, 64, 64]))

    def test_type_mismatch(self):
        with self.assertRaises(TypeError):
            prepare_mask_and_masked_image(paddle.rand(shape=[3, 32, 32]), paddle.rand(shape=[3, 32, 32]).numpy())
        with self.assertRaises(TypeError):
            prepare_mask_and_masked_image(paddle.rand(shape=[3, 32, 32]).numpy(), paddle.rand(shape=[3, 32, 32]))

    def test_channels_first(self):
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(paddle.rand(shape=[32, 32, 3]), paddle.rand(shape=[3, 32, 32]))

    def test_tensor_range(self):
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(paddle.ones(shape=[3, 32, 32]) * 2, paddle.rand(shape=[32, 32]))
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(paddle.ones(shape=[3, 32, 32]) * -2, paddle.rand(shape=[32, 32]))
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(paddle.rand(shape=[3, 32, 32]), paddle.ones(shape=[32, 32]) * 2)
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(paddle.rand(shape=[3, 32, 32]), paddle.ones(shape=[32, 32]) * -1)
