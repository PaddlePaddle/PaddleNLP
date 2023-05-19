# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np
import paddle
from ppdiffusers_test.pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import load_image, load_numpy, randn_tensor, slow
from ppdiffusers.utils.import_utils import is_ppxformers_available
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class StableDiffusionControlNetPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionControlNetPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS

    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        paddle.seed(0)
        controlnet = ControlNetModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
        )
        paddle.seed(0)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
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
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)

        controlnet_embedder_scale_factor = 2
        image = randn_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            generator=generator,
        )

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
            "image": image,
        }

        return inputs

    def test_attention_slicing_forward_pass(self):
        return self._test_attention_slicing_forward_pass(expected_max_diff=2e-3)

    @unittest.skipIf(
        not is_ppxformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=1e-2)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=2e-3)


@slow
@require_paddle_gpu
class StableDiffusionControlNetPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_canny(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        prompt = "bird"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np")

        image = output.images[0]

        assert image.shape == (768, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny_out.npy"
        )

        assert np.abs(expected_image - image).max() < 5e-3

    def test_depth(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        prompt = "Stormtrooper's lecture"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np")

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )

        assert np.abs(expected_image - image).max() < 5e-3

    def test_hed(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        prompt = "oil painting of handsome old man, masterpiece"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/man_hed.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np")

        image = output.images[0]

        assert image.shape == (704, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/man_hed_out.npy"
        )

        assert np.abs(expected_image - image).max() < 5e-3

    def test_mlsd(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        prompt = "room"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/room_mlsd.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np")

        image = output.images[0]

        assert image.shape == (704, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/room_mlsd_out.npy"
        )

        assert np.abs(expected_image - image).max() < 5e-3

    def test_normal(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        prompt = "cute toy"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/cute_toy_normal.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np")

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/cute_toy_normal_out.npy"
        )

        assert np.abs(expected_image - image).max() < 5e-3

    def test_openpose(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        prompt = "Chef in the kitchen"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np")

        image = output.images[0]

        assert image.shape == (768, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/chef_pose_out.npy"
        )

        assert np.abs(expected_image - image).max() < 5e-3

    def test_scribble(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(5)
        prompt = "bag"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bag_scribble.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np")

        image = output.images[0]

        assert image.shape == (640, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bag_scribble_out.npy"
        )

        assert np.abs(expected_image - image).max() < 5e-3

    def test_seg(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(5)
        prompt = "house"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/house_seg.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np")

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/house_seg_out.npy"
        )

        assert np.abs(expected_image - image).max() < 5e-3
