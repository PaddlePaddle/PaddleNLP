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
import random
import unittest

import numpy as np
import paddle

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    PNDMScheduler,
    StableDiffusionAdapterPipeline,
    T2IAdapter,
    UNet2DConditionModel,
)
from ppdiffusers.utils import floats_tensor, load_image, load_numpy, slow
from ppdiffusers.utils.import_utils import is_ppxformers_available

from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin


class StableDiffusionAdapterPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionAdapterPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS

    def get_dummy_components(self):
        paddle.seed(seed=0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = PNDMScheduler(skip_prk_steps=True)
        paddle.Generator().manual_seed(seed=0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        vae_scale_factor = 2
        paddle.Generator().manual_seed(seed=0)
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
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        paddle.Generator().manual_seed(seed=0)
        adapter = T2IAdapter(
            block_out_channels=[32, 64],
            channels_in=3,
            num_res_blocks=2,
            kernel_size=1,
            res_block_skip=True,
            use_conv=False,
            input_scale_factor=vae_scale_factor,
        )
        components = {
            "adapter": adapter,
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
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(seed))
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_adapter_default_case(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionAdapterPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.9088084, 0.6012194, 0.43046606, 0.7228667, 0.46428588, 0.30164504, 0.508494, 0.6241546, 0.55453974]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.005

    def test_attention_slicing_forward_pass(self):
        return self._test_attention_slicing_forward_pass(expected_max_diff=0.002)

    @unittest.skipIf(
        not is_ppxformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=0.002)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=0.002)


@slow
class StableDiffusionAdapterPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, revision="segmentation", dtype="float32", seed=0):
        generator = paddle.Generator().manual_seed(seed)
        image_urls = {
            "segmentation": "https://huggingface.co/RzZ/sd-v1-4-adapter-pipeline/resolve/segmentation/sample_input.png",
            "keypose": "https://huggingface.co/RzZ/sd-v1-4-adapter-pipeline/resolve/keypose/sample_input.png",
            "depth": "https://huggingface.co/RzZ/sd-v1-4-adapter-pipeline/resolve/depth/sample_input.png",
        }
        prompt_by_rev = {
            "segmentation": "A black Honda motorcycle parked in front of a garage",
            "keypose": "An astronaut on the moon",
            "depth": "An office room with nice view",
        }
        cond_image = load_image(image_urls[revision])
        inputs = {
            "prompt": prompt_by_rev[revision],
            "image": cond_image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_segmentation_adapter(self):
        adapter = T2IAdapter.from_pretrained("RzZ/sd-v1-4-adapter-seg")
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", adapter=adapter, safety_checker=None
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(revision="segmentation")
        image = pipe(**inputs).images
        assert image.shape == (1, 512, 512, 3)
        expected_image = load_numpy(
            "https://huggingface.co/RzZ/sd-v1-4-adapter-pipeline/resolve/segmentation/sample_output.npy"
        )
        assert np.abs(expected_image - image).max() < 0.005

    def test_stable_diffusion_keypose_adapter(self):
        adapter = T2IAdapter.from_pretrained("RzZ/sd-v1-4-adapter-keypose")
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", adapter=adapter, safety_checker=None
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(revision="keypose")
        image = pipe(**inputs).images
        assert image.shape == (1, 512, 512, 3)
        expected_image = load_numpy(
            "https://huggingface.co/RzZ/sd-v1-4-adapter-pipeline/resolve/keypose/sample_output.npy"
        )
        assert np.abs(expected_image - image).max() < 0.005

    def test_stable_diffusion_depth_adapter(self):
        adapter = T2IAdapter.from_pretrained("RzZ/sd-v1-4-adapter-depth")
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", adapter=adapter, safety_checker=None
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(revision="depth")
        image = pipe(**inputs).images
        assert image.shape == (1, 512, 512, 3)
        expected_image = load_numpy(
            "https://huggingface.co/RzZ/sd-v1-4-adapter-pipeline/resolve/depth/sample_output.npy"
        )
        assert np.abs(expected_image - image).max() < 0.005
