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

from paddlenlp.transformers import XLMRobertaTokenizer
from ppdiffusers import (
    AltDiffusionImg2ImgPipeline,
    AutoencoderKL,
    PNDMScheduler,
    UNet2DConditionModel,
)
from ppdiffusers.pipelines.alt_diffusion.modeling_roberta_series import (
    RobertaSeriesConfig,
    RobertaSeriesModelWithTransformation,
)
from ppdiffusers.utils import floats_tensor, load_image, slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class AltDiffusionImg2ImgPipelineFastTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 3
        sizes = 32, 32
        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0))
        return image

    @property
    def dummy_cond_unet(self):
        paddle.seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        return model

    @property
    def dummy_vae(self):
        paddle.seed(0)
        model = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        return model

    @property
    def dummy_text_encoder(self):
        paddle.seed(0)
        config = RobertaSeriesConfig(
            hidden_size=32,
            project_dim=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=5006,
        )
        return RobertaSeriesModelWithTransformation(config)

    @property
    def dummy_extractor(self):
        def extract(*args, **kwargs):
            class Out:
                def __init__(self):
                    self.pixel_values = paddle.ones(shape=[0])

                def to(self, device):
                    self.pixel_values
                    return self

            return Out()

        return extract

    def test_stable_diffusion_img2img_default_case(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = XLMRobertaTokenizer.from_pretrained("hf-internal-testing/tiny-xlm-roberta")
        tokenizer.model_max_length = 77
        init_image = self.dummy_image
        alt_pipe = AltDiffusionImg2ImgPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )
        alt_pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        output = alt_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
        )
        image = output.images
        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = alt_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            return_dict=False,
        )[0]
        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [0.920333, 0.53369606, 0.56038886, 0.47739977, 0.18425128, 0.47001246, 0.5406687, 0.4329021, 0.6154301]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.005
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.005

    def test_stable_diffusion_img2img_fp16(self):
        """Test that stable diffusion img2img works with fp16"""
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = XLMRobertaTokenizer.from_pretrained("hf-internal-testing/tiny-xlm-roberta")
        tokenizer.model_max_length = 77
        init_image = self.dummy_image
        unet = unet.to(dtype=paddle.float16)
        vae = vae.to(dtype=paddle.float16)
        bert = bert.to(dtype=paddle.float16)
        alt_pipe = AltDiffusionImg2ImgPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )
        alt_pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        image = alt_pipe(
            [prompt], generator=generator, num_inference_steps=2, output_type="np", image=init_image
        ).images
        assert image.shape == (1, 32, 32, 3)

    def test_stable_diffusion_img2img_pipeline_multiple_of_8(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((760, 504))
        model_id = "BAAI/AltDiffusion"
        pipe = AltDiffusionImg2ImgPipeline.from_pretrained(model_id, safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        prompt = "A fantasy landscape, trending on artstation"
        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5, generator=generator, output_type="np"
        )
        image = output.images[0]
        image_slice = image[255:258, 383:386, -1]
        assert image.shape == (504, 760, 3)
        expected_slice = np.array(
            [0.3251649, 0.3340174, 0.3418343, 0.32628638, 0.33462793, 0.3300547, 0.31628466, 0.3470268, 0.34273332]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001


@slow
@require_paddle_gpu
class AltDiffusionImg2ImgPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_stable_diffusion_img2img_pipeline_default(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((768, 512))
        # expected_image = load_numpy(
        #     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/fantasy_landscape_alt.npy"
        # )
        model_id = "BAAI/AltDiffusion"
        pipe = AltDiffusionImg2ImgPipeline.from_pretrained(model_id, safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        prompt = "A fantasy landscape, trending on artstation"
        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5, generator=generator, output_type="np"
        )
        image = output.images
        assert image.shape == (1, 512, 768, 3)
        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array(
            [
                0.09987255930900574,
                0.09875822067260742,
                0.12803134322166443,
                0.10067081451416016,
                0.1142435073852539,
                0.11815103888511658,
                0.14216548204421997,
                0.16465380787849426,
                0.15393462777137756,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
