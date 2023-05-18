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
import unittest

import numpy as np
import paddle

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    Transformer2DModel,
    VQDiffusionPipeline,
    VQDiffusionScheduler,
    VQModel,
)
from ppdiffusers.pipelines.vq_diffusion.pipeline_vq_diffusion import (
    LearnedClassifierFreeSamplingEmbeddings,
)
from ppdiffusers.utils import load_numpy, slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class VQDiffusionPipelineFastTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    @property
    def num_embed(self):
        return 12

    @property
    def num_embeds_ada_norm(self):
        return 12

    @property
    def text_embedder_hidden_size(self):
        return 32

    @property
    def dummy_vqvae(self):
        paddle.seed(0)
        model = VQModel(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=3,
            num_vq_embeddings=self.num_embed,
            vq_embed_dim=3,
        )
        return model

    @property
    def dummy_tokenizer(self):
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        return tokenizer

    @property
    def dummy_text_encoder(self):
        paddle.seed(0)
        config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=self.text_embedder_hidden_size,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        return CLIPTextModel(config).eval()

    @property
    def dummy_transformer(self):
        paddle.seed(0)
        height = 12
        width = 12
        model_kwargs = {
            "attention_bias": True,
            "cross_attention_dim": 32,
            "attention_head_dim": height * width,
            "num_attention_heads": 1,
            "num_vector_embeds": self.num_embed,
            "num_embeds_ada_norm": self.num_embeds_ada_norm,
            "norm_num_groups": 32,
            "sample_size": width,
            "activation_fn": "geglu-approximate",
        }
        model = Transformer2DModel(**model_kwargs)
        return model

    def test_vq_diffusion(self):
        vqvae = self.dummy_vqvae
        text_encoder = self.dummy_text_encoder
        tokenizer = self.dummy_tokenizer
        transformer = self.dummy_transformer
        scheduler = VQDiffusionScheduler(self.num_embed)
        learned_classifier_free_sampling_embeddings = LearnedClassifierFreeSamplingEmbeddings(learnable=False)
        pipe = VQDiffusionPipeline(
            vqvae=vqvae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            learned_classifier_free_sampling_embeddings=learned_classifier_free_sampling_embeddings,
        )
        pipe.set_progress_bar_config(disable=None)
        prompt = "teddy bear playing in the pool"
        generator = paddle.Generator().manual_seed(0)
        output = pipe([prompt], generator=generator, num_inference_steps=2, output_type="np")
        image = output.images
        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = pipe(
            [prompt], generator=generator, output_type="np", return_dict=False, num_inference_steps=2
        )[0]
        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
        assert image.shape == (1, 24, 24, 3)
        expected_slice = np.array(
            [0.828682, 0.9523265, 0.9386728, 1.0, 0.670735, 0.69808894, 0.57591987, 0.5695971, 0.6580568]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01

    def test_vq_diffusion_classifier_free_sampling(self):
        vqvae = self.dummy_vqvae
        text_encoder = self.dummy_text_encoder
        tokenizer = self.dummy_tokenizer
        transformer = self.dummy_transformer
        scheduler = VQDiffusionScheduler(self.num_embed)
        learned_classifier_free_sampling_embeddings = LearnedClassifierFreeSamplingEmbeddings(
            learnable=True, hidden_size=self.text_embedder_hidden_size, length=tokenizer.model_max_length
        )
        pipe = VQDiffusionPipeline(
            vqvae=vqvae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            learned_classifier_free_sampling_embeddings=learned_classifier_free_sampling_embeddings,
        )
        pipe.set_progress_bar_config(disable=None)
        prompt = "teddy bear playing in the pool"
        generator = paddle.Generator().manual_seed(0)
        output = pipe([prompt], generator=generator, num_inference_steps=2, output_type="np")
        image = output.images
        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = pipe(
            [prompt], generator=generator, output_type="np", return_dict=False, num_inference_steps=2
        )[0]
        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
        assert image.shape == (1, 24, 24, 3)
        expected_slice = np.array(
            [0.51841056, 0.8037737, 0.71010727, 0.8259821, 0.6837207, 0.53903896, 0.80566585, 0.79730004, 0.7192132]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01


@slow
@require_paddle_gpu
class VQDiffusionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_vq_diffusion_classifier_free_sampling(self):
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/vq_diffusion/teddy_bear_pool_classifier_free_sampling.npy"
        )
        pipeline = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq")
        pipeline = pipeline
        pipeline.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        output = pipeline(
            "teddy bear playing in the pool", num_images_per_prompt=1, generator=generator, output_type="np"
        )
        image = output.images[0]
        assert image.shape == (256, 256, 3)
        assert np.abs(expected_image - image).max() < 0.01
