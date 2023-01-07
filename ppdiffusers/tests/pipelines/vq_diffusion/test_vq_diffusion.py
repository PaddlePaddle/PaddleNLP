# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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


class VQDiffusionPipelineFastTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
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
        config = dict(
            text_embed_dim=self.text_embedder_hidden_size,
            text_heads=4,
            text_layers=5,
            vocab_size=1000,
        )
        config = CLIPTextConfig.from_dict(config)
        model = CLIPTextModel(config)
        model.eval()
        return model

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
            [
                0.6901105642318726,
                0.5934056639671326,
                0.5921000838279724,
                0.4860895872116089,
                0.6917713284492493,
                0.5599623918533325,
                0.6187697649002075,
                0.728931188583374,
                0.6061633229255676,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

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
            [
                0.6959998607635498,
                0.7165096998214722,
                0.739770770072937,
                0.7538110017776489,
                0.8786328434944153,
                0.6491624712944031,
                0.6340304017066956,
                0.7169196605682373,
                0.7034797668457031,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2


@slow
class VQDiffusionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_vq_diffusion_classifier_free_sampling(self):
        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/teddy_bear_pool_classifier_free_sampling.npy"
        )

        pipeline = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq")
        pipeline.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        output = pipeline(
            "teddy bear playing in the pool",
            num_images_per_prompt=1,
            generator=generator,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (256, 256, 3)
        assert np.abs(expected_image - image).max() < 1e-2
