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

import paddle

from ppdiffusers import (
    IFImg2ImgPipeline,
    IFImg2ImgSuperResolutionPipeline,
    IFInpaintingPipeline,
    IFInpaintingSuperResolutionPipeline,
    IFPipeline,
    IFSuperResolutionPipeline,
)
from ppdiffusers.models.attention_processor import AttnAddedKVProcessor
from ppdiffusers.utils.testing_utils import (
    floats_tensor,
    load_numpy,
    require_paddle_gpu,
    slow,
)

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, assert_mean_pixel_difference
from . import IFPipelineTesterMixin


class IFPipelineFastTests(PipelineTesterMixin, IFPipelineTesterMixin, unittest.TestCase):
    pipeline_class = IFPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"width", "height", "latents"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}

    def get_dummy_components(self):
        return self._get_dummy_components()

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "numpy",
        }

        return inputs

    def test_save_load_optional_components(self):
        self._test_save_load_optional_components()

    def test_save_load_float16(self):
        # Due to non-determinism in save load of the hf-internal-testing/tiny-random-t5 text encoder
        pass

    def test_attention_slicing_forward_pass(self):
        self._test_attention_slicing_forward_pass(expected_max_diff=1e-2)

    def test_save_load_local(self):
        self._test_save_load_local()

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(
            expected_max_diff=1e-2,
        )

    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=1e-3)


@slow
@require_paddle_gpu
class IFPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_all(self):
        # if

        pipe_1 = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", paddle_dtype=paddle.float16)

        pipe_2 = IFSuperResolutionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0", variant="fp16", paddle_dtype=paddle.float16, text_encoder=None, tokenizer=None
        )

        # pre compute text embeddings and remove T5 to save memory

        pipe_1.text_encoder

        prompt_embeds, negative_prompt_embeds = pipe_1.encode_prompt("anime turtle")

        del pipe_1.tokenizer
        del pipe_1.text_encoder
        gc.collect()

        pipe_1.tokenizer = None
        pipe_1.text_encoder = None

        pipe_1.enable_model_cpu_offload()
        pipe_2.enable_model_cpu_offload()

        pipe_1.unet.set_attn_processor(AttnAddedKVProcessor())
        pipe_2.unet.set_attn_processor(AttnAddedKVProcessor())

        self._test_if(pipe_1, pipe_2, prompt_embeds, negative_prompt_embeds)

        pipe_1.remove_all_hooks()
        pipe_2.remove_all_hooks()

        # img2img

        pipe_1 = IFImg2ImgPipeline(**pipe_1.components)
        pipe_2 = IFImg2ImgSuperResolutionPipeline(**pipe_2.components)

        pipe_1.enable_model_cpu_offload()
        pipe_2.enable_model_cpu_offload()

        pipe_1.unet.set_attn_processor(AttnAddedKVProcessor())
        pipe_2.unet.set_attn_processor(AttnAddedKVProcessor())

        self._test_if_img2img(pipe_1, pipe_2, prompt_embeds, negative_prompt_embeds)

        pipe_1.remove_all_hooks()
        pipe_2.remove_all_hooks()

        # inpainting

        pipe_1 = IFInpaintingPipeline(**pipe_1.components)
        pipe_2 = IFInpaintingSuperResolutionPipeline(**pipe_2.components)

        pipe_1.enable_model_cpu_offload()
        pipe_2.enable_model_cpu_offload()

        pipe_1.unet.set_attn_processor(AttnAddedKVProcessor())
        pipe_2.unet.set_attn_processor(AttnAddedKVProcessor())

        self._test_if_inpainting(pipe_1, pipe_2, prompt_embeds, negative_prompt_embeds)

    def _test_if(self, pipe_1, pipe_2, prompt_embeds, negative_prompt_embeds):
        # pipeline 1

        generator = paddle.Generator().manual_seed(0)
        output = pipe_1(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=2,
            generator=generator,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (64, 64, 3)

        mem_bytes = paddle.cuda.max_memory_allocated()
        assert mem_bytes < 13 * 10**9

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/if/test_if.npy"
        )
        assert_mean_pixel_difference(image, expected_image)

        # pipeline 2

        generator = paddle.Generator().manual_seed(0)

        image = floats_tensor((1, 3, 64, 64), rng=random.Random(0))

        output = pipe_2(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=image,
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (256, 256, 3)

        mem_bytes = paddle.cuda.max_memory_allocated()
        assert mem_bytes < 4 * 10**9

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/if/test_if_superresolution_stage_II.npy"
        )
        assert_mean_pixel_difference(image, expected_image)

    def _test_if_img2img(self, pipe_1, pipe_2, prompt_embeds, negative_prompt_embeds):
        # pipeline 1

        image = floats_tensor((1, 3, 64, 64), rng=random.Random(0))

        generator = paddle.Generator().manual_seed(0)

        output = pipe_1(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=image,
            num_inference_steps=2,
            generator=generator,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (64, 64, 3)

        mem_bytes = paddle.cuda.max_memory_allocated()
        assert mem_bytes < 10 * 10**9

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/if/test_if_img2img.npy"
        )
        assert_mean_pixel_difference(image, expected_image)

        # pipeline 2

        generator = paddle.Generator().manual_seed(0)

        original_image = floats_tensor((1, 3, 256, 256), rng=random.Random(0))
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(0))

        output = pipe_2(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=image,
            original_image=original_image,
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (256, 256, 3)

        mem_bytes = paddle.cuda.max_memory_allocated()
        assert mem_bytes < 4 * 10**9

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/if/test_if_img2img_superresolution_stage_II.npy"
        )
        assert_mean_pixel_difference(image, expected_image)

    def _test_if_inpainting(self, pipe_1, pipe_2, prompt_embeds, negative_prompt_embeds):
        # pipeline 1

        image = floats_tensor((1, 3, 64, 64), rng=random.Random(0))
        mask_image = floats_tensor((1, 3, 64, 64), rng=random.Random(1))

        generator = paddle.Generator().manual_seed(0)
        output = pipe_1(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=image,
            mask_image=mask_image,
            num_inference_steps=2,
            generator=generator,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (64, 64, 3)

        mem_bytes = paddle.cuda.max_memory_allocated()
        assert mem_bytes < 10 * 10**9

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/if/test_if_inpainting.npy"
        )
        assert_mean_pixel_difference(image, expected_image)

        # pipeline 2

        generator = paddle.Generator().manual_seed(0)

        image = floats_tensor((1, 3, 64, 64), rng=random.Random(0))
        original_image = floats_tensor((1, 3, 256, 256), rng=random.Random(0))
        mask_image = floats_tensor((1, 3, 256, 256), rng=random.Random(1))

        output = pipe_2(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=image,
            mask_image=mask_image,
            original_image=original_image,
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (256, 256, 3)

        mem_bytes = paddle.device.cuda.max_memory_allocated()
        assert mem_bytes < 4 * 10**9

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/if/test_if_inpainting_superresolution_stage_II.npy"
        )
        assert_mean_pixel_difference(image, expected_image)
