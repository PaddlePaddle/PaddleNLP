# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Callable, List, Optional, Union

import numpy as np
import PIL.Image

from ...utils import logging
from .pipeline_fastdeploy_upainting import FastDeployUPaintingPipeline
from .pipeline_fastdeploy_upainting_img2img import FastDeployUPaintingImg2ImgPipeline
from .pipeline_fastdeploy_upainting_inpaint_legacy import (
    FastDeployUPaintingInpaintPipelineLegacy,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FastDeployUPaintingMegaPipeline(FastDeployUPaintingPipeline):
    r"""
    Pipeline for generation using FastDeployUPaintingMegaPipeline.

    This model inherits from [`FastDeployUPaintingPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving etc.)

    Args:
        vae_encoder ([`FastDeployRuntimeModel`]):
            Variational Auto-Encoder (VAE) Model to encode images to latent representations.
        vae_decoder ([`FastDeployRuntimeModel`]):
            Variational Auto-Encoder (VAE) Model to decode images from latent representations.
        text_encoder ([`FastDeployRuntimeModel`]):
            Text-encoder model based on [BERT](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.bert.modeling.html#paddlenlp.transformers.bert.modeling.BertModel) architecture.
        tokenizer (`BertTokenizer`):
            Tokenizer of class
            [BertTokenizer](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.bert.tokenizer.html#paddlenlp.transformers.bert.tokenizer.BertTokenizer).
        unet ([`FastDeployRuntimeModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`PNDMScheduler`], [`EulerDiscreteScheduler`], [`EulerAncestralDiscreteScheduler`]
            or [`DPMSolverMultistepScheduler`].
    """
    _optional_components = ["vae_encoder"]

    def __call__(self, *args, **kwargs):
        return self.text2img(*args, **kwargs)

    def text2img(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: Optional[int] = 1,
    ):

        expected_components = inspect.signature(FastDeployUPaintingPipeline.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        temp_pipeline = FastDeployUPaintingPipeline(
            **components,
        )
        output = temp_pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
        )
        return output

    def img2img(
        self,
        prompt: Union[str, List[str]],
        image: Union[np.ndarray, PIL.Image.Image],
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[np.random.RandomState] = None,
        noise: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        expected_components = inspect.signature(FastDeployUPaintingImg2ImgPipeline.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        temp_pipeline = FastDeployUPaintingImg2ImgPipeline(
            **components,
        )
        output = temp_pipeline(
            prompt=prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            noise=noise,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
        )

        return output

    def inpaint_legacy(
        self,
        prompt: Union[str, List[str]],
        image: Union[np.ndarray, PIL.Image.Image],
        mask_image: Union[np.ndarray, PIL.Image.Image],
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[np.random.RandomState] = None,
        noise: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        expected_components = inspect.signature(FastDeployUPaintingInpaintPipelineLegacy.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        temp_pipeline = FastDeployUPaintingInpaintPipelineLegacy(
            **components,
        )
        output = temp_pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            noise=noise,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
        )

        return output
