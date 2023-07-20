# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Callable, Dict, List, Optional, Union

import paddle
import PIL.Image

from ...utils import logging
from .pipeline_fastdeploy_cycle_diffusion import FastDeployCycleDiffusionPipeline
from .pipeline_fastdeploy_stable_diffusion import FastDeployStableDiffusionPipeline
from .pipeline_fastdeploy_stable_diffusion_img2img import (
    FastDeployStableDiffusionImg2ImgPipeline,
)
from .pipeline_fastdeploy_stable_diffusion_inpaint import (
    FastDeployStableDiffusionInpaintPipeline,
)
from .pipeline_fastdeploy_stable_diffusion_inpaint_legacy import (
    FastDeployStableDiffusionInpaintPipelineLegacy,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FastDeployStableDiffusionMegaPipeline(FastDeployStableDiffusionPipeline):
    r"""
    Pipeline for generation using FastDeployStableDiffusion.

    This model inherits from [`FastDeployStableDiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving etc.)

    Args:
        vae_encoder ([`FastDeployRuntimeModel`]):
            Variational Auto-Encoder (VAE) Model to encode images to latent representations.
        vae_decoder ([`FastDeployRuntimeModel`]):
            Variational Auto-Encoder (VAE) Model to decode images from latent representations.
        text_encoder ([`FastDeployRuntimeModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`FastDeployRuntimeModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`PNDMScheduler`], [`EulerDiscreteScheduler`], [`EulerAncestralDiscreteScheduler`]
            or [`DPMSolverMultistepScheduler`].
        safety_checker ([`FastDeployRuntimeModel`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["vae_encoder", "safety_checker", "feature_extractor"]

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
        generator: Optional[paddle.Generator] = None,
        latents: Optional[paddle.Tensor] = None,
        parse_prompt_type: Optional[str] = "lpw",
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        controlnet_cond: Union[paddle.Tensor, PIL.Image.Image] = None,
        controlnet_conditioning_scale: float = 1.0,
        infer_op_dict: Dict[str, str] = None,
    ):

        expected_components = inspect.signature(FastDeployStableDiffusionPipeline.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        temp_pipeline = FastDeployStableDiffusionPipeline(
            **components, requires_safety_checker=self.config.requires_safety_checker
        )
        temp_pipeline._progress_bar_config = self._progress_bar_config
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
            parse_prompt_type=parse_prompt_type,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            controlnet_cond=controlnet_cond,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            infer_op_dict=infer_op_dict,
        )
        return output

    def img2img(
        self,
        prompt: Union[str, List[str]],
        image: Union[paddle.Tensor, PIL.Image.Image],
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[paddle.Generator] = None,
        latents: Optional[paddle.Tensor] = None,
        parse_prompt_type: Optional[str] = "lpw",
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        controlnet_cond: Union[paddle.Tensor, PIL.Image.Image] = None,
        controlnet_conditioning_scale: float = 1.0,
        infer_op_dict: Dict[str, str] = None,
    ):
        expected_components = inspect.signature(FastDeployStableDiffusionImg2ImgPipeline.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        temp_pipeline = FastDeployStableDiffusionImg2ImgPipeline(
            **components, requires_safety_checker=self.config.requires_safety_checker
        )
        temp_pipeline._progress_bar_config = self._progress_bar_config
        output = temp_pipeline(
            prompt=prompt,
            image=image,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            parse_prompt_type=parse_prompt_type,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            controlnet_cond=controlnet_cond,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            infer_op_dict=infer_op_dict,
        )

        return output

    def inpaint_legacy(
        self,
        prompt: Union[str, List[str]],
        image: Union[paddle.Tensor, PIL.Image.Image],
        mask_image: Union[paddle.Tensor, PIL.Image.Image],
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[paddle.Generator] = None,
        latents: Optional[paddle.Tensor] = None,
        parse_prompt_type: Optional[str] = "lpw",
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        controlnet_cond: Union[paddle.Tensor, PIL.Image.Image] = None,
        controlnet_conditioning_scale: float = 1.0,
        infer_op_dict: Dict[str, str] = None,
    ):
        assert (
            self.unet_num_latent_channels == 4
        ), f"Detected `unet_num_latent_channels` is {self.unet_num_latent_channels}, Plese use `inpaint` method."
        expected_components = inspect.signature(
            FastDeployStableDiffusionInpaintPipelineLegacy.__init__
        ).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        temp_pipeline = FastDeployStableDiffusionInpaintPipelineLegacy(
            **components, requires_safety_checker=self.config.requires_safety_checker
        )
        temp_pipeline._progress_bar_config = self._progress_bar_config
        output = temp_pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            parse_prompt_type=parse_prompt_type,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            controlnet_cond=controlnet_cond,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            infer_op_dict=infer_op_dict,
        )

        return output

    def inpaint(
        self,
        prompt: Union[str, List[str]],
        image: Union[paddle.Tensor, PIL.Image.Image],
        mask_image: Union[paddle.Tensor, PIL.Image.Image],
        height=None,
        width=None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[paddle.Generator] = None,
        latents: Optional[paddle.Tensor] = None,
        parse_prompt_type: Optional[str] = "lpw",
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        controlnet_cond: Union[paddle.Tensor, PIL.Image.Image] = None,
        controlnet_conditioning_scale: float = 1.0,
        infer_op_dict: Dict[str, str] = None,
    ):
        assert self.unet_num_latent_channels in [4, 9]
        expected_components = inspect.signature(FastDeployStableDiffusionInpaintPipeline.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        temp_pipeline = FastDeployStableDiffusionInpaintPipeline(
            **components, requires_safety_checker=self.config.requires_safety_checker
        )
        temp_pipeline._progress_bar_config = self._progress_bar_config
        output = temp_pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            parse_prompt_type=parse_prompt_type,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            controlnet_cond=controlnet_cond,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            infer_op_dict=infer_op_dict,
        )

        return output

    def cycle_diffusion(
        self,
        prompt: Union[str, List[str]],
        source_prompt: Union[str, List[str]],
        image: Union[paddle.Tensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[paddle.Tensor] = None,
        source_guidance_scale: Optional[float] = 1,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.1,
        latents: Optional[paddle.Tensor] = None,
        parse_prompt_type: Optional[str] = "lpw",
        max_embeddings_multiples: Optional[int] = 3,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        infer_op_dict: Dict[str, str] = None,
    ):
        expected_components = inspect.signature(FastDeployCycleDiffusionPipeline.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        temp_pipeline = FastDeployCycleDiffusionPipeline(
            **components, requires_safety_checker=self.config.requires_safety_checker
        )
        temp_pipeline._progress_bar_config = self._progress_bar_config
        output = temp_pipeline(
            prompt=prompt,
            source_prompt=source_prompt,
            source_guidance_scale=source_guidance_scale,
            image=image,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            latents=latents,
            parse_prompt_type=parse_prompt_type,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            infer_op_dict=infer_op_dict,
        )

        return output
