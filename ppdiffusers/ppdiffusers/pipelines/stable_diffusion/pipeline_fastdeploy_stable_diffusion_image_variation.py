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

from typing import Callable, Dict, List, Optional, Union

import paddle
import PIL

from paddlenlp.transformers import CLIPImageProcessor
from ppdiffusers.pipelines.fastdeploy_utils import (
    FastDeployDiffusionPipelineMixin,
    FastDeployRuntimeModel,
)

from ...schedulers import KarrasDiffusionSchedulers
from ...utils import logging
from ..pipeline_utils import DiffusionPipeline
from . import StableDiffusionPipelineOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FastDeployStableDiffusionImageVariationPipeline(DiffusionPipeline, FastDeployDiffusionPipelineMixin):
    r"""
    Pipeline to generate variations from an input image using Stable Diffusion.

    Args:
        vae_encoder ([`FastDeployRuntimeModel`]):
            Variational Auto-Encoder (VAE) Model to encode images to latent representations.
        vae_decoder ([`FastDeployRuntimeModel`]):
            Variational Auto-Encoder (VAE) Model to decode images from latent representations.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder. Stable Diffusion Image Variation uses the vision portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        unet ([`FastDeployRuntimeModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`FastDeployRuntimeModel`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    # TODO: feature_extractor is required to encode images (if they are in PIL format),
    # we should give a descriptive message if the pipeline doesn't have one.
    _optional_components = ["safety_checker"]

    def __init__(
        self,
        vae_encoder: FastDeployRuntimeModel,
        vae_decoder: FastDeployRuntimeModel,
        image_encoder: FastDeployRuntimeModel,
        unet: FastDeployRuntimeModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: FastDeployRuntimeModel,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = False,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warn(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. PaddleNLP team, diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                f"Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.post_init()

    def _encode_image(self, image, num_images_per_prompt, do_classifier_free_guidance, infer_op_dict):
        if not isinstance(image, paddle.Tensor):
            image = self.feature_extractor(images=image, return_tensors="pd").pixel_values

        image_encoder_inputs = dict(
            pixel_values=image,
            infer_op=infer_op_dict.get("image_encoder", None),
            output_shape=[image.shape[0], 768],
        )
        image_embeddings = self.image_encoder(**image_encoder_inputs)[0]
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.tile([1, num_images_per_prompt, 1])
        image_embeddings = image_embeddings.reshape([bs_embed * num_images_per_prompt, seq_len, -1])

        if do_classifier_free_guidance:
            negative_prompt_embeds = paddle.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = paddle.concat([negative_prompt_embeds, image_embeddings])

        return image_embeddings

    def check_inputs(self, image, height, width, callback_steps):
        if (
            not isinstance(image, paddle.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `paddle.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    @paddle.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], paddle.Tensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        infer_op_dict: Dict[str, str] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `paddle.Tensor`):
                The image or images to guide the image generation. If you provide a tensor, it needs to comply with the
                configuration of
                [this](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json)
                `CLIPImageProcessor`
            height (`int`, *optional*, defaults to None):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to None):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`paddle.Generator`, *optional*):
                One or a list of paddle generator(s) to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: paddle.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            infer_op_dict: The parameter infer_op_dict is a dictionary that maps module to it's inference op. The purpose of this dictionary is to store inferred operations or operations that have been deduced or determined during some process. The op are choosen from the following: 'None', 'zero_copy_infer', 'raw'.


        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        infer_op_dict = self.prepare_infer_op_dict(infer_op_dict)

        # 0. Default height and width to unet
        height = height or 512
        width = width or 512

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width, callback_steps)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input image
        image_embeddings = self._encode_image(image, num_images_per_prompt, do_classifier_free_guidance, infer_op_dict)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps)

        # 5. Prepare latent variables
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            height,
            width,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_scheduler_support_step_index = self.is_scheduler_support_step_index()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
                if is_scheduler_support_step_index:
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t, step_index=i)
                else:
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                unet_inputs = dict(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=image_embeddings,
                    infer_op=infer_op_dict.get("unet", None),
                    output_shape=latent_model_input.shape,
                )
                noise_pred = self.unet(**unet_inputs)[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if is_scheduler_support_step_index:
                    scheduler_output = self.scheduler.step(
                        noise_pred, t, latents, step_index=i, return_pred_original_sample=False, **extra_step_kwargs
                    )
                else:
                    scheduler_output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents = scheduler_output.prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                    if i == len(timesteps) - 1:
                        # sync for accuracy it/s measure
                        paddle.device.cuda.synchronize()

        # 8. Post-processing
        image = self._decode_vae_latents(
            latents / self.vae_scaling_factor, infer_op=infer_op_dict.get("vae_decoder", None)
        )

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image)

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
