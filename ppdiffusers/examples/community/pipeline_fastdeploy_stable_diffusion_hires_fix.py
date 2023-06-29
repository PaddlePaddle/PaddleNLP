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

import copy
from typing import Callable, Dict, List, Optional, Union

import paddle
import paddle.nn.functional as F
import PIL

from paddlenlp.transformers import CLIPImageProcessor, CLIPTokenizer
from ppdiffusers import DiffusionPipeline
from ppdiffusers.pipelines.fastdeploy_utils import (
    FastDeployDiffusionPipelineMixin,
    FastDeployRuntimeModel,
)
from ppdiffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from ppdiffusers.schedulers import KarrasDiffusionSchedulers
from ppdiffusers.utils import logging, randn_tensor

logger = logging.get_logger(__name__)


class FastStableDiffusionHiresFixPipeline(DiffusionPipeline, FastDeployDiffusionPipelineMixin):
    r"""
    Pipeline for text-to-image generation with high resolution fixing(hires.fix) based on Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
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
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["vae_encoder", "safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae_encoder: FastDeployRuntimeModel,
        vae_decoder: FastDeployRuntimeModel,
        text_encoder: FastDeployRuntimeModel,
        tokenizer: CLIPTokenizer,
        unet: FastDeployRuntimeModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: FastDeployRuntimeModel,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = False,
    ):
        super().__init__()
        if safety_checker is None and requires_safety_checker:
            logger.warning(
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
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.post_init()

    def get_timesteps(self, denoising_steps, denoising_strength):
        steps = int(denoising_steps / min(denoising_strength, 0.999))
        self.scheduler.set_timesteps(steps)

        t_start = max(steps - denoising_steps, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        if hasattr(self.scheduler, "step_index_offset"):
            self.scheduler.step_index_offset = t_start * self.scheduler.order

        return timesteps.cast("float32"), denoising_steps

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        hr_scale,
        hr_resize_height,
        hr_resize_width,
        denoising_strength,
        latent_scale_mode,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if hr_scale < 0:
            raise ValueError("hr_scale shoule be greater that 0, but acceived {hr_scale}")

        if hr_resize_height % 8 != 0 or hr_resize_width % 8 != 0:
            raise ValueError(
                f"`hr_resize_height` and `hr_resize_width` have to be divisible by 8 but are {hr_resize_height} and {hr_resize_width}."
            )

        if denoising_strength > 1 or denoising_strength < 0:
            raise ValueError(f"denoising_strength should be set between 0 and 1., but acceived {denoising_strength}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if latent_scale_mode not in ["nearest", "bilinear", "bicubic", "area"]:
            raise ValueError(
                f"Only such interpolate method supported for latent_scale_mode in [nearest, bilinear, bicubic, area]. but acceived {latent_scale_mode}."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def get_upscaled_width_and_height(self, width, height, hr_scale=2, hr_resize_width=0, hr_resize_height=0):
        if hr_resize_width == 0 and hr_resize_height == 0:
            hr_upscale_to_width = int(width * hr_scale)
            hr_upscale_to_height = int(height * hr_scale)
        else:
            if hr_resize_height == 0:
                hr_upscale_to_width = hr_resize_width
                hr_upscale_to_height = hr_resize_width * height // width
            elif hr_resize_width == 0:
                hr_upscale_to_width = hr_resize_height * width // height
                hr_upscale_to_height = hr_resize_height
            else:
                src_ratio = width / height
                dst_ratio = hr_resize_width / hr_resize_height

                if src_ratio < dst_ratio:
                    hr_upscale_to_width = hr_resize_width
                    hr_upscale_to_height = hr_resize_width * height // width
                else:
                    hr_upscale_to_width = hr_resize_height * width // height
                    hr_upscale_to_height = hr_resize_height

        return hr_upscale_to_width, hr_upscale_to_height

    @paddle.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 40,
        hires_ratio: Optional[float] = 0.5,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        parse_prompt_type: Optional[str] = "lpw",
        max_embeddings_multiples: Optional[int] = 3,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        enable_hr: Optional[bool] = True,
        hr_scale: Optional[float] = 2.0,
        hr_resize_width: Optional[int] = 0,
        hr_resize_height: Optional[int] = 0,
        denoising_strength: Optional[float] = 0.7,
        latent_scale_mode: Optional[str] = "nearest",
        controlnet_cond: Union[paddle.Tensor, PIL.Image.Image] = None,
        controlnet_conditioning_scale: float = 1.0,
        infer_op_dict: Dict[str, str] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 40):
                The number of denoising steps, equal to sample_steps and hr_steps. samples_steps means the initial
                denoising steps, and hr_steps means hires denoising steps. More denoising steps usually lead to a
                higher quality image at the expense of slower inference.
            hires_ratio (`float`, *optional*, defaults to 0.5):
                The step proportion of hires.fix, that means hr_steps = int(num_inference_steps * hires_ratio).
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
                One or a list of paddle generator(s) to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
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
            hr_steps (`int`, *optional*, defaults to 30):
                The number of second denoising steps about high resolution fixing.
            hr_scale (`float`, *optional*, defaults to 2.0):
                The upscaler to expand the width and height of image. if set 2.0, it means that expand width and height of a image to width*2.0 and height*2.0.
            hr_resize_width (`int`, *optional*, defaults to 0):
                It enable users to specify the upscaled width mannually. if hr_resize_width!=0, program will use it to compute scaled width and height instead of hr_scale.
            hr_resize_height (`int`, *optional*, defaults to 0):
                It enable users to specify the upscaled height mannually. if hr_resize_height!=0, program will use it to compute scaled width and height instead of hr_scale.
            denoising_strength (`float`, *optional*, defaults to 0.7):
                The denoising strength applying on hires.fix steps. It take a value between 0 and 1.
            latent_scale_mode (`str`, *optional*, defaults to nearest):
                The interpolate method applying upscale initial images, you can set it in [nearest, bilinear, bicubic, area].

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or 512
        width = width or 512

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            hr_scale,
            hr_resize_height,
            hr_resize_width,
            denoising_strength,
            latent_scale_mode,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )
        infer_op_dict = self.prepare_infer_op_dict(infer_op_dict)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # do_controlnet
        do_controlnet = controlnet_cond is not None
        if do_controlnet:
            control_image, control_conditioning_scale = self.prepare_controlnet_cond(
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                width=width,
                height=height,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            parse_prompt_type=parse_prompt_type,
            max_embeddings_multiples=max_embeddings_multiples,
            infer_op=infer_op_dict.get("text_encoder", None),
        )

        # 4. Prepare timesteps
        if enable_hr:
            hr_steps = int(num_inference_steps * hires_ratio)
            sample_steps = num_inference_steps - hr_steps
        else:
            hr_steps = 0
            sample_steps = num_inference_steps

        self.scheduler.set_timesteps(sample_steps)
        timesteps = self.scheduler.timesteps.cast("float32")

        # 5. Prepare latent variables
        if generator is None:
            generator_state = paddle.get_cuda_rng_state()
            paddle.Generator().states_["initial_generator"] = copy.deepcopy(generator_state)
        else:
            paddle.Generator().states_["initial_generator"] = copy.deepcopy(paddle.Generator().states_[generator])

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
        num_warmup_steps = len(timesteps) - sample_steps * self.scheduler.order
        is_scheduler_support_step_index = self.is_scheduler_support_step_index()
        with self.progress_bar(total=sample_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
                if is_scheduler_support_step_index:
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t, step_index=i)
                else:
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                unet_inputs = dict(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    infer_op=infer_op_dict.get("unet", None),
                    output_shape=latent_model_input.shape,
                )
                if do_controlnet:
                    unet_inputs["controlnet_cond"] = control_image
                    unet_inputs["controlnet_conditioning_scale"] = control_conditioning_scale
                # predict the noise residual
                noise_pred_unet = self.unet(**unet_inputs)[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred_unet.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_unet

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

        # start to apply hires.fix on initial latents
        if enable_hr:
            # 8. determine the upscaled width and height for upscaled images
            truncate_width = 0
            truncate_height = 0
            hr_upscale_to_width, hr_upscale_to_height = self.get_upscaled_width_and_height(
                width, height, hr_scale=hr_scale, hr_resize_width=hr_resize_width, hr_resize_height=hr_resize_height
            )
            if hr_resize_width != 0 and hr_resize_height != 0:
                truncate_width = (hr_upscale_to_width - hr_resize_width) // self.vae_scale_factor
                truncate_height = (hr_upscale_to_height - hr_resize_height) // self.vae_scale_factor

            # 9. special case: do nothing if upscaling is not nesscessary
            if hr_upscale_to_width == width and hr_upscale_to_height == height:
                enable_hr = False
                denoising_strength = None

        if enable_hr:
            if do_controlnet:
                control_image, control_conditioning_scale = self.prepare_controlnet_cond(
                    controlnet_cond=controlnet_cond,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    width=hr_upscale_to_width,
                    height=hr_upscale_to_height,
                    batch_size=batch_size,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )

            # 10. prepare init latents
            timesteps, hr_steps = self.get_timesteps(hr_steps, denoising_strength)
            init_timestep = timesteps[:1].tile([latents.shape[0]])

            latents = F.interpolate(
                latents,
                size=(
                    hr_upscale_to_height // self.vae_scale_factor,
                    hr_upscale_to_width // self.vae_scale_factor,
                ),
                mode=latent_scale_mode,
            )
            latents = latents[
                :,
                :,
                truncate_height // 2 : latents.shape[2] - (truncate_height + 1) // 2,
                truncate_width // 2 : latents.shape[3] - (truncate_width + 1) // 2,
            ]

            noise = randn_tensor(latents.shape, dtype=latents.dtype, generator="initial_generator")
            latents = self.scheduler.add_noise(latents, noise, init_timestep)

            # 11. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs("initial_generator", eta)

            # 12. denoising on hires.fix steps
            num_warmup_steps = len(timesteps) - hr_steps * self.scheduler.order
            with self.progress_bar(total=hr_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
                    if is_scheduler_support_step_index:
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t, step_index=i)
                    else:
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    unet_inputs = dict(
                        sample=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        infer_op=infer_op_dict.get("unet", None),
                        output_shape=latent_model_input.shape,
                    )
                    if do_controlnet:
                        unet_inputs["controlnet_cond"] = control_image
                        unet_inputs["controlnet_conditioning_scale"] = control_conditioning_scale
                    # predict the noise residual
                    noise_pred_unet = self.unet(**unet_inputs)[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred_unet.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_unet

                    # compute the previous noisy sample x_t -> x_t-1
                    if is_scheduler_support_step_index:
                        scheduler_output = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            step_index=i,
                            return_pred_original_sample=False,
                            **extra_step_kwargs,
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

        if not output_type == "latent":
            image = self._decode_vae_latents(
                latents / self.vae_scaling_factor, infer_op=infer_op_dict.get("vae_decoder", None)
            )
            image, has_nsfw_concept = self.run_safety_checker(image)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
