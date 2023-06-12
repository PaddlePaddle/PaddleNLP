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

from typing import Any, Callable, List, Optional, Union

import numpy as np
import paddle
import PIL

from ...pipeline_utils import DiffusionPipeline
from ...schedulers import DDPMScheduler
from ...utils import logging
from ..fastdeploy_utils import FastDeployDiffusionPipelineMixin, FastDeployRuntimeModel
from ..pipeline_utils import ImagePipelineOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FastDeployStableDiffusionUpscalePipeline(DiffusionPipeline, FastDeployDiffusionPipelineMixin):
    def __init__(
        self,
        vae: FastDeployRuntimeModel,
        text_encoder: FastDeployRuntimeModel,
        tokenizer: Any,
        unet: FastDeployRuntimeModel,
        low_res_scheduler: DDPMScheduler,
        scheduler: Any,
        max_noise_level: int = 350,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            watermarker=None,
            max_noise_level=max_noise_level,
        )
        self.post_init(vae_scaling_factor=0.08333)

    def check_inputs(self, prompt, image, noise_level, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if (
            not isinstance(image, paddle.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                f"`image` has to be of type `paddle.Tensor`, `PIL.Image.Image` or `list` but is {type(image)}"
            )

        # verify batch size of prompt and image are same if image is a list or tensor
        if isinstance(image, list) or isinstance(image, paddle.Tensor):
            if isinstance(prompt, str):
                batch_size = 1
            else:
                batch_size = len(prompt)
            if isinstance(image, list):
                image_batch_size = len(image)
            else:
                image_batch_size = image.shape[0]
            if batch_size != image_batch_size:
                raise ValueError(
                    f"`prompt` has batch size {batch_size} and `image` has batch size {image_batch_size}."
                    " Please make sure that passed `prompt` matches the batch size of `image`."
                )

        # check noise level
        if noise_level > self.config.max_noise_level:
            raise ValueError(f"`noise_level` has to be <= {self.config.max_noise_level} but is {noise_level}")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[paddle.Tensor, PIL.Image.Image, List[PIL.Image.Image]],
        num_inference_steps: int = 75,
        guidance_scale: float = 9.0,
        noise_level: int = 20,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`np.ndarray` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            noise_level TODO
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`np.random.RandomState`, *optional*):
                A np.random.RandomState to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`np.ndarray`, *optional*):
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
                called with the following arguments: `callback(step: int, timestep: int, latents: np.ndarray)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # 1. Check inputs
        self.check_inputs(prompt, image, noise_level, callback_steps)

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

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps)

        # 5. Add noise to image
        noise_level = paddle.to_tensor([noise_level], dtype="int64")
        noise = paddle.randn(image.shape, generator=generator, dtype=text_embeddings.dtype)
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)

        batch_multiplier = 2 if do_classifier_free_guidance else 1
        image = paddle.concat([image] * batch_multiplier * num_images_per_prompt)
        noise_level = paddle.concat([noise_level] * image.shape[0])

        # 6. Prepare latent variables
        height, width = image.shape[2:]
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            height,
            width,
            generator,
            latents,
        )
        NUM_UNET_INPUT_CHANNELS = self.unet_num_latent_channels
        NUM_LATENT_CHANNELS = self.vae_decoder_num_latent_channels

        # 7. Check that sizes of image and latents match
        num_channels_image = image.shape[1]
        if NUM_LATENT_CHANNELS + num_channels_image != NUM_UNET_INPUT_CHANNELS:
            raise ValueError(
                "Incorrect configuration settings! The config of `pipeline.unet` expects"
                f" {NUM_UNET_INPUT_CHANNELS} but received `num_channels_latents`: {NUM_LATENT_CHANNELS} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {NUM_LATENT_CHANNELS+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        unet_output_name = self.unet.model.get_output_info(0).name
        unet_input_names = [self.unet.model.get_input_info(i).name for i in range(self.unet.model.num_inputs())]
        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                noise_pred_unet = paddle.zeros_like(latent_model_input)
                # concat latents, image in the channel dimension
                latent_model_input = paddle.concat([latent_model_input, image], axis=1)

                unet_inputs = {
                    unet_input_names[0]: latent_model_input,
                    unet_input_names[1]: t,
                    unet_input_names[2]: prompt_embeds,
                }
                # predict the noise residual
                self.unet.zero_copy_infer(
                    prebinded_inputs=unet_inputs,
                    prebinded_outputs={unet_output_name: noise_pred_unet},
                    share_with_raw_ptr=True,
                )
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred_unet.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_unet

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                    if i == len(timesteps) - 1:
                        # sync for accuracy it/s measure
                        paddle.device.cuda.synchronize()

        if not output_type == "latent":
            image = self._decode_vae_latents(latents / self.vae_scaling_factor)
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(
            images=image,
        )
