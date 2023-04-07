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

import inspect
from typing import Callable, List, Optional, Union

import numpy as np
import paddle
import PIL

from paddlenlp.transformers import CLIPFeatureExtractor, CLIPTokenizer

from ...configuration_utils import FrozenDict
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import DDIMScheduler
from ...utils import PIL_INTERPOLATION, deprecate, logging
from ..fastdeploy_utils import FastDeployRuntimeModel
from . import StableDiffusionPipelineOutput

logger = logging.get_logger(__name__)


# Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess
def preprocess(image):
    if isinstance(image, paddle.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = paddle.to_tensor(image)
    elif isinstance(image[0], paddle.Tensor):
        image = paddle.concat(image, axis=0)
    return image


def posterior_sample(scheduler, latents, timestep, clean_latents, generator, eta):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    if prev_timestep <= 0:
        return clean_latents

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    )

    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    # direction pointing to x_t
    e_t = (latents - alpha_prod_t ** (0.5) * clean_latents) / (1 - alpha_prod_t) ** (0.5)
    dir_xt = (1.0 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * e_t
    noise = std_dev_t * paddle.randn(clean_latents.shape, dtype=clean_latents.dtype, generator=generator)
    prev_latents = alpha_prod_t_prev ** (0.5) * clean_latents + dir_xt + noise

    return prev_latents


def compute_noise(scheduler, prev_latents, latents, timestep, noise_pred, eta):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    )

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

    # 4. Clip "predicted x_0"
    if scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clip(-1, 1)

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred

    noise = (prev_latents - (alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction)) / (
        variance ** (0.5) * eta
    )
    return noise


class FastDeployCycleDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-guided image to image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular xxxx, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae_encoder: FastDeployRuntimeModel,
        vae_decoder: FastDeployRuntimeModel,
        text_encoder: FastDeployRuntimeModel,
        tokenizer: CLIPTokenizer,
        unet: FastDeployRuntimeModel,
        scheduler: DDIMScheduler,
        safety_checker: FastDeployRuntimeModel,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

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
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
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

    # Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[Union[paddle.Tensor, np.ndarray]] = None,
        negative_prompt_embeds: Optional[Union[paddle.Tensor, np.ndarray]] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="np").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not paddle.equal_all(
                paddle.to_tensor(text_input_ids), paddle.to_tensor(untruncated_ids)
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
            prompt_embeds = self.text_encoder(input_ids=text_input_ids.astype(np.int64))
            prompt_embeds = paddle.to_tensor(prompt_embeds[0])

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile([1, num_images_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape([bs_embed * num_images_per_prompt, seq_len, -1])

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )

            negative_prompt_embeds = self.text_encoder(
                input_ids=uncond_input.input_ids.astype(np.int64),
            )
            negative_prompt_embeds = paddle.to_tensor(negative_prompt_embeds[0])

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.tile([1, num_images_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape([batch_size * num_images_per_prompt, seq_len, -1])

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = paddle.concat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.check_inputs
    def check_inputs(
        self, prompt, strength, callback_steps, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

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

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pd")
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.cast(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    # Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        latents_shape = latents.shape
        vae_output_shape = [latents_shape[0], 3, latents_shape[2] * 8, latents_shape[3] * 8]
        images_vae = paddle.zeros(vae_output_shape, dtype="float32")

        vae_input_name = self.vae_decoder.model.get_input_info(0).name
        vae_output_name = self.vae_decoder.model.get_output_info(0).name

        self.vae_decoder.zero_copy_infer(
            prebinded_inputs={vae_input_name: latents},
            prebinded_outputs={vae_output_name: images_vae},
            share_with_raw_ptr=True,
        )

        images_vae = paddle.clip(images_vae / 2 + 0.5, 0, 1)
        images = images_vae.transpose([0, 2, 3, 1])
        return images.numpy()

    # Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, generator=None):
        image = image.cast(dtype)

        batch_size = image.shape[0]
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image = image.astype(dtype)
        init_latents = self.vae_encoder(sample=image)[0]
        init_latents = 0.18215 * init_latents
        init_latents = paddle.to_tensor(init_latents)

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = paddle.concat([init_latents] * additional_image_per_prompt * num_images_per_prompt, axis=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = paddle.concat([init_latents] * num_images_per_prompt, axis=0)

        # add noise to latents using the timestep
        shape = init_latents.shape
        if isinstance(generator, list):
            shape = [
                1,
            ] + shape[1:]
            noise = [paddle.randn(shape, generator=generator[i], dtype=dtype) for i in range(batch_size)]
            noise = paddle.concat(noise, axis=0)
        else:
            noise = paddle.randn(shape, generator=generator, dtype=dtype)

        # get latents
        clean_latents = init_latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents, clean_latents

    def __call__(
        self,
        prompt: Union[str, List[str]],
        source_prompt: Union[str, List[str]],
        image: Union[paddle.Tensor, PIL.Image.Image, np.ndarray] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[paddle.Tensor, np.ndarray]] = None,
        source_guidance_scale: Optional[float] = 1,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.1,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        prompt_embeds: Optional[Union[paddle.Tensor, np.ndarray]] = None,
        negative_prompt_embeds: Optional[Union[paddle.Tensor, np.ndarray]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The target prompt or prompts to guide the image generation.
            source_prompt (`str` or `List[str]`):
                The source prompt or prompts describe the input image.
            image (`paddle.Tensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1.
                `image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The negative prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            source_guidance_scale (`float`, *optional*, defaults to 1):
                Guidance scale for the source prompt. This is useful to control the amount of influence the source
                prompt for encoding.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.1):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`paddle.Generator`, *optional*):
                One or a list of paddle generator(s) to make generation deterministic.
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

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 1. Check inputs
        self.check_inputs(prompt, strength, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode target prompt and source prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        source_prompt_embeds = self._encode_prompt(
            source_prompt, num_images_per_prompt, do_classifier_free_guidance, None
        )

        # 4. Preprocess image
        image = preprocess(image)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        latent_timestep = timesteps[:1].tile([batch_size * num_images_per_prompt])

        # 6. Prepare latent variables
        latents, clean_latents = self.prepare_latents(
            image, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, generator
        )
        source_latents = latents

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        generator = extra_step_kwargs.pop("generator", None)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # unet_output_name = self.unet.model.get_output_info(0).name
        # unet_input_names = [self.unet.model.get_input_info(i).name for i in range(self.unet.model.num_inputs())]
        # height, width = image.shape[-2:]
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # concat_noise_pred = paddle.zeros(
                #     [4 * batch_size * num_images_per_prompt, 4, height // 8, width // 8], dtype="float32"
                # )
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat([latents] * 2)
                source_latent_model_input = paddle.concat([source_latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                source_latent_model_input = self.scheduler.scale_model_input(source_latent_model_input, t)

                # predict the noise residual
                concat_latent_model_input = paddle.stack(
                    [
                        source_latent_model_input[0],
                        latent_model_input[0],
                        source_latent_model_input[1],
                        latent_model_input[1],
                    ],
                    axis=0,
                )
                concat_prompt_embeds = paddle.stack(
                    [
                        source_prompt_embeds[0],
                        prompt_embeds[0],
                        source_prompt_embeds[1],
                        prompt_embeds[1],
                    ],
                    axis=0,
                )

                # predict the noise residual
                # TODO(zhoushunjie): Use zero copy infer in the future
                # self.unet.zero_copy_infer(
                #     prebinded_inputs={
                #         unet_input_names[0]: concat_latent_model_input,
                #         unet_input_names[1]: t,
                #         unet_input_names[2]: concat_prompt_embeds,
                #     },
                #     prebinded_outputs={unet_output_name: concat_noise_pred},
                #     share_with_raw_ptr=True,
                # )
                concat_noise_pred = self.unet(
                    sample=concat_latent_model_input.numpy(),
                    timestep=t.cast("float32").numpy(),
                    encoder_hidden_states=concat_prompt_embeds.numpy(),
                )[0]
                concat_noise_pred = paddle.to_tensor(concat_noise_pred)

                # perform guidance
                (
                    source_noise_pred_uncond,
                    noise_pred_uncond,
                    source_noise_pred_text,
                    noise_pred_text,
                ) = concat_noise_pred.chunk(4, axis=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                source_noise_pred = source_noise_pred_uncond + source_guidance_scale * (
                    source_noise_pred_text - source_noise_pred_uncond
                )

                # Sample source_latents from the posterior distribution.
                prev_source_latents = posterior_sample(
                    self.scheduler, source_latents, t, clean_latents, generator=generator, **extra_step_kwargs
                )
                # Compute noise.
                noise = compute_noise(
                    self.scheduler, prev_source_latents, source_latents, t, source_noise_pred, **extra_step_kwargs
                )
                source_latents = prev_source_latents

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, variance_noise=noise, **extra_step_kwargs
                ).prev_sample
                if i == len(timesteps) - 1:
                    # sync for accuracy it/s measure
                    paddle.device.cuda.synchronize()
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 9. Post-processing
        image = self.decode_latents(latents)

        # 10. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, prompt_embeds.dtype)

        # 11. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
