# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Callable, List, Optional, Union

import numpy as np
import paddle
import PIL

from paddlenlp.transformers import CLIPImageProcessor, CLIPTokenizer

from ...pipeline_utils import DiffusionPipeline
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import PIL_INTERPOLATION, logging
from ..fastdeploy_utils import FastDeployDiffusionPipelineMixin, FastDeployRuntimeModel
from . import StableDiffusionPipelineOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def prepare_mask_and_masked_image(image, mask, height=None, width=None, return_image: bool = False):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``paddle.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``paddle.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``paddle.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, paddle.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``paddle.Tensor`` or a ``batch x channels x height x width`` ``paddle.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``paddle.Tensor`` or a ``batch x 1 x height x width`` ``paddle.Tensor``.


    Raises:
        ValueError: ``paddle.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``paddle.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``paddle.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[paddle.Tensor]: The pair (mask, masked_image) as ``paddle.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """

    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, paddle.Tensor):
        if not isinstance(mask, paddle.Tensor):
            raise TypeError(f"`image` is a paddle.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.cast(dtype=paddle.float32)
    elif isinstance(mask, paddle.Tensor):
        raise TypeError(f"`mask` is a paddle.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            if width is None or height is None:
                w, h = image[0].size
            else:
                w, h = width, height
            w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
            image = [i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = paddle.to_tensor(image, dtype=paddle.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            if width is None or height is None:
                w, h = mask[0].size
            else:
                w, h = width, height
            w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
            mask = [i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = paddle.to_tensor(mask)

    masked_image = image * (mask < 0.5)

    # n.b. ensure backwards compatibility as old function does not return image
    if return_image:
        return mask, masked_image, image

    return mask, masked_image


class FastDeployStableDiffusionInpaintPipeline(DiffusionPipeline, FastDeployDiffusionPipelineMixin):
    r"""
    Pipeline for text-guided image inpainting using Stable Diffusion.

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
    _optional_components = ["safety_checker", "feature_extractor"]

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

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[paddle.Tensor, PIL.Image.Image] = None,
        mask_image: Union[paddle.Tensor, PIL.Image.Image] = None,
        height: int = None,
        width: int = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        add_predicted_noise: Optional[bool] = False,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        controlnet_cond: Union[paddle.Tensor, PIL.Image.Image] = None,
        controlnet_conditioning_scale: float = 1.0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`paddle.Tensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. This is the image whose masked region will be inpainted.
            mask_image (`paddle.Tensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
                PIL image, it will be converted to a single channel (luminance) before use. If mask is a tensor, the
                expected shape should be either `(B, H, W, C)` or `(B, C, H, W)`, where C is 1 or 3.
            height (`int`, *optional*, defaults to None):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to None):
                The width in pixels of the generated image.
            strength (`float`, *optional*, defaults to 1.0):
                Conceptually, indicates how much to inpaint the masked area. Must be between 0 and 1. When `strength`
                is 1, the denoising process will be run on the masked area for the full number of iterations specified
                in `num_inference_steps`. `image` will be used as a reference for the masked area, adding more noise to
                that region the larger the `strength`. If `strength` is 0, no inpainting will occur.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The reference number of denoising steps. More denoising steps usually lead to a higher quality image at
                the expense of slower inference. This parameter will be modulated by `strength`, as explained above.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. Ignored when not using guidance (i.e., ignored if `guidance_scale`
                is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            add_predicted_noise (`bool`, *optional*, defaults to False):
                Use predicted noise instead of random noise when constructing noisy versions of the original image in
                the reverse diffusion process
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`paddle.Generator`, *optional*):
                One or a list of [paddle generator(s)] to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noise tensor, sampled from a Gaussian distribution, to be used as inputs for image
                generation. If not provided, a noise tensor will ge generated by sampling using the supplied random
                `generator`.
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
        # 0. Preprocess mask and image
        mask, masked_image, init_image = prepare_mask_and_masked_image(
            image,
            mask_image,
            height,
            width,
            return_image=True,
        )
        height, width = init_image.shape[-2:]

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            strength,
        )

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
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].tile([batch_size * num_images_per_prompt])
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Prepare latent variables
        num_channels_latents = self.vae_decoder_num_latent_channels
        num_channels_unet = self.unet_num_latent_channels
        is_legacy = return_image_latents = num_channels_unet == 4

        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            height,
            width,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 6. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            do_classifier_free_guidance,
            return_masked_image_latents=True,
        )

        # 7. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != num_channels_unet:
                raise ValueError(
                    f"Incorrect configuration settings! Received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(f"The unet should have either 4 or 9 input channels, not {num_channels_unet}.")
        # do_controlnet
        do_controlnet = controlnet_cond is not None and num_channels_unet == 4
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

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        unet_output_name = self.unet.model.get_output_info(0).name
        unet_input_names = [self.unet.model.get_input_info(i).name for i in range(self.unet.model.num_inputs())]

        if do_classifier_free_guidance:
            init_mask = mask[: mask.shape[0] // 2]
        else:
            init_mask = mask

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                noise_pred_unet = paddle.zeros_like(latent_model_input)

                if not is_legacy:
                    # concat latents, mask, masked_image_latents in the channel dimension
                    latent_model_input = paddle.concat([latent_model_input, mask, masked_image_latents], axis=1)

                unet_inputs = {
                    unet_input_names[0]: latent_model_input,
                    unet_input_names[1]: t,
                    unet_input_names[2]: prompt_embeds,
                }
                if do_controlnet:
                    unet_inputs[unet_input_names[3]] = control_image
                    unet_inputs[unet_input_names[4]] = control_conditioning_scale

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
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if is_legacy:
                    if i < len(timesteps) - 1:
                        # masking
                        if add_predicted_noise:
                            init_latents_proper = self.scheduler.add_noise(image_latents, noise_pred_uncond, t)
                        else:
                            init_latents_proper = self.scheduler.add_noise(image_latents, noise, t)
                    else:
                        init_latents_proper = image_latents
                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

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
