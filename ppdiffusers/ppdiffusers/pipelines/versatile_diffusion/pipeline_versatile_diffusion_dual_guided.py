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
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import paddle
import PIL

from paddlenlp.transformers import (
    CLIPFeatureExtractor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ...models import (
    AutoencoderKL,
    DualTransformer2DModel,
    Transformer2DModel,
    UNet2DConditionModel,
)
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import logging, randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from .modeling_text_unet import UNetFlatConditionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VersatileDiffusionDualGuidedPipeline(DiffusionPipeline):
    r"""
    Pipeline for dual-guided generation using Versatile Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder. Versatile Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen vision-encoder. Versatile Diffusion uses the vision portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModelWithProjection), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        image_unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        text_unet ([`UNetFlatConditionModel`]): xxx.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        image_feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    tokenizer: CLIPTokenizer
    image_feature_extractor: CLIPFeatureExtractor
    text_encoder: CLIPTextModelWithProjection
    image_encoder: CLIPVisionModelWithProjection
    image_unet: UNet2DConditionModel
    text_unet: UNetFlatConditionModel
    vae: AutoencoderKL
    scheduler: KarrasDiffusionSchedulers

    _optional_components = ["text_unet"]

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        image_feature_extractor: CLIPFeatureExtractor,
        text_encoder: CLIPTextModelWithProjection,
        image_encoder: CLIPVisionModelWithProjection,
        image_unet: UNet2DConditionModel,
        text_unet: UNetFlatConditionModel,
        vae: AutoencoderKL,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer,
            image_feature_extractor=image_feature_extractor,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            image_unet=image_unet,
            text_unet=text_unet,
            vae=vae,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        if self.text_unet is not None and (
            "dual_cross_attention" not in self.image_unet.config or not self.image_unet.config.dual_cross_attention
        ):
            # if loading from a universal checkpoint rather than a saved dual-guided pipeline
            self._convert_to_dual_attention()

    def remove_unused_weights(self):
        self.register_modules(text_unet=None)

    def _convert_to_dual_attention(self):
        """
        Replace image_unet's `Transformer2DModel` blocks with `DualTransformer2DModel` that contains transformer blocks
        from both `image_unet` and `text_unet`
        """
        for name, module in self.image_unet.named_sublayers(include_self=True):
            if isinstance(module, Transformer2DModel):
                parent_name, index = name.rsplit(".", 1)
                index = int(index)

                image_transformer = self.image_unet.get_sublayer(parent_name)[index]
                text_transformer = self.text_unet.get_sublayer(parent_name)[index]

                config = image_transformer.config
                dual_transformer = DualTransformer2DModel(
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                    in_channels=config.in_channels,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                    norm_num_groups=config.norm_num_groups,
                    cross_attention_dim=config.cross_attention_dim,
                    attention_bias=config.attention_bias,
                    sample_size=config.sample_size,
                    num_vector_embeds=config.num_vector_embeds,
                    activation_fn=config.activation_fn,
                    num_embeds_ada_norm=config.num_embeds_ada_norm,
                )
                dual_transformer.transformers[0] = image_transformer
                dual_transformer.transformers[1] = text_transformer

                self.image_unet.get_sublayer(parent_name)[index] = dual_transformer
                self.image_unet.register_to_config(dual_cross_attention=True)

    def _revert_dual_attention(self):
        """
        Revert the image_unet `DualTransformer2DModel` blocks back to `Transformer2DModel` with image_unet weights Call
        this function if you reuse `image_unet` in another pipeline, e.g. `VersatileDiffusionPipeline`
        """
        for name, module in self.image_unet.named_sublayers(include_self=True):
            if isinstance(module, DualTransformer2DModel):
                parent_name, index = name.rsplit(".", 1)
                index = int(index)
                self.image_unet.get_sublayer(parent_name)[index] = module.transformers[0]
        self.image_unet.register_to_config(dual_cross_attention=False)

    def _encode_text_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
        """

        def normalize_embeddings(encoder_output):
            embeds = paddle.matmul(encoder_output.last_hidden_state, self.text_encoder.text_projection)
            embeds_pooled = encoder_output.text_embeds
            embeds = embeds / paddle.norm(embeds_pooled.unsqueeze(1), axis=-1, keepdim=True)
            return embeds

        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pd",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pd").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not paddle.equal_all(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )
        prompt_embeds = normalize_embeddings(prompt_embeds)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.tile([1, num_images_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape([bs_embed * num_images_per_prompt, seq_len, -1])

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size
            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pd",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids,
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = normalize_embeddings(negative_prompt_embeds)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.tile([1, num_images_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape([batch_size * num_images_per_prompt, seq_len, -1])

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = paddle.concat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def _encode_image_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance):
        r"""
        Encodes the prompt into vision encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
        """

        def normalize_embeddings(encoder_output):
            embeds = self.image_encoder.vision_model.ln_post(encoder_output.last_hidden_state)
            embeds = paddle.matmul(embeds, self.image_encoder.vision_projection)
            embeds_pooled = embeds[:, 0:1]
            embeds = embeds / paddle.norm(embeds_pooled, axis=-1, keepdim=True)
            return embeds

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings
        image_input = self.image_feature_extractor(images=prompt, return_tensors="pd")
        pixel_values = image_input.pixel_values.cast(self.image_encoder.dtype)
        image_embeddings = self.image_encoder(pixel_values)
        image_embeddings = normalize_embeddings(image_embeddings)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.tile([1, num_images_per_prompt, 1])
        image_embeddings = image_embeddings.reshape([bs_embed * num_images_per_prompt, seq_len, -1])

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_images = [np.zeros((512, 512, 3)) + 0.5] * batch_size
            uncond_images = self.image_feature_extractor(images=uncond_images, return_tensors="pd")
            pixel_values = uncond_images.pixel_values.cast(self.image_encoder.dtype)
            negative_prompt_embeds = self.image_encoder(pixel_values)
            negative_prompt_embeds = normalize_embeddings(negative_prompt_embeds)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.tile([1, num_images_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape([batch_size * num_images_per_prompt, seq_len, -1])

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and conditional embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = paddle.concat([negative_prompt_embeds, image_embeddings])

        return image_embeddings

    # Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.transpose([0, 2, 3, 1]).cast("float32").numpy()
        return image

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

    def check_inputs(self, prompt, image, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, PIL.Image.Image) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` `PIL.Image` or `list` but is {type(prompt)}")
        if not isinstance(image, str) and not isinstance(image, PIL.Image.Image) and not isinstance(image, list):
            raise ValueError(f"`image` has to be of type `str` `PIL.Image` or `list` but is {type(image)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def set_transformer_params(self, mix_ratio: float = 0.5, condition_types: Tuple = ("text", "image")):
        for name, module in self.image_unet.named_sublayers(include_self=True):
            if isinstance(module, DualTransformer2DModel):
                module.mix_ratio = mix_ratio

                for i, type in enumerate(condition_types):
                    if type == "text":
                        module.condition_lengths[i] = self.text_encoder.config.max_position_embeddings
                        module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
                    else:
                        module.condition_lengths[i] = 257
                        module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

    @paddle.no_grad()
    def __call__(
        self,
        prompt: Union[PIL.Image.Image, List[PIL.Image.Image]],
        image: Union[str, List[str]],
        text_to_image_strength: float = 0.5,
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
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.image_unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.image_unet.config.sample_size * self.vae_scale_factor):
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
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
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

        Examples:

        ```py
        >>> from ppdiffusers import VersatileDiffusionDualGuidedPipeline
        >>> import paddle
        >>> import requests
        >>> from io import BytesIO
        >>> from PIL import Image

        >>> # let's download an initial image
        >>> url = "https://huggingface.co/datasets/diffusers/images/resolve/main/benz.jpg"

        >>> response = requests.get(url)
        >>> image = Image.open(BytesIO(response.content)).convert("RGB")
        >>> text = "a red car in the sun"

        >>> pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(
        ...     "shi-labs/versatile-diffusion", paddle_dtype=paddle.float16
        ... )
        >>> pipe.remove_unused_weights()

        >>> generator = paddle.Generator().manual_seed(0)
        >>> text_to_image_strength = 0.75

        >>> image = pipe(
        ...     prompt=text, image=image, text_to_image_strength=text_to_image_strength, generator=generator
        ... ).images[0]
        >>> image.save("./car_variation.png")
        ```

        Returns:
            [`~pipelines.stable_diffusion.ImagePipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.ImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple. When
            returning a tuple, the first element is a list with the generated images.
        """
        # 0. Default height and width to unet
        height = height or self.image_unet.config.sample_size * self.vae_scale_factor
        width = width or self.image_unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, image, height, width, callback_steps)

        # 2. Define call parameters
        prompt = [prompt] if not isinstance(prompt, list) else prompt
        image = [image] if not isinstance(image, list) else image
        batch_size = len(prompt)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompts
        prompt_embeds = self._encode_text_prompt(prompt, num_images_per_prompt, do_classifier_free_guidance)
        image_embeddings = self._encode_image_prompt(image, num_images_per_prompt, do_classifier_free_guidance)
        dual_prompt_embeddings = paddle.concat([prompt_embeds, image_embeddings], axis=1)
        prompt_types = ("text", "image")

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.image_unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dual_prompt_embeddings.dtype,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Combine the attention blocks of the image and text UNets
        self.set_transformer_params(text_to_image_strength, prompt_types)

        # 8. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.image_unet(latent_model_input, t, encoder_hidden_states=dual_prompt_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 9. Post-processing
        image = self.decode_latents(latents)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
