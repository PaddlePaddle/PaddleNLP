# Copyright 2022 Kakao Brain and The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Union

import paddle
import paddle.nn.functional as F

from paddlenlp.transformers import CLIPTextModelWithProjection, CLIPTokenizer

from ...models import PriorTransformer, UNet2DConditionModel, UNet2DModel
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import UnCLIPScheduler
from ...utils import logging
from .text_proj import UnCLIPTextProjModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class UnCLIPPipeline(DiffusionPipeline):
    """
    Pipeline for text-to-image generation using unCLIP

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        prior ([`PriorTransformer`]):
            The canonincal unCLIP prior to approximate the image embedding from the text embedding.
        decoder ([`UNet2DConditionModel`]):
            The decoder to invert the image embedding into an image.
        super_res_first ([`UNet2DModel`]):
            Super resolution unet. Used in all but the last step of the super resolution diffusion process.
        super_res_last ([`UNet2DModel`]):
            Super resolution unet. Used in the last step of the super resolution diffusion process.
        prior_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the prior denoising process. Just a modified DDPMScheduler.
        decoder_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the decoder denoising process. Just a modified DDPMScheduler.
        super_res_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the super resolution denoising process. Just a modified DDPMScheduler.

    """

    prior: PriorTransformer
    decoder: UNet2DConditionModel
    text_proj: UnCLIPTextProjModel
    text_encoder: CLIPTextModelWithProjection
    tokenizer: CLIPTokenizer
    super_res_first: UNet2DModel
    super_res_last: UNet2DModel

    prior_scheduler: UnCLIPScheduler
    decoder_scheduler: UnCLIPScheduler
    super_res_scheduler: UnCLIPScheduler

    def __init__(
        self,
        prior: PriorTransformer,
        decoder: UNet2DConditionModel,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_proj: UnCLIPTextProjModel,
        super_res_first: UNet2DModel,
        super_res_last: UNet2DModel,
        prior_scheduler: UnCLIPScheduler,
        decoder_scheduler: UnCLIPScheduler,
        super_res_scheduler: UnCLIPScheduler,
    ):
        super().__init__()

        self.register_modules(
            prior=prior,
            decoder=decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_proj=text_proj,
            super_res_first=super_res_first,
            super_res_last=super_res_last,
            prior_scheduler=prior_scheduler,
            decoder_scheduler=decoder_scheduler,
            super_res_scheduler=super_res_scheduler,
        )

    def prepare_latents(self, shape, dtype, generator, latents, scheduler):
        batch_size = shape[0]
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            if isinstance(generator, list):
                shape = [
                    1,
                ] + shape[1:]
                latents = [paddle.randn(shape, generator=generator[i], dtype=dtype) for i in range(batch_size)]
                latents = paddle.concat(latents, axis=0)
            else:
                latents = paddle.randn(shape, generator=generator, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma
        return latents

    def _encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pd",
            return_attention_mask=True,
        )
        text_input_ids = text_inputs.input_ids
        text_mask = text_inputs.attention_mask

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

        text_encoder_output = self.text_encoder(text_input_ids)

        text_embeddings = text_encoder_output.text_embeds
        text_encoder_hidden_states = text_encoder_output.last_hidden_state

        # duplicate text embeddings for each generation per prompt
        seq_len = text_embeddings.shape[1]
        text_embeddings = text_embeddings.tile([1, num_images_per_prompt])
        text_embeddings = text_embeddings.reshape([batch_size * num_images_per_prompt, seq_len])

        # duplicate text_encoder_hidden_states for each generation per prompt
        seq_len = text_encoder_hidden_states.shape[1]
        text_encoder_hidden_states = text_encoder_hidden_states.tile([1, num_images_per_prompt, 1])
        text_encoder_hidden_states = text_encoder_hidden_states.reshape(
            [batch_size * num_images_per_prompt, seq_len, -1]
        )

        # duplicate text_mask for each generation per prompt
        seq_len = text_mask.shape[1]
        text_mask = text_mask.tile([1, num_images_per_prompt])
        text_mask = text_mask.reshape([batch_size * num_images_per_prompt, seq_len])

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pd",
                return_attention_mask=True,
            )
            uncond_text_mask = uncond_input.attention_mask
            uncond_embeddings_text_encoder_output = self.text_encoder(uncond_input.input_ids)

            uncond_embeddings = uncond_embeddings_text_encoder_output.text_embeds
            uncond_text_encoder_hidden_states = uncond_embeddings_text_encoder_output.last_hidden_state

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.tile([1, num_images_per_prompt])
            uncond_embeddings = uncond_embeddings.reshape([batch_size * num_images_per_prompt, seq_len])

            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.tile([1, num_images_per_prompt, 1])
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.reshape(
                [batch_size * num_images_per_prompt, seq_len, -1]
            )

            # duplicate uncond_text_mask for each generation per prompt
            seq_len = uncond_text_mask.shape[1]
            uncond_text_mask = uncond_text_mask.tile([1, num_images_per_prompt])
            uncond_text_mask = uncond_text_mask.reshape([batch_size * num_images_per_prompt, seq_len])

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = paddle.concat([uncond_embeddings, text_embeddings])
            text_encoder_hidden_states = paddle.concat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

            text_mask = paddle.concat([uncond_text_mask, text_mask])

        return text_embeddings, text_encoder_hidden_states, text_mask

    @paddle.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        prior_num_inference_steps: int = 25,
        decoder_num_inference_steps: int = 25,
        super_res_num_inference_steps: int = 7,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        prior_latents: Optional[paddle.Tensor] = None,
        decoder_latents: Optional[paddle.Tensor] = None,
        super_res_latents: Optional[paddle.Tensor] = None,
        prior_guidance_scale: float = 4.0,
        decoder_guidance_scale: float = 8.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            prior_num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps for the prior. More denoising steps usually lead to a higher quality
                image at the expense of slower inference.
            decoder_num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps for the decoder. More denoising steps usually lead to a higher quality
                image at the expense of slower inference.
            super_res_num_inference_steps (`int`, *optional*, defaults to 7):
                The number of denoising steps for super resolution. More denoising steps usually lead to a higher
                quality image at the expense of slower inference.
            generator (`paddle.Generator`, *optional*):
                One or a list of paddle generator(s) to make generation deterministic.
            prior_latents (`paddle.Tensor` of shape (batch size, embeddings dimension), *optional*):
                Pre-generated noisy latents to be used as inputs for the prior.
            decoder_latents (`paddle.Tensor` of shape (batch size, channels, height, width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            super_res_latents (`paddle.Tensor` of shape (batch size, channels, super res height, super res width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            prior_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            decoder_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        batch_size = batch_size * num_images_per_prompt

        do_classifier_free_guidance = prior_guidance_scale > 1.0 or decoder_guidance_scale > 1.0

        text_embeddings, text_encoder_hidden_states, text_mask = self._encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance
        )

        # prior

        self.prior_scheduler.set_timesteps(prior_num_inference_steps)
        prior_timesteps_tensor = self.prior_scheduler.timesteps

        embedding_dim = self.prior.config.embedding_dim
        prior_latents = self.prepare_latents(
            (batch_size, embedding_dim),
            text_embeddings.dtype,
            generator,
            prior_latents,
            self.prior_scheduler,
        )

        for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = paddle.concat([prior_latents] * 2) if do_classifier_free_guidance else prior_latents

            predicted_image_embedding = self.prior(
                latent_model_input,
                timestep=t,
                proj_embedding=text_embeddings,
                encoder_hidden_states=text_encoder_hidden_states,
                attention_mask=text_mask,
            ).predicted_image_embedding

            if do_classifier_free_guidance:
                predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                predicted_image_embedding = predicted_image_embedding_uncond + prior_guidance_scale * (
                    predicted_image_embedding_text - predicted_image_embedding_uncond
                )

            if i + 1 == prior_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = prior_timesteps_tensor[i + 1]

            prior_latents = self.prior_scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=prior_latents,
                generator=generator,
                prev_timestep=prev_timestep,
            ).prev_sample

        prior_latents = self.prior.post_process_latents(prior_latents)

        image_embeddings = prior_latents

        # done prior

        # decoder

        text_encoder_hidden_states, additive_clip_time_embeddings = self.text_proj(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            text_encoder_hidden_states=text_encoder_hidden_states,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        decoder_text_mask = F.pad(
            text_mask.unsqueeze(0), (self.text_proj.clip_extra_context_tokens, 0), value=1, data_format="NCL"
        ).squeeze(0)

        self.decoder_scheduler.set_timesteps(decoder_num_inference_steps)
        decoder_timesteps_tensor = self.decoder_scheduler.timesteps

        num_channels_latents = self.decoder.in_channels
        height = self.decoder.sample_size
        width = self.decoder.sample_size
        decoder_latents = self.prepare_latents(
            (batch_size, num_channels_latents, height, width),
            text_encoder_hidden_states.dtype,
            generator,
            decoder_latents,
            self.decoder_scheduler,
        )

        for i, t in enumerate(self.progress_bar(decoder_timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                paddle.concat([decoder_latents] * 2) if do_classifier_free_guidance else decoder_latents
            )

            noise_pred = self.decoder(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_encoder_hidden_states,
                class_labels=additive_clip_time_embeddings,
                attention_mask=decoder_text_mask,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # paddle.split is not equal torch.split
                noise_pred_uncond, _ = noise_pred_uncond.split(
                    [latent_model_input.shape[1], noise_pred_uncond.shape[1] - latent_model_input.shape[1]], axis=1
                )
                noise_pred_text, predicted_variance = noise_pred_text.split(
                    [latent_model_input.shape[1], noise_pred_text.shape[1] - latent_model_input.shape[1]], axis=1
                )
                noise_pred = noise_pred_uncond + decoder_guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = paddle.concat([noise_pred, predicted_variance], axis=1)

            if i + 1 == decoder_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = decoder_timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            decoder_latents = self.decoder_scheduler.step(
                noise_pred, t, decoder_latents, prev_timestep=prev_timestep, generator=generator
            ).prev_sample

        decoder_latents = decoder_latents.clip(-1, 1)

        image_small = decoder_latents

        # done decoder

        # super res

        self.super_res_scheduler.set_timesteps(super_res_num_inference_steps)
        super_res_timesteps_tensor = self.super_res_scheduler.timesteps

        channels = self.super_res_first.in_channels // 2
        height = self.super_res_first.sample_size
        width = self.super_res_first.sample_size
        super_res_latents = self.prepare_latents(
            (batch_size, channels, height, width),
            image_small.dtype,
            generator,
            super_res_latents,
            self.super_res_scheduler,
        )

        interpolate_antialias = {}
        if "antialias" in inspect.signature(F.interpolate).parameters:
            interpolate_antialias["antialias"] = True

        image_upscaled = F.interpolate(
            image_small, size=[height, width], mode="bicubic", align_corners=False, **interpolate_antialias
        )

        for i, t in enumerate(self.progress_bar(super_res_timesteps_tensor)):
            # no classifier free guidance

            if i == super_res_timesteps_tensor.shape[0] - 1:
                unet = self.super_res_last
            else:
                unet = self.super_res_first

            latent_model_input = paddle.concat([super_res_latents, image_upscaled], axis=1)

            noise_pred = unet(
                sample=latent_model_input,
                timestep=t,
            ).sample

            if i + 1 == super_res_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = super_res_timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            super_res_latents = self.super_res_scheduler.step(
                noise_pred, t, super_res_latents, prev_timestep=prev_timestep, generator=generator
            ).prev_sample

        image = super_res_latents

        # done super res

        # post processing

        image = image * 0.5 + 0.5
        image = image.clip(0, 1)
        image = image.transpose([0, 2, 3, 1]).cast("float32").numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
