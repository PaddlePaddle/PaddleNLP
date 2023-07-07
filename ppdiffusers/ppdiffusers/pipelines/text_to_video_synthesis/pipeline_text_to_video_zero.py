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

import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import paddle
import paddle.nn.functional as F
import PIL

from paddlenlp.transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from ppdiffusers.models import AutoencoderKL, UNet2DConditionModel
from ppdiffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionSafetyChecker,
)
from ppdiffusers.schedulers import KarrasDiffusionSchedulers
from ppdiffusers.utils import BaseOutput


def rearrange_0(tensor, f):
    F, C, H, W = tensor.shape
    tensor = paddle.transpose(x=paddle.reshape(x=tensor, shape=(F // f, f, C, H, W)), perm=(0, 2, 1, 3, 4))
    return tensor


def rearrange_1(tensor):
    B, C, F, H, W = tensor.shape
    return paddle.reshape(x=paddle.transpose(x=tensor, perm=(0, 2, 1, 3, 4)), shape=(B * F, C, H, W))


def rearrange_3(tensor, f):
    F, D, C = tensor.shape
    return paddle.reshape(x=tensor, shape=(F // f, f, D, C))


def rearrange_4(tensor):
    B, F, D, C = tensor.shape
    return paddle.reshape(x=tensor, shape=(B * F, D, C))


class CrossFrameAttnProcessor:
    """
    Cross frame attention processor. For each frame the self-attention is replaced with attention with first frame

    Args:
        batch_size: The number that represents actual batch size, other than the frames.
            For example, using calling unet with a single prompt and num_images_per_prompt=1, batch_size should be
            equal to 2, due to classifier-free guidance.
    """

    def __init__(self, batch_size=2):
        self.batch_size = batch_size

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Sparse Attention
        if not is_cross_attention:
            video_length = key.shape[0] // self.batch_size
            first_frame_index = [0] * video_length

            # rearrange keys to have batch and frames in the 1st and 2nd dims respectively
            key = rearrange_3(key, video_length)
            key = key.index_select(paddle.to_tensor(first_frame_index), 1)
            # rearrange values to have batch and frames in the 1st and 2nd dims respectively
            value = rearrange_3(value, video_length)
            value = value.index_select(paddle.to_tensor(first_frame_index), 1)

            # rearrange back to original shape
            key = rearrange_4(key)
            value = rearrange_4(value)
        query = attn.head_to_batch_dim(query, out_dim=3)
        key = attn.head_to_batch_dim(key, out_dim=3)
        value = attn.head_to_batch_dim(value, out_dim=3)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = paddle.bmm(x=attention_probs, y=value)
        hidden_states = attn.batch_to_head_dim(hidden_states, in_dim=3)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


@dataclass
class TextToVideoPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


def coords_grid(batch, ht, wd):
    coords = paddle.meshgrid(paddle.arange(end=ht), paddle.arange(end=wd))
    coords = paddle.stack(x=coords[::-1], axis=0).astype(dtype="float32")
    return coords[None].tile(repeat_times=[batch, 1, 1, 1])


def warp_single_latent(latent, reference_flow):
    """
    Warp latent of a single frame with given flow

    Args:
        latent: latent code of a single frame
        reference_flow: flow which to warp the latent with

    Returns:
        warped: warped latent
    """
    _, _, H, W = reference_flow.shape
    _, _, h, w = latent.shape
    if isinstance(latent.dtype, paddle.dtype):
        dtype = latent.dtype
    elif isinstance(latent.dtype, str) and latent.dtype not in ["cpu", "cuda", "ipu", "xpu"]:
        dtype = latent.dtype
    elif isinstance(latent.dtype, paddle.Tensor):
        dtype = latent.dtype.dtype
    else:
        dtype = coords_grid(1, H, W).dtype
    coords0 = coords_grid(1, H, W).cast(dtype)
    coords_t0 = coords0 + reference_flow
    coords_t0[:, (0)] /= W
    coords_t0[:, (1)] /= H
    coords_t0 = coords_t0 * 2.0 - 1.0
    coords_t0 = F.interpolate(x=coords_t0, size=(h, w), mode="bilinear")
    coords_t0 = paddle.transpose(x=coords_t0, perm=(0, 2, 3, 1))
    warped = F.grid_sample(x=latent, grid=coords_t0, mode="nearest", padding_mode="reflection")
    return warped


def create_motion_field(motion_field_strength_x, motion_field_strength_y, frame_ids, dtype):
    """
    Create translation motion field

    Args:
        motion_field_strength_x: motion strength along x-axis
        motion_field_strength_y: motion strength along y-axis
        frame_ids: indexes of the frames the latents of which are being processed.
            This is needed when we perform chunk-by-chunk inference
        dtype: dtype

    Returns:

    """
    seq_length = len(frame_ids)
    reference_flow = paddle.zeros(shape=(seq_length, 2, 512, 512), dtype=dtype)
    for fr_idx in range(seq_length):
        reference_flow[(fr_idx), (0), :, :] = motion_field_strength_x * frame_ids[fr_idx]
        reference_flow[(fr_idx), (1), :, :] = motion_field_strength_y * frame_ids[fr_idx]
    return reference_flow


def create_motion_field_and_warp_latents(motion_field_strength_x, motion_field_strength_y, frame_ids, latents):
    """
    Creates translation motion and warps the latents accordingly

    Args:
        motion_field_strength_x: motion strength along x-axis
        motion_field_strength_y: motion strength along y-axis
        frame_ids: indexes of the frames the latents of which are being processed.
            This is needed when we perform chunk-by-chunk inference
        latents: latent codes of frames

    Returns:
        warped_latents: warped latents
    """
    motion_field = create_motion_field(
        motion_field_strength_x=motion_field_strength_x,
        motion_field_strength_y=motion_field_strength_y,
        frame_ids=frame_ids,
        dtype=latents.dtype,
    )
    warped_latents = latents.clone().detach()
    for i in range(len(warped_latents)):
        warped_latents[i] = warp_single_latent(latents[i][None], motion_field[i][None])
    return warped_latents


class TextToVideoZeroPipeline(StableDiffusionPipeline):
    """
    Pipeline for zero-shot text-to-video generation using Stable Diffusion.

    This model inherits from [`StableDiffusionPipeline`]. Check the superclass documentation for the generic methods
    the library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of CLIP, specifically
            the clip-vit-large-patch14 variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class CLIPTokenizer.
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker
        )
        self.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

    def forward_loop(self, x_t0, t0, t1, generator):
        """
        Perform ddpm forward process from time t0 to t1. This is the same as adding noise with corresponding variance.

        Args:
            x_t0: latent code at time t0
            t0: t0
            t1: t1
            generator: paddle.Generator object

        Returns:
            x_t1: forward process applied to x_t0 from time t0 to t1.
        """
        eps = paddle.randn(shape=x_t0.shape, generator=generator, dtype=x_t0.dtype)
        alpha_vec = paddle.prod(x=self.scheduler.alphas[t0:t1])
        x_t1 = paddle.sqrt(x=alpha_vec) * x_t0 + paddle.sqrt(x=1 - alpha_vec) * eps
        return x_t1

    def backward_loop(
        self,
        latents,
        timesteps,
        prompt_embeds,
        guidance_scale,
        callback,
        callback_steps,
        num_warmup_steps,
        extra_step_kwargs,
        cross_attention_kwargs=None,
    ):
        """
        Perform backward process given list of time steps

        Args:
            latents: Latents at time timesteps[0].
            timesteps: time steps, along which to perform backward process.
            prompt_embeds: Pre-generated text embeddings
            guidance_scale:
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: paddle.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            extra_step_kwargs: extra_step_kwargs.
            cross_attention_kwargs: cross_attention_kwargs.
            num_warmup_steps: number of warmup steps.

        Returns:
            latents: latents of backward process output at time timesteps[-1]
        """
        do_classifier_free_guidance = guidance_scale > 1.0
        num_steps = (len(timesteps) - num_warmup_steps) // self.scheduler.order
        with self.progress_bar(total=num_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat(x=[latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(chunks=2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        return latents.clone().detach()

    @paddle.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int] = 8,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        motion_field_strength_x: float = 12,
        motion_field_strength_y: float = 12,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        t0: int = 44,
        t1: int = 47,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            video_length (`int`, *optional*, defaults to 8): The number of generated video frames
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
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
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
                One or a list of paddle generator(s)
                to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"numpy"`):
                The output format of the generated image. Choose between `"latent"` and `"numpy"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: paddle.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            motion_field_strength_x (`float`, *optional*, defaults to 12):
                Strength of motion in generated video along x-axis. See the [paper](https://arxiv.org/abs/2303.13439),
                Sect. 3.3.1.
            motion_field_strength_y (`float`, *optional*, defaults to 12):
                Strength of motion in generated video along y-axis. See the [paper](https://arxiv.org/abs/2303.13439),
                Sect. 3.3.1.
            t0 (`int`, *optional*, defaults to 44):
                Timestep t0. Should be in the range [0, num_inference_steps - 1]. See the
                [paper](https://arxiv.org/abs/2303.13439), Sect. 3.3.1.
            t1 (`int`, *optional*, defaults to 47):
                Timestep t0. Should be in the range [t0 + 1, num_inference_steps - 1]. See the
                [paper](https://arxiv.org/abs/2303.13439), Sect. 3.3.1.

        Returns:
            [`~pipelines.text_to_video_synthesis.TextToVideoPipelineOutput`]:
                The output contains a ndarray of the generated images, when output_type != 'latent', otherwise a latent
                codes of generated image, and a list of `bool`s denoting whether the corresponding generated image
                likely represents "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        assert video_length > 0
        frame_ids = list(range(video_length))
        assert num_videos_per_prompt == 1
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(
            num_inference_steps,
        )
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # Perform the first backward process up to time T_1
        x_1_t1 = self.backward_loop(
            timesteps=timesteps[: -t1 - 1],
            prompt_embeds=prompt_embeds,
            latents=latents,
            guidance_scale=guidance_scale,
            callback=callback,
            callback_steps=callback_steps,
            extra_step_kwargs=extra_step_kwargs,
            num_warmup_steps=num_warmup_steps,
        )
        scheduler_copy = copy.deepcopy(self.scheduler)

        # Perform the second backward process up to time T_0
        x_1_t0 = self.backward_loop(
            timesteps=timesteps[-t1 - 1 : -t0 - 1],
            prompt_embeds=prompt_embeds,
            latents=x_1_t1,
            guidance_scale=guidance_scale,
            callback=callback,
            callback_steps=callback_steps,
            extra_step_kwargs=extra_step_kwargs,
            num_warmup_steps=0,
        )

        # Propagate first frame latents at time T_0 to remaining frames
        x_2k_t0 = x_1_t0.tile(repeat_times=[video_length - 1, 1, 1, 1])

        # Add motion in latents at time T_0
        x_2k_t0 = create_motion_field_and_warp_latents(
            motion_field_strength_x=motion_field_strength_x,
            motion_field_strength_y=motion_field_strength_y,
            latents=x_2k_t0,
            frame_ids=frame_ids[1:],
        )

        # Perform forward process up to time T_1
        x_2k_t1 = self.forward_loop(
            x_t0=x_2k_t0, t0=timesteps[-t0 - 1].item(), t1=timesteps[-t1 - 1].item(), generator=generator
        )

        # Perform backward process from time T_1 to 0
        x_1k_t1 = paddle.concat(x=[x_1_t1, x_2k_t1])
        b, l, d = prompt_embeds.shape
        prompt_embeds = (
            prompt_embeds[:, (None)].tile(repeat_times=[1, video_length, 1, 1]).reshape([b * video_length, l, d])
        )
        self.scheduler = scheduler_copy
        x_1k_0 = self.backward_loop(
            timesteps=timesteps[-t1 - 1 :],
            prompt_embeds=prompt_embeds,
            latents=x_1k_t1,
            guidance_scale=guidance_scale,
            callback=callback,
            callback_steps=callback_steps,
            extra_step_kwargs=extra_step_kwargs,
            num_warmup_steps=0,
        )
        latents = x_1k_0
        paddle.device.cuda.empty_cache()
        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        else:
            image = self.decode_latents(latents)
            image, has_nsfw_concept = self.run_safety_checker(image, prompt_embeds.dtype)
        if not return_dict:
            return image, has_nsfw_concept
        return TextToVideoPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
