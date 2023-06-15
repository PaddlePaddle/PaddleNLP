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

import inspect
from typing import Callable, List, Optional, Union

import einops
import numpy as np
import paddle
import PIL
from PIL import Image

from paddlenlp.transformers import (
    CLIPFeatureExtractor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    GPTTokenizer,
)

from ...models import AutoencoderKL, UViTModel
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import DPMSolverUniDiffuserScheduler
from ...utils import logging, randn_tensor
from . import ImageTextPipelineOutput
from .caption_decoder import CaptionDecoder

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def center_crop(width, height, img):
    resample = {"box": Image.BOX, "lanczos": Image.LANCZOS}["lanczos"]
    crop = np.min(img.shape[:2])
    img = img[
        (img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2,
        (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2,
    ]  # center crop
    try:
        img = Image.fromarray(img, "RGB")
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)  # resize the center crop from [crop, crop] to [width, height]
    return np.array(img).astype(np.uint8)


class UniDiffuserPipeline(DiffusionPipeline):

    image_encoder: CLIPVisionModelWithProjection
    image_feature_extractor: CLIPFeatureExtractor
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    unet: UViTModel
    vae: AutoencoderKL
    caption_decoder: CaptionDecoder
    caption_tokenizer: GPTTokenizer
    scheduler: DPMSolverUniDiffuserScheduler

    def __init__(
        self,
        image_encoder: CLIPVisionModelWithProjection,
        image_feature_extractor: CLIPFeatureExtractor,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UViTModel,
        vae: AutoencoderKL,
        caption_decoder: CaptionDecoder,
        caption_tokenizer: GPTTokenizer,
        scheduler: DPMSolverUniDiffuserScheduler,
    ):
        super().__init__()
        self.register_modules(
            image_encoder=image_encoder,
            image_feature_extractor=image_feature_extractor,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
            caption_decoder=caption_decoder,
            caption_tokenizer=caption_tokenizer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.num_channels_latents = vae.latent_channels  # 4
        self.image_encoder_clip_img_dim = image_encoder.config.projection_dim  # 512
        self.text_encoder_seq_len = tokenizer.model_max_length  # 77
        self.text_encoder_text_dim = text_encoder.config.hidden_size // text_encoder.config.num_attention_heads  # 64

    # Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
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

    def _infer_batch_size(self, mode, image, prompt, prompt_embeds, num_samples):
        if mode in ["t2i", "t2i2t"]:
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]
        elif mode in ["i2t", "i2t2i"]:
            if isinstance(image, PIL.Image.Image):
                batch_size = 1
            else:
                batch_size = image.shape[0]
        else:
            # For unconditional (and marginal) generation, set as num_samples
            batch_size = num_samples
        return batch_size

    def _split(self, x, height, width):
        r"""
        Splits a flattened embedding x of shape (B, C * H * W + clip_img_dim) into two tensors of shape (B, C, H, W)
        and (B, 1, clip_img_dim)
        """
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        img_vae_dim = self.num_channels_latents * latent_height * latent_width

        img_vae, img_clip = x.split([img_vae_dim, self.image_encoder_clip_img_dim], axis=1)

        img_vae = einops.rearrange(
            img_vae, "B (C H W) -> B C H W", C=self.num_channels_latents, H=latent_height, W=latent_width
        )
        img_clip = einops.rearrange(img_clip, "B (L D) -> B L D", L=1, D=self.image_encoder_clip_img_dim)
        return img_vae, img_clip

    def _combine(self, img_vae, img_clip):
        r"""
        Combines a latent iamge img_vae of shape (B, C, H, W) and a CLIP-embedded image img_clip of shape (B, 1,
        clip_img_dim) into a single tensor of shape (B, C * H * W + clip_img_dim).
        """
        img_vae = einops.rearrange(img_vae, "B C H W -> B (C H W)")
        img_clip = einops.rearrange(img_clip, "B L D -> B (L D)")
        return paddle.concat([img_vae, img_clip], axis=-1)

    def _split_joint(self, x, height, width):
        r"""
        Splits a flattened embedding x of shape (B, C * H * W + clip_img_dim + text_seq_len * text_dim] into (img_vae,
        img_clip, text) where img_vae is of shape (B, C, H, W), img_clip is of shape (B, 1, clip_img_dim), and text is
        of shape (B, text_seq_len, text_dim).
        """
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        img_vae_dim = self.num_channels_latents * latent_height * latent_width
        text_dim = self.text_encoder_seq_len * self.text_encoder_text_dim

        img_vae, img_clip, text = x.split([img_vae_dim, self.image_encoder_clip_img_dim, text_dim], axis=1)
        img_vae = einops.rearrange(
            img_vae, "B (C H W) -> B C H W", C=self.num_channels_latents, H=latent_height, W=latent_width
        )
        img_clip = einops.rearrange(img_clip, "B (L D) -> B L D", L=1, D=self.image_encoder_clip_img_dim)
        text = einops.rearrange(text, "B (L D) -> B L D", L=self.text_encoder_seq_len, D=self.text_encoder_text_dim)
        return img_vae, img_clip, text

    def _combine_joint(self, img_vae, img_clip, text):
        r"""
        Combines a latent image img_vae of shape (B, C, H, W), a CLIP-embedded image img_clip of shape (B, L_img,
        clip_img_dim), and a text embedding text of shape (B, L_text, text_dim) into a single embedding x of shape (B,
        C * H * W + L_img * clip_img_dim + L_text * text_dim).
        """
        img_vae = einops.rearrange(img_vae, "B C H W -> B (C H W)")
        img_clip = einops.rearrange(img_clip, "B L D -> B (L D)")
        text = einops.rearrange(text, "B L D -> B (L D)")
        return paddle.concat([img_vae, img_clip, text], axis=-1)

    # Modified from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def encode_text_latents(
        self,
        prompt,
        num_images_per_prompt,
        negative_prompt=None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
    ):
        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pd",
            )
            prompt_embeds = self.text_encoder(text_inputs.input_ids)[0]

        return prompt_embeds

    # Modified from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix.StableDiffusionInstructPix2PixPipeline.prepare_image_latents
    def encode_image_vae_latents(self, image, batch_size, num_images_per_prompt, dtype, generator=None):
        if not isinstance(image, paddle.Tensor):
            raise ValueError(f"`image` has to be of type `paddle.Tensor`, but is {type(image)}")
        image = image.cast(dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # vae encode
        if isinstance(generator, list):
            image_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) * self.vae.scaling_factor
                for i in range(batch_size)
            ]
            image_latents = paddle.concat(image_latents, axis=0)
        else:
            image_latents = self.vae.encode(image).latent_dist.sample(generator) * self.vae.scaling_factor

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = paddle.concat([image_latents], axis=0)

        return image_latents

    # Modified from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix.StableDiffusionInstructPix2PixPipeline.prepare_image_latents
    def encode_image_clip_latents(
        self,
        image,
        batch_size,
        num_images_per_prompt,
        dtype,
    ):
        batch_size = batch_size * num_images_per_prompt

        # clip encode
        inputs = self.image_feature_extractor(images=Image.fromarray(image), return_tensors="pd").pixel_values
        # TODO junnyu, support float16 we need cast dtype
        image_latents = self.image_encoder(inputs.cast(self.image_encoder.dtype)).image_embeds.unsqueeze(1)

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = paddle.concat([image_latents], axis=0)

        return image_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_image_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.transpose([0, 2, 3, 1]).cast("float32").numpy()
        return image

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_text_latents(self, batch_size, seq_len, hidden_size, dtype, generator, latents=None):
        # Prepare text latents for the CLIP embedded prompt.
        shape = [batch_size, seq_len, hidden_size]
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

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_image_vae_latents(
        self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None
    ):
        # Prepare latents for the VAE embedded image.
        shape = [batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor]
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

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_image_clip_latents(self, batch_size, clip_img_dim, dtype, generator, latents=None):
        # Prepare latents for the CLIP embedded image.
        shape = [batch_size, 1, clip_img_dim]
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

    def get_noise_pred(
        self,
        mode,
        latents,
        t,
        img_vae,
        img_clip,
        prompt_embeds,
        N,
        guidance_scale,
        height,
        width,
        data_type=1,
        generator=None,
    ):
        dtype = self.unet.dtype
        if mode == "joint":
            img_vae_latents, img_clip_latents, text_latents = self._split_joint(latents, height, width)
            img_vae_out, img_clip_out, text_out = self.unet(
                img=img_vae_latents,
                clip_img=img_clip_latents,
                text=text_latents,
                t_img=t,
                t_text=t,
                data_type=paddle.zeros_like(t, dtype=paddle.int32) + data_type,
            )
            x_out = self._combine_joint(img_vae_out, img_clip_out, text_out)

            if guidance_scale == 0.0:
                return x_out

            img_vae_T = randn_tensor(img_vae.shape, generator=generator, dtype=dtype)
            img_clip_T = randn_tensor(img_clip.shape, generator=generator, dtype=dtype)
            _, _, text_out_uncond = self.unet(
                img=img_vae_T,
                clip_img=img_clip_T,
                text=text_latents,
                t_img=paddle.ones_like(t) * N,
                t_text=t,
                data_type=paddle.zeros_like(t, dtype=paddle.int32) + data_type,
            )
            text_T = randn_tensor(prompt_embeds.shape, generator=generator, dtype=dtype)
            img_vae_out_uncond, img_clip_out_uncond, _ = self.unet(
                img=img_vae_latents,
                clip_img=img_clip_latents,
                text=text_T,
                t_img=t,
                t_text=paddle.ones_like(t) * N,
                data_type=paddle.zeros_like(t, dtype=paddle.int32) + data_type,
            )
            x_out_uncond = self._combine_joint(img_vae_out_uncond, img_clip_out_uncond, text_out_uncond)

            return x_out + guidance_scale * (x_out - x_out_uncond)

        elif mode == "t2i":
            img_vae_latents, img_clip_latents = self._split(latents, height, width)
            t_text = paddle.zeros([t.shape[0]], dtype=paddle.int32)
            img_vae_out, img_clip_out, text_out = self.unet(
                img=img_vae_latents,
                clip_img=img_clip_latents,
                text=prompt_embeds,
                t_img=t,
                t_text=t_text,
                data_type=paddle.zeros_like(t_text, dtype=paddle.int32) + data_type,
            )
            img_out = self._combine(img_vae_out, img_clip_out)

            if guidance_scale == 0.0:
                return img_out

            text_T = randn_tensor(prompt_embeds.shape, generator=generator, dtype=dtype)
            img_vae_out_uncond, img_clip_out_uncond, text_out_uncond = self.unet(
                img=img_vae_latents,
                clip_img=img_clip_latents,
                text=text_T,
                t_img=t,
                t_text=paddle.ones_like(t) * N,
                data_type=paddle.zeros_like(t_text, dtype=paddle.int32) + data_type,
            )
            img_out_uncond = self._combine(img_vae_out_uncond, img_clip_out_uncond)

            return img_out + guidance_scale * (img_out - img_out_uncond)

        elif mode == "i2t":
            t_img = paddle.zeros([t.shape[0]], dtype=paddle.int32)
            img_vae_out, img_clip_out, text_out = self.unet(
                img=img_vae,
                clip_img=img_clip,
                text=latents,
                t_img=t_img,
                t_text=t,
                data_type=paddle.zeros_like(t_img, dtype=paddle.int32) + data_type,
            )
            if guidance_scale == 0.0:
                return text_out

            img_vae_T = randn_tensor(img_vae.shape, generator=generator, dtype=dtype)
            img_clip_T = randn_tensor(img_clip.shape, generator=generator, dtype=dtype)
            img_vae_out_uncond, img_clip_out_uncond, text_out_uncond = self.unet(
                img=img_vae_T,
                clip_img=img_clip_T,
                text=latents,
                t_img=paddle.ones_like(t) * N,
                t_text=t,
                data_type=paddle.zeros_like(t, dtype=paddle.int32) + data_type,
            )
            return text_out + guidance_scale * (text_out - text_out_uncond)

        elif mode == "t":
            img_vae_out, img_clip_out, text_out = self.unet(
                img=img_vae,
                clip_img=img_clip,
                text=latents,
                t_img=paddle.ones_like(t) * N,
                t_text=t,
                data_type=paddle.zeros_like(t, dtype=paddle.int32) + data_type,
            )
            return text_out

        elif mode == "i":
            img_vae_latents, img_clip_latents = self._split(latents, height, width)
            t_text = paddle.ones_like(t) * N
            img_vae_out, img_clip_out, text_out = self.unet(
                img=img_vae_latents,
                clip_img=img_clip_latents,
                text=prompt_embeds,
                t_img=t,
                t_text=t_text,
                data_type=paddle.zeros_like(t_text, dtype=paddle.int32) + data_type,
            )
            img_out = self._combine(img_vae_out, img_clip_out)
            return img_out

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
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

    def _denoising_sample_fn(
        self,
        mode,
        image_vae_latents,
        image_clip_latents,
        prompt_embeds,
        num_inference_steps,
        extra_step_kwargs,
        guidance_scale,
        height,
        width,
        callback,
        callback_steps,
    ):
        # Prepare latent variables
        if mode == "joint":
            latents = self._combine_joint(image_vae_latents, image_clip_latents, prompt_embeds)
        elif mode in ["t2i", "i"]:
            latents = self._combine(image_vae_latents, image_clip_latents)
        elif mode in ["i2t", "t"]:
            latents = prompt_embeds
        else:
            raise ValueError

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        N = self.scheduler.config.num_train_timesteps

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = self.get_noise_pred(
                    mode,
                    latents,
                    t * N,
                    image_vae_latents,
                    image_clip_latents,
                    prompt_embeds,
                    N,
                    guidance_scale,
                    height,
                    width,
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if mode == "joint":
            image_vae_latents, image_clip_latents, text_latents = self._split_joint(latents, height, width)
            return image_vae_latents, image_clip_latents, text_latents
        elif mode in ["t2i", "i"]:
            image_vae_latents, image_clip_latents = self._split(latents, height, width)
            return image_vae_latents, image_clip_latents
        elif mode in ["i2t", "t"]:
            text_latents = latents
            return text_latents

    @paddle.no_grad()
    def __call__(
        self,
        mode: str = "t2i",  # t2i, i2t, t2i2t, i2t2i, joint, i, t
        image: Optional[Union[paddle.Tensor, PIL.Image.Image]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        num_prompts_per_image: Optional[int] = 1,
        num_samples: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        prompt_latents: Optional[paddle.Tensor] = None,
        vae_latents: Optional[paddle.Tensor] = None,
        clip_latents: Optional[paddle.Tensor] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        use_beam_search: Optional[bool] = True,
        **kwargs,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.img_size * self.vae_scale_factor
        width = width or self.unet.config.img_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        if mode in ["i2t", "i2t2i"]:
            self.check_inputs([image], height, width, callback_steps)

        if mode in ["t2i", "t2i2t"]:
            self.check_inputs([prompt], height, width, callback_steps)

        # 2. Define call parameters
        batch_size = self._infer_batch_size(mode, image, prompt, prompt_embeds, num_samples)

        # 3. Encode input prompt if available; otherwise prepare text latents
        if mode in ["t2i", "t2i2t"]:
            # 3.1. Encode input prompt(text)
            assert prompt is not None or prompt_embeds is not None
            prompt_embeds = self.encode_text_latents(
                prompt,
                num_images_per_prompt,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )
            # Encode contexts to lower text dim, 768 -> 64
            prompt_embeds = self.unet.encode_prefix(prompt_embeds)
        else:
            # 3.2. Prepare text latents
            prompt_embeds = self.prepare_text_latents(
                batch_size,
                self.text_encoder_seq_len,
                self.text_encoder_text_dim,
                paddle.float32,  # Placeholder, need to determine correct thing to do for dtype
                generator,
                prompt_latents,
            )

        # 4. Encode input image if available; otherwise prepare image latents
        if mode in ["i2t", "i2t2i"]:
            assert image is not None and isinstance(image, PIL.Image.Image)
            # 4.1. Encode images, if available
            image = np.array(image).astype(np.uint8)
            image_crop = center_crop(height, width, image)
            # Encode image using CLIP
            image_clip_latents = self.encode_image_clip_latents(
                image_crop,
                batch_size,
                num_prompts_per_image,  # not num_images_per_prompt
                prompt_embeds.dtype,
            )
            # Encode image using VAE
            image_vae = (image_crop / 127.5 - 1.0).astype(np.float32)
            image_vae = einops.rearrange(image_vae, "h w c -> 1 c h w")
            image_vae_latents = self.encode_image_vae_latents(
                paddle.to_tensor(image_vae),
                batch_size,
                num_prompts_per_image,  # not num_images_per_prompt
                prompt_embeds.dtype,
                generator,
            )

        else:
            # 4.2. Prepare image latent variables, if necessary
            # Prepare image CLIP latents
            image_clip_latents = self.prepare_image_clip_latents(
                batch_size * num_images_per_prompt,
                self.image_encoder_clip_img_dim,
                prompt_embeds.dtype,
                generator,
                clip_latents,
            )
            # Prepare image VAE latents
            image_vae_latents = self.prepare_image_vae_latents(
                batch_size * num_images_per_prompt,
                self.num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                generator,
                vae_latents,
            )

        # 5. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6. Prepare timesteps and Denoising loop
        if mode in ["i", "t", "i2t", "t2i", "joint"]:
            outs = self._denoising_sample_fn(
                mode,
                image_vae_latents,
                image_clip_latents,
                prompt_embeds,
                num_inference_steps,
                extra_step_kwargs,
                guidance_scale,
                height,
                width,
                callback,
                callback_steps,
            )
        elif mode in ["i2t2i"]:
            # 'i2t2i' should do 'i2t' first
            outs = self._denoising_sample_fn(
                "i2t",
                image_vae_latents,
                image_clip_latents,
                prompt_embeds,
                num_inference_steps,
                extra_step_kwargs,
                guidance_scale,
                height,
                width,
                callback,
                callback_steps,
            )
        elif mode in ["t2i2t"]:
            # 't2i2t' should do 't2i' first
            outs = self._denoising_sample_fn(
                "t2i",
                image_vae_latents,
                image_clip_latents,
                prompt_embeds,
                num_inference_steps,
                extra_step_kwargs,
                guidance_scale,
                height,
                width,
                callback,
                callback_steps,
            )
        else:
            raise ValueError

        # 7. Generate image or text and Post-processing
        gen_image, gen_text = None, None
        if mode == "joint":
            image_vae_latents, image_clip_latents, text_latents = outs
            gen_image = self.decode_image_latents(image_vae_latents)
            gen_text = self.caption_decoder.generate_captions(
                self.caption_tokenizer, text_latents, use_beam_search=use_beam_search
            )

        elif mode in ["t2i", "i", "t2i2t"]:
            image_vae_latents, image_clip_latents = outs
            if mode in ["t2i", "i"]:
                gen_image = self.decode_image_latents(image_vae_latents)
            else:
                # 't2i2t' should do 'i2t' later
                prompt_embeds = self.prepare_text_latents(
                    batch_size,
                    self.text_encoder_seq_len,
                    self.text_encoder_text_dim,
                    paddle.float32,  # Placeholder, need to determine correct thing to do for dtype
                    generator,
                    prompt_latents,
                )
                text_latents = self._denoising_sample_fn(
                    "i2t",
                    image_vae_latents,
                    image_clip_latents,
                    prompt_embeds,
                    num_inference_steps,
                    extra_step_kwargs,
                    guidance_scale,
                    height,
                    width,
                    callback,
                    callback_steps,
                )
                gen_text = self.caption_decoder.generate_captions(
                    self.caption_tokenizer, text_latents, use_beam_search=use_beam_search
                )

        elif mode in ["i2t", "t", "i2t2i"]:
            text_latents = outs
            if mode in ["i2t", "t"]:
                gen_text = self.caption_decoder.generate_captions(
                    self.caption_tokenizer, text_latents, use_beam_search=use_beam_search
                )
            else:
                # 'i2t2i' should do 't2i' later
                # Prepare image CLIP latents
                image_clip_latents = self.prepare_image_clip_latents(
                    batch_size * num_images_per_prompt,
                    self.image_encoder_clip_img_dim,
                    prompt_embeds.dtype,
                    generator,
                    clip_latents,
                )
                # Prepare image VAE latents
                image_vae_latents = self.prepare_image_vae_latents(
                    batch_size * num_images_per_prompt,
                    self.num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    generator,
                    vae_latents,
                )
                image_vae_latents, image_clip_latents = self._denoising_sample_fn(
                    "t2i",
                    image_vae_latents,
                    image_clip_latents,
                    text_latents,
                    num_inference_steps,
                    extra_step_kwargs,
                    guidance_scale,
                    height,
                    width,
                    callback,
                    callback_steps,
                )
                gen_image = self.decode_image_latents(image_vae_latents)

        # 8. Convert gen_image to PIL, gen_text has no else processing
        if output_type == "pil" and gen_image is not None:
            gen_image = self.numpy_to_pil(gen_image)

        if not return_dict:
            return (gen_image, gen_text)

        return ImageTextPipelineOutput(images=gen_image, texts=gen_text)
