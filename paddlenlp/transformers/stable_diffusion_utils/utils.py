# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
import random
from typing import Optional

import numpy as np
import paddle
from PIL import Image
from tqdm.auto import tqdm

from ..image_utils import load_image
from .schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler

__all__ = ["StableDiffusionMixin"]


class StableDiffusionMixin:
    def set_scheduler(self, scheduler):
        if isinstance(scheduler, (PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler)):
            self.scheduler = scheduler
        elif isinstance(scheduler, str):
            if scheduler == "pndm":
                self.scheduler = PNDMScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    skip_prk_steps=True,
                )
            elif scheduler == "ddim":
                self.scheduler = DDIMScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    clip_sample=False,
                    set_alpha_to_one=False,
                )
            elif scheduler == "k-lms":
                self.scheduler = LMSDiscreteScheduler(
                    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
                )
            else:
                raise ValueError('scheduler must be in ["pndm", "ddim", "k-lms"].')
        else:
            raise ValueError("scheduler error.")

    @classmethod
    def preprocess_image(cls, image):
        image = load_image(image)
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose([0, 3, 1, 2])
        image = paddle.to_tensor(image)
        return 2.0 * image - 1.0

    @classmethod
    def preprocess_mask(cls, mask):
        mask = load_image(mask)
        mask = mask.convert("L")
        w, h = mask.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        mask = mask.resize((w // 8, h // 8), resample=Image.NEAREST)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.tile(mask, (4, 1, 1))
        mask = mask[None].transpose([0, 1, 2, 3])  # what does this step do?
        mask = 1 - mask  # repaint white, keep black
        mask = paddle.to_tensor(mask)
        return mask

    @paddle.no_grad()
    def stable_diffusion_text2image(
        self,
        input_ids,
        seed: Optional[int] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        latents: Optional[paddle.Tensor] = None,
        fp16: Optional[bool] = False,
    ):
        batch_size = input_ids.shape[0]
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 64 but are {height} and {width}.")

        with paddle.amp.auto_cast(enable=fp16, level="O1"):
            text_embeddings = self.clip.text_model(input_ids)[0]

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0
            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance:
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                uncond_embeddings = self.clip.text_model(self.input_ids_uncond.expand([batch_size, -1]))[0]
                text_embeddings = paddle.concat([uncond_embeddings, text_embeddings])

            # get the initial random noise unless the user supplied it
            latents_shape = [batch_size, self.unet_model.in_channels, height // 8, width // 8]
            if latents is None:
                if seed is None:
                    seed = random.randint(0, 2**32)
                paddle.seed(seed)
                latents = paddle.randn(latents_shape)
            else:
                if latents.shape != latents_shape:
                    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

            # set timesteps
            accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
            extra_set_kwargs = {}
            if accepts_offset:
                extra_set_kwargs["offset"] = 1

            self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

            # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = latents * self.scheduler.sigmas[0]

            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
            # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
            # and should be between [0, 1]
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            for i, t in tqdm(enumerate(self.scheduler.timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    sigma = self.scheduler.sigmas[i]
                    latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

                # predict the noise residual
                noise_pred = self.unet_model(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)["prev_sample"]
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

            # scale and decode the image latents with vae
            image = self.vae_model.decode(1 / 0.18215 * latents)
            image = (image / 2 + 0.5).clip(0, 1)
            image = image.transpose([0, 2, 3, 1]).cpu().numpy()
            image = (image * 255).round().astype(np.uint8)
            image = [Image.fromarray(img) for img in image]

        return image

    @paddle.no_grad()
    def stable_diffusion_image2image(
        self,
        input_ids,
        init_image,
        strength=0.8,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        seed=None,
        fp16=False,
    ):
        batch_size = input_ids.shape[0]

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        with paddle.amp.auto_cast(enable=fp16, level="O1"):
            # set timesteps
            accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
            extra_set_kwargs = {}
            offset = 0
            if accepts_offset:
                offset = 1
                extra_set_kwargs["offset"] = 1

            self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

            # encode the init image into latents and scale the latents
            init_latents = self.vae_model.encode(init_image).sample()
            init_latents = 0.18215 * init_latents

            # prepare init_latents noise to latents
            init_latents = paddle.concat([init_latents] * batch_size)

            # get the original timestep using init_timestep
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                timesteps = paddle.to_tensor([num_inference_steps - init_timestep] * batch_size, dtype="int64")
            else:
                timesteps = self.scheduler.timesteps[-init_timestep]
                timesteps = paddle.to_tensor([timesteps] * batch_size, dtype="int64")

            # add noise to latents using the timesteps
            if seed is None:
                seed = random.randint(0, 2**32)
            paddle.seed(seed)
            noise = paddle.randn(init_latents.shape)
            init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

            text_embeddings = self.clip.text_model(input_ids)[0]

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0
            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance:
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                uncond_embeddings = self.clip.text_model(self.input_ids_uncond.expand([batch_size, -1]))[0]
                text_embeddings = paddle.concat([uncond_embeddings, text_embeddings])

            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
            # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
            # and should be between [0, 1]
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            latents = init_latents
            t_start = max(num_inference_steps - init_timestep + offset, 0)
            for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
                t_index = t_start + i
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    sigma = self.scheduler.sigmas[t_index]
                    # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                    latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
                    latent_model_input = latent_model_input.astype(paddle.get_default_dtype())
                    t = t.astype(paddle.get_default_dtype())

                # predict the noise residual
                noise_pred = self.unet_model(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    latents = self.scheduler.step(noise_pred, t_index, latents, **extra_step_kwargs)["prev_sample"]
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            image = self.vae_model.decode(latents.astype(paddle.get_default_dtype()))
            image = (image / 2 + 0.5).clip(0, 1)
            image = image.transpose([0, 2, 3, 1]).cpu().numpy()
            image = (image * 255).round().astype(np.uint8)
            image = [Image.fromarray(img) for img in image]
        return image

    @paddle.no_grad()
    def stable_diffusion_inpainting(
        self,
        input_ids,
        init_image,
        mask_image,
        strength=0.8,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        seed=None,
        fp16=False,
    ):
        batch_size = input_ids.shape[0]

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        with paddle.amp.auto_cast(enable=fp16, level="O1"):
            # set timesteps
            accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
            extra_set_kwargs = {}
            offset = 0
            if accepts_offset:
                offset = 1
                extra_set_kwargs["offset"] = 1

            self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

            # encode the init image into latents and scale the latents
            init_latents = self.vae_model.encode(init_image).sample()
            init_latents = 0.18215 * init_latents

            # prepare init_latents noise to latents
            init_latents = paddle.concat([init_latents] * batch_size)
            init_latents_orig = init_latents

            mask = paddle.concat([mask_image] * batch_size)

            # check sizes
            if not mask.shape == init_latents.shape:
                raise ValueError("The mask and init_image should be the same size!")

            # get the original timestep using init_timestep
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                timesteps = paddle.to_tensor([num_inference_steps - init_timestep] * batch_size, dtype="int64")
            else:
                timesteps = self.scheduler.timesteps[-init_timestep]
                timesteps = paddle.to_tensor([timesteps] * batch_size, dtype="int64")

            # add noise to latents using the timesteps
            if seed is None:
                seed = random.randint(0, 2**32)
            paddle.seed(seed)
            noise = paddle.randn(init_latents.shape)
            init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

            # get prompt text embeddings
            text_embeddings = self.clip.text_model(input_ids)[0]

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0
            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance:
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                uncond_embeddings = self.clip.text_model(self.input_ids_uncond.expand([batch_size, -1]))[0]
                text_embeddings = paddle.concat([uncond_embeddings, text_embeddings])

            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
            # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
            # and should be between [0, 1]
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            latents = init_latents
            t_start = max(num_inference_steps - init_timestep + offset, 0)
            for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
                t_index = t_start + i
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents

                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    sigma = self.scheduler.sigmas[t_index]
                    # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                    latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
                    latent_model_input = latent_model_input.astype(paddle.get_default_dtype())
                    t = t.astype(paddle.get_default_dtype())

                # predict the noise residual
                noise_pred = self.unet_model(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    latents = self.scheduler.step(noise_pred, t_index, latents, **extra_step_kwargs)["prev_sample"]
                    # masking
                    init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, t_index)
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

                    # masking
                    init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, t)
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

            # scale and decode the image latents with vae
            image = self.vae_model.decode(1 / 0.18215 * latents)
            image = (image / 2 + 0.5).clip(0, 1)
            image = image.transpose([0, 2, 3, 1]).cpu().numpy()
            image = (image * 255).round().astype(np.uint8)
            image = [Image.fromarray(img) for img in image]
        return image
