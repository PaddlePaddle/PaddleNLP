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
import time
from typing import Callable, List, Optional, Union

import einops
import numpy as np
import paddle
import PIL
from IPython import embed

from ...models import CaptionDecoder, FrozenAutoencoderKL, UViT
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from ...utils import logging
from . import JointPipelineOutput
from .dpm_solver_pp import DPM_Solver, NoiseScheduleVP
from .unidiffuser_common import stable_diffusion_beta_schedule

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_betas = stable_diffusion_beta_schedule()
N = len(_betas)


def split(x, z_shape=(4, 64, 64), clip_img_dim=512):
    C, H, W = z_shape
    z_dim = C * H * W
    z, clip_img = x.split([z_dim, clip_img_dim], axis=1)
    z = einops.rearrange(z, "B (C H W) -> B C H W", C=C, H=H, W=W)
    clip_img = einops.rearrange(clip_img, "B (L D) -> B L D", L=1, D=clip_img_dim)
    return z, clip_img


def combine(z, clip_img):
    z = einops.rearrange(z, "B C H W -> B (C H W)")
    clip_img = einops.rearrange(clip_img, "B L D -> B (L D)")
    return paddle.concat([z, clip_img], axis=-1)


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.0)
    v.clip_(0.0, 1.0)
    return v


def split_joint(x):
    z_shape = (4, 64, 64)
    clip_img_dim = 512
    text_dim = 64

    C, H, W = z_shape
    z_dim = C * H * W
    z, clip_img, text = x.split([z_dim, clip_img_dim, 77 * text_dim], axis=1)
    z = einops.rearrange(z, "B (C H W) -> B C H W", C=C, H=H, W=W)
    clip_img = einops.rearrange(clip_img, "B (L D) -> B L D", L=1, D=clip_img_dim)
    text = einops.rearrange(text, "B (L D) -> B L D", L=77, D=text_dim)
    return z, clip_img, text


def combine_joint(z, clip_img, text):
    z = einops.rearrange(z, "B C H W -> B (C H W)")
    clip_img = einops.rearrange(clip_img, "B L D -> B (L D)")
    text = einops.rearrange(text, "B L D -> B (L D)")
    return paddle.concat([z, clip_img, text], axis=-1)


class UniDiffuserJointGenerationPipeline(DiffusionPipeline):

    unet: UViT
    vae: FrozenAutoencoderKL
    caption_decoder: CaptionDecoder
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]

    def __init__(
        self,
        unet: UViT,
        vae: FrozenAutoencoderKL,
        caption_decoder: CaptionDecoder,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            caption_decoder=CaptionDecoder,
            scheduler=scheduler,
        )

    def joint_nnet(self, x, timesteps):
        data_type = 1
        text_dim = 64
        z_shape = (4, 64, 64)
        clip_img_dim = 512
        sample_scale = 7

        z, clip_img, text = split_joint(x)
        z_out, clip_img_out, text_out = self.unet(
            z,
            clip_img,
            text=text,
            t_img=timesteps,
            t_text=timesteps,
            data_type=paddle.zeros_like(timesteps, dtype=paddle.int32) + data_type,
        )
        x_out = combine_joint(z_out, clip_img_out, text_out)

        if sample_scale == 0.0:
            return x_out

        z_noise = paddle.randn([x.shape[0], *z_shape])
        clip_img_noise = paddle.randn([x.shape[0], 1, clip_img_dim])
        text_noise = paddle.randn([x.shape[0], 77, text_dim])

        _, _, text_out_uncond = self.unet(
            z_noise,
            clip_img_noise,
            text=text,
            t_img=paddle.ones_like(timesteps) * N,
            t_text=timesteps,
            data_type=paddle.zeros_like(timesteps, dtype=paddle.int32) + data_type,
        )
        z_out_uncond, clip_img_out_uncond, _ = self.unet(
            z,
            clip_img,
            text=text_noise,
            t_img=timesteps,
            t_text=paddle.ones_like(timesteps) * N,
            data_type=paddle.zeros_like(timesteps, dtype=paddle.int32) + data_type,
        )

        x_out_uncond = combine_joint(z_out_uncond, clip_img_out_uncond, text_out_uncond)

        return x_out + sample_scale * (x_out - x_out_uncond)

    def sample_fn(self, z=None, clip_img=None, text=None):
        _n_samples = 1
        clip_img_dim = 512
        z_shape = (4, 64, 64)
        sample_steps = 50
        text_dim = 64

        _z_init = paddle.randn([_n_samples, *z_shape])
        _clip_img_init = paddle.randn([_n_samples, 1, clip_img_dim])
        _text_init = paddle.randn([_n_samples, 77, text_dim])

        _x_init = combine_joint(_z_init, _clip_img_init, _text_init)

        noise_schedule = NoiseScheduleVP(schedule="discrete", betas=paddle.to_tensor(_betas))

        def model_fn(x, t_continuous):
            t = t_continuous * N
            return self.joint_nnet(x, t)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        with paddle.no_grad():
            with paddle.amp.auto_cast():
                start_time = time.time()
                x = dpm_solver.sample(_x_init, steps=50, eps=1.0 / N, T=1.0)
                end_time = time.time()
                print(f"\ngenerate {_n_samples} samples with {sample_steps} steps takes {end_time - start_time:.2f}s")

        _z, _clip_img, _text = split_joint(x)
        return _z, _clip_img, _text

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        with paddle.amp.auto_cast():
            image = self.vae.decode(latents)
        # samples = unpreprocess(image)
        # image = samples[0]
        # image_ndarr = (image * 255 + 0.5).clip_(0, 255).transpose([1, 2, 0]).astype('uint8').numpy()
        # return image_ndarr

        image = (image / 2 + 0.5).clip(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.transpose([0, 2, 3, 1]).cast("float32").numpy()
        return image

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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
        shape = [batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor]
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
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @paddle.no_grad()
    def __call__(
        self,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,  # 7.5
        negative_prompt: Optional[Union[str, List[str]]] = None,
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
        # 0. Default height and width to unet
        # height = height #or self.image_unet.config.sample_size * self.vae_scale_factor
        # width = width #or self.image_unet.config.sample_size * self.vae_scale_factor

        # # 1. Check inputs. Raise error if not correct
        # self.check_inputs(prompt, height, width, callback_steps)

        # # 2. Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        # # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # # corresponds to doing no classifier free guidance.
        # do_classifier_free_guidance = guidance_scale > 1.0

        # # 3. Encode input prompt
        # text_embeddings = self._encode_text_prompt(
        #     prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        # )
        # contexts_low_dim = text_embeddings
        # _n_samples = contexts_low_dim.shape[0]

        # config = dict(
        #     n_samples=1,
        #     clip_img_dim=512,
        #     clip_text_dim=64,
        #     z_shape=(4, 64, 64),
        # )
        # contexts = paddle.randn([config.n_samples, 77, config.clip_text_dim])
        # img_contexts = paddle.randn([config.n_samples, 2 * config.z_shape[0], config.z_shape[1], config.z_shape[2]])
        # clip_imgs = paddle.randn([config.n_samples, 1, config.clip_img_dim])

        _z, _clip_img, _text = self.sample_fn()

        # samples = unpreprocess(decode(_z))
        # os.makedirs(os.path.join(config.output_path, config.mode), exist_ok=True)

        # 9. Post-processing
        image = self.decode_latents(_z)
        text = self.caption_decoder.generate_captions(_text)
        print(text)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, text)

        return JointPipelineOutput(images=image, texts=text)
