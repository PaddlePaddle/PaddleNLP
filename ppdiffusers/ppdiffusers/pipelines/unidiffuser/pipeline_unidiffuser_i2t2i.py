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

from paddlenlp.transformers import CLIPFeatureExtractor, CLIPVisionModelWithProjection

from ...models import AutoencoderKL, UViTModel
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from ...utils import logging
from .dpm_solver_pp import DPM_Solver, NoiseScheduleVP
from .unidiffuser_common import (
    combine,
    combine_joint,
    split,
    split_joint,
    stable_diffusion_beta_schedule,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_betas = stable_diffusion_beta_schedule()
N = len(_betas)


class UniDiffuserImageVariationPipeline(DiffusionPipeline):

    image_encoder: CLIPVisionModelWithProjection
    image_feature_extractor: CLIPFeatureExtractor
    unet: UViTModel
    vae: AutoencoderKL
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]

    def __init__(
        self,
        image_encoder: CLIPVisionModelWithProjection,
        image_feature_extractor: CLIPFeatureExtractor,
        unet: UViTModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()
        self.register_modules(
            image_encoder=image_encoder,
            image_feature_extractor=image_feature_extractor,
            unet=unet,
            vae=vae,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def i2t_nnet(self, x, timesteps, z, clip_img):
        sample_scale = 7
        data_type = 1

        t_img = paddle.zeros([timesteps.shape[0]], dtype=paddle.int32)

        z_out, clip_img_out, text_out = self.unet(
            z,
            clip_img,
            text=x,
            t_img=t_img,
            t_text=timesteps,
            data_type=paddle.zeros_like(t_img, dtype=paddle.int32) + data_type,
        )

        if sample_scale == 0.0:
            return text_out

        z_N = paddle.randn(z.shape)  # 3 other possible choices
        clip_img_N = paddle.randn(clip_img.shape)
        z_out_uncond, clip_img_out_uncond, text_out_uncond = self.unet(
            z_N,
            clip_img_N,
            text=x,
            t_img=paddle.ones_like(timesteps) * N,
            t_text=timesteps,
            data_type=paddle.zeros_like(timesteps, dtype=paddle.int32) + data_type,
        )
        return text_out + sample_scale * (text_out - text_out_uncond)

    def t2i_nnet(self, x, timesteps, text):  # text is the low dimension version of the text clip embedding
        data_type = 1
        sample_scale = 7

        z, clip_img = split(x)
        t_text = paddle.zeros([timesteps.shape[0]], dtype=paddle.int32)
        z_out, clip_img_out, text_out = self.unet(
            z,
            clip_img,
            text=text,
            t_img=timesteps,
            t_text=t_text,
            data_type=paddle.zeros_like(t_text, dtype=paddle.int32) + data_type,
        )
        x_out = combine(z_out, clip_img_out)

        if sample_scale == 0.0:
            return x_out

        text_N = paddle.randn(text.shape)
        z_out_uncond, clip_img_out_uncond, text_out_uncond = self.unet(
            z,
            clip_img,
            text=text_N,
            t_img=timesteps,
            t_text=paddle.ones_like(timesteps) * N,
            data_type=paddle.zeros_like(t_text, dtype=paddle.int32) + data_type,
        )
        x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)

        return x_out + sample_scale * (x_out - x_out_uncond)

    def sample_fn(self, mode, z=None, clip_img=None, text=None):
        _n_samples = 1
        clip_img_dim = 512
        z_shape = (4, 64, 64)
        sample_steps = 50
        text_dim = 64

        _z_init = paddle.randn([_n_samples, *z_shape])
        _clip_img_init = paddle.randn([_n_samples, 1, clip_img_dim])
        _text_init = paddle.randn([_n_samples, 77, text_dim])
        if mode == "joint":
            _x_init = combine_joint(_z_init, _clip_img_init, _text_init)
        elif mode in ["t2i", "i"]:
            _x_init = combine(_z_init, _clip_img_init)
        elif mode in ["i2t", "t"]:
            _x_init = _text_init

        noise_schedule = NoiseScheduleVP(schedule="discrete", betas=paddle.to_tensor(_betas))

        def model_fn(x, t_continuous):
            t = t_continuous * N
            if mode == "t2i":
                return self.t2i_nnet(x, t, text)
            elif mode == "i2t":
                return self.i2t_nnet(x, t, z, clip_img)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        with paddle.no_grad():
            with paddle.amp.auto_cast():
                start_time = time.time()
                x = dpm_solver.sample(_x_init, steps=50, eps=1.0 / N, T=1.0)
                end_time = time.time()
                print(f"\ngenerate {_n_samples} samples with {sample_steps} steps takes {end_time - start_time:.2f}s")

        if mode == "joint":
            _z, _clip_img, _text = split_joint(x)
            return _z, _clip_img, _text
        elif mode in ["t2i", "i"]:
            _z, _clip_img = split(x)
            return _z, _clip_img
        elif mode in ["i2t", "t"]:
            return x

    def _encode_image(self, image, num_images_per_prompt):
        dtype = self.image_encoder.dtype

        if not isinstance(image, paddle.Tensor):
            image = self.image_feature_extractor(images=image, return_tensors="pd").pixel_values

        image = image.cast(dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.tile([1, num_images_per_prompt, 1])
        image_embeddings = image_embeddings.reshape([bs_embed * num_images_per_prompt, seq_len, -1])
        return image_embeddings

    def _encode_image_contexts(self, image, num_images_per_prompt):
        img_contexts = []
        # image = load_image(raw_image)
        image = np.array(image)  # .astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, "h w c -> 1 c h w")  # (1, 3, 512, 512)
        moments = self.vae.encode(paddle.to_tensor(image)).latent_dist.sample()  # encode_moments
        moments = moments * self.vae.scaling_factor
        img_contexts.append(moments)
        img_contexts = img_contexts * num_images_per_prompt
        img_contexts = paddle.concat(img_contexts, axis=0)
        return img_contexts

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        with paddle.amp.auto_cast():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
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

    def check_inputs(self, image, height, width, callback_steps):
        if (
            not isinstance(image, paddle.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `paddle.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

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
        image: Union[paddle.Tensor, PIL.Image.Image],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
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
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width, callback_steps)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]

        # 3. Encode input image
        clip_imgs = self._encode_image(image, num_images_per_prompt)
        z_img = self._encode_image_contexts(image, num_images_per_prompt)

        _text = self.sample_fn("i2t", z=z_img, clip_img=clip_imgs)

        _z, _clip_img = self.sample_fn("t2i", text=_text)
        image = self.decode_latents(_z)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
