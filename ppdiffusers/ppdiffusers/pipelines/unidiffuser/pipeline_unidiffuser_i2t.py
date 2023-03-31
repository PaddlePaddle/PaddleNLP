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

from paddlenlp.transformers import CLIPModel, CLIPProcessor

from ...models import CaptionDecoder, FrozenAutoencoderKL, UViT
from ...pipeline_utils import DiffusionPipeline, TextPipelineOutput
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from ...utils import logging
from .dpm_solver_pp import DPM_Solver, NoiseScheduleVP
from .unidiffuser_common import (
    center_crop,
    combine,
    combine_joint,
    split,
    split_joint,
    stable_diffusion_beta_schedule,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_betas = stable_diffusion_beta_schedule()
N = len(_betas)


class UniDiffuserImageToTextPipeline(DiffusionPipeline):

    image_encoder: CLIPModel  # clip_img_model clip_img_model_preprocess = CLIPProcessor.from_pretrained(model_name)
    image_feature_extractor: CLIPProcessor
    unet: UViT
    vae: FrozenAutoencoderKL
    caption_decoder: CaptionDecoder
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]

    def __init__(
        self,
        image_encoder: CLIPModel,
        image_feature_extractor: CLIPProcessor,
        unet: UViT,
        vae: FrozenAutoencoderKL,
        caption_decoder: CaptionDecoder,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()
        self.register_modules(
            image_encoder=image_encoder,
            image_feature_extractor=image_feature_extractor,
            unet=unet,
            vae=vae,
            caption_decoder=caption_decoder,
            scheduler=scheduler,
        )

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

    def prepare_contexts(self, raw_image, clip_img_model, clip_img_model_preprocess, autoencoder):
        n_samples = 1
        z_shape = (4, 64, 64)
        resolution = z_shape[-1] * 8

        from PIL import Image

        img_contexts = []
        clip_imgs = []

        def get_img_feature(image):
            image = np.array(image).astype(np.uint8)
            image = center_crop(resolution, resolution, image)
            inputs = clip_img_model_preprocess(images=Image.fromarray(image), return_tensors="pd")
            clip_img_feature = clip_img_model.get_image_features(**inputs)

            image = (image / 127.5 - 1.0).astype(np.float32)
            image = einops.rearrange(image, "h w c -> 1 c h w")
            image = paddle.to_tensor(image)
            moments = autoencoder.encode_moments(image)

            return clip_img_feature, moments

        image = Image.open(raw_image).convert("RGB")
        clip_img, img_context = get_img_feature(image)

        img_contexts.append(img_context)
        clip_imgs.append(clip_img)
        img_contexts = img_contexts * n_samples
        clip_imgs = clip_imgs * n_samples

        img_contexts = paddle.concat(img_contexts, axis=0)
        clip_imgs = paddle.stack(clip_imgs, axis=0)

        return img_contexts, clip_imgs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        with paddle.amp.auto_cast():
            image = self.vae.decode(latents)

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
        image: Union[paddle.Tensor, PIL.Image.Image],
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

        img_contexts, clip_imgs = self.prepare_contexts(
            raw_image=image,
            clip_img_model=self.image_encoder,
            clip_img_model_preprocess=self.image_feature_extractor,
            autoencoder=self.vae,
        )

        z_img = self.vae.sample(img_contexts)

        _text = self.sample_fn("i2t", z=z_img, clip_img=clip_imgs)
        text = self.caption_decoder.generate_captions(_text)
        print(text)

        if not return_dict:
            return (text,)

        return TextPipelineOutput(texts=text)
