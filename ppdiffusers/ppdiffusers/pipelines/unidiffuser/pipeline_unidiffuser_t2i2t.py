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

import paddle

from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer

from ...models import UViTModel
from ...pipeline_utils import DiffusionPipeline, TextPipelineOutput
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from ...utils import logging
from .caption_decoder import CaptionDecoder
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


class UniDiffuserTextVariationPipeline(DiffusionPipeline):

    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    unet: UViTModel
    caption_decoder: CaptionDecoder
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]

    def __init__(
        self,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UViTModel,
        caption_decoder: CaptionDecoder,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()
        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            caption_decoder=caption_decoder,
            scheduler=scheduler,
        )
        self.use_caption_decoder = True

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

        z_N = paddle.randn(z.shape)
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
        prompt: Union[str, List[str]],
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

        n_samples = 1
        clip_text_dim = 64

        contexts = paddle.randn([n_samples, 77, clip_text_dim])
        prompts = [prompt] * n_samples

        batch_encoding = self.tokenizer(
            prompts,
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pd",
        )
        outputs = self.text_encoder(input_ids=batch_encoding["input_ids"])
        contexts = outputs.last_hidden_state

        contexts_low_dim = self.caption_decoder.encode_prefix(contexts)
        _z, _clip_img = self.sample_fn("t2i", text=contexts_low_dim)

        _text = self.sample_fn("i2t", z=_z, clip_img=_clip_img)

        text = self.caption_decoder.generate_captions(_text)

        if not return_dict:
            return (text,)

        return TextPipelineOutput(texts=text)
