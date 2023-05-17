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

import contextlib
import inspect
import json
import os

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import AutoTokenizer, CLIPTextModel
from paddlenlp.utils.log import logger
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    LDMBertModel,
    T2IAdapter,
    UNet2DConditionModel,
    is_ppxformers_available,
)

# from ppdiffusers.initializer import reset_initialized_parameter
from ppdiffusers.models.ema import LitEma
from ppdiffusers.training_utils import freeze_params

from .annotator_utils import create_annotator


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


generator = np.random.RandomState(42)


class AdapterLDM(nn.Layer):
    def __init__(self, model_args):
        super().__init__()
        # init control image processor
        self.control_image_processor = create_annotator(model_args.control_type)

        # init tokenizer
        tokenizer_name_or_path = (
            model_args.tokenizer_name
            if model_args.pretrained_model_name_or_path is None
            else os.path.join(model_args.pretrained_model_name_or_path, "tokenizer")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, model_max_length=model_args.model_max_length
        )

        vae_name = "vqvae" if model_args.is_ldmbert else "vae"
        # init vae
        vae_name_or_path = (
            model_args.vae_name_or_path
            if model_args.pretrained_model_name_or_path is None
            else os.path.join(model_args.pretrained_model_name_or_path, vae_name)
        )

        self.vae = AutoencoderKL.from_pretrained(vae_name_or_path)
        freeze_params(self.vae.parameters())
        logger.info("Freeze vae parameters!")

        if model_args.is_ldmbert:
            text_encoder_name_or_path = (
                model_args.text_encoder_name_or_path
                if model_args.pretrained_model_name_or_path is None
                else os.path.join(model_args.pretrained_model_name_or_path, "bert")
            )
            # init text_encoder
            self.text_encoder = LDMBertModel.from_pretrained(text_encoder_name_or_path)
        else:
            text_encoder_name_or_path = (
                model_args.text_encoder_name_or_path
                if model_args.pretrained_model_name_or_path is None
                else os.path.join(model_args.pretrained_model_name_or_path, "text_encoder")
            )
            self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name_or_path)

        freeze_params(self.text_encoder.parameters())
        logger.info("Freeze text_encoder parameters!")

        unet_name_or_path = (
            model_args.unet_name_or_path
            if model_args.pretrained_model_name_or_path is None
            else os.path.join(model_args.pretrained_model_name_or_path, "unet")
        )

        self.unet = UNet2DConditionModel.from_pretrained(unet_name_or_path)

        freeze_params(self.unet.parameters())
        logger.info("Freeze unet parameters!")

        if model_args.pretrained_adapter_name_or_path:
            self.adapter = T2IAdapter.from_pretrained(model_args.pretrained_adapter_name_or_path)
        else:
            self.adapter = T2IAdapter(**read_json(model_args.adapter_config_file))

        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )
        self.eval_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        self.eval_scheduler.set_timesteps(model_args.num_inference_steps)
        self.use_ema = model_args.use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.adapter)
        self.adapter_conditioning_scale = 1.0

        if model_args.enable_xformers_memory_efficient_attention and is_ppxformers_available():
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                self.adapter.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warn(
                    "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                    f" correctly and a GPU is available: {e}"
                )
        self.use_preconfig_latents = False
        if model_args.latents_path:
            self.use_preconfig_latents = True
            self.register_buffer("preconfig_latents", paddle.load(model_args.latents_path))
        self.random_alignment = model_args.random_alignment

    @contextlib.contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.controlnet.parameters())
            self.model_ema.copy_to(self.controlnet)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.controlnet.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self):
        if self.use_ema:
            self.model_ema(self.adapter)

    def forward(self, input_ids=None, pixel_values=None, adapter_cond=None, **kwargs):
        with paddle.no_grad():
            adapter_cond = self.control_image_processor.process_model_forward(adapter_cond)
        self.train()
        with paddle.amp.auto_cast(enable=False):
            with paddle.no_grad():
                self.vae.eval()
                self.text_encoder.eval()
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215
                if self.random_alignment:
                    timesteps = paddle.to_tensor(
                        generator.randint(0, self.noise_scheduler.num_train_timesteps, size=(latents.shape[0],)),
                        dtype="int64",
                    )
                    noise = paddle.to_tensor(generator.randn(*latents.shape), dtype="float32")
                else:
                    timesteps = paddle.randint(
                        0, self.noise_scheduler.num_train_timesteps, (latents.shape[0],)
                    ).astype("int64")
                    noise = paddle.randn(latents.shape)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = self.text_encoder(input_ids)[0]
        adapter_state = self.adapter(adapter_cond)

        for k, v in enumerate(adapter_state):
            adapter_state[k] = v * self.adapter_conditioning_scale

        # predict the noise residual
        noise_pred = self.unet(
            noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=adapter_state,
        ).sample
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        return loss

    @paddle.no_grad()
    def decode_image(self, pixel_values=None, **kwargs):
        self.eval()
        if pixel_values.shape[0] > 8:
            pixel_values = pixel_values[:8]
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1])
        image = (image * 255.0).cast("float32").numpy().round()
        return image

    @paddle.no_grad()
    def decode_control_image(self, adapter_cond=None, **kwargs):
        adapter_cond = self.control_image_processor.process_model_forward(adapter_cond)  # (0, 1)
        return 255 * (adapter_cond.transpose([0, 2, 3, 1])).cast("float32").numpy().round()

    @paddle.no_grad()
    def log_image(self, input_ids=None, adapter_cond=None, height=512, width=512, eta=0.0, guidance_scale=9, **kwargs):
        adapter_cond = self.control_image_processor.process_model_forward(adapter_cond)
        self.eval()
        with self.ema_scope():
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
            # only log 8 image
            if input_ids.shape[0] > 4:
                input_ids = input_ids[:4]

            text_embeddings = self.text_encoder(input_ids)[0]
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                batch_size, max_length = input_ids.shape
                uncond_input = self.tokenizer(
                    [""] * batch_size,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pd",
                )
                uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
                text_embeddings = paddle.concat([uncond_embeddings, text_embeddings], axis=0)
            if self.use_preconfig_latents:
                latents = self.preconfig_latents
            else:
                latents = paddle.randn((input_ids.shape[0], self.unet.in_channels, height // 8, width // 8))
            # ddim donot use this
            latents = latents * self.eval_scheduler.init_noise_sigma

            accepts_eta = "eta" in set(inspect.signature(self.eval_scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            for t in self.eval_scheduler.timesteps:
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents

                # ddim donot use this
                latent_model_input = self.eval_scheduler.scale_model_input(latent_model_input, t)

                # Adapter predict the noise residual
                adapter_state = self.adapter(adapter_cond)
                for k, v in enumerate(adapter_state):
                    adapter_state[k] = v * self.adapter_conditioning_scale
                if do_classifier_free_guidance:
                    for k, v in enumerate(adapter_state):
                        adapter_state[k] = paddle.concat(x=[v] * 2, axis=0)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=[state.clone() for state in adapter_state],
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.eval_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1]) * 255.0

        return image.cast("float32").numpy().round()

    def set_recompute(self, value=False):
        def fn(layer):
            if hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = value
                print("Set", layer.__class__, "recompute", layer.gradient_checkpointing)

        self.adapter.apply(fn)
