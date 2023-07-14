# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import AutoTokenizer, CLIPTextModel
from paddlenlp.utils.log import logger
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    is_ppxformers_available,
)
from ppdiffusers.initializer import reset_initialized_parameter, zeros_
from ppdiffusers.models.attention import AttentionBlock
from ppdiffusers.models.ema import LitEma
from ppdiffusers.models.resnet import ResnetBlock2D
from ppdiffusers.models.transformer_2d import Transformer2DModel
from ppdiffusers.training_utils import freeze_params


class StableDiffusionModel(nn.Layer):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args
        tokenizer_name_or_path = (
            model_args.tokenizer_name
            if model_args.tokenizer_name is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "tokenizer")
        )
        vae_name_or_path = (
            model_args.vae_name_or_path
            if model_args.vae_name_or_path is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "vae")
        )
        text_encoder_name_or_path = (
            model_args.text_encoder_name_or_path
            if model_args.text_encoder_name_or_path is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "text_encoder")
        )
        unet_name_or_path = (
            model_args.unet_name_or_path
            if model_args.unet_name_or_path is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "unet")
        )
        # init model and tokenizer
        tokenizer_kwargs = {}
        if model_args.model_max_length is not None:
            tokenizer_kwargs["model_max_length"] = model_args.model_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
        self.vae = AutoencoderKL.from_pretrained(vae_name_or_path)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name_or_path)
        try:
            self.unet = UNet2DConditionModel.from_pretrained(unet_name_or_path)
        except Exception:
            self.unet = UNet2DConditionModel.from_config(unet_name_or_path)
            self.init_unet_weights()
            logger.info("Init unet model from scratch!")

        freeze_params(self.vae.parameters())
        logger.info("Freeze vae parameters!")
        if not self.model_args.train_text_encoder:
            freeze_params(self.text_encoder.parameters())
            logger.info("Freeze text_encoder parameters!")
            self.text_encoder.eval()
            self.train_text_encoder = False
        else:
            self.text_encoder.train()
            self.train_text_encoder = True
        self.unet.train()
        self.vae.eval()

        # init noise_scheduler and eval_scheduler
        assert self.model_args.prediction_type in ["epsilon", "v_prediction"]
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=self.model_args.prediction_type,
        )
        self.register_buffer("alphas_cumprod", self.noise_scheduler.alphas_cumprod)
        self.eval_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type=self.model_args.prediction_type,
        )
        self.eval_scheduler.set_timesteps(self.model_args.num_inference_steps)
        self.use_ema = False
        self.model_ema = None

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        sqrt_alphas_cumprod = self.alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[timesteps].cast("float32")
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timesteps].cast("float32")
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        self.vae.eval()
        if not self.model_args.train_text_encoder:
            self.text_encoder.eval()

        # vae encode
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = paddle.randn(latents.shape)
        if self.model_args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.model_args.noise_offset * paddle.randn(
                (latents.shape[0], latents.shape[1], 1, 1), dtype=noise.dtype
            )
        if self.model_args.input_perturbation:
            new_noise = noise + self.model_args.input_perturbation * paddle.randn(noise.shape, dtype=noise.dtype)

        timesteps = paddle.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],)).cast(
            "int64"
        )
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        if self.model_args.input_perturbation:
            noisy_latents = self.add_noise(latents, new_noise, timesteps)
        else:
            noisy_latents = self.add_noise(latents, noise, timesteps)

        # text encode
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        # unet
        model_pred = self.unet(
            sample=noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states
        ).sample

        # Get the target for loss depending on the prediction type
        if self.model_args.prediction_type == "epsilon":
            target = noise
        elif self.model_args.prediction_type == "v_prediction":
            target = self.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.model_args.prediction_type}")

        # compute loss
        if self.model_args.snr_gamma is None:
            loss = (
                F.mse_loss(model_pred.cast("float32"), target.cast("float32"), reduction="none").mean([1, 2, 3]).mean()
            )
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                paddle.stack([snr, self.model_args.snr_gamma * paddle.ones_like(timesteps)], axis=1).min(axis=1)[0]
                / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.cast("float32"), target.cast("float32"), reduction="none")
            loss = loss.mean(list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        return loss

    def add_noise(
        self,
        original_samples: paddle.Tensor,
        noise: paddle.Tensor,
        timesteps: paddle.Tensor,
    ) -> paddle.Tensor:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(self, sample: paddle.Tensor, noise: paddle.Tensor, timesteps: paddle.Tensor) -> paddle.Tensor:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def init_unet_weights(self):
        reset_initialized_parameter(self.unet)
        zeros_(self.unet.conv_out.weight)
        zeros_(self.unet.conv_out.bias)
        for _, m in self.unet.named_sublayers():
            if isinstance(m, AttentionBlock):
                zeros_(m.proj_attn.weight)
                zeros_(m.proj_attn.bias)
            if isinstance(m, ResnetBlock2D):
                zeros_(m.conv2.weight)
                zeros_(m.conv2.bias)
            if isinstance(m, Transformer2DModel):
                zeros_(m.proj_out.weight)
                zeros_(m.proj_out.bias)

    @contextlib.contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.unet.parameters())
            self.model_ema.copy_to(self.unet)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.unet.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self):
        if self.use_ema:
            self.model_ema(self.unet)

    @paddle.no_grad()
    def decode_image(self, pixel_values=None, max_batch=8, **kwargs):
        self.eval()
        if pixel_values.shape[0] > max_batch:
            pixel_values = pixel_values[:max_batch]
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1])
        image = (image * 255.0).cast("float32").numpy().round()
        return image

    @paddle.no_grad()
    def log_image(self, input_ids=None, height=256, width=256, eta=0.0, guidance_scale=7.5, max_batch=8, **kwargs):
        self.eval()
        with self.ema_scope():
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
            # only log max_batch image
            if input_ids.shape[0] > max_batch:
                input_ids = input_ids[:max_batch]
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

            latents = paddle.randn((input_ids.shape[0], self.unet.in_channels, height // 8, width // 8))
            latents = latents * self.eval_scheduler.init_noise_sigma
            accepts_eta = "eta" in set(inspect.signature(self.eval_scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
            for t in self.eval_scheduler.timesteps:
                latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.eval_scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.eval_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            latents = 1 / self.vae.config.scaling_factor * latents
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1]) * 255.0
        return image.cast("float32").numpy().round()

    def set_recompute(self, use_recompute=False):
        if use_recompute:
            self.unet.enable_gradient_checkpointing()
            if self.model_args.train_text_encoder and hasattr(self.text_encoder, "gradient_checkpointing_enable"):
                self.text_encoder.gradient_checkpointing_enable()

    def gradient_checkpointing_enable(self):
        self.set_recompute(True)

    def set_xformers(self, use_xformers=False):
        if use_xformers:
            if not is_ppxformers_available():
                raise ValueError(
                    'Please run `python -m pip install "paddlepaddle-gpu>=2.5.0.post117" -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html first.'
                )
            else:
                try:
                    self.unet.enable_xformers_memory_efficient_attention()
                    if hasattr(self.vae, "enable_xformers_memory_efficient_attention"):
                        self.vae.enable_xformers_memory_efficient_attention()
                    if hasattr(self.text_encoder, "enable_xformers_memory_efficient_attention"):
                        self.text_encoder.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warn(
                        "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                        f" correctly and a GPU is available: {e}"
                    )

    def set_ema(self, use_ema=False):
        self.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(self.unet)
