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

import os
import inspect
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.trainer import PdArgumentParser, TrainingArguments, get_last_checkpoint
from paddlenlp.transformers import AutoTokenizer, CLIPTextModel
from paddlenlp.utils.log import logger

from text_image_pair_dataset import TextImagePair
from ldm_args import DataArguments, ModelArguments
from ldm_trainer import LatentDiffusionTrainer

from ppdiffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
from ppdiffusers.pipelines.latent_diffusion import LDMBertModel


def freeze_params(params):
    for param in params:
        param.stop_gradient = True


class LatentDiffusionModel(nn.Layer):

    def __init__(self,
                 text_encoder,
                 vqvae,
                 unet,
                 tokenizer,
                 noise_scheduler,
                 eval_scheduler,
                 freeze_text_encoder=False):
        super().__init__()
        self.text_encoder = text_encoder
        self.vqvae = vqvae
        self.unet = unet
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.eval_scheduler = eval_scheduler
        self.freeze_text_encoder = freeze_text_encoder
        freeze_params(self.vqvae.parameters())
        print("Freeze vqvae!")
        if freeze_text_encoder:
            freeze_params(self.text_encoder.parameters())
            print("Freeze text_encoder!")

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        self.train()
        with paddle.no_grad():
            self.vqvae.eval()
            latents = self.vqvae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215
            noise = paddle.randn(latents.shape)
            timesteps = paddle.randint(0,
                                       self.noise_scheduler.num_train_timesteps,
                                       (latents.shape[0], )).astype("int64")
            noisy_latents = self.noise_scheduler.add_noise(
                latents, noise, timesteps)

        if self.freeze_text_encoder:
            self.text_encoder.eval()
            with paddle.no_grad():
                encoder_hidden_states = self.text_encoder(input_ids)[0]
        else:
            encoder_hidden_states = self.text_encoder(input_ids)[0]
        noise_pred = self.unet(noisy_latents, timesteps,
                               encoder_hidden_states).sample
        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2,
                                                                     3]).mean()
        return loss

    @paddle.no_grad()
    def decode_image(self, pixel_values=None):
        self.eval()
        if pixel_values.shape[0] > 8:
            pixel_values = pixel_values[:8]
        latents = self.vqvae.encode(pixel_values).latent_dist.sample()
        image = self.vqvae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1])
        image = (image * 255.).numpy().round()
        return image

    @paddle.no_grad()
    def log_image(self,
                  input_ids=None,
                  height=256,
                  width=256,
                  eta=1.0,
                  guidance_scale=7.5,
                  **kwargs):
        self.eval()
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )
        # only log 8 image
        if input_ids.shape[0] > 8:
            input_ids = input_ids[:8]

        text_embeddings = self.text_encoder(input_ids)[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            batch_size, max_length = input_ids.shape
            uncond_input = self.tokenizer([""] * batch_size,
                                          padding="max_length",
                                          truncation=True,
                                          max_length=max_length,
                                          return_tensors="pd")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
            text_embeddings = paddle.concat(
                [uncond_embeddings, text_embeddings], axis=0)

        latents = paddle.randn((input_ids.shape[0], self.unet.in_channels,
                                height // 8, width // 8))
        # ddim donot use this
        latents = latents * self.eval_scheduler.init_noise_sigma

        accepts_eta = "eta" in set(
            inspect.signature(self.eval_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for t in self.eval_scheduler.timesteps:
            # expand the latents if we are doing classifier free guidance
            latent_model_input = paddle.concat(
                [latents] * 2) if do_classifier_free_guidance else latents
            # ddim donot use this
            latent_model_input = self.eval_scheduler.scale_model_input(
                latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input,
                                   t,
                                   encoder_hidden_states=text_embeddings).sample
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.eval_scheduler.step(noise_pred, t, latents,
                                               **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vqvae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1]) * 255.
        return image.numpy().round()


def main():
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    # donot use max_grad_norm
    training_args.max_grad_norm = None

    paddle.set_device(training_args.device)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir
    ) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(
                training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if model_args.pretrained_text_encoder_name is not None:
        text_encoder = CLIPTextModel.from_pretrained(
            model_args.model_name_or_path)
        training_args.freeze_text_encoder = True
        if model_args.tokenizer_name is None:
            model_args.tokenizer_name = model_args.model_name_or_path
    else:
        text_encoder = LDMBertModel.from_pretrained(
            os.path.join(model_args.model_name_or_path, "ldmbert"))
        training_args.freeze_text_encoder = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name, model_max_length=model_args.model_max_length)
    train_dataset = TextImagePair(
        file_list=data_args.file_list,
        size=data_args.resolution,
        num_records=data_args.num_records,
        buffer_size=data_args.buffer_size,
        shuffle_every_n_samples=data_args.shuffle_every_n_samples,
        interpolation="lanczos",
        tokenizer=tokenizer)
    vqvae = AutoencoderKL.from_pretrained(model_args.model_name_or_path,
                                          subfolder="vqvae")
    unet = UNet2DConditionModel.from_pretrained(model_args.model_name_or_path,
                                                subfolder="unet")
    noise_scheduler = DDPMScheduler(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    num_train_timesteps=1000)
    eval_scheduler = DDIMScheduler(beta_start=0.00085,
                                   beta_end=0.012,
                                   beta_schedule="scaled_linear",
                                   clip_sample=False,
                                   set_alpha_to_one=False)
    eval_scheduler.set_timesteps(model_args.num_inference_steps)
    model = LatentDiffusionModel(text_encoder, vqvae, unet, tokenizer,
                                 noise_scheduler, eval_scheduler,
                                 training_args.freeze_text_encoder)

    trainer = LatentDiffusionTrainer(model=model,
                                     args=training_args,
                                     train_dataset=train_dataset,
                                     tokenizer=tokenizer)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Training
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
