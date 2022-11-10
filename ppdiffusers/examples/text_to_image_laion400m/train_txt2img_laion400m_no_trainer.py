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
import argparse
import time
import inspect
import itertools
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
import contextlib
import sys
from paddlenlp.utils.log import logger
from paddlenlp.trainer import set_seed
from ppdiffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
from ppdiffusers.optimization import get_scheduler

from ppdiffusers.modeling_utils import unwrap_model, freeze_params
from text_image_pair_dataset import TextImagePair, worker_init_fn
from paddle.optimizer import AdamW
from ppdiffusers.pipelines.latent_diffusion import LDMBertModel
from paddlenlp.transformers import AutoTokenizer


def get_writer(args):
    if args.writer_type == "visualdl":
        from visualdl import LogWriter
        writer = LogWriter(logdir=args.logging_dir)
    elif args.writer_type == "tensorboard":
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(logdir=args.logging_dir)
    else:
        raise ValueError("writer_type must be in ['visualdl', 'tensorboard']")
    return writer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training latent diffusion script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/ldm_laion400M_pretrain",
        required=False,
        help="Path to pretrained model or model identifier from local models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=77,
        help="Pretrained tokenizer model_max_length",
    )
    parser.add_argument("--file_list",
                        type=str,
                        default="./data/filelist/train.filelist.list",
                        required=False,
                        help="A train file list.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output-model",
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=23,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=
        "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"
    )
    parser.add_argument("--num_records",
                        type=int,
                        default=10000000,
                        help="num_records")
    parser.add_argument("--buffer_size",
                        type=int,
                        default=100,
                        help="buffer_size")
    parser.add_argument("--shuffle_every_n_samples",
                        type=int,
                        default=5,
                        help="shuffle_every_n_samples")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=6,
        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=10,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Number of save steps to save model.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Number of log steps to logging model results.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=200,
        help=
        "The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-05,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Dataloader num_workers.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=
        ('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
         ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100000000000,
        help=
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--adam_beta1",
                        type=float,
                        default=0.9,
                        help="The beta1 parameter for the AdamW optimizer.")
    parser.add_argument("--adam_beta2",
                        type=float,
                        default=0.999,
                        help="The beta2 parameter for the AdamW optimizer.")
    parser.add_argument("--adam_weight_decay",
                        type=float,
                        default=1e-2,
                        help="Weight decay to use.")
    parser.add_argument("--adam_epsilon",
                        type=float,
                        default=1e-08,
                        help="Epsilon value for the AdamW optimizer")
    parser.add_argument("--max_grad_norm",
                        default=None,
                        type=float,
                        help="Max gradient norm.")

    parser.add_argument(
        "--freeze_text_encoder",
        default=False,
        action="store_true",
        help="Flag to freeze text_encoder.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=
        ("[TensorBoard](https://www.tensorflow.org/tensorboard) or [VisualDL](https://www.paddlepaddle.org.cn/paddle/visualdl) log directory. Will default to"
         "*output_dir/logs"),
    )
    parser.add_argument("--writer_type",
                        type=str,
                        default="visualdl",
                        choices=["tensorboard", "visualdl"],
                        help="Log writer type.")
    args = parser.parse_args()
    return args


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
    args = parse_args()

    rank = paddle.distributed.get_rank()
    num_processes = paddle.distributed.get_world_size()

    if num_processes > 1:
        paddle.distributed.init_parallel_env()

    args.logging_dir = os.path.join(args.output_dir, args.logging_dir)

    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, model_max_length=args.model_max_length)
    text_encoder = LDMBertModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "ldmbert"))
    vqvae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                          subfolder="vqvae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    num_train_timesteps=1000)
    eval_scheduler = DDIMScheduler(beta_start=0.00085,
                                   beta_end=0.012,
                                   beta_schedule="scaled_linear",
                                   clip_sample=False,
                                   set_alpha_to_one=False)
    eval_scheduler.set_timesteps(args.num_inference_steps)
    model = LatentDiffusionModel(text_encoder, vqvae, unet, tokenizer,
                                 noise_scheduler, eval_scheduler,
                                 args.freeze_text_encoder)

    if args.freeze_text_encoder:
        params_to_train = model.unet.parameters()
    else:
        params_to_train = itertools.chain(model.text_encoder.parameters(),
                                          model.unet.parameters())

    if num_processes > 1:
        model = paddle.DataParallel(model)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps *
        args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps *
        args.gradient_accumulation_steps,
    )

    optimizer = AdamW(
        learning_rate=lr_scheduler,
        parameters=params_to_train,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        epsilon=args.adam_epsilon,
        grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm)
        if args.max_grad_norm is not None else None,
    )
    train_dataset = TextImagePair(
        file_list=args.file_list,
        size=args.resolution,
        num_records=args.num_records,
        buffer_size=args.buffer_size,
        shuffle_every_n_samples=args.shuffle_every_n_samples,
        interpolation="lanczos",
        tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn)

    # logger arguments
    logger.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        logger.info('%s: %s' % (arg, value))
    logger.info('------------------------------------------------')

    if rank == 0:
        writer = get_writer(args)

    # Train!
    total_batch_size = args.train_batch_size * num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    global_steps = 0
    tic_train = time.time()

    for epoch in range(args.num_train_epochs):
        if epoch == args.num_train_epochs:
            logger.info("***** Trainging Done *****")
            break

        for step, batch in enumerate(train_dataloader):
            if num_processes > 1 and (
                (step + 1) % args.gradient_accumulation_steps != 0):
                # grad acc, no_sync when (step + 1) % args.gradient_accumulation_steps != 0:
                # gradient_checkpointing, no_sync every where
                # gradient_checkpointing + grad_acc, no_sync every where
                ctx_manager = model.no_sync()
            else:
                ctx_manager = contextlib.nullcontext() if sys.version_info >= (
                    3, 7) else contextlib.suppress()

            with ctx_manager:
                loss = model(**batch)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                global_steps += 1

                # train log
                if rank == 0:
                    if global_steps % args.logging_steps == 0:
                        # add scalar
                        logs = {
                            "train/loss":
                            loss.item() * args.gradient_accumulation_steps,
                            "train/lr_abs": lr_scheduler.get_lr(),
                            "train/global_steps": global_steps
                        }
                        for name, val in logs.items():
                            writer.add_scalar(name, val, step=global_steps)
                        log_str = 'Train: global_steps {0:d}/{1:d}, epoch: {2:d}, batch: {3:d}, train_loss: {4:.10f}, lr_abs: {5:.10f}, speed: {6:.2f} s/it.'.format(
                            global_steps, args.max_train_steps, epoch, step + 1,
                            logs["train/loss"], logs["train/lr_abs"],
                            (time.time() - tic_train) / args.logging_steps)
                        logger.info(log_str)

                        if global_steps % (args.logging_steps * 20) == 0:
                            reconstruction_img = unwrap_model(
                                model).decode_image(
                                    pixel_values=batch["pixel_values"])
                            ddim_10_img = unwrap_model(model).log_image(
                                input_ids=batch["input_ids"],
                                guidance_scale=1.0)
                            ddim_75_img = unwrap_model(model).log_image(
                                input_ids=batch["input_ids"],
                                guidance_scale=7.5)
                            writer.add_image("reconstruction",
                                             reconstruction_img,
                                             global_steps,
                                             dataformats="NHWC")
                            writer.add_image("ddim-samples-1.0",
                                             ddim_10_img,
                                             global_steps,
                                             dataformats="NHWC")
                            writer.add_image("ddim-samples-7.5",
                                             ddim_75_img,
                                             global_steps,
                                             dataformats="NHWC")
                        tic_train = time.time()

                    if global_steps % args.save_steps == 0:
                        os.makedirs(os.path.join(
                            args.output_dir, f"global-steps-{global_steps}"),
                                    exist_ok=True)
                        paddle.save(
                            model.state_dict(),
                            os.path.join(args.output_dir,
                                         f"global-steps-{global_steps}",
                                         "model_state.pdparams"))

    if rank == 0:
        paddle.save(model.state_dict(),
                    os.path.join(args.output_dir, "model_state.pdparams"))
        writer.close()


if __name__ == "__main__":
    main()
