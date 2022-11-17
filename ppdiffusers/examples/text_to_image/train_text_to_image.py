# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import argparse
import math
import os
import random

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import contextlib
import sys
from paddle.io import DataLoader, BatchSampler, DistributedBatchSampler

from paddlenlp.utils.log import logger
from paddlenlp.trainer import set_seed
from ppdiffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from ppdiffusers.optimization import get_scheduler
from ppdiffusers.modeling_utils import unwrap_model, freeze_params
from ppdiffusers.training_utils import main_process_first
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

from datasets import load_dataset

from paddle.vision import transforms, BaseTransform
from paddle.optimizer import AdamW
from tqdm.auto import tqdm
from paddlenlp.transformers import CLIPTextModel, AutoTokenizer, BertModel


class Lambda(BaseTransform):

    def __init__(self, fn, keys=None):
        super().__init__(keys)
        self.fn = fn

    def _apply_image(self, img):
        return self.fn(img)


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
        description="Simple example of a training a text to image model script."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=True,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=
        ("The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
         " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
         " or to a folder containing files that 🤗 Datasets can understand."),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help=
        "The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=
        ("A folder containing the training data. Folder contents must follow the structure described in"
         " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
         " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
         ),
    )
    parser.add_argument("--image_column",
                        type=str,
                        default="image",
                        help="The column of the dataset containing an image.")
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help=
        "The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=
        ("For debugging purposes or quicker training, truncate the number of training examples to this "
         "value if set."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=
        "The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=
        ("The resolution for input images, all the images in the train/validation dataset will be resized to this"
         " resolution"),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help=
        "Whether to center crop images before resizing to resolution (if not set, random crop will be used)",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=
        "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=
        "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
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
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.")

    parser.add_argument("--use_ema",
                        action="store_true",
                        help="Whether to use EMA model.")
    parser.add_argument("--adam_beta1",
                        type=float,
                        default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2",
                        type=float,
                        default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay",
                        type=float,
                        default=1e-2,
                        help="Weight decay to use.")
    parser.add_argument("--adam_epsilon",
                        type=float,
                        default=1e-08,
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")

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

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    args.logging_dir = os.path.join(args.output_dir, args.logging_dir)

    return args


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters, decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay

        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @paddle.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if not param.stop_gradient:
                s_param.copy_(s_param - self.decay * (s_param - param), True)
            else:
                s_param.copy_(param, True)

        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()

    def copy_to(self, parameters) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `paddle.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.copy_(s_param, True)


def main():
    args = parse_args()
    rank = paddle.distributed.get_rank()
    num_processes = paddle.distributed.get_world_size()
    if num_processes > 1:
        paddle.distributed.init_parallel_env()

    # If passed along, set the training seed now.
    if args.seed is not None:
        seed = args.seed + rank
        set_seed(seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load models and create wrapper for stable diffusion
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "tokenizer"))

    if "Taiyi-Stable-Diffusion-1B-Chinese-v0.1" in args.pretrained_model_name_or_path:
        model_cls = BertModel
    else:
        model_cls = CLIPTextModel
    text_encoder = model_cls.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "text_encoder"))
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                        subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet")

    # Freeze vae and text_encoder
    freeze_params(vae.parameters())
    freeze_params(text_encoder.parameters())

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (args.learning_rate *
                              args.gradient_accumulation_steps *
                              args.train_batch_size * num_processes)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps *
        args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps *
        args.gradient_accumulation_steps,
    )
    if num_processes > 1:
        unet = paddle.DataParallel(unet)

    # Initialize the optimizer
    optimizer = AdamW(learning_rate=lr_scheduler,
                      parameters=unet.parameters(),
                      beta1=args.adam_beta1,
                      beta2=args.adam_beta2,
                      weight_decay=args.adam_weight_decay,
                      epsilon=args.adam_epsilon,
                      grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm)
                      if args.max_grad_norm is not None else None)

    noise_scheduler = DDPMScheduler(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    num_train_timesteps=1000)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[
            0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[
            1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(
                    random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(captions,
                           max_length=tokenizer.model_max_length,
                           padding="do_not_pad",
                           truncation=True,
                           return_attention_mask=False)
        return inputs.input_ids

    train_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution),
                          interpolation="bilinear"),
        transforms.CenterCrop(args.resolution)
        if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip()
        if args.random_flip else Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(
                range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = paddle.stack(
            [example["pixel_values"] for example in examples])
        pixel_values = pixel_values.astype("float32")
        input_ids = [example["input_ids"] for example in examples]
        padded_tokens = tokenizer.pad({"input_ids": input_ids},
                                      padding="max_length",
                                      max_length=tokenizer.model_max_length,
                                      return_tensors="pd")
        return {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
        }

    train_sampler = DistributedBatchSampler(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=True) if num_processes > 1 else BatchSampler(
            train_dataset, batch_size=args.train_batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_sampler,
                                  collate_fn=collate_fn)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps /
                                      num_update_steps_per_epoch)

    if rank == 0:
        logger.info('-----------  Configuration Arguments -----------')
        for arg, value in sorted(vars(args).items()):
            logger.info('%s: %s' % (arg, value))
        logger.info('------------------------------------------------')
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
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=rank > 0)
    progress_bar.set_description("Steps")
    global_step = 0

    # Keep vae and text_encoder in eval model as we don't train these
    vae.eval()
    text_encoder.eval()
    unet.train()

    for epoch in range(args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = paddle.randn(latents.shape)
            batch_size = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = paddle.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size, )).astype("int64")

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            attention_mask = paddle.ones_like(batch["input_ids"])
            encoder_hidden_states = text_encoder(
                batch["input_ids"], attention_mask=attention_mask)[0]

            if num_processes > 1 and (args.gradient_checkpointing or (
                (step + 1) % args.gradient_accumulation_steps != 0)):
                # grad acc, no_sync when (step + 1) % args.gradient_accumulation_steps != 0:
                # gradient_checkpointing, no_sync every where
                # gradient_checkpointing + grad_acc, no_sync every where
                ctx_manager = unet.no_sync()
            else:
                ctx_manager = contextlib.nullcontext() if sys.version_info >= (
                    3, 7) else contextlib.suppress()

            with ctx_manager:
                # Predict the noise residual and compute loss
                noise_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred, noise, reduction="mean")

                train_loss += loss.item()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if num_processes > 1 and args.gradient_checkpointing:
                    fused_allreduce_gradients(unet.parameters(), None)
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                progress_bar.update(1)
                global_step += 1

                if args.use_ema:
                    ema_unet.step(unet.parameters())
                logs = {
                    "epoch":
                    str(epoch).zfill(4),
                    "train_loss":
                    "{0:.10f}".format(train_loss),
                    "step_loss":
                    "{0:.10f}".format(loss.item() *
                                      args.gradient_accumulation_steps),
                    "lr":
                    lr_scheduler.get_lr()
                }
                progress_bar.set_postfix(**logs)
                train_loss = 0.0
                if rank == 0:
                    for name, val in logs.items():
                        if name == "epoch": continue
                        writer.add_scalar(f"train/{name}",
                                          val,
                                          step=global_step)

            if global_step >= args.max_train_steps:
                break

    if rank == 0:
        writer.close()
        unet = unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        # Create the pipeline using using the trained modules and save it.
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            safety_checker=None,
        )
        pipeline.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
