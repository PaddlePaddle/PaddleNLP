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
import contextlib
import hashlib
import itertools
import math
import os
import sys
import warnings
from pathlib import Path

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)
from paddle.io import BatchSampler, DataLoader, Dataset, DistributedBatchSampler
from paddle.optimizer import AdamW
from paddle.vision import transforms
from PIL import Image
from tqdm.auto import tqdm

from paddlenlp.trainer import set_seed
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.log import logger
from ppdiffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.modeling_utils import freeze_params, unwrap_model
from ppdiffusers.optimization import get_scheduler


def url_or_path_join(*path_list):
    return os.path.join(*path_list) if os.path.isdir(os.path.join(*path_list)) else "/".join(path_list)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    pipeline_config = DiffusionPipeline.load_config(pretrained_model_name_or_path)
    model_class = pipeline_config.get("text_encoder", pipeline_config.get("bert"))[-1]

    if model_class == "CLIPTextModel":
        from paddlenlp.transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from ppdiffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    elif model_class == "LDMBertModel":
        from ppdiffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import (
            LDMBertModel,
        )

        return LDMBertModel
    elif model_class == "BertModel":
        from paddlenlp.transformers import BertModel

        return BertModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def set_recompute(model, value=False):
    def fn(layer):
        # ldmbert
        if hasattr(layer, "enable_recompute"):
            layer.enable_recompute = value
            print("Set", layer.__class__, "recompute", layer.enable_recompute)
        # unet
        if hasattr(layer, "gradient_checkpointing"):
            layer.gradient_checkpointing = value
            print("Set", layer.__class__, "recompute", layer.gradient_checkpointing)

    model.apply(fn)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training dreambooth script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save pipe every X updates steps.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help=(
            "The height for input images, all the images in the train/validation dataset will be resized to this"
            " height"
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help=(
            "The width for input images, all the images in the train/validation dataset will be resized to this"
            " width"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) or [VisualDL](https://www.paddlepaddle.org.cn/paddle/visualdl) log directory. Will default to"
            "*output_dir/logs"
        ),
    )
    parser.add_argument(
        "--report_to", type=str, default="visualdl", choices=["tensorboard", "visualdl"], help="Log writer type."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    args.logging_dir = os.path.join(args.output_dir, args.logging_dir)
    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        height=512,
        width=512,
        center_crop=False,
        interpolation="bilinear",
    ):
        self.height = height
        self.width = width
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        ext = ["png", "jpg", "jpeg", "bmp", "PNG", "JPG", "JPEG", "BMP"]
        self.instance_images_path = []
        for p in Path(instance_data_root).iterdir():
            if any(suffix in p.name for suffix in ext):
                self.instance_images_path.append(p)
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = []
            for p in Path(class_data_root).iterdir():
                if any(suffix in p.name for suffix in ext):
                    self.class_images_path.append(p)
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=interpolation),
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_attention_mask=False,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_attention_mask=False,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_report_to(args):
    if args.report_to == "visualdl":
        from visualdl import LogWriter

        writer = LogWriter(logdir=args.logging_dir)
    elif args.report_to == "tensorboard":
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir=args.logging_dir)
    else:
        raise ValueError("report_to must be in ['visualdl', 'tensorboard']")
    return writer


def main():
    args = parse_args()
    rank = paddle.distributed.get_rank()
    is_main_process = rank == 0
    num_processes = paddle.distributed.get_world_size()
    if num_processes > 1:
        paddle.distributed.init_parallel_env()

    # If passed along, set the training seed now.
    if args.seed is not None:
        seed = args.seed + rank
        set_seed(seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
            )
            pipeline.safety_checker = None
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            batch_sampler = (
                DistributedBatchSampler(sample_dataset, batch_size=args.sample_batch_size, shuffle=False)
                if num_processes > 1
                else BatchSampler(sample_dataset, batch_size=args.sample_batch_size, shuffle=False)
            )
            sample_dataloader = DataLoader(sample_dataset, batch_sampler=batch_sampler)

            for example in tqdm(sample_dataloader, desc="Generating class images", disable=not is_main_process):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            import gc

            gc.collect()

    if is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        # support windows "\"
        tokenizer = AutoTokenizer.from_pretrained(url_or_path_join(args.pretrained_model_name_or_path, "tokenizer"))
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)
    text_encoder = text_encoder_cls.from_pretrained(
        url_or_path_join(args.pretrained_model_name_or_path, "text_encoder")
    )
    text_config = text_encoder.config if isinstance(text_encoder.config, dict) else text_encoder.config.to_dict()
    if text_config.get("use_attention_mask", None) is not None and text_config["use_attention_mask"]:
        use_attention_mask = True
    else:
        use_attention_mask = False
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    freeze_params(vae.parameters())
    if not args.train_text_encoder:
        freeze_params(text_encoder.parameters())
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            set_recompute(text_encoder, True)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * num_processes
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if num_processes > 1:
        unet = paddle.DataParallel(unet)
        if args.train_text_encoder:
            text_encoder = paddle.DataParallel(text_encoder)

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )

    optimizer = AdamW(
        learning_rate=lr_scheduler,
        parameters=params_to_optimize,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        epsilon=args.adam_epsilon,
        grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm) if args.max_grad_norm > 0 else None,
    )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        height=args.height,
        width=args.width,
        center_crop=args.center_crop,
        interpolation="bilinear",
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = paddle.stack(pixel_values).astype("float32")

        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pd"
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_sampler = (
        DistributedBatchSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        if num_processes > 1
        else BatchSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    )
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=1)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if is_main_process:
        logger.info("-----------  Configuration Arguments -----------")
        for arg, value in sorted(vars(args).items()):
            logger.info("%s: %s" % (arg, value))
        logger.info("------------------------------------------------")
        writer = get_report_to(args)

    # Train!
    total_batch_size = args.train_batch_size * num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Train Steps")
    global_step = 0

    # Keep vae in eval model as we don't train these
    vae.eval()
    if args.train_text_encoder:
        text_encoder.train()
    else:
        text_encoder.eval()
    unet.train()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = paddle.randn(latents.shape)
            batch_size = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = paddle.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,)).cast("int64")

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            if num_processes > 1 and (
                args.gradient_checkpointing or ((step + 1) % args.gradient_accumulation_steps != 0)
            ):
                # grad acc, no_sync when (step + 1) % args.gradient_accumulation_steps != 0:
                # gradient_checkpointing, no_sync every where
                # gradient_checkpointing + grad_acc, no_sync every where
                unet_ctx_manager = unet.no_sync()
                if args.train_text_encoder:
                    text_encoder_ctx_manager = text_encoder.no_sync()
                else:
                    text_encoder_ctx_manager = (
                        contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()
                    )
            else:
                unet_ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()
                text_encoder_ctx_manager = (
                    contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()
                )

            with text_encoder_ctx_manager:
                # Get the text embedding for conditioning
                if use_attention_mask:
                    attention_mask = (batch["input_ids"] != tokenizer.pad_token_id).cast("int64")
                else:
                    attention_mask = None
                encoder_hidden_states = text_encoder(batch["input_ids"], attention_mask=attention_mask)[0]

                with unet_ctx_manager:
                    # Predict the noise residual / sample
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = model_pred.chunk(2, axis=0)
                        target, target_prior = target.chunk(2, axis=0)

                        # Compute instance loss
                        loss = F.mse_loss(model_pred, target, reduction="mean")

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior, target_prior, reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(model_pred, target, reduction="mean")

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if num_processes > 1 and args.gradient_checkpointing:
                    fused_allreduce_gradients(params_to_optimize, None)
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                progress_bar.update(1)
                global_step += 1
                step_loss = loss.item() * args.gradient_accumulation_steps
                logs = {
                    "epoch": str(epoch).zfill(4),
                    "step_loss": round(step_loss, 10),
                    "lr": lr_scheduler.get_lr(),
                }
                progress_bar.set_postfix(**logs)
                if is_main_process:
                    for name, val in logs.items():
                        if name == "epoch":
                            continue
                        writer.add_scalar(f"train/{name}", val, step=global_step)

                    if global_step % args.save_steps == 0:
                        # Create the pipeline using using the trained modules and save it.
                        pipeline = DiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=unwrap_model(unet),
                            text_encoder=unwrap_model(text_encoder),
                        )
                        output_dir = os.path.join(args.output_dir, f"ckpt-{global_step}")
                        os.makedirs(output_dir, exist_ok=True)
                        pipeline.save_pretrained(output_dir)

            if global_step >= args.max_train_steps:
                break

    if is_main_process:
        writer.close()
        # Create the pipeline using using the trained modules and save it.
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unwrap_model(unet),
            text_encoder=unwrap_model(text_encoder),
        )
        pipeline.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
