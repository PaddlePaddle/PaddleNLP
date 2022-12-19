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
import glob
import itertools
import math
import os
import random
from pathlib import Path

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import PIL
from paddle.io import BatchSampler, DataLoader, Dataset, DistributedBatchSampler
from PIL import Image

from paddlenlp.trainer import set_seed
from paddlenlp.utils.log import logger
from paddlenlp.utils.tools import compare_version
from ppdiffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.modeling_utils import freeze_params, unwrap_model
from ppdiffusers.optimization import get_scheduler

if compare_version(PIL.__version__, "9.1.0") >= 0:
    Resampling = PIL.Image.Resampling
else:
    Resampling = PIL.Image
from paddle.optimizer import AdamW
from paddle.vision.transforms import RandomHorizontalFlip
from tqdm.auto import tqdm

from paddlenlp.transformers import AutoTokenizer, BertModel, CLIPTextModel


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


def save_progress(text_encoder, placeholder_token_id, args):
    logger.info("Saving embeddings")
    learned_embeds = unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
    paddle.save(learned_embeds_dict, os.path.join(args.output_dir, "learned_embeds.pdparams"))


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.pdparams every X updates steps.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=True,
        help="Path to pretrained model or model identifier from local models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
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
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
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
        default=True,
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
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=None, type=float, help="Max gradient norm.")

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
        "--writer_type", type=str, default="visualdl", choices=["tensorboard", "visualdl"], help="Log writer type."
    )
    parser.add_argument("--language", default="en", choices=["en", "zh", "zh_en"], help="Model language.")

    args = parser.parse_args()

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.language == "en":
        if "chinese-en" in args.pretrained_model_name_or_path.lower():
            args.language = "zh_en"
            logger.info("Detect Chinese-English Model, we will set language to 'zh_en'. ")
        elif "chinese" in args.pretrained_model_name_or_path.lower():
            args.language = "zh"
            logger.info("Detect Chinese Model, we will set language to 'zh'. ")

    args.logging_dir = os.path.join(args.output_dir, args.logging_dir)
    return args


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

zh_imagenet_templates_small = [
    "一张{}的照片",
    "{}的渲染",
    "{}裁剪过的照片",
    "一张干净的{}的照片",
    "{}的黑暗照片",
    "我的{}的照片",
    "酷的{}的照片",
    "{}的特写照片",
    "{}的明亮照片",
    "{}的裁剪照片",
    "{}的照片",
    "{}的好照片",
    "一张{}的照片",
    "干净的照片{}",
    "一张漂亮的{}的照片",
    "漂亮的照片{}",
    "一张很酷的照片{}",
    "一张奇怪的照片{}",
]

zh_imagenet_style_templates_small = [
    "一幅{}风格的画",
    "{}风格的渲染",
    "{}风格的裁剪画",
    "{}风格的绘画",
    "{}风格的一幅干净的画",
    "{}风格的黑暗画作",
    "{}风格的图片",
    "{}风格的一幅很酷的画",
    "{}风格的特写画",
    "一幅{}风格的明亮画作",
    "{}风格的一幅好画",
    "{}风格的特写画",
    "{}风格的艺术画",
    "一幅{}风格的漂亮画",
    "一幅{}风格的奇怪的画",
]


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        height=512,
        width=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        language="en",
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.height = height
        self.width = width
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        if not Path(data_root).exists():
            raise ValueError(f"{data_root} dir doesn't exists.")

        ext = ["png", "jpg", "jpeg", "bmp", "PNG", "JPG", "JPEG", "BMP"]
        self.image_paths = []
        for e in ext:
            self.image_paths.extend(glob.glob(os.path.join(data_root, "*." + e)))

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": Resampling.BILINEAR,
            "bilinear": Resampling.BILINEAR,
            "bicubic": Resampling.BICUBIC,
            "lanczos": Resampling.LANCZOS,
        }[interpolation]

        self.templates = []
        if learnable_property == "style":
            if "en" in language:
                self.templates.extend(imagenet_style_templates_small)
            if "zh" in language:
                self.templates.extend(zh_imagenet_style_templates_small)
        else:
            if "en" in language:
                self.templates.extend(imagenet_templates_small)
            if "zh" in language:
                self.templates.extend(zh_imagenet_templates_small)

        self.flip_transform = RandomHorizontalFlip(prob=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.width, self.height), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32).transpose([2, 0, 1])

        example["pixel_values"] = image
        return example


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

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.pretrained_model_name_or_path, "tokenizer"))

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)["input_ids"]
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)

    # Load models and create wrapper for stable diffusion
    if "Taiyi-Stable-Diffusion-1B-Chinese-v0.1" in args.pretrained_model_name_or_path:
        model_cls = BertModel
    else:
        model_cls = CLIPTextModel
    text_encoder = model_cls.from_pretrained(os.path.join(args.pretrained_model_name_or_path, "text_encoder"))
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    with paddle.no_grad():
        token_embeds = text_encoder.get_input_embeddings()
        token_embeds.weight[placeholder_token_id] = token_embeds.weight[initializer_token_id]

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())

    # Freeze all parameters except for the token embeddings in text encoder
    if isinstance(text_encoder, BertModel):
        # bert text_encoder
        params_to_freeze = itertools.chain(
            text_encoder.encoder.parameters(),
            text_encoder.pooler.parameters(),
            text_encoder.embeddings.position_embeddings.parameters(),
            text_encoder.embeddings.token_type_embeddings.parameters(),
            text_encoder.embeddings.layer_norm.parameters(),
        )
    else:
        # clip text_encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.transformer.parameters(),
            text_encoder.text_model.ln_final.parameters(),
            text_encoder.text_model.positional_embedding.parameters(),
        )
    freeze_params(params_to_freeze)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * num_processes
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if num_processes > 1:
        text_encoder = paddle.DataParallel(text_encoder)

    # Initialize the optimizer
    optimizer = AdamW(
        learning_rate=lr_scheduler,
        parameters=unwrap_model(text_encoder).get_input_embeddings().parameters(),
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        epsilon=args.adam_epsilon,
        grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm) if args.max_grad_norm is not None else None,
    )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        height=args.height,
        width=args.width,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
        language=args.language,
    )

    def collate_fn(examples):
        input_ids = [example["input_ids"] for example in examples]
        pixel_values = paddle.to_tensor([example["pixel_values"] for example in examples], dtype="float32")
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
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)

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

    if rank == 0:
        logger.info("-----------  Configuration Arguments -----------")
        for arg, value in sorted(vars(args).items()):
            logger.info("%s: %s" % (arg, value))
        logger.info("------------------------------------------------")
        writer = get_writer(args)

    # Train!
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=rank > 0)
    progress_bar.set_description("Train Steps")
    global_step = 0

    text_encoder_embedding_clone = unwrap_model(text_encoder).get_input_embeddings().weight.clone()

    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.eval()
    text_encoder.train()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = paddle.randn(latents.shape)
            batch_size = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = paddle.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,)).astype("int64")

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            attention_mask = paddle.ones_like(batch["input_ids"])
            encoder_hidden_states = text_encoder(batch["input_ids"], attention_mask=attention_mask)[0]
            # Predict the noise residual
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            with paddle.no_grad():
                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = (
                    (paddle.arange(len(tokenizer)) == placeholder_token_id).astype("float32").unsqueeze(-1)
                )
                unwrap_model(text_encoder).get_input_embeddings().weight.grad = (
                    unwrap_model(text_encoder).get_input_embeddings().weight.grad * index_grads_to_zero
                )

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                with paddle.no_grad():
                    unwrap_model(text_encoder).get_input_embeddings().weight[:-1] = text_encoder_embedding_clone[:-1]

                lr_scheduler.step()
                optimizer.clear_grad()
                progress_bar.update(1)
                global_step += 1
                logs = {
                    "epoch": str(epoch).zfill(4),
                    "step_loss": round(loss.item() * args.gradient_accumulation_steps, 10),
                    "lr": lr_scheduler.get_lr(),
                }
                progress_bar.set_postfix(**logs)
                if rank == 0:
                    for name, val in logs.items():
                        if name == "epoch":
                            continue
                        writer.add_scalar(f"train/{name}", val, step=global_step)

                    if global_step % args.save_steps == 0:
                        save_progress(text_encoder, placeholder_token_id, args)

            if global_step >= args.max_train_steps:
                break

    if rank == 0:
        writer.close()
        # Create the pipeline using using the trained modules and save it.
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=unwrap_model(text_encoder),
            safety_checker=None,
            tokenizer=tokenizer,
        )
        pipeline.save_pretrained(args.output_dir)
        # Also save the newly trained embeddings
        save_progress(text_encoder, placeholder_token_id, args)


if __name__ == "__main__":
    main()
