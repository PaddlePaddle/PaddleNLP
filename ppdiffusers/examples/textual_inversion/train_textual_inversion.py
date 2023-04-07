# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import gc
import glob
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)
from paddle.io import BatchSampler, DataLoader, Dataset, DistributedBatchSampler
from paddle.optimizer import AdamW
from paddle.vision.transforms import RandomHorizontalFlip
from PIL import Image
from tqdm.auto import tqdm

from paddlenlp.trainer import set_seed
from paddlenlp.transformers import AutoTokenizer, PretrainedConfig
from paddlenlp.utils.log import logger
from ppdiffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    is_ppxformers_available,
)
from ppdiffusers.optimization import get_scheduler
from ppdiffusers.training_utils import freeze_params, unfreeze_params, unwrap_model
from ppdiffusers.utils import PIL_INTERPOLATION


def url_or_path_join(*path_list):
    return os.path.join(*path_list) if os.path.isdir(os.path.join(*path_list)) else "/".join(path_list)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    try:
        text_encoder_config = PretrainedConfig.from_pretrained(
            url_or_path_join(pretrained_model_name_or_path, "text_encoder")
        )
        model_class = text_encoder_config.architectures[0]
    except Exception:
        model_class = "LDMBertModel"
    if model_class == "CLIPTextModel":
        from paddlenlp.transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from ppdiffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    elif model_class == "BertModel":
        from paddlenlp.transformers import BertModel

        return BertModel
    elif model_class == "LDMBertModel":
        from ppdiffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import (
            LDMBertModel,
        )

        return LDMBertModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def set_recompute(model, value=False):
    def fn(layer):
        # ldmbert
        if hasattr(layer, "enable_recompute"):
            layer.enable_recompute = value
            print("Set", layer.__class__, "recompute", layer.enable_recompute)
        # # unet
        # if hasattr(layer, "gradient_checkpointing"):
        #     layer.gradient_checkpointing = value
        #     print("Set", layer.__class__, "recompute", layer.gradient_checkpointing)

    model.apply(fn)


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


def save_progress(text_encoder, placeholder_token_id, args, save_path):
    logger.info("Saving embeddings")
    learned_embeds = unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach()}
    paddle.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.pdparams every X updates steps.",
    )
    parser.add_argument(
        "--only_save_embeds",
        action="store_true",
        default=False,
        help="Save only the embeddings for the new concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
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
        default=None,
        help=(
            "The height for input images, all the images in the train/validation dataset will be resized to this"
            " height"
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "The width for input images, all the images in the train/validation dataset will be resized to this"
            " width"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
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
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
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
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
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
    parser.add_argument("--max_grad_norm", default=-1, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
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
        "--report_to",
        type=str,
        default="visualdl",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"visualdl"`'
            ' (default), `"tensorboard"`.'
        ),
    )
    parser.add_argument("--language", default="en", choices=["en", "zh", "zh_en"], help="Model language.")
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
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
    if args.height is None or args.width is None and args.resolution is not None:
        args.height = args.width = args.resolution
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
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
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
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_attention_mask=False,
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


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main():
    paddle_dtype = paddle.float32
    args = parse_args()
    rank = paddle.distributed.get_rank()
    is_main_process = rank == 0
    num_processes = paddle.distributed.get_world_size()
    if num_processes > 1:
        paddle.distributed.init_parallel_env()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        # support windows "\"
        tokenizer = AutoTokenizer.from_pretrained(url_or_path_join(args.pretrained_model_name_or_path, "tokenizer"))
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    initializer_token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)["input_ids"]
    if len(initializer_token_ids) < 1:
        raise ValueError("The initializer token must be a greater equal than one.")

    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)

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

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    with paddle.no_grad():
        token_embeds = text_encoder.get_input_embeddings()
        # we will compute mean
        token_embeds.weight[placeholder_token_id] = paddle.stack(
            [token_embeds.weight[each] for each in initializer_token_ids]
        ).mean(0)

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    freeze_params(text_encoder.parameters())
    unfreeze_params(text_encoder.get_input_embeddings().parameters())

    if args.gradient_checkpointing:
        # unet.enable_gradient_checkpointing()
        set_recompute(text_encoder, True)

    if args.enable_xformers_memory_efficient_attention and is_ppxformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warn(
                "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                f" correctly and a GPU is available: {e}"
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
        interpolation="bilinear",
    )

    def collate_fn(examples):
        input_ids = [example["input_ids"] for example in examples]
        pixel_values = paddle.to_tensor([example["pixel_values"] for example in examples], dtype="float32")
        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pd"
        ).input_ids
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

    train_sampler = (
        DistributedBatchSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        if num_processes > 1
        else BatchSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=args.dataloader_num_workers
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * num_processes
        )

    # Initialize the lr_scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # Initialize the optimizer
    optimizer = AdamW(
        learning_rate=lr_scheduler,
        parameters=text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        epsilon=args.adam_epsilon,
        grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm) if args.max_grad_norm > 0 else None,
    )

    if num_processes > 1:
        text_encoder = paddle.DataParallel(text_encoder)

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

    # keep original embeddings as reference
    orig_embeds_params = unwrap_model(text_encoder).get_input_embeddings().weight.clone()
    index_no_updates = (paddle.arange(len(tokenizer)) != placeholder_token_id).cast(paddle.int64).sum()

    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.train()
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
                # unet_ctx_manager = unet.no_sync()
                text_encoder_ctx_manager = text_encoder.no_sync()
            else:
                # unet_ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()
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

                # with unet_ctx_manager:
                # Predict the noise or sample
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred, target, reduction="mean")

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if num_processes > 1 and args.gradient_checkpointing:
                    fused_allreduce_gradients(unwrap_model(text_encoder).get_input_embeddings().parameters(), None)
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                # Let's make sure we don't update any embedding weights besides the newly added token
                with paddle.no_grad():
                    unwrap_model(text_encoder).get_input_embeddings().weight[:index_no_updates] = orig_embeds_params[
                        :index_no_updates
                    ]

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
                        writer.add_scalar(f"train/{name}", val, global_step)

                    if global_step % args.save_steps == 0:
                        save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.pdparams")
                        save_progress(text_encoder, placeholder_token_id, args, save_path)

                if global_step >= args.max_train_steps:
                    break

        if is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # create pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    text_encoder=unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    paddle_dtype=paddle_dtype,
                    safety_checker=None,
                )
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = paddle.Generator().manual_seed(args.seed) if args.seed else None
                images = [
                    pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                    for _ in range(args.num_validation_images)
                ]
                np_images = np.stack([np.asarray(img) for img in images])

                if args.report_to == "tensorboard":
                    writer.add_images("test", np_images, epoch, dataformats="NHWC")
                else:
                    writer.add_image("test", np_images, epoch, dataformats="NHWC")

                del pipeline
                gc.collect()
                vae.eval()
                unet.train()
                text_encoder.train()

    if is_main_process:
        writer.close()
        if args.push_to_hub and args.only_save_embeds:
            logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = not args.only_save_embeds
        if save_full_model:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=unwrap_model(text_encoder),
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.output_dir)
        # Save the newly trained embeddings
        save_path = os.path.join(args.output_dir, "learned_embeds.pdparams")
        save_progress(text_encoder, placeholder_token_id, args, save_path)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)


if __name__ == "__main__":
    main()
