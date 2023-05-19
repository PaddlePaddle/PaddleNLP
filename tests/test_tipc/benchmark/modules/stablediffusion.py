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

import os
import random

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from datasets import DatasetDict, concatenate_datasets
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler
from paddle.vision import BaseTransform, transforms

from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer
from paddlenlp.utils.downloader import get_path_from_url_with_filelock
from paddlenlp.utils.log import logger
from ppdiffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from ppdiffusers.training_utils import main_process_first
from ppdiffusers.utils import PPDIFFUSERS_CACHE

from .model_base import BenchmarkBase


def freeze_params(params):
    for param in params:
        param.stop_gradient = True


def url_or_path_join(*path_list):
    return os.path.join(*path_list) if os.path.isdir(os.path.join(*path_list)) else "/".join(path_list)


class Lambda(BaseTransform):
    def __init__(self, fn, keys=None):
        super().__init__(keys)
        self.fn = fn

    def _apply_image(self, img):
        return self.fn(img)


class StableDiffusion(nn.Layer):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.unet = UNet2DConditionModel.from_pretrained(url_or_path_join(args.model_name_or_path, "unet"))
        self.vae = AutoencoderKL.from_pretrained(url_or_path_join(args.model_name_or_path, "vae"))
        self.text_encoder = CLIPTextModel.from_pretrained(url_or_path_join(args.model_name_or_path, "text_encoder"))
        # we only use self.noise_scheduler.alphas_cumprod
        self.noise_scheduler = DDPMScheduler.from_pretrained(url_or_path_join(args.model_name_or_path, "scheduler"))
        self.register_buffer("alphas_cumprod", self.noise_scheduler.alphas_cumprod)
        freeze_params(self.vae.parameters())
        freeze_params(self.text_encoder.parameters())
        self.unet.train()
        self.vae.eval()
        self.text_encoder.eval()
        if args.use_amp and args.amp_level == "O2":
            self.vae.to(dtype=paddle.float16)
            self.text_encoder.to(dtype=paddle.float16)

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

    def forward(self, input_ids=None, pixel_values=None):
        with paddle.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215
            noise = paddle.randn(latents.shape)
            timesteps = paddle.randint(0, self.noise_scheduler.num_train_timesteps, (latents.shape[0],))
            noisy_latents = self.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.text_encoder(input_ids)[0]
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.cast(paddle.float32), target.cast(paddle.float32), reduction="mean")
        return loss


class StableDiffusionBenchmark(BenchmarkBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(args, parser):
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            default="CompVis/stable-diffusion-v1-4",
            help="Model name. Defaults to CompVis/stable-diffusion-v1-4. ",
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
            "--center_crop",
            action="store_true",
            help="Whether to center crop images before resizing to resolution (if not set, random crop will be used)",
        )
        parser.add_argument(
            "--random_flip",
            action="store_true",
            help="whether to randomly flip images horizontally",
        )
        parser.add_argument(
            "--dataloader_num_workers",
            type=int,
            default=4,
            help=(
                "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
            ),
        )

    def create_input_specs(self):
        input_ids = paddle.static.InputSpec(name="input_ids", shape=[-1, self.model_max_length], dtype="int64")
        dtype = "float16" if self.args.use_amp and self.args.amp_level == "O2" else "float32"
        pixel_values = paddle.static.InputSpec(
            name="pixel_values", shape=[-1, 3, self.args.resolution, self.args.resolution], dtype=dtype
        )
        return [input_ids, pixel_values]

    def create_data_loader(self, args, **kwargs):
        caption_column = "text"
        image_column = "image"
        self.tokenizer = tokenizer = CLIPTokenizer.from_pretrained(
            url_or_path_join(args.model_name_or_path, "tokenizer")
        )

        def tokenize_captions(examples, is_train=True):
            captions = []
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column `{caption_column}` should contain either strings or lists of strings."
                    )
            inputs = tokenizer(
                captions,
                max_length=tokenizer.model_max_length,
                padding="do_not_pad",
                truncation=True,
                return_attention_mask=False,
            )
            return inputs.input_ids

        # Preprocessing the datasets.
        train_transforms = transforms.Compose(
            [
                transforms.Resize((args.resolution, args.resolution), interpolation="bilinear"),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples)

            return examples

        file_path = get_path_from_url_with_filelock(
            "https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/pokemon-blip-captions.tar.gz",
            PPDIFFUSERS_CACHE,
        )
        dataset = DatasetDict.load_from_disk(file_path)

        with main_process_first():
            repeat_dataset = concatenate_datasets([dataset["train"]] * 250)
            dataset["train"] = repeat_dataset
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)

        train_sampler = (
            DistributedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
            if paddle.distributed.get_world_size() > 1
            else BatchSampler(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
        )

        def collate_fn(examples):
            pixel_values = paddle.stack([example["pixel_values"] for example in examples])
            if args.use_amp and args.amp_level == "O2":
                pixel_values = pixel_values.cast(paddle.float16)
            input_ids = [example["input_ids"] for example in examples]
            input_ids = tokenizer.pad(
                {"input_ids": input_ids},
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pd",
                return_attention_mask=False,
            ).input_ids
            return {"pixel_values": pixel_values, "input_ids": input_ids}

        train_dataloader = DataLoader(
            train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=args.dataloader_num_workers
        )
        self.num_batch = len(train_dataloader)
        self.model_max_length = tokenizer.model_max_length
        return train_dataloader, None

    def build_model(self, args, **kwargs):
        model = StableDiffusion(args)
        self.args = args
        return model

    def forward(self, model, args, input_data=None, **kwargs):
        loss = model(**input_data)
        return (
            loss,
            input_data["input_ids"].shape[0],
        )

    def logger(
        self,
        args,
        step_id=None,
        pass_id=None,
        batch_id=None,
        loss=None,
        batch_cost=None,
        reader_cost=None,
        num_samples=None,
        ips=None,
        **kwargs
    ):
        logger.info(
            "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sample/sec"
            % (step_id, args.epoch * self.num_batch, loss, reader_cost, batch_cost, num_samples, ips)
        )
