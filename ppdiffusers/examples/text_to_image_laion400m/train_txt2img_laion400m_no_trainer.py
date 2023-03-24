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
import itertools
import math
import os
import sys
import time

import paddle
import paddle.nn as nn
from ldm import (
    DataArguments,
    LatentDiffusionModel,
    ModelArguments,
    NoTrainerTrainingArguments,
    TextImagePair,
    worker_init_fn,
)
from paddle.io import DataLoader
from paddle.optimizer import AdamW

from paddlenlp.trainer import PdArgumentParser, set_seed
from paddlenlp.utils.log import logger
from ppdiffusers.optimization import get_scheduler
from ppdiffusers.training_utils import unwrap_model


def get_writer(training_args):
    if training_args.report_to == "visualdl":
        from visualdl import LogWriter

        writer = LogWriter(logdir=training_args.logging_dir)
    elif training_args.report_to == "tensorboard":
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir=training_args.logging_dir)
    else:
        raise ValueError("writer_type must be in ['visualdl', 'tensorboard']")
    return writer


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, NoTrainerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.image_logging_steps = model_args.image_logging_steps = (
        math.ceil(model_args.image_logging_steps / training_args.logging_steps) * training_args.logging_steps
    )
    training_args.resolution = data_args.resolution
    training_args.print_config(training_args, "Training")
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    rank = paddle.distributed.get_rank()
    num_processes = paddle.distributed.get_world_size()

    if num_processes > 1:
        paddle.distributed.init_parallel_env()

    training_args.logging_dir = os.path.join(training_args.output_dir, training_args.logging_dir)

    if training_args.seed is not None:
        set_seed(training_args.seed)

    if training_args.output_dir is not None:
        os.makedirs(training_args.output_dir, exist_ok=True)

    model = LatentDiffusionModel(model_args)
    model.set_recompute(training_args.recompute)
    params_to_train = itertools.chain(model.text_encoder.parameters(), model.unet.parameters())

    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        learning_rate=training_args.learning_rate,
        num_warmup_steps=training_args.warmup_steps * training_args.gradient_accumulation_steps,
        num_training_steps=training_args.max_steps * training_args.gradient_accumulation_steps,
    )

    optimizer = AdamW(
        learning_rate=lr_scheduler,
        parameters=params_to_train,
        beta1=training_args.adam_beta1,
        beta2=training_args.adam_beta2,
        weight_decay=training_args.weight_decay,
        epsilon=training_args.adam_epsilon,
        grad_clip=nn.ClipGradByGlobalNorm(training_args.max_grad_norm)
        if training_args.max_grad_norm is not None and training_args.max_grad_norm > 0
        else None,
    )
    train_dataset = TextImagePair(
        file_list=data_args.file_list,
        size=data_args.resolution,
        num_records=data_args.num_records,
        buffer_size=data_args.buffer_size,
        shuffle_every_n_samples=data_args.shuffle_every_n_samples,
        interpolation="lanczos",
        tokenizer=model.tokenizer,
    )

    if num_processes > 1:
        model = paddle.DataParallel(model)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=training_args.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
    )

    if rank == 0:
        writer = get_writer(training_args)

    # Train!
    total_batch_size = (
        training_args.per_device_train_batch_size * num_processes * training_args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")

    global_steps = 0
    tic_train = time.time()

    for epoch in range(training_args.num_train_epochs):
        if epoch == training_args.num_train_epochs:
            logger.info("***** Trainging Done *****")
            break

        for step, batch in enumerate(train_dataloader):
            if (
                num_processes > 1 and ((step + 1) % training_args.gradient_accumulation_steps != 0)
            ) or training_args.recompute:
                # grad acc, no_sync when (step + 1) % training_args.gradient_accumulation_steps != 0:
                ctx_manager = model.no_sync()
            else:
                ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()

            with ctx_manager:
                loss = model(**batch)
                if training_args.gradient_accumulation_steps > 1:
                    loss = loss / training_args.gradient_accumulation_steps
                loss.backward()

            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                global_steps += 1
                unwrap_model(model).on_train_batch_end()

                # train log
                if global_steps % training_args.logging_steps == 0:
                    logs = {
                        "train/loss": loss.item() * training_args.gradient_accumulation_steps,
                        "train/lr_abs": lr_scheduler.get_lr(),
                        "train/global_steps": global_steps,
                    }
                    if rank == 0:
                        # add scalar
                        for name, val in logs.items():
                            writer.add_scalar(name, val, global_steps)
                    log_str = "Train: global_steps {0:d}/{1:d}, epoch: {2:d}, batch: {3:d}, train_loss: {4:.10f}, lr_abs: {5:.10f}, speed: {6:.2f} s/it.".format(
                        global_steps,
                        training_args.max_steps,
                        epoch,
                        step + 1,
                        logs["train/loss"],
                        logs["train/lr_abs"],
                        (time.time() - tic_train) / training_args.logging_steps,
                    )
                    logger.info(log_str)

                    if global_steps % training_args.image_logging_steps == 0:
                        reconstruction_img = unwrap_model(model).decode_image(pixel_values=batch["pixel_values"])
                        ddim_10_img = unwrap_model(model).log_image(input_ids=batch["input_ids"], guidance_scale=1.0)
                        ddim_75_img = unwrap_model(model).log_image(input_ids=batch["input_ids"], guidance_scale=7.5)
                        if rank == 0:
                            writer.add_image("reconstruction", reconstruction_img, global_steps, dataformats="NHWC")
                            writer.add_image("ddim-samples-1.0", ddim_10_img, global_steps, dataformats="NHWC")
                            writer.add_image("ddim-samples-7.5", ddim_75_img, global_steps, dataformats="NHWC")
                    tic_train = time.time()

                    if rank == 0 and global_steps % training_args.save_steps == 0:
                        os.makedirs(
                            os.path.join(training_args.output_dir, f"global-steps-{global_steps}"), exist_ok=True
                        )
                        paddle.save(
                            model.state_dict(),
                            os.path.join(
                                training_args.output_dir, f"global-steps-{global_steps}", "model_state.pdparams"
                            ),
                        )

                if global_steps >= training_args.max_steps:
                    break
    if rank == 0:
        paddle.save(model.state_dict(), os.path.join(training_args.output_dir, "model_state.pdparams"))
        writer.close()


if __name__ == "__main__":
    main()
