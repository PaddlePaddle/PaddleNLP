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

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import paddle
from ldm import AutoencoderKLWithLoss, TextImagePair, worker_init_fn
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler
from paddle.optimizer import Adam
from tqdm.auto import tqdm

from paddlenlp.trainer import set_seed
from paddlenlp.utils.log import logger
from ppdiffusers.training_utils import freeze_params, main_process_first, unwrap_model


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_writer(args):
    if args.report_to == "visualdl":
        from visualdl import LogWriter

        writer = LogWriter(logdir=args.logging_dir)
    elif args.report_to == "tensorboard":
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir=args.logging_dir)
    else:
        raise ValueError("report_to must be in ['visualdl', 'tensorboard']")
    return writer


def run_evaluate(vae, val_dataloader, writer, global_step):
    log_dict_ae_all = defaultdict(list)
    log_dict_disc_all = defaultdict(list)
    for batch in val_dataloader:
        log_dict_ae, log_dict_disc = unwrap_model(vae).validation_step(batch["image"], global_step=global_step)
        for k, v in log_dict_ae.items():
            if "loss" not in k:
                continue
            log_dict_ae_all[k].append(v)
        for k, v in log_dict_disc.items():
            log_dict_disc_all[k].append(v)
    for name, val in log_dict_ae_all.items():
        writer.add_scalar(name, np.mean(val), global_step)
    for name, val in log_dict_disc_all.items():
        writer.add_scalar(name, np.mean(val), global_step)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training a autoencoder model script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from bos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="autoencoder_outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=23, help="A seed for reproducible training.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training/validation dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4.5e-06,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale base-lr by ngpu * batch_size",
    )
    parser.add_argument("--freeze_encoder", action="store_true", help="Whether to freeze encoder layer.")
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Whether to train new model from scratch. ",
    )
    parser.add_argument("--vae_config_file", default=None, type=str, help="Path to the vae_config_file.")
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
        choices=["tensorboard", "visualdl"],
        help="Log writer type.",
    )
    parser.add_argument("--logging_steps", default=100, type=int, help="The interval steps to logging.")
    parser.add_argument(
        "--image_logging_steps",
        default=500,
        type=int,
        help="The interval steps to logging images.",
    )
    parser.add_argument("--save_steps", default=2000, type=int, help="The interval steps to saveing.")
    parser.add_argument(
        "--ignore_keys",
        default=[],
        type=str,
        nargs="*",
        help="The prefix keys to be ignored when we resume from a pretrained model, e.g. ignore_keys = ['decoder.'], we will ignore 'decoder.xxx', 'decoder.xxx.xxx'.",
    )
    parser.add_argument(
        "--input_size", default=None, type=int, nargs="*", help="The height and width of the input at the encoder."
    )
    # dataset
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="text_image_pair",
        choices=["imagenet", "text_image_pair"],
        help="The type of dataset.",
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
        "--degradation",
        type=str,
        default="pil_nearest",
        help="Degradation_fn, e.g. cv_bicubic, bsrgan_light, or pil_nearest",
    )
    parser.add_argument(
        "--file_list",
        type=str,
        default="./data/filelist/train.filelist.list",
        help="Path to the train file_list.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="The number of subprocess to load data.",
    )
    parser.add_argument(
        "--num_records",
        type=int,
        default=62500,
        help="The num_records of the text_image_pair dataset.",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=100,
        help="The buffer size of the text_image_pair dataset.",
    )
    parser.add_argument(
        "--shuffle_every_n_samples",
        type=int,
        default=5,
        help="The shuffle_every_n_samples of the text_image_pair dataset.",
    )
    parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")

    # loss fn
    parser.add_argument("--disc_start", type=int, default=50001, help="The number of steps the discriminator started.")
    parser.add_argument("--kl_weight", type=float, default=1.0e-6, help="The weight ratio of the kl_loss.")
    parser.add_argument("--disc_weight", type=float, default=0.5, help="The weight ratio of the disc_loss.")
    parser.add_argument("--logvar_init", type=float, default=0.0, help="The init value of the output log variances.")
    parser.add_argument("--pixelloss_weight", type=float, default=1.0, help="The weight ratio of the pixelloss.")
    parser.add_argument("--disc_num_layers", type=int, default=3, help="The num layers of the discriminator.")
    parser.add_argument("--disc_in_channels", type=int, default=3, help="The in channels of the discriminator.")
    parser.add_argument("--disc_factor", type=float, default=1.0, help="The factor of the discriminator loss.")
    parser.add_argument(
        "--perceptual_weight", type=float, default=1.0, help="The weight ratio of the perceptual loss."
    )
    parser.add_argument(
        "--use_actnorm", action="store_true", help="Whether to use actnorm in NLayerDiscriminator layer."
    )
    parser.add_argument("--disc_conditional", action="store_true", help="Whether to use conditional discriminator.")
    parser.add_argument(
        "--disc_loss", type=str, choices=["hinge", "vanilla"], default="hinge", help="The type of discriminator loss."
    )
    args = parser.parse_args()

    args.logging_dir = os.path.join(args.output_dir, args.logging_dir)
    args.image_logging_steps = math.ceil(args.image_logging_steps / args.logging_steps) * args.logging_steps

    return args


def check_keys(model, state_dict):
    cls_name = model.__class__.__name__
    missing_keys = []
    mismatched_keys = []
    for k, v in model.state_dict().items():
        if k not in state_dict.keys():
            missing_keys.append(k)
        if list(v.shape) != list(state_dict[k].shape):
            mismatched_keys.append(k)
    if len(missing_keys):
        missing_keys_str = ", ".join(missing_keys)
        print(f"{cls_name} Found missing_keys {missing_keys_str}!")
    if len(mismatched_keys):
        mismatched_keys_str = ", ".join(mismatched_keys)
        print(f"{cls_name} Found mismatched_keys {mismatched_keys_str}!")
    if len(missing_keys) == 0 and len(mismatched_keys) == 0:
        print(f"{cls_name} All model state_dict are loaded!")


def main():
    args = parse_args()
    rank = paddle.distributed.get_rank()
    num_processes = paddle.distributed.get_world_size()
    if num_processes > 1:
        paddle.distributed.init_parallel_env()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if not args.from_scratch:
        # Load pretrained model
        if args.vae_config_file is not None:
            model_kwargs = read_json(args.vae_config_file)
        else:
            model_kwargs = {}
        vae = AutoencoderKLWithLoss.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            ignore_keys=args.ignore_keys,
            input_size=args.input_size,
            disc_start=args.disc_start,
            kl_weight=args.kl_weight,
            disc_weight=args.disc_weight,
            logvar_init=args.logvar_init,
            pixelloss_weight=args.pixelloss_weight,
            disc_num_layers=args.disc_num_layers,
            disc_in_channels=args.disc_in_channels,
            disc_factor=args.disc_factor,
            perceptual_weight=args.perceptual_weight,
            use_actnorm=args.use_actnorm,
            disc_conditional=args.disc_conditional,
            disc_loss=args.disc_loss,
            **model_kwargs,
        )
    else:
        assert args.vae_config_file is not None, "We must supply vae_config_file!"
        # Load config: train model from scatch
        vae = AutoencoderKLWithLoss.from_config(
            read_json(args.vae_config_file),
            input_size=args.input_size,
            disc_start=args.disc_start,
            kl_weight=args.kl_weight,
            disc_weight=args.disc_weight,
            logvar_init=args.logvar_init,
            pixelloss_weight=args.pixelloss_weight,
            disc_num_layers=args.disc_num_layers,
            disc_in_channels=args.disc_in_channels,
            disc_factor=args.disc_factor,
            perceptual_weight=args.perceptual_weight,
            use_actnorm=args.use_actnorm,
            disc_conditional=args.disc_conditional,
            disc_loss=args.disc_loss,
        )

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        vae.set_dict(state_dict)
        check_keys(vae, state_dict)
        del state_dict

    if args.scale_lr:
        args.learning_rate = num_processes * args.batch_size * args.learning_rate

    # configure_optimizers
    parameters = list(vae.decoder.parameters()) + list(vae.post_quant_conv.parameters())
    # we may freeze_encoder
    if not args.freeze_encoder:
        parameters += list(vae.encoder.parameters())
        parameters += list(vae.quant_conv.parameters())
    else:
        freeze_params(vae.encoder.parameters())
        freeze_params(vae.quant_conv.parameters())
        print("Freeze vae.encoder.parameters and vae.quant_conv.parameters!")

    opt_ae = Adam(parameters=parameters, learning_rate=args.learning_rate, beta1=0.5, beta2=0.9)
    opt_disc = Adam(
        parameters=vae.loss.discriminator.parameters(),
        learning_rate=args.learning_rate,
        beta1=0.5,
        beta2=0.9,
    )

    optimizers = [opt_ae, opt_disc]

    if num_processes > 1:
        vae = paddle.DataParallel(vae, find_unused_parameters=True)

    if args.dataset_type == "imagenet":
        from ldm import ImageNetSRTrain, ImageNetSRValidation

        with main_process_first():
            train_dataset = ImageNetSRTrain(size=args.resolution, degradation=args.degradation)
            val_dataset = ImageNetSRValidation(size=args.resolution, degradation=args.degradation)
        train_sampler = (
            DistributedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
            if num_processes > 1
            else BatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
        )
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers)

        val_sampler = BatchSampler(val_dataset, batch_size=args.batch_size * 2, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=args.num_workers)
    else:
        train_dataset = TextImagePair(
            file_list=args.file_list,
            size=args.resolution,
            num_records=args.num_records,
            buffer_size=args.buffer_size,
            shuffle_every_n_samples=args.shuffle_every_n_samples,
            interpolation="lanczos",
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
        )
        val_dataloader = val_dataset = None
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = (
        len(train_dataloader) if args.dataset_type == "imagenet" else math.ceil(len(train_dataset) / args.batch_size)
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

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
    total_batch_size = args.batch_size * num_processes

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(
        f"  Number of trainable parameters = {sum(p.numel().item() for p in vae.parameters() if not p.stop_gradient) }"
    )
    logger.info(
        f"  Number of non-trainable parameters = {sum(p.numel().item() for p in vae.parameters() if p.stop_gradient) }"
    )
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=rank > 0)
    progress_bar.set_description("Steps")
    global_step = 0

    vae.train()
    for epoch in range(args.num_train_epochs):
        for batch in train_dataloader:
            logs = {"epoch": str(epoch).zfill(4)}
            for optimizer_idx in [0, 1]:
                # pytorch_lightning use this `toggle_optimizer` method
                # ref: https://github.com/Lightning-AI/lightning/blob/a58639ce7e864dd70484e7d34c37730ae204183c/src/pytorch_lightning/core/module.py#L1419-L1447
                unwrap_model(vae).toggle_optimizer(optimizers, optimizer_idx)
                loss, log_dict = vae(batch["image"], optimizer_idx=optimizer_idx, global_step=global_step)
                optimizers[optimizer_idx].clear_grad()
                loss.backward()
                optimizers[optimizer_idx].step()
                # pytorch_lightning use this `untoggle_optimizer` method
                # ref: https://github.com/Lightning-AI/lightning/blob/a58639ce7e864dd70484e7d34c37730ae204183c/src/pytorch_lightning/core/module.py#L1449-L1464
                unwrap_model(vae).untoggle_optimizer(optimizers, optimizer_idx)
                logs.update(log_dict)

            progress_bar.update(1)
            global_step += 1
            # progress_bar.set_postfix(**logs)

            if rank == 0:
                # logging
                if global_step % args.logging_steps == 0:
                    for name, val in logs.items():
                        if name == "epoch":
                            continue
                        writer.add_scalar(name, val, global_step)

                if global_step % args.image_logging_steps == 0:
                    images_log = unwrap_model(vae).log_images(batch["image"])
                    for name, val in images_log.items():
                        writer.add_image(name, val, global_step, dataformats="NHWC")

                # saving
                if global_step % args.save_steps == 0:
                    if val_dataloader is not None:
                        run_evaluate(unwrap_model(vae), val_dataloader, writer, global_step)
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    unwrap_model(vae).save_pretrained(output_dir)

            del logs
            if global_step >= args.max_train_steps:
                break

    if rank == 0:
        writer.close()
        unwrap_model(vae).save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
