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
import math
import numpy as np
import os
from collections import defaultdict
import paddle

from paddle.io import DataLoader, BatchSampler, DistributedBatchSampler

from paddlenlp.utils.log import logger
from paddlenlp.trainer import set_seed

from ppdiffusers.modeling_utils import unwrap_model
from ppdiffusers.training_utils import main_process_first

from paddle.optimizer import Adam
from tqdm.auto import tqdm
from autoencoder_datasets import ImageNetSRTrain, ImageNetSRValidation

# AutoencoderKLWithLoss Model
from ppdiffusers.configuration_utils import register_to_config
from ppdiffusers import AutoencoderKL
from losses import LPIPSWithDiscriminator
from typing import Tuple


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
        description="Simple example of a training a autoencoder model script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=False,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="autoencoder_outputs",
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training/validation dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",
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
    parser.add_argument("--freeze_encoder",
                        action="store_true",
                        help="Whether to freeze_encoder.")
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Whether to train new model from scratch. ",
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
    parser.add_argument("--logging_steps",
                        default=100,
                        type=int,
                        help="The interval steps to logging.")
    parser.add_argument("--logging_image_steps",
                        default=1000,
                        type=int,
                        help="The interval steps to logging images.")
    parser.add_argument("--save_steps",
                        default=5000,
                        type=int,
                        help="The interval steps to saveing.")

    # dataset
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=
        ("The resolution for input images, all the images in the train/validation dataset will be resized to this"
         " resolution"),
    )
    parser.add_argument("--degradation",
                        type=str,
                        default="pil_nearest",
                        help="degradation")
    # loss fn
    parser.add_argument("--disc_start", type=int, default=10000)
    parser.add_argument("--kl_weight", type=float, default=1.0e-6)
    parser.add_argument("--disc_weight", type=float, default=0.5)
    parser.add_argument("--logvar_init", type=float, default=0.0)
    parser.add_argument("--pixelloss_weight", type=float, default=1.0)
    parser.add_argument("--disc_num_layers", type=int, default=3)
    parser.add_argument("--disc_in_channels", type=int, default=3)
    parser.add_argument("--disc_factor", type=float, default=1.0)
    parser.add_argument("--perceptual_weight", type=float, default=1.0)
    parser.add_argument("--use_actnorm", action="store_true")
    parser.add_argument("--disc_conditional", action="store_true")
    parser.add_argument("--disc_loss", type=str, default="hinge")
    args = parser.parse_args()

    args.logging_dir = os.path.join(args.output_dir, args.logging_dir)

    return args


def run_evaluate(vae, val_dataloader, writer, global_step):
    log_dict_ae_all = defaultdict(list)
    log_dict_disc_all = defaultdict(list)
    for batch in val_dataloader:
        log_dict_ae, log_dict_disc = vae.validation_step(
            batch["image"], global_step=global_step)
        for k, v in log_dict_ae.items():
            if 'loss' not in k:
                continue
            log_dict_ae_all[k].append(v)
        for k, v in log_dict_disc.items():
            log_dict_disc_all[k].append(v)
    for name, val in log_dict_ae_all.items():
        writer.add_scalar(name, np.mean(val), step=global_step)
    for name, val in log_dict_disc_all.items():
        writer.add_scalar(name, np.mean(val), step=global_step)


class AutoencoderKLWithLoss(AutoencoderKL):

    @register_to_config
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str] = ("DownEncoderBlock2D", ),
            up_block_types: Tuple[str] = ("UpDecoderBlock2D", ),
            block_out_channels: Tuple[int] = (64, ),
            layers_per_block: int = 1,
            act_fn: str = "silu",
            latent_channels: int = 4,
            norm_num_groups: int = 32,
            sample_size: int = 32,
            # loss arguments
            disc_start=10000,
            kl_weight=1.0e-6,
            disc_weight=0.5,
            logvar_init=0.0,
            pixelloss_weight=1.0,
            disc_num_layers=3,
            disc_in_channels=3,
            disc_factor=1.0,
            perceptual_weight=1.0,
            use_actnorm=False,
            disc_conditional=False,
            disc_loss="hinge"):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         down_block_types=down_block_types,
                         up_block_types=up_block_types,
                         block_out_channels=block_out_channels,
                         layers_per_block=layers_per_block,
                         act_fn=act_fn,
                         latent_channels=latent_channels,
                         norm_num_groups=norm_num_groups,
                         sample_size=sample_size)
        self.loss = LPIPSWithDiscriminator(disc_start=disc_start,
                                           kl_weight=kl_weight,
                                           disc_weight=disc_weight,
                                           logvar_init=logvar_init,
                                           pixelloss_weight=pixelloss_weight,
                                           disc_num_layers=disc_num_layers,
                                           disc_in_channels=disc_in_channels,
                                           disc_factor=disc_factor,
                                           perceptual_weight=perceptual_weight,
                                           use_actnorm=use_actnorm,
                                           disc_conditional=disc_conditional,
                                           disc_loss=disc_loss)

    def forward(
        self,
        sample: paddle.Tensor,
        sample_posterior: bool = True,
    ):
        posterior = self.encode(sample).latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z).sample
        return dec, posterior

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def training_step(self, pixel_values, optimizer_idx=0, global_step=0):
        reconstructions, posterior = self(pixel_values)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(pixel_values,
                                            reconstructions,
                                            posterior,
                                            optimizer_idx,
                                            global_step,
                                            last_layer=self.get_last_layer(),
                                            split="train")
            return aeloss, log_dict_ae

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                pixel_values,
                reconstructions,
                posterior,
                optimizer_idx,
                global_step,
                last_layer=self.get_last_layer(),
                split="train")
            return discloss, log_dict_disc

    @paddle.no_grad()
    def log_images(self, pixel_values, only_inputs=False, **kwargs):
        self.eval()
        log = dict()
        if not only_inputs:
            xrec, posterior = self(pixel_values)
            log["samples"] = self.decode_image(
                self.decode(paddle.randn(posterior.sample().shape)).sample)
            log["reconstructions"] = self.decode_image(xrec)
        log["inputs"] = self.decode_image(pixel_values)
        self.train()
        return log

    def decode_image(self, image):
        image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1])
        image = (image * 255.).numpy().round()
        return image

    @paddle.no_grad()
    def validation_step(self, pixel_values, global_step=0):
        self.eval()
        reconstructions, posterior = self(pixel_values)
        aeloss, log_dict_ae = self.loss(pixel_values,
                                        reconstructions,
                                        posterior,
                                        0,
                                        global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val")

        discloss, log_dict_disc = self.loss(pixel_values,
                                            reconstructions,
                                            posterior,
                                            1,
                                            global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val")
        self.train()
        return log_dict_ae, log_dict_disc

    def toggle_optimizer(self, optimizers, optimizer_idx):
        """
        Makes sure only the gradients of the current optimizer's parameters are calculated
        in the training step to prevent dangling gradients in multiple-optimizer setup.
        It works with :meth:`untoggle_optimizer` to make sure ``param_stop_gradient_state`` is properly reset.
        Override for your own behavior.

        Args:
            optimizer: Current optimizer used in the training loop
            optimizer_idx: Current optimizer idx in the training loop

        Note:
            Only called when using multiple optimizers
        """
        # Iterate over all optimizer parameters to preserve their `stop_gradient` information
        # in case these are pre-defined during `configure_optimizers`
        param_stop_gradient_state = {}
        for opt in optimizers:
            for param in opt._parameter_list:
                # If a param already appear in param_stop_gradient_state, continue
                if param in param_stop_gradient_state:
                    continue
                param_stop_gradient_state[param] = param.stop_gradient
                param.stop_gradient = True

        # Then iterate over the current optimizer's parameters and set its `stop_gradient`
        # properties accordingly
        for param in optimizers[optimizer_idx]._parameter_list:
            param.stop_gradient = param_stop_gradient_state[param]
        self._param_stop_gradient_state = param_stop_gradient_state

    def untoggle_optimizer(self, optimizers, optimizer_idx):
        """
        Resets the state of required gradients that were toggled with :meth:`toggle_optimizer`.
        Override for your own behavior.

        Args:
            optimizer_idx: Current optimizer idx in the training loop

        Note:
            Only called when using multiple optimizers
        """
        for opt_idx, opt in enumerate(optimizers):
            if optimizer_idx != opt_idx:
                for param in opt._parameter_list:
                    if param in self._param_stop_gradient_state:
                        param.stop_gradient = self._param_stop_gradient_state[
                            param]
        # save memory
        self._param_stop_gradient_state = {}


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
        vae = AutoencoderKLWithLoss.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
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
            disc_loss=args.disc_loss)
    else:
        # Load config: train model from scatch
        vae = AutoencoderKLWithLoss.from_config(
            args.pretrained_model_name_or_path,
            subfolder="vae",
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
            disc_loss=args.disc_loss)

    # configure_optimizers
    parameters = list(vae.decoder.parameters()) + list(
        vae.post_quant_conv.parameters())
    if not args.freeze_encoder:
        parameters += list(vae.encoder.parameters())
        parameters += list(vae.quant_conv.parameters())

    if args.scale_lr:
        args.learning_rate = num_processes * args.batch_size * args.learning_rate

    opt_ae = Adam(parameters=parameters,
                  learning_rate=args.learning_rate,
                  beta1=0.5,
                  beta2=0.9)
    opt_disc = Adam(parameters=vae.loss.discriminator.parameters(),
                    learning_rate=args.learning_rate,
                    beta1=0.5,
                    beta2=0.9)

    optimizers = [opt_ae, opt_disc]

    if num_processes > 1:
        vae = paddle.DataParallel(vae)

    with main_process_first():
        train_dataset = ImageNetSRTrain(size=args.resolution,
                                        degradation=args.degradation,
                                        output_LR_image=False)
        val_dataset = ImageNetSRValidation(size=args.resolution,
                                           degradation=args.degradation,
                                           output_LR_image=False)

    train_sampler = DistributedBatchSampler(
        train_dataset, batch_size=args.batch_size,
        shuffle=True) if num_processes > 1 else BatchSampler(
            train_dataset, batch_size=args.batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)

    val_sampler = BatchSampler(val_dataset,
                               batch_size=args.batch_size,
                               shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = len(train_dataloader)
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
    total_batch_size = args.batch_size * num_processes

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed) = {total_batch_size}"
    )
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=rank > 0)
    progress_bar.set_description("Steps")
    global_step = 0

    vae.train()
    for epoch in range(args.num_train_epochs):
        for batch in train_dataloader:
            logs = {"epoch": str(epoch).zfill(4)}
            for optimizer_idx in [0, 1]:
                # toggle_optimizer
                vae.toggle_optimizer(optimizers, optimizer_idx)
                loss, log_dict = vae.training_step(batch["image"],
                                                   optimizer_idx=optimizer_idx,
                                                   global_step=global_step)
                loss.backward()
                optimizers[optimizer_idx].step()
                # untoggle_optimizer
                vae.untoggle_optimizer(optimizers, optimizer_idx)

                optimizers[optimizer_idx].clear_grad()
                logs.update(log_dict)

            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix(**logs)

            if rank == 0:
                # logging
                if global_step % args.logging_steps == 0:
                    for name, val in logs.items():
                        if name == "epoch": continue
                        writer.add_scalar(name, val, step=global_step)

                if global_step % args.logging_image_steps == 0:
                    images_log = unwrap_model(vae).log_images(batch["image"])
                    for name, val in images_log.items():
                        writer.add_image(name,
                                         val,
                                         global_step,
                                         dataformats="NHWC")

                # saving
                if global_step % args.save_steps == 0:
                    run_evaluate(unwrap_model(vae), val_dataloader, writer,
                                 global_step)
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step))
                    unwrap_model(vae).save_pretrained(output_dir)

            del logs
            if global_step >= args.max_train_steps:
                break

    if rank == 0:
        writer.close()
        vae = unwrap_model(vae)
        vae.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
