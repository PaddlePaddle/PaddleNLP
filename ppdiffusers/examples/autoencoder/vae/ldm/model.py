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
from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdiffusers.configuration_utils import ConfigMixin, register_to_config
from ppdiffusers.initializer import reset_initialized_parameter
from ppdiffusers.models.autoencoder_kl import (
    AutoencoderKLOutput,
    Decoder,
    DecoderOutput,
    DiagonalGaussianDistribution,
    Encoder,
)
from ppdiffusers.models.modeling_utils import ModelMixin

from .losses import LPIPSWithDiscriminator


def count_params(model, verbose=True):
    total_params = sum(p.numel() for p in model.parameters()).item()
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


# regist a new model
class AutoencoderKLWithLoss(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = (
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        down_block_out_channels: Tuple[int] = None,
        up_block_types: Tuple[str] = (
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
        up_block_out_channels: Tuple[int] = None,
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 512,
        # new add
        input_size: Tuple[int] = None,
        # loss arguments
        disc_start=50001,
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
        disc_loss="hinge",
    ):
        super().__init__()
        self.input_size = [int(_) for _ in input_size] if input_size is not None else None
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=down_block_out_channels
            if down_block_out_channels
            is not None  # if down_block_out_channels not givien, we will use block_out_channels
            else block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=up_block_out_channels  # if up_block_out_channels not givien, we will use block_out_channels
            if up_block_out_channels is not None
            else block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.quant_conv = nn.Conv2D(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2D(latent_channels, latent_channels, 1)

        # register a loss function
        self.loss = LPIPSWithDiscriminator(
            disc_start=disc_start,
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
            disc_loss=disc_loss,
        )
        count_params(self)
        self.init_weights()

    def init_weights(self):
        reset_initialized_parameter(self.encoder)
        reset_initialized_parameter(self.decoder)
        reset_initialized_parameter(self.quant_conv)
        reset_initialized_parameter(self.post_quant_conv)

    def custom_forward(
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

    def forward(self, pixel_values, optimizer_idx=0, global_step=0):
        # make sure we are in train mode
        self.train()
        if self.input_size is None:
            encoder_inputs = pixel_values
        else:
            encoder_inputs = F.interpolate(pixel_values, size=self.input_size, mode="bilinear")

        reconstructions, posterior = self.custom_forward(encoder_inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(
                pixel_values,
                reconstructions,
                posterior,
                optimizer_idx,
                global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
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
                split="train",
            )
            return discloss, log_dict_disc

    @paddle.no_grad()
    def log_images(self, pixel_values, only_inputs=False, **kwargs):
        self.eval()
        log = dict()
        if self.input_size is None:
            encoder_inputs = pixel_values
        else:
            encoder_inputs = F.interpolate(pixel_values, size=self.input_size, mode="bilinear")

        if not only_inputs:
            xrec, posterior = self.custom_forward(encoder_inputs)
            log["samples"] = self.decode_image(self.decode(paddle.randn(posterior.sample().shape)).sample)
            log["reconstructions"] = self.decode_image(xrec)
        # update
        log["encoder_inputs"] = self.decode_image(encoder_inputs)
        self.train()
        return log

    def decode_image(self, image):
        image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1])
        image = (image * 255.0).cast("float32").numpy().round()
        return image

    @paddle.no_grad()
    def validation_step(self, pixel_values, global_step=0):
        self.eval()
        if self.input_size is None:
            encoder_inputs = pixel_values
        else:
            encoder_inputs = F.interpolate(pixel_values, size=self.input_size, mode="bilinear")

        reconstructions, posterior = self.custom_forward(encoder_inputs)
        aeloss, log_dict_ae = self.loss(
            pixel_values,
            reconstructions,
            posterior,
            0,
            global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        discloss, log_dict_disc = self.loss(
            pixel_values,
            reconstructions,
            posterior,
            1,
            global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )
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
                        param.stop_gradient = self._param_stop_gradient_state[param]
        # save memory
        self._param_stop_gradient_state = {}

    def encode(self, x: paddle.Tensor, return_dict: bool = True):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: paddle.Tensor, return_dict: bool = True):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
