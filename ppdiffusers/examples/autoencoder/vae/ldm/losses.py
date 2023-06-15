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

import functools
from collections import namedtuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.utils.download import get_weights_path_from_url

from ppdiffusers.initializer import constant_, normal_, reset_initialized_parameter

model_urls = {
    "vgg16": (
        "https://paddlenlp.bj.bcebos.com/models/lpips_vgg16/lpips_vgg16.pdparams",
        "a1583475db9e49334735f2866847ae41",
    ),
    "vgg_netlin": (
        "https://paddlenlp.bj.bcebos.com/models/lpips_vgg16/vgg_netlin.pdparams",
        "f3ae85f16a1a243e789606ae0c4a59a1",
    ),
}


class ActNorm(nn.Layer):
    def __init__(self, num_features, logdet=False, affine=True, allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = self.create_parameter((1, num_features, 1, 1), default_initializer=nn.initializer.Constant(0))
        self.scale = self.create_parameter((1, num_features, 1, 1), default_initializer=nn.initializer.Constant(1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", paddle.to_tensor(0, dtype=paddle.int64))

    @paddle.no_grad()
    def initialize(self, input):
        flatten = input.transpose([1, 0, 2, 3]).reshape([input.shape[1], -1])
        mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).transpose([1, 0, 2, 3])
        std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).transpose([1, 0, 2, 3])

        self.loc.set_value(-mean)
        self.scale.set_value(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.set_value(paddle.to_tensor(1, dtype=self.initialized.dtype))

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = paddle.log(paddle.abs(self.scale))
            logdet = height * width * paddle.sum(log_abs)
            logdet = logdet * input.shape[0]
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.set_value(paddle.to_tensor(1, dtype=self.initialized.dtype))

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = paddle.mean(F.relu(1.0 - logits_real))
    loss_fake = paddle.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (paddle.mean(F.softplus(-logits_real)) + paddle.mean(F.softplus(logits_fake)))
    return d_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        normal_(m.weight, 1.0, 0.02)
        constant_(m.bias, 0.0)


class NLayerDiscriminator(nn.Layer):
    r"""Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    Construct a PatchGAN discriminator

    Parameters:
        input_nc (int)  -- the number of channels in input images
        ndf (int)       -- the number of filters in the last conv layer
        n_layers (int)  -- the number of conv layers in the discriminator
        norm_layer      -- normalization layer
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        super().__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2D
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2D
        else:
            use_bias = norm_layer != nn.BatchNorm2D

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2D(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2D(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias_attr=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2D(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias_attr=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2),
        ]

        sequence += [
            nn.Conv2D(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
    return nn.Upsample(size=out_HW, mode="bilinear", align_corners=False)(in_tens)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = paddle.sqrt(paddle.sum(in_feat**2, axis=1, keepdim=True))
    return in_feat / (norm_factor + eps)


class NetLinLayer(nn.Layer):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2D(chn_in, chn_out, 1, stride=1, padding=0, bias_attr=False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ScalingLayer(nn.Layer):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift",
            paddle.to_tensor(np.asarray([-0.030, -0.088, -0.188]).astype("float32")[None, :, None, None]),
        )
        self.register_buffer(
            "scale",
            paddle.to_tensor(np.asarray([0.458, 0.448, 0.450]).astype("float32")[None, :, None, None]),
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class VGG16(nn.Layer):
    def __init__(self, pretrained=True, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_model = paddle.vision.models.vgg16(pretrained=False)
        if pretrained:
            state_dict = paddle.load(get_weights_path_from_url(*model_urls["vgg16"]))
            vgg_model.set_state_dict(state_dict)
        vgg_pretrained_features = vgg_model.features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_sublayer(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.stop_gradient = True

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class LPIPS(nn.Layer):
    def __init__(
        self,
        pretrained=True,
        net="alex",
        lpips=True,
        spatial=False,
        pnet_rand=False,
        pnet_tune=False,
        use_dropout=True,
        model_path=None,
        eval_mode=True,
        verbose=True,
    ):
        # lpips - [True] means with linear calibration on top of base network
        # pretrained - [True] means load linear weights

        super(LPIPS, self).__init__()
        if verbose:
            print(
                "Setting up [%s] perceptual loss: trunk [%s], spatial [%s]"
                % ("LPIPS" if lpips else "baseline", net, "on" if spatial else "off")
            )

        self.pnet_type = net.lower()
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips  # false means baseline of just averaging all layers
        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ["vgg", "vgg16"]:
            net_type = VGG16
            self.chns = [64, 128, 256, 512, 512]
        else:
            raise NotImplementedError
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if lpips:
            lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            lins = [lin0, lin1, lin2, lin3, lin4]
            if self.pnet_type == "squeeze":  # 7 layers for squeezenet
                lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                lins += [lin5, lin6]
            self.lins = nn.LayerList(lins)

            if pretrained:
                if model_path is None:
                    model_path = get_weights_path_from_url(*model_urls["vgg_netlin"])
                if verbose:
                    print("Loading model from: %s" % model_path)
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.set_state_dict(paddle.load(model_path))

        if eval_mode:
            self.eval()
        for param in self.parameters():
            param.stop_gradient = True

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1))
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if self.spatial:
                res = [upsample(diffs[kk].sum(axis=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(axis=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1, self.L):
            val += res[l]

        if retPerLayer:
            return (val, res)
        else:
            return val


class LPIPSWithDiscriminator(nn.Layer):
    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        # LPIPS
        self.perceptual_loss = LPIPS(net="vgg")
        self.perceptual_loss.eval()

        self.perceptual_weight = perceptual_weight
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        )
        reset_initialized_parameter(self.discriminator)
        self.discriminator.apply(weights_init)

        # output log variance
        self.logvar = self.create_parameter((1,), default_initializer=nn.initializer.Constant(logvar_init))

        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = paddle.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = paddle.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = paddle.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = paddle.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = paddle.norm(nll_grads) / (paddle.norm(g_grads) + 1e-4)
        d_weight = paddle.clip(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        weights=None,
    ):
        rec_loss = paddle.abs(inputs - reconstructions)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)

            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / paddle.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = paddle.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = paddle.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = paddle.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions)
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(paddle.concat((reconstructions, cond), axis=1))
            g_loss = -paddle.mean(logits_fake)
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except Exception:
                    assert not self.training
                    d_weight = paddle.to_tensor(0.0)
            else:
                d_weight = paddle.to_tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean().item(),
                "{}/logvar".format(split): self.logvar.detach().item(),
                "{}/kl_loss".format(split): kl_loss.detach().mean().item(),
                "{}/nll_loss".format(split): nll_loss.detach().mean().item(),
                "{}/rec_loss".format(split): rec_loss.detach().mean().item(),
                "{}/d_weight".format(split): d_weight.detach().item(),
                "{}/disc_factor".format(split): paddle.to_tensor(disc_factor).item(),
                "{}/g_loss".format(split): g_loss.detach().mean().item(),
            }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.detach())
                logits_fake = self.discriminator(reconstructions.detach())
            else:
                logits_real = self.discriminator(paddle.concat((inputs.detach(), cond), axis=1))
                logits_fake = self.discriminator(paddle.concat((reconstructions.detach(), cond), axis=1))
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean().item(),
                "{}/logits_real".format(split): logits_real.detach().mean().item(),
                "{}/logits_fake".format(split): logits_fake.detach().mean().item(),
            }
            return d_loss, log
