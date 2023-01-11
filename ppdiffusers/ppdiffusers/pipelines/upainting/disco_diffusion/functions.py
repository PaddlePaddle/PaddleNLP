# -*- coding: utf-8 -*-
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

"""
modified from https://github.com/alembics/disco-diffusion/blob/main/disco.py
"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T

from ..paddle_patches.functional import hflip
from ..paddle_patches.transforms import (
    ColorJitter,
    Grayscale,
    Lambda,
    RandomAffine,
    RandomGrayscale,
    RandomHorizontalFlip,
)

# from .resize_right import resize


padargs = {}


def parse_prompt(prompt):
    """
    Parse prompts and weights.
    """
    if prompt.startswith("http://") or prompt.startswith("https://"):
        vals = prompt.rsplit(":", 2)
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", "1"][len(vals) :]
    return vals[0], float(vals[1])


def resize(x, out_shape):
    if len(out_shape) == 4:
        out_shape = out_shape[2:]
    return paddle.nn.functional.interpolate(x, out_shape)


class MakeCutoutsDango(nn.Layer):
    """
    Make cutouts.
    """

    def __init__(self, cut_size, animation_mode, skip_augs, Overview=4, InnerCrop=0, IC_Size_Pow=0.5, IC_Grey_P=0.2):
        super().__init__()
        self.cut_size = cut_size
        self.skip_augs = skip_augs
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        if animation_mode == "None":
            self.augs = T.Compose(
                [
                    RandomHorizontalFlip(prob=0.5),
                    Lambda(lambda x: x + paddle.randn(x.shape) * 0.01),
                    RandomAffine(degrees=10, translate=(0.05, 0.05), interpolation="bilinear"),
                    Lambda(lambda x: x + paddle.randn(x.shape) * 0.01),
                    RandomGrayscale(prob=0.1),
                    Lambda(lambda x: x + paddle.randn(x.shape) * 0.01),
                    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                ]
            )
        elif animation_mode == "Video Input":
            raise NotImplementedError
        elif animation_mode == "2D" or animation_mode == "3D":
            raise NotImplementedError

    def forward(self, input):
        """
        input: BCHW
        """
        cutouts = []
        gray = Grayscale(num_output_channels=3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(
            input,
            ((sideY - max_size) // 2, (sideY - max_size) // 2, (sideX - max_size) // 2, (sideX - max_size) // 2),
            **padargs,
        )
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1:
                    cutouts.append(cutout)
                if self.Overview >= 2:
                    cutouts.append(gray(cutout))
                if self.Overview >= 3:
                    cutouts.append(hflip(cutout))
                if self.Overview == 4:
                    cutouts.append(gray(hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int((np.random.rand()) ** self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = int(np.random.randint(0, sideX - size + 1, [1]))
                offsety = int(np.random.randint(0, sideY - size + 1, [1]))
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
        cutouts = paddle.concat(cutouts)

        if self.skip_augs is not True:
            cutouts = self.augs(cutouts)
        return cutouts


def spherical_dist_loss(x, y):
    """spherical_dist_loss"""
    x = F.normalize(x, axis=-1)
    y = F.normalize(y, axis=-1)
    return ((x - y).norm(axis=-1) / 2.0).asin().pow(2) * 2


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    """range_loss"""
    return (input - input.clip(-1, 1)).pow(2).mean([1, 2, 3])


def sat_loss(input):
    """sat_loss"""
    return (input - input.clip(-1, 1)).abs().mean()
