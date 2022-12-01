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
"""
This code is rewritten by Paddle based on Jina-ai/discoart.
https://github.com/jina-ai/discoart/blob/main/discoart/nn/make_cutouts.py
"""
import paddle
import paddle.nn as nn
from paddle.nn import functional as F
from .resize_right import resize

from . import transforms as T

skip_augs = False
padargs = {}


class MakeCutoutsDango(nn.Layer):
    def __init__(self, cut_size, Overview=4, InnerCrop=0, IC_Size_Pow=0.5, IC_Grey_P=0.2):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.augs = nn.Sequential(
            *[
                T.RandomHorizontalFlip(prob=0.5),
                T.Lambda(lambda x: x + paddle.randn(x.shape) * 0.01),
                T.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.Lambda(lambda x: x + paddle.randn(x.shape) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + paddle.randn(x.shape) * 0.01),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ]
        )

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(
            input,
            (
                (sideY - max_size) // 2,
                (sideY - max_size) // 2,
                (sideX - max_size) // 2,
                (sideX - max_size) // 2,
            ),
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
                    cutouts.append(cutout[:, :, :, ::-1])
                if self.Overview == 4:
                    cutouts.append(gray(cutout[:, :, :, ::-1]))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(paddle.rand([1]) ** self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = paddle.randint(0, sideX - size + 1)
                offsety = paddle.randint(0, sideY - size + 1)
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)

        cutouts = paddle.concat(cutouts)
        if skip_augs is not True:
            cutouts = self.augs(cutouts)
        return cutouts
