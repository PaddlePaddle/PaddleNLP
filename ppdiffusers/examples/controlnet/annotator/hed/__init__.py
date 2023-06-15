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

import os

import cv2
import numpy as np
import paddle

from ..util import annotator_ckpts_path


class Network(paddle.nn.Layer):
    def __init__(self, model_path=None):
        super().__init__()

        self.netVggOne = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
        )

        self.netVggTwo = paddle.nn.Sequential(
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
        )

        self.netVggThr = paddle.nn.Sequential(
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
        )

        self.netVggFou = paddle.nn.Sequential(
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
        )

        self.netVggFiv = paddle.nn.Sequential(
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
        )

        self.netScoreOne = paddle.nn.Conv2D(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = paddle.nn.Conv2D(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = paddle.nn.Conv2D(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = paddle.nn.Conv2D(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = paddle.nn.Conv2D(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0), paddle.nn.Sigmoid()
        )

        if model_path:
            self.set_state_dict(paddle.load(model_path))

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - paddle.to_tensor(
            [104.00698793, 116.66876762, 122.67891434],
            dtype=tenInput.dtype,
        ).reshape([1, 3, 1, 1])

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = paddle.nn.functional.interpolate(
            tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode="bilinear", align_corners=False
        )
        tenScoreTwo = paddle.nn.functional.interpolate(
            tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode="bilinear", align_corners=False
        )
        tenScoreThr = paddle.nn.functional.interpolate(
            tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode="bilinear", align_corners=False
        )
        tenScoreFou = paddle.nn.functional.interpolate(
            tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode="bilinear", align_corners=False
        )
        tenScoreFiv = paddle.nn.functional.interpolate(
            tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode="bilinear", align_corners=False
        )

        return self.netCombine(paddle.concat([tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1))


remote_model_path = (
    "https://paddlenlp.bj.bcebos.com/models/community/westfish/network-bsds500-paddle/network-bsds500.pdparams"
)


class HEDdetector:
    def __init__(self, modelpath=None):
        modelpath = os.path.join(annotator_ckpts_path, "network-bsds500.pdparams")
        if not os.path.exists(modelpath):
            from paddlenlp.utils.downloader import get_path_from_url_with_filelock

            get_path_from_url_with_filelock(remote_model_path, root_dir=annotator_ckpts_path)
        self.model_path = modelpath
        self.netNetwork = Network(modelpath)
        self.netNetwork.eval()

    def __call__(self, input_image):
        assert input_image.ndim == 3
        input_image = input_image[:, :, ::-1].copy()
        with paddle.no_grad():
            image_hed = paddle.to_tensor(input_image).astype(paddle.float32)
            image_hed = image_hed / 255.0
            image_hed = image_hed.transpose([2, 0, 1]).unsqueeze(axis=0)
            edge = self.netNetwork(image_hed)[0]
            edge = (edge.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            return edge[0]


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z
