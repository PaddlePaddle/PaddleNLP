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

import numpy as np
import paddle
import paddlehub as hub
from annotator.util import annotator_ckpts_path

from . import util


class OpenposePaddleDetector:
    def __init__(self):
        self.body_estimation = hub.Module(name="openpose_body_estimation")
        self.hand_estimation = hub.Module(name="openpose_hands_estimation")

    def __call__(self, oriImg, hand=False):
        oriImg = oriImg[:, :, ::-1].copy()
        with paddle.no_grad():
            canvas = oriImg[:, :, ::-1].copy()
            canvas.fill(0)
            result = self.body_estimation.predict(oriImg, save_path="saved_images", visualization=False)
            canvas = self.body_estimation.draw_pose(canvas, result["candidate"], result["subset"])
            if hand:
                hands_list = util.hand_detect(result["candidate"], result["subset"], oriImg)
                all_hand_peaks = []
                for x, y, w, is_left in hands_list:
                    scale_search = [0.5, 1.0, 1.5, 2.0]
                    peaks = self.hand_estimation.hand_estimation(
                        oriImg[y : y + w, x : x + w, :], scale_search=scale_search
                    )
                    peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                    peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                    all_hand_peaks.append(peaks)
                canvas = self.hand_estimation.draw_hand(canvas, all_hand_peaks)

            return canvas, dict(candidate=result["candidate"].tolist(), subset=result["subset"].tolist())
