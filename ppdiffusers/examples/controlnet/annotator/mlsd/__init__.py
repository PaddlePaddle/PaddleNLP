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

import cv2
import numpy as np
import paddle
from annotator.util import annotator_ckpts_path

from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .utils import pred_lines

remote_model_path = "https://bj.bcebos.com/v1/paddlenlp/models/community/ppdiffusers/mlsd_large_512_fp32.pdparams"


class MLSDdetector:
    def __init__(self):
        model_path = os.path.join(annotator_ckpts_path, "mlsd_large_512_fp32.pdparams")
        if not os.path.exists(model_path):
            from basicsr.utils.download_util import load_file_from_url

            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        self.model = MobileV2_MLSD_Large()
        self.model.eval()
        self.model.set_dict(paddle.load(model_path))

    def __call__(self, input_image, thr_v, thr_d):
        assert input_image.ndim == 3
        img = input_image
        img_output = np.zeros_like(img)
        with paddle.no_grad():
            lines = pred_lines(img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d)
            for line in lines:
                x_start, y_start, x_end, y_end = [int(val) for val in line]
                cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        return img_output[:, :, (0)]
