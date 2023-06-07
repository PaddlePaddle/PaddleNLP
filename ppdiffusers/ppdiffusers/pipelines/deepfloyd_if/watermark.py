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

from typing import List

import paddle
import PIL
from PIL import Image

from ...configuration_utils import ConfigMixin
from ...models.modeling_utils import ModelMixin
from ...utils import PIL_INTERPOLATION


class IFWatermarker(ModelMixin, ConfigMixin):
    def __init__(self):
        super().__init__()

        self.register_buffer("watermark_image", paddle.zeros((62, 62, 4), dtype=paddle.get_default_dtype()))
        self.watermark_image_as_pil = None

    def apply_watermark(self, images: List[PIL.Image.Image], sample_size=None):
        # copied from https://github.com/deep-floyd/IF/blob/b77482e36ca2031cb94dbca1001fc1e6400bf4ab/deepfloyd_if/modules/base.py#L287

        h = images[0].height
        w = images[0].width

        sample_size = sample_size or h

        coef = min(h / sample_size, w / sample_size)
        img_h, img_w = (int(h / coef), int(w / coef)) if coef < 1 else (h, w)

        S1, S2 = 1024**2, img_w * img_h
        K = (S2 / S1) ** 0.5
        wm_size, wm_x, wm_y = int(K * 62), img_w - int(14 * K), img_h - int(14 * K)

        if self.watermark_image_as_pil is None:
            watermark_image = self.watermark_image.cpu().numpy().astype("uint8")
            watermark_image = Image.fromarray(watermark_image, mode="RGBA")
            self.watermark_image_as_pil = watermark_image

        wm_img = self.watermark_image_as_pil.resize(
            (wm_size, wm_size), PIL_INTERPOLATION["bicubic"], reducing_gap=None
        )

        for pil_img in images:
            pil_img.paste(wm_img, box=(wm_x - wm_size, wm_y - wm_size, wm_x, wm_y), mask=wm_img.split()[-1])

        return images
