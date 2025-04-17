# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import random
from functools import partial

import cv2
import numpy as np
from paddle.vision.transforms import ColorJitter as PPColorJitter
from paddle.vision.transforms import functional as F
from PIL import Image, ImageFilter
from ppfleetx.utils.log import logger


class OperatorParamError(ValueError):
    """OperatorParamError"""

    pass


class DecodeImage(object):
    """decode image"""

    def __init__(self, to_rgb=True, channel_first=False):
        self.to_rgb = to_rgb
        self.channel_first = channel_first

    def __call__(self, img):
        assert type(img) is bytes and len(img) > 0, "invalid input 'img' in DecodeImage"
        data = np.frombuffer(img, dtype="uint8")
        img = cv2.imdecode(data, 1)
        if self.to_rgb:
            assert img.shape[2] == 3, "invalid shape of image[%s]" % (img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        return img


class UnifiedResize(object):
    def __init__(self, interpolation=None, backend="cv2"):
        _cv2_interp_from_str = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        _pil_interp_from_str = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "box": Image.BOX,
            "lanczos": Image.LANCZOS,
            "hamming": Image.HAMMING,
        }

        def _pil_resize(src, size, resample):
            pil_img = Image.fromarray(src)
            pil_img = pil_img.resize(size, resample)
            return np.asarray(pil_img)

        if backend.lower() == "cv2":
            if isinstance(interpolation, str):
                interpolation = _cv2_interp_from_str[interpolation.lower()]
            # compatible with opencv < version 4.4.0
            elif interpolation is None:
                interpolation = cv2.INTER_LINEAR
            self.resize_func = partial(cv2.resize, interpolation=interpolation)
        elif backend.lower() == "pil":
            if isinstance(interpolation, str):
                interpolation = _pil_interp_from_str[interpolation.lower()]
            self.resize_func = partial(_pil_resize, resample=interpolation)
        else:
            logger.warning(
                f'The backend of Resize only support "cv2" or "PIL". "f{backend}" is unavailable. Use "cv2" instead.'
            )
            self.resize_func = cv2.resize

    def __call__(self, src, size):
        return self.resize_func(src, size)


class ResizeImage(object):
    """resize image"""

    def __init__(self, size=None, resize_short=None, interpolation=None, backend="cv2"):
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w = None
            self.h = None
        elif size is not None:
            self.resize_short = None
            self.w = size if type(size) is int else size[0]
            self.h = size if type(size) is int else size[1]
        else:
            raise OperatorParamError(
                "invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None"
            )

        self._resize_func = UnifiedResize(interpolation=interpolation, backend=backend)

    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        return self._resize_func(img, (w, h))


class CenterCropImage(object):
    """crop image"""

    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


class RandCropImage(object):
    """random crop image"""

    def __init__(self, size, scale=None, ratio=None, interpolation=None, backend="cv2"):
        if type(size) is int:
            self.size = (size, size)  # (h, w)
        else:
            self.size = size

        self.scale = [0.08, 1.0] if scale is None else scale
        self.ratio = [3.0 / 4.0, 4.0 / 3.0] if ratio is None else ratio

        self._resize_func = UnifiedResize(interpolation=interpolation, backend=backend)

    def __call__(self, img):
        size = self.size
        scale = self.scale
        ratio = self.ratio

        aspect_ratio = math.sqrt(random.uniform(*ratio))
        w = 1.0 * aspect_ratio
        h = 1.0 / aspect_ratio

        img_h, img_w = img.shape[:2]

        bound = min((float(img_w) / img_h) / (w**2), (float(img_h) / img_w) / (h**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = img_w * img_h * random.uniform(scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = random.randint(0, img_w - w)
        j = random.randint(0, img_h - h)

        img = img[j : j + h, i : i + w, :]

        return self._resize_func(img, size)


class RandFlipImage(object):
    """random flip image
    flip_code:
        1: Flipped Horizontally
        0: Flipped Vertically
        -1: Flipped Horizontally & Vertically
    """

    def __init__(self, flip_code=1):
        assert flip_code in [-1, 0, 1], "flip_code should be a value in [-1, 0, 1]"
        self.flip_code = flip_code

    def __call__(self, img):
        if random.randint(0, 1) == 1:
            return cv2.flip(img, self.flip_code)
        else:
            return img


class NormalizeImage(object):
    """normalize image such as substract mean, divide std"""

    def __init__(self, scale=None, mean=None, std=None, order="chw", output_fp16=False, channel_num=3):
        if isinstance(scale, str):
            scale = eval(scale)
        assert channel_num in [3, 4], "channel number of input image should be set to 3 or 4."
        self.channel_num = channel_num
        self.output_dtype = "float16" if output_fp16 else "float32"
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if self.order == "chw" else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"

        img = (img.astype("float32") * self.scale - self.mean) / self.std

        if self.channel_num == 4:
            img_h = img.shape[1] if self.order == "chw" else img.shape[0]
            img_w = img.shape[2] if self.order == "chw" else img.shape[1]
            pad_zeros = np.zeros((1, img_h, img_w)) if self.order == "chw" else np.zeros((img_h, img_w, 1))
            img = (
                np.concatenate((img, pad_zeros), axis=0)
                if self.order == "chw"
                else np.concatenate((img, pad_zeros), axis=2)
            )
        return img.astype(self.output_dtype)


class ToCHWImage(object):
    """convert hwc image to chw image"""

    def __init__(self):
        pass

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        return img.transpose((2, 0, 1))


class ColorJitter(PPColorJitter):
    """ColorJitter."""

    def __init__(self, *args, **kwargs):
        self.p = kwargs.pop("p", 1.0)
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        if random.random() < self.p:
            if not isinstance(img, Image.Image):
                img = np.ascontiguousarray(img)
                img = Image.fromarray(img)
            img = super()._apply_image(img)
            if isinstance(img, Image.Image):
                img = np.asarray(img)
        return img


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0], p=1.0):
        self.p = p
        self.sigma = sigma

    def __call__(self, img):
        if random.random() < self.p:
            if not isinstance(img, Image.Image):
                img = np.ascontiguousarray(img)
                img = Image.fromarray(img)
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            if isinstance(img, Image.Image):
                img = np.asarray(img)
        return img


class Pixels(object):
    def __init__(self, mode="const", mean=[0.0, 0.0, 0.0]):
        self._mode = mode
        self._mean = mean

    def __call__(self, h=224, w=224, c=3):
        if self._mode == "rand":
            return np.random.normal(size=(1, 1, 3))
        elif self._mode == "pixel":
            return np.random.normal(size=(h, w, c))
        elif self._mode == "const":
            return self._mean
        else:
            raise Exception('Invalid mode in RandomErasing, only support "const", "rand", "pixel"')


class RandomErasing(object):
    """RandomErasing.
    This code is adapted from https://github.com/zhunzhong07/Random-Erasing, and refer to Timm.
    """

    def __init__(
        self,
        EPSILON=0.5,
        sl=0.02,
        sh=0.4,
        r1=0.3,
        mean=[0.0, 0.0, 0.0],
        attempt=100,
        use_log_aspect=False,
        mode="const",
    ):
        self.EPSILON = eval(EPSILON) if isinstance(EPSILON, str) else EPSILON
        self.sl = eval(sl) if isinstance(sl, str) else sl
        self.sh = eval(sh) if isinstance(sh, str) else sh
        r1 = eval(r1) if isinstance(r1, str) else r1
        self.r1 = (math.log(r1), math.log(1 / r1)) if use_log_aspect else (r1, 1 / r1)
        self.use_log_aspect = use_log_aspect
        self.attempt = attempt
        self.get_pixels = Pixels(mode, mean)

    def __call__(self, img):
        if random.random() > self.EPSILON:
            return img

        for _ in range(self.attempt):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(*self.r1)
            if self.use_log_aspect:
                aspect_ratio = math.exp(aspect_ratio)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                pixels = self.get_pixels(h, w, img.shape[2])
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if img.shape[2] == 3:
                    img[x1 : x1 + h, y1 : y1 + w, :] = pixels
                else:
                    img[x1 : x1 + h, y1 : y1 + w, 0] = pixels[0]
                return img
        return img


class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    Args:
        p (float): probability that image should be converted to grayscale.
    Returns:
        PIL Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b
    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.
        Returns:
            PIL Image: Randomly grayscaled image.
        """

        flag = False
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)
            flag = True

        num_output_channels = 1 if img.mode == "L" else 3

        if random.random() < self.p:
            img = F.to_grayscale(img, num_output_channels=num_output_channels)

        if flag:
            img = np.asarray(img)

        return img
