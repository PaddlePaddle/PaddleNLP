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
modified from
https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/vision/transforms/transforms.py
"""

from __future__ import division

import collections
import numbers
import random
import sys

from paddle.vision.transforms import BaseTransform, Compose

from . import functional as F

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif F._is_numpy_image(img):
        return img.shape[:2][::-1]
    elif F._is_tensor_image(img):
        if len(img.shape) == 3:
            return img.shape[1:][::-1]  # chw -> wh
        elif len(img.shape) == 4:
            return img.shape[2:][::-1]  # nchw -> wh
        else:
            raise ValueError("The dim for input Tensor should be 3-D or 4-D, but received {}".format(len(img.shape)))
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


def _check_input(value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError("If {} is a single number, it must be non negative.".format(name))
        value = [center - value, center + value]
        if clip_first_on_zero:
            value[0] = max(value[0], 0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError("{} values should be between {}".format(name, bound))
    else:
        raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

    if value[0] == value[1] == center:
        value = None
    return value


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError(f"{name} should be a sequence of length {msg}.")
    if len(x) not in req_sizes:
        raise ValueError(f"{name} should be sequence of length {msg}.")


def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(f"If {name} is a single number, it must be positive.")
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


class RandomAffine(BaseTransform):
    """Random affine transformation of the image.

    Args:
        degrees (int|float|tuple): The angle interval of the random rotation.
            If set as a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees) in clockwise order. If set 0, will not rotate.
        translate (tuple, optional): Maximum absolute fraction for horizontal and vertical translations.
            For example translate=(a, b), then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a
            and vertical shift is randomly sampled in the range -img_height * b < dy < img_height * b.
            Default is None, will not translate.
        scale (tuple, optional): Scaling factor interval, e.g (a, b), then scale is randomly sampled from the range a <= scale <= b.
            Default is None, will keep original scale and not scale.
        shear (sequence or number, optional): Range of degrees to shear, ranges from -180 to 180 in clockwise order.
            If set as a number, a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            Else if set as a sequence of 2 values a shear parallel to the x axis in the range (shear[0], shear[1]) will be applied.
            Else if set as a sequence of 4 values, a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Default is None, will not apply shear.
        interpolation (str, optional): Interpolation method. If omitted, or if the
            image has only one channel, it is set to PIL.Image.NEAREST or cv2.INTER_NEAREST
            according the backend.
            When use pil backend, support method are as following:
            - "nearest": Image.NEAREST,
            - "bilinear": Image.BILINEAR,
            - "bicubic": Image.BICUBIC
            When use cv2 backend, support method are as following:
            - "nearest": cv2.INTER_NEAREST,
            - "bilinear": cv2.INTER_LINEAR,
            - "bicubic": cv2.INTER_CUBIC
        fill (int|list|tuple, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        center (2-tuple, optional): Optional center of rotation, (x, y).
            Origin is the upper left corner.
            Default is the center of the image.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): An affined image.

    Returns:
        A callable object of RandomAffine.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.vision.transforms import RandomAffine

            transform = RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[-10, 10])

            fake_img = paddle.randn((3, 256, 300)).astype(paddle.float32)

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(
        self, degrees, translate=None, scale=None, shear=None, interpolation="nearest", fill=0, center=None, keys=None
    ):
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        super(RandomAffine, self).__init__(keys)
        assert interpolation in ["nearest", "bilinear", "bicubic"]
        self.interpolation = interpolation

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")
        self.fill = fill

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))
        self.center = center

    def _get_param(self, img_size, degrees, translate=None, scale_ranges=None, shears=None):
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])

        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(random.uniform(-max_dx, max_dx))
            ty = int(random.uniform(-max_dy, max_dy))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        shear_x, shear_y = 0.0, 0.0
        if shears is not None:
            shear_x = random.uniform(shears[0], shears[1])
            if len(shears) == 4:
                shear_y = random.uniform(shears[2], shears[3])
        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def _apply_image(self, img):
        """
        Args:
            img (PIL.Image|np.array): Image to be affine transformed.

        Returns:
            PIL.Image or np.array: Affine transformed image.
        """

        w, h = _get_image_size(img)
        img_size = [w, h]

        ret = self._get_param(img_size, self.degrees, self.translate, self.scale, self.shear)

        return F.affine(img, *ret, interpolation=self.interpolation, fill=self.fill, center=self.center)


class BrightnessTransform(BaseTransform):
    """Adjust brightness of the image.

    Args:
        value (float): How much to adjust the brightness. Can be any
            non negative number. 0 gives the original image
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): An image with a transform in brghtness.

    Returns:
        A callable object of BrightnessTransform.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import BrightnessTransform

            transform = BrightnessTransform(0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)

    """

    def __init__(self, value, keys=None):
        super(BrightnessTransform, self).__init__(keys)
        self.value = _check_input(value, "brightness")

    def _apply_image(self, img):
        if self.value is None:
            return img

        brightness_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_brightness(img, brightness_factor)


class ContrastTransform(BaseTransform):
    """Adjust contrast of the image.

    Args:
        value (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives the original image
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): An image with a transform in contrast.

    Returns:
        A callable object of ContrastTransform.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import ContrastTransform

            transform = ContrastTransform(0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)

    """

    def __init__(self, value, keys=None):
        super(ContrastTransform, self).__init__(keys)
        if value < 0:
            raise ValueError("contrast value should be non-negative")
        self.value = _check_input(value, "contrast")

    def _apply_image(self, img):
        if self.value is None:
            return img

        contrast_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_contrast(img, contrast_factor)


class SaturationTransform(BaseTransform):
    """Adjust saturation of the image.

    Args:
        value (float): How much to adjust the saturation. Can be any
            non negative number. 0 gives the original image
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): An image with a transform in saturation.

    Returns:
        A callable object of SaturationTransform.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import SaturationTransform

            transform = SaturationTransform(0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)

    """

    def __init__(self, value, keys=None):
        super(SaturationTransform, self).__init__(keys)
        self.value = _check_input(value, "saturation")

    def _apply_image(self, img):
        if self.value is None:
            return img

        saturation_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_saturation(img, saturation_factor)


class HueTransform(BaseTransform):
    """Adjust hue of the image.

    Args:
        value (float): How much to adjust the hue. Can be any number
            between 0 and 0.5, 0 gives the original image
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): An image with a transform in hue.

    Returns:
        A callable object of HueTransform.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import HueTransform

            transform = HueTransform(0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)

    """

    def __init__(self, value, keys=None):
        super(HueTransform, self).__init__(keys)
        self.value = _check_input(value, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _apply_image(self, img):
        if self.value is None:
            return img

        hue_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_hue(img, hue_factor)


class ColorJitter(BaseTransform):
    """Randomly change the brightness, contrast, saturation and hue of an image.

    Args:
        brightness (float): How much to jitter brightness.
            Chosen uniformly from [max(0, 1 - brightness), 1 + brightness]. Should be non negative numbers.
        contrast (float): How much to jitter contrast.
            Chosen uniformly from [max(0, 1 - contrast), 1 + contrast]. Should be non negative numbers.
        saturation (float): How much to jitter saturation.
            Chosen uniformly from [max(0, 1 - saturation), 1 + saturation]. Should be non negative numbers.
        hue (float): How much to jitter hue.
            Chosen uniformly from [-hue, hue]. Should have 0<= hue <= 0.5.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A color jittered image.

    Returns:
        A callable object of ColorJitter.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import ColorJitter

            transform = ColorJitter(0.4, 0.4, 0.4, 0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)

    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, keys=None):
        super(ColorJitter, self).__init__(keys)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def _get_param(self, brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            transforms.append(BrightnessTransform(brightness, self.keys))

        if contrast is not None:
            transforms.append(ContrastTransform(contrast, self.keys))

        if saturation is not None:
            transforms.append(SaturationTransform(saturation, self.keys))

        if hue is not None:
            transforms.append(HueTransform(hue, self.keys))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def _apply_image(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self._get_param(self.brightness, self.contrast, self.saturation, self.hue)
        return transform(img)


class Lambda(BaseTransform):
    """
    Modified from
    https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/
    pytorch_project_convertor/API_docs/vision/torchvision.transforms.Lambda.md
    """

    def __init__(self, lambd, keys=None):
        super(Lambda, self).__init__(keys)
        if not callable(lambd):
            raise TypeError("Argument lambd should be callable, got {}".format(repr(type(lambd).__name__)))
        self.lambd = lambd

    def _apply_image(self, img):
        return self.lambd(img)


class Grayscale(BaseTransform):
    """Converts image to grayscale.
    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.
    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): Grayscale version of the input image.
            - If output_channels == 1 : returned image is single channel
            - If output_channels == 3 : returned image is 3 channel with r == g == b
    Returns:
        A callable object of Grayscale.
    Examples:

        .. code-block:: python
            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import Grayscale
            transform = Grayscale()
            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))
            fake_img = transform(fake_img)
            print(np.array(fake_img).shape)
    """

    def __init__(self, num_output_channels=1, keys=None):
        super(Grayscale, self).__init__(keys)
        self.num_output_channels = num_output_channels

    def _apply_image(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.
        Returns:
            PIL Image: Randomly grayscaled image.
        """
        return F.to_grayscale(img, self.num_output_channels)


class RandomGrayscale(BaseTransform):
    """
    RandomGrayscale vision transforms.
    """

    def __init__(self, prob, keys=None):
        super(RandomGrayscale, self).__init__(keys)
        assert 0 <= prob <= 1, "probability must be between 0 and 1"
        self.prob = prob

    def _apply_image(self, img):
        if random.random() < self.prob:
            return F.to_grayscale(img, num_output_channels=3)
        return img


class RandomHorizontalFlip(BaseTransform):
    """Horizontally flip the input data randomly with a given probability.
    Args:
        prob (float, optional): Probability of the input data being flipped. Should be in [0, 1]. Default: 0.5
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.
    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A horiziotal flipped image.
    Returns:
        A callable object of RandomHorizontalFlip.
    Examples:

        .. code-block:: python
            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import RandomHorizontalFlip
            transform = RandomHorizontalFlip(0.5)
            fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))
            fake_img = transform(fake_img)
            print(fake_img.size)
    """

    def __init__(self, prob=0.5, keys=None):
        super(RandomHorizontalFlip, self).__init__(keys)
        assert 0 <= prob <= 1, "probability must be between 0 and 1"
        self.prob = prob

    def _apply_image(self, img):
        if random.random() < self.prob:
            return F.hflip(img)
        return img
