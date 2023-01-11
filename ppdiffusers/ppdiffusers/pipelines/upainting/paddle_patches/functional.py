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
https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/vision/transforms/functional.py
"""

from __future__ import division

import math
import numbers

import numpy as np
import paddle
from PIL import Image

from . import functional_tensor as F_t

__all__ = []


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return isinstance(img, paddle.Tensor)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _get_affine_matrix(center, angle, translate, scale, shear):
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    # Ihe inverse one is : M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1
    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    # Rotate and Shear without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Center Translation
    cx, cy = center
    tx, ty = translate

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x / scale for x in matrix]
    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    return matrix


def affine(img, angle, translate, scale, shear, interpolation="nearest", fill=0, center=None):
    """Apply affine transformation on the image.

    Args:
        img (PIL.Image|np.array|paddle.Tensor): Image to be affined.
        angle (int|float): The angle of the random rotation in clockwise order.
        translate (list[float]): Maximum absolute fraction for horizontal and vertical translations.
        scale (float): Scale factor for the image, scale should be positive.
        shear (list[float]): Shear angle values which are parallel to the x-axis and y-axis in clockwise order.
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

    Returns:
        PIL.Image|np.array|paddle.Tensor: Affine Transformed image.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.transforms import functional as F

            fake_img = paddle.randn((3, 256, 300)).astype(paddle.float32)

            affined_img = F.affine(fake_img, 45, translate=[0.2, 0.2], scale=0.5, shear=[-10, 10])
            print(affined_img.shape)
    """

    if not (_is_pil_image(img) or _is_numpy_image(img) or _is_tensor_image(img)):
        raise TypeError(
            "img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}".format(type(img))
        )

    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if not isinstance(translate, (list, tuple)):
        raise TypeError("Argument translate should be a sequence")

    if len(translate) != 2:
        raise ValueError("Argument translate should be a sequence of length 2")

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    if not isinstance(shear, (numbers.Number, (list, tuple))):
        raise TypeError("Shear should be either a single value or a sequence of two values")

    if not isinstance(interpolation, str):
        raise TypeError("Argument interpolation should be a string")

    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, tuple):
        translate = list(translate)

    if isinstance(shear, numbers.Number):
        shear = [shear, 0.0]

    if isinstance(shear, tuple):
        shear = list(shear)

    if len(shear) == 1:
        shear = [shear[0], shear[0]]

    if len(shear) != 2:
        raise ValueError(f"Shear should be a sequence containing two values. Got {shear}")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    if _is_pil_image(img):
        raise NotImplementedError

    if _is_numpy_image(img):
        raise NotImplementedError

    if _is_tensor_image(img):
        center_f = [0.0, 0.0]
        if center is not None:
            height, width = img.shape[-1], img.shape[-2]
            # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
            center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]
        translate_f = [1.0 * t for t in translate]
        matrix = _get_affine_matrix(center_f, angle, translate_f, scale, shear)
        return F_t.affine(img, matrix, interpolation, fill)


def adjust_brightness(img, brightness_factor):
    """Adjusts brightness of an Image.

    Args:
        img (PIL.Image|np.array|paddle.Tensor): Image to be adjusted.
        brightness_factor (float): How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL.Image|np.array|paddle.Tensor: Brightness adjusted image.

    Examples:
        .. code-block:: python
           :name: code-example1

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)
            print(fake_img.size) # (300, 256)
            print(fake_img.load()[1,1]) # (95, 127, 202)
            converted_img = F.adjust_brightness(fake_img, 0.5)
            print(converted_img.size) # (300, 256)
            print(converted_img.load()[1,1]) # (47, 63, 101)


    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or _is_tensor_image(img)):
        raise TypeError(
            "img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}".format(type(img))
        )

    if _is_pil_image(img):
        raise NotImplementedError
    elif _is_numpy_image(img):
        raise NotImplementedError
    else:
        return F_t.adjust_brightness(img, brightness_factor)


def adjust_contrast(img, contrast_factor):
    """Adjusts contrast of an Image.

    Args:
        img (PIL.Image|np.array|paddle.Tensor): Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL.Image|np.array|paddle.Tensor: Contrast adjusted image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            converted_img = F.adjust_contrast(fake_img, 0.4)
            print(converted_img.size)
    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or _is_tensor_image(img)):
        raise TypeError(
            "img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}".format(type(img))
        )

    if _is_pil_image(img):
        raise NotImplementedError
    elif _is_numpy_image(img):
        raise NotImplementedError
    else:
        return F_t.adjust_contrast(img, contrast_factor)


def adjust_saturation(img, saturation_factor):
    """Adjusts color saturation of an image.

    Args:
        img (PIL.Image|np.array|paddle.Tensor): Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL.Image|np.array|paddle.Tensor: Saturation adjusted image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            converted_img = F.adjust_saturation(fake_img, 0.4)
            print(converted_img.size)

    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or _is_tensor_image(img)):
        raise TypeError(
            "img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}".format(type(img))
        )

    if _is_pil_image(img):
        raise NotImplementedError
    elif _is_numpy_image(img):
        raise NotImplementedError
    else:
        return F_t.adjust_saturation(img, saturation_factor)


def adjust_hue(img, hue_factor):
    """Adjusts hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    Args:
        img (PIL.Image|np.array|paddle.Tensor): Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL.Image|np.array|paddle.Tensor: Hue adjusted image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            converted_img = F.adjust_hue(fake_img, 0.4)
            print(converted_img.size)

    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or _is_tensor_image(img)):
        raise TypeError(
            "img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}".format(type(img))
        )

    if _is_pil_image(img):
        raise NotImplementedError
    elif _is_numpy_image(img):
        raise NotImplementedError
    else:
        return F_t.adjust_hue(img, hue_factor)


def to_grayscale(img, num_output_channels=1):
    """Converts image to grayscale version of image.
    Args:
        img (PIL.Image|np.array): Image to be converted to grayscale.
    Returns:
        PIL.Image or np.array: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel
            if num_output_channels = 3 : returned image is 3 channel with r = g = b

    Examples:
        .. code-block:: python
            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F
            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')
            fake_img = Image.fromarray(fake_img)
            gray_img = F.to_grayscale(fake_img)
            print(gray_img.size)
    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or _is_tensor_image(img)):
        raise TypeError(
            "img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}".format(type(img))
        )

    if _is_pil_image(img):
        raise NotImplementedError
    elif _is_tensor_image(img):
        return F_t.to_grayscale(img, num_output_channels)
    else:
        raise NotImplementedError


def hflip(img):
    """Horizontally flips the given Image or np.array.
    Args:
        img (PIL.Image|np.array): Image to be flipped.
    Returns:
        PIL.Image or np.array:  Horizontall flipped image.
    Examples:
        .. code-block:: python
            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F
            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')
            fake_img = Image.fromarray(fake_img)
            flpped_img = F.hflip(fake_img)
            print(flpped_img.size)
    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or _is_tensor_image(img)):
        raise TypeError(
            "img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}".format(type(img))
        )

    if _is_pil_image(img):
        raise NotImplementedError
    elif _is_tensor_image(img):
        return F_t.hflip(img)
    else:
        raise NotImplementedError
