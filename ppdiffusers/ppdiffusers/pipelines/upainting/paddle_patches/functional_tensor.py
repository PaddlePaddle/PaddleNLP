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
https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/vision/transforms/functional_tensor.py
"""

from __future__ import division

import paddle
import paddle.nn.functional as F

rgb_weights = paddle.to_tensor([0.2989, 0.5870, 0.1140]).astype("float32")


def _assert_image_tensor(img, data_format):
    if not isinstance(img, paddle.Tensor) or img.ndim < 3 or img.ndim > 4 or not data_format.lower() in ("chw", "hwc"):
        raise RuntimeError(
            "not support [type={}, ndim={}, data_format={}] paddle image".format(type(img), img.ndim, data_format)
        )


def _get_image_h_axis(data_format):
    if data_format.lower() == "chw":
        return -2
    elif data_format.lower() == "hwc":
        return -3


def _get_image_w_axis(data_format):
    if data_format.lower() == "chw":
        return -1
    elif data_format.lower() == "hwc":
        return -2


def _get_image_c_axis(data_format):
    if data_format.lower() == "chw":
        return -3
    elif data_format.lower() == "hwc":
        return -1


def _get_image_n_axis(data_format):
    if len(data_format) == 3:
        return None
    elif len(data_format) == 4:
        return 0


def _is_channel_first(data_format):
    return _get_image_c_axis(data_format) == -3


def _get_image_num_channels(img, data_format):
    return img.shape[_get_image_c_axis(data_format)]


def _rgb_to_hsv(img):
    """Convert a image Tensor from RGB to HSV. This implementation is based on Pillow (
    https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Convert.c)
    """
    maxc = img.max(axis=-3)
    minc = img.min(axis=-3)

    is_equal = paddle.equal(maxc, minc)
    one_divisor = paddle.ones_like(maxc)
    c_delta = maxc - minc
    # s is 0 when maxc == minc, set the divisor to 1 to avoid zero divide.
    s = c_delta / paddle.where(is_equal, one_divisor, maxc)

    r, g, b = img.unbind(axis=-3)
    c_delta_divisor = paddle.where(is_equal, one_divisor, c_delta)
    # when maxc == minc, there is r == g == b, set the divisor to 1 to avoid zero divide.
    rc = (maxc - r) / c_delta_divisor
    gc = (maxc - g) / c_delta_divisor
    bc = (maxc - b) / c_delta_divisor

    hr = (maxc == r).astype(maxc.dtype) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)).astype(maxc.dtype) * (rc - bc + 2.0)
    hb = ((maxc != r) & (maxc != g)).astype(maxc.dtype) * (gc - rc + 4.0)
    h = (hr + hg + hb) / 6.0 + 1.0
    h = h - h.trunc()
    return paddle.stack([h, s, maxc], axis=-3)


def _hsv_to_rgb(img):
    """Convert a image Tensor from HSV to RGB."""
    h, s, v = img.unbind(axis=-3)
    f = h * 6.0
    i = paddle.floor(f)
    f = f - i
    i = i.astype(paddle.int32) % 6

    p = paddle.clip(v * (1.0 - s), 0.0, 1.0)
    q = paddle.clip(v * (1.0 - s * f), 0.0, 1.0)
    t = paddle.clip(v * (1.0 - s * (1.0 - f)), 0.0, 1.0)

    mask = paddle.equal(i.unsqueeze(axis=-3), paddle.arange(6, dtype=i.dtype).reshape((-1, 1, 1))).astype(img.dtype)
    matrix = paddle.stack(
        [
            paddle.stack([v, q, p, p, t, v], axis=-3),
            paddle.stack([t, v, v, q, p, p], axis=-3),
            paddle.stack([p, p, t, v, v, q], axis=-3),
        ],
        axis=-4,
    )
    return paddle.einsum("...ijk, ...xijk -> ...xjk", mask, matrix)


def _blend_images(img1, img2, ratio):
    max_value = 1.0 if paddle.is_floating_point(img1) else 255.0
    ratio = float(ratio)
    return (ratio * img1 + (1.0 - ratio) * img2).clip(0, max_value).astype(img1.dtype)


def to_grayscale(img, num_output_channels=1, data_format="CHW"):
    """Converts image to grayscale version of image.

    Args:
        img (paddel.Tensor): Image to be converted to grayscale.
        num_output_channels (int, optionl[1, 3]):
            if num_output_channels = 1 : returned image is single channel
            if num_output_channels = 3 : returned image is 3 channel
        data_format (str, optional): Data format of img, should be 'HWC' or
            'CHW'. Default: 'CHW'.

    Returns:
        paddle.Tensor: Grayscale version of the image.
    """
    _assert_image_tensor(img, data_format)

    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels should be either 1 or 3")

    global rgb_weights

    if _is_channel_first(data_format):
        rgb_weights = rgb_weights.reshape((-1, 1, 1))

    _c_index = _get_image_c_axis(data_format)

    img = (img * rgb_weights).sum(axis=_c_index, keepdim=True)
    _shape = img.shape
    _shape[_c_index] = num_output_channels

    return img.expand(_shape)


def _affine_grid(theta, w, h, ow, oh):
    d = 0.5
    base_grid = paddle.ones((1, oh, ow, 3), dtype=theta.dtype)

    x_grid = paddle.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, ow)
    base_grid[..., 0] = x_grid
    y_grid = paddle.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, oh).unsqueeze_(-1)
    base_grid[..., 1] = y_grid

    scaled_theta = theta.transpose((0, 2, 1)) / paddle.to_tensor([0.5 * w, 0.5 * h])
    output_grid = base_grid.reshape((1, oh * ow, 3)).bmm(scaled_theta)

    return output_grid.reshape((1, oh, ow, 2))


def _grid_transform(img, grid, mode, fill):
    if img.shape[0] > 1:
        grid = grid.expand(shape=[img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]])

    if fill is not None:
        dummy = paddle.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype)
        img = paddle.concat((img, dummy), axis=1)

    img = F.grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # n 1 h w
        img = img[:, :-1, :, :]  # n c h w
        mask = mask.expand_as(img)
        len_fill = len(fill) if isinstance(fill, (tuple, list)) else 1
        fill_img = paddle.to_tensor(fill).reshape((1, len_fill, 1, 1)).expand_as(img)

        if mode == "nearest":
            mask = paddle.cast(mask < 0.5, img.dtype)
            img = img * (1.0 - mask) + mask * fill_img
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img

    return img


def affine(img, matrix, interpolation="nearest", fill=None, data_format="CHW"):
    """Affine to the image by matrix.

    Args:
        img (paddle.Tensor): Image to be rotated.
        matrix (float or int): Affine matrix.
        interpolation (str, optional): Interpolation method. If omitted, or if the
            image has only one channel, it is set NEAREST . when use pil backend,
            support method are as following:
            - "nearest"
            - "bilinear"
            - "bicubic"
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.
        data_format (str, optional): Data format of img, should be 'HWC' or
            'CHW'. Default: 'CHW'.

    Returns:
        paddle.Tensor: Affined image.

    """
    ndim = len(img.shape)
    if ndim == 3:
        img = img.unsqueeze(0)

    img = img if data_format.lower() == "chw" else img.transpose((0, 3, 1, 2))

    matrix = paddle.to_tensor(matrix, place=img.place)
    matrix = matrix.reshape((1, 2, 3))
    shape = img.shape

    grid = _affine_grid(matrix, w=shape[-1], h=shape[-2], ow=shape[-1], oh=shape[-2])

    if isinstance(fill, int):
        fill = tuple([fill] * 3)

    out = _grid_transform(img, grid, mode=interpolation, fill=fill)

    out = out if data_format.lower() == "chw" else out.transpose((0, 2, 3, 1))
    out = out.squeeze(0) if ndim == 3 else out

    return out


def adjust_brightness(img, brightness_factor):
    """Adjusts brightness of an Image.

    Args:
        img (paddle.Tensor): Image to be adjusted.
        brightness_factor (float): How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        paddle.Tensor: Brightness adjusted image.

    """
    _assert_image_tensor(img, "CHW")
    assert brightness_factor >= 0, "brightness_factor should be non-negative."
    assert _get_image_num_channels(img, "CHW") in [1, 3], "channels of input should be either 1 or 3."

    extreme_target = paddle.zeros_like(img, img.dtype)
    return _blend_images(img, extreme_target, brightness_factor)


def adjust_contrast(img, contrast_factor):
    """Adjusts contrast of an image.

    Args:
        img (paddle.Tensor): Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        paddle.Tensor: Contrast adjusted image.

    """
    _assert_image_tensor(img, "chw")
    assert contrast_factor >= 0, "contrast_factor should be non-negative."

    channels = _get_image_num_channels(img, "CHW")
    dtype = img.dtype if paddle.is_floating_point(img) else paddle.float32
    if channels == 1:
        extreme_target = paddle.mean(img.astype(dtype), axis=(-3, -2, -1), keepdim=True)
    elif channels == 3:
        extreme_target = paddle.mean(to_grayscale(img).astype(dtype), axis=(-3, -2, -1), keepdim=True)
    else:
        raise ValueError("channels of input should be either 1 or 3.")

    return _blend_images(img, extreme_target, contrast_factor)


def adjust_saturation(img, saturation_factor):
    """Adjusts color saturation of an image.

    Args:
        img (paddle.Tensor): Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        paddle.Tensor: Saturation adjusted image.

    """
    _assert_image_tensor(img, "CHW")
    assert saturation_factor >= 0, "saturation_factor should be non-negative."
    channels = _get_image_num_channels(img, "CHW")
    if channels == 1:
        return img
    elif channels == 3:
        extreme_target = to_grayscale(img)
    else:
        raise ValueError("channels of input should be either 1 or 3.")

    return _blend_images(img, extreme_target, saturation_factor)


def adjust_hue(img, hue_factor):
    """Adjusts hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    Args:
        img (paddle.Tensor): Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        paddle.Tensor: Hue adjusted image.

    """
    _assert_image_tensor(img, "CHW")
    assert hue_factor >= -0.5 and hue_factor <= 0.5, "hue_factor should be in range [-0.5, 0.5]"
    channels = _get_image_num_channels(img, "CHW")
    if channels == 1:
        return img
    elif channels == 3:
        dtype = img.dtype
        if dtype == paddle.uint8:
            img = img.astype(paddle.float32) / 255.0

        img_hsv = _rgb_to_hsv(img)
        h, s, v = img_hsv.unbind(axis=-3)
        h = h + hue_factor
        h = h - h.floor()
        img_adjusted = _hsv_to_rgb(paddle.stack([h, s, v], axis=-3))

        if dtype == paddle.uint8:
            img_adjusted = (img_adjusted * 255.0).astype(dtype)
    else:
        raise ValueError("channels of input should be either 1 or 3.")

    return img_adjusted


def hflip(img, data_format="CHW"):
    """Horizontally flips the given paddle.Tensor Image.
    Args:
        img (paddle.Tensor): Image to be flipped.
        data_format (str, optional): Data format of img, should be 'HWC' or
            'CHW'. Default: 'CHW'.
    Returns:
        paddle.Tensor:  Horizontall flipped image.
    """
    _assert_image_tensor(img, data_format)

    w_axis = _get_image_w_axis(data_format)

    return img.flip(axis=[w_axis])
