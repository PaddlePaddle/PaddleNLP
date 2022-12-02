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
This code is rewritten by Paddle based on
https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
"""
import math
import numbers
import warnings
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import paddle
import paddle.nn as nn
from paddle.nn.functional import grid_sample


class Normalize(nn.Layer):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = paddle.to_tensor(mean)
        self.std = paddle.to_tensor(std)

    def forward(self, tensor):
        dtype = tensor.dtype
        mean = paddle.cast(self.mean, dtype=dtype).reshape([1, -1, 1, 1])
        std = paddle.cast(self.std, dtype=dtype).reshape([1, -1, 1, 1])
        result = tensor.subtract(mean).divide(std)
        return result


class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``bilinear``, ``bicubic``, ``box``, ``hamming``, and ``lanczos``.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


class Grayscale(nn.Layer):
    def __init__(self, num_output_channels):
        super(Grayscale, self).__init__()
        self.num_output_channels = num_output_channels

    def forward(self, x):
        output = 0.2989 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        if self.num_output_channels == 3:
            return output.expand(x.shape)

        return output


class Lambda(nn.Layer):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.transform = func

    def forward(self, x):
        return self.transform(x)


class RandomGrayscale(nn.Layer):
    def __init__(self, p):
        super(RandomGrayscale, self).__init__()
        self.prob = p
        self.transform = Grayscale(3)

    def forward(self, x):
        if paddle.rand([1]) < self.prob:
            return self.transform(x)
        else:
            return x


class RandomHorizontalFlip(nn.Layer):
    def __init__(self, prob):
        super(RandomHorizontalFlip, self).__init__()
        self.prob = prob

    def forward(self, x):
        if paddle.rand([1]) < self.prob:
            return x[:, :, :, ::-1]
        else:
            return x


def _blend(img1, img2, ratio: float):
    ratio = float(ratio)
    bound = 1.0
    return (ratio * img1 + (1.0 - ratio) * img2).clip(0, bound)


def trunc_div(a, b):
    ipt = paddle.divide(a, b)
    sign_ipt = paddle.sign(ipt)
    abs_ipt = paddle.abs(ipt)
    abs_ipt = paddle.floor(abs_ipt)
    out = paddle.multiply(sign_ipt, abs_ipt)
    return out


def fmod(a, b):
    return a - trunc_div(a, b) * b


def _rgb2hsv(img):
    r, g, b = img.unbind(axis=-3)

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = paddle.max(img, axis=-3)
    minc = paddle.min(img, axis=-3)

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = paddle.ones_like(maxc)
    s = cr / paddle.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = paddle.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r).cast("float32") * (bc - gc)
    hg = ((maxc == g) & (maxc != r)).cast("float32") * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)).cast("float32") * (4.0 + gc - rc)
    h = hr + hg + hb
    h = fmod((h / 6.0 + 1.0), paddle.to_tensor(1.0))
    return paddle.stack((h, s, maxc), axis=-3)


def _hsv2rgb(img):
    h, s, v = img.unbind(axis=-3)
    i = paddle.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.cast(dtype="int32")

    p = paddle.clip((v * (1.0 - s)), 0.0, 1.0)
    q = paddle.clip((v * (1.0 - s * f)), 0.0, 1.0)
    t = paddle.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i.unsqueeze(axis=-3) == paddle.arange(6).reshape([-1, 1, 1])

    a1 = paddle.stack((v, q, p, p, t, v), axis=-3)
    a2 = paddle.stack((t, v, v, q, p, p), axis=-3)
    a3 = paddle.stack((p, p, t, v, v, q), axis=-3)
    a4 = paddle.stack((a1, a2, a3), axis=-4)

    return paddle.einsum("...ijk, ...xijk -> ...xjk", mask.cast(dtype=img.dtype), a4)


def adjust_brightness(img, brightness_factor: float):
    if brightness_factor < 0:
        raise ValueError(f"brightness_factor ({brightness_factor}) is not non-negative.")

    return _blend(img, paddle.zeros_like(img), brightness_factor)


def adjust_contrast(img, contrast_factor: float):
    if contrast_factor < 0:
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")

    c = img.shape[1]

    if c == 3:
        output = 0.2989 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        mean = paddle.mean(output, axis=(-3, -2, -1), keepdim=True)

    else:
        mean = paddle.mean(img, axis=(-3, -2, -1), keepdim=True)

    return _blend(img, mean, contrast_factor)


def adjust_hue(img, hue_factor: float):
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(f"hue_factor ({hue_factor}) is not in [-0.5, 0.5].")

    img = _rgb2hsv(img)
    h, s, v = img.unbind(axis=-3)
    h = fmod(h + hue_factor, paddle.to_tensor(1.0))
    img = paddle.stack((h, s, v), axis=-3)
    img_hue_adj = _hsv2rgb(img)
    return img_hue_adj


def adjust_saturation(img, saturation_factor: float):
    if saturation_factor < 0:
        raise ValueError(f"saturation_factor ({saturation_factor}) is not non-negative.")

    output = 0.2989 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]

    return _blend(img, output, saturation_factor)


class ColorJitter(nn.Layer):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(ColorJitter, self).__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(
        brightness: Optional[List[float]],
        contrast: Optional[List[float]],
        saturation: Optional[List[float]],
        hue: Optional[List[float]],
    ) -> Tuple[paddle.Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = paddle.randperm(4)

        b = None if brightness is None else paddle.empty([1]).uniform_(brightness[0], brightness[1])
        c = None if contrast is None else paddle.empty([1]).uniform_(contrast[0], contrast[1])
        s = None if saturation is None else paddle.empty([1]).uniform_(saturation[0], saturation[1])
        h = None if hue is None else paddle.empty([1]).uniform_(hue[0], hue[1])

        return fn_idx, b, c, s, h

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = adjust_hue(img, hue_factor)

        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue})"
        )
        return s


def _apply_grid_transform(img, grid, mode: str, fill: Optional[List[float]]):

    if img.shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand([img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]])

    # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
    if fill is not None:
        dummy = paddle.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype)
        img = paddle.concat((img, dummy), axis=1)

    img = grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # N * 1 * H * W
        img = img[:, :-1, :, :]  # N * C * H * W
        mask = mask.expand_as(img)
        len_fill = len(fill) if isinstance(fill, (tuple, list)) else 1
        fill_img = paddle.to_tensor(fill, dtype=img.dtype).reshape([1, len_fill, 1, 1]).expand_as(img)
        if mode == "nearest":
            mask = mask < 0.5
            img[mask] = fill_img[mask]
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img
    return img


def _gen_affine_grid(
    theta,
    w: int,
    h: int,
    ow: int,
    oh: int,
):
    # https://github.com/pytorch/pytorch/blob/74b65c32be68b15dc7c9e8bb62459efbfbde33d8/aten/src/ATen/native/
    # AffineGridGenerator.cpp#L18
    # Difference with AffineGridGenerator is that:
    # 1) we normalize grid values after applying theta
    # 2) we can normalize by other image size, such that it covers "extend" option like in PIL.Image.rotate

    d = 0.5
    base_grid = paddle.empty([1, oh, ow, 3], dtype=theta.dtype)
    x_grid = paddle.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, num=ow)
    base_grid[..., 0] = x_grid
    y_grid = paddle.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, num=oh).unsqueeze_(-1)
    base_grid[..., 1] = y_grid
    base_grid[..., 2] = 1.0
    rescaled_theta = theta.transpose([0, 2, 1]) / paddle.to_tensor([0.5 * w, 0.5 * h], dtype=theta.dtype)
    output_grid = base_grid.reshape([1, oh * ow, 3]).bmm(rescaled_theta)
    return output_grid.reshape([1, oh, ow, 2])


def affine_impl(img, matrix: List[float], interpolation: str = "nearest", fill: Optional[List[float]] = None):
    theta = paddle.to_tensor(matrix, dtype=img.dtype).reshape([1, 2, 3])
    shape = img.shape
    # grid will be generated on the same device as theta and img
    grid = _gen_affine_grid(theta, w=shape[-1], h=shape[-2], ow=shape[-1], oh=shape[-2])
    return _apply_grid_transform(img, grid, interpolation, fill=fill)


def _get_inverse_affine_matrix(
    center: List[float], angle: float, translate: List[float], scale: float, shear: List[float], inverted: bool = True
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation

    # Pillow requires inverse affine transformation matrix:
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    #
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RotateScaleShear is rotation with scale and shear matrix
    #
    #       RotateScaleShear(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    if inverted:
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
    else:
        matrix = [a, b, 0.0, c, d, 0.0]
        matrix = [x * scale for x in matrix]
        # Apply inverse of center translation: RSS * C^-1
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        # Apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx + tx
        matrix[5] += cy + ty

    return matrix


def affine(
    img,
    angle: float,
    translate: List[int],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Optional[List[float]] = None,
    resample: Optional[int] = None,
    fillcolor: Optional[List[float]] = None,
    center: Optional[List[int]] = None,
):
    """Apply affine transformation on the image keeping image center invariant.
    If the image is paddle Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): image to transform.
        angle (number): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (sequence of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float or sequence): shear angle value in degrees between -180 to 180, clockwise direction.
            If a sequence is specified, the first value corresponds to a shear parallel to the x axis, while
            the second value corresponds to a shear parallel to the y axis.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.[Resampling.]NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

            .. note::
                In torchscript mode single int/float value is not supported, please use a sequence
                of length 1: ``[value, ]``.
        fillcolor (sequence or number, optional):
            .. warning::
                This parameter was deprecated in ``0.12`` and will be removed in ``0.14``. Please use ``fill`` instead.
        resample (int, optional):
            .. warning::
                This parameter was deprecated in ``0.12`` and will be removed in ``0.14``. Please use ``interpolation``
                instead.
        center (sequence, optional): Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.

    Returns:
        PIL Image or Tensor: Transformed image.
    """

    # Backward compatibility with integer value
    if isinstance(interpolation, int):
        warnings.warn(
            "Argument 'interpolation' of type int is deprecated since 0.13 and will be removed in 0.15. "
            "Please use InterpolationMode enum."
        )
        interpolation = _interpolation_modes_from_int(interpolation)

    if fillcolor is not None:
        warnings.warn(
            "The parameter 'fillcolor' is deprecated since 0.12 and will be removed in 0.14. "
            "Please use 'fill' instead."
        )
        fill = fillcolor

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

    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

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
    center_f = [0.0, 0.0]
    if center is not None:
        _, height, width = img.shape[0], img.shape[1], img.shape[2]
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

    translate_f = [1.0 * t for t in translate]
    matrix = _get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear)
    return affine_impl(img, matrix=matrix, interpolation=interpolation.value, fill=fill)


def _interpolation_modes_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]


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


class RandomAffine(nn.Layer):
    """Random affine transformation of the image keeping center invariant.
    If the image is paddle Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or number, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be applied. Else if shear is a sequence of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a sequence of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.[Resampling.]NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
        fillcolor (sequence or number, optional):
            .. warning::
                This parameter was deprecated in ``0.12`` and will be removed in ``0.14``. Please use ``fill`` instead.
        resample (int, optional):
            .. warning::
                This parameter was deprecated in ``0.12`` and will be removed in ``0.14``. Please use ``interpolation``
                instead.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(
        self,
        degrees,
        translate=None,
        scale=None,
        shear=None,
        interpolation=InterpolationMode.NEAREST,
        fill=0,
        fillcolor=None,
        resample=None,
        center=None,
    ):
        super(RandomAffine, self).__init__()
        if resample is not None:
            warnings.warn(
                "The parameter 'resample' is deprecated since 0.12 and will be removed in 0.14. "
                "Please use 'interpolation' instead."
            )
            interpolation = _interpolation_modes_from_int(resample)

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument 'interpolation' of type int is deprecated since 0.13 and will be removed in 0.15. "
                "Please use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        if fillcolor is not None:
            warnings.warn(
                "The parameter 'fillcolor' is deprecated since 0.12 and will be removed in 0.14. "
                "Please use 'fill' instead."
            )
            fill = fillcolor

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

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

        self.resample = self.interpolation = interpolation

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fillcolor = self.fill = fill

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

    @staticmethod
    def get_params(
        degrees: List[float],
        translate: Optional[List[float]],
        scale_ranges: Optional[List[float]],
        shears: Optional[List[float]],
        img_size: List[int],
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(paddle.empty([1]).uniform_(float(degrees[0]), float(degrees[1])))
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(float(paddle.empty([1]).uniform_(-max_dx, max_dx)))
            ty = int(float(paddle.empty([1]).uniform_(-max_dy, max_dy)))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(paddle.empty([1]).uniform_(scale_ranges[0], scale_ranges[1]))
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(paddle.empty([1]).uniform_(shears[0], shears[1]))
            if len(shears) == 4:
                shear_y = float(paddle.empty([1]).uniform_(shears[2], shears[3]))

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def forward(self, img):
        fill = self.fill
        channels, height, width = img.shape[1], img.shape[2], img.shape[3]
        if isinstance(fill, (int, float)):
            fill = [float(fill)] * channels
        else:
            fill = [float(f) for f in fill]

        img_size = [width, height]  # flip for keeping BC on get_params call

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        return affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(degrees={self.degrees}"
        s += f", translate={self.translate}" if self.translate is not None else ""
        s += f", scale={self.scale}" if self.scale is not None else ""
        s += f", shear={self.shear}" if self.shear is not None else ""
        s += f", interpolation={self.interpolation.value}" if self.interpolation != InterpolationMode.NEAREST else ""
        s += f", fill={self.fill}" if self.fill != 0 else ""
        s += f", center={self.center}" if self.center is not None else ""
        s += ")"

        return s
