# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
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
from collections import UserDict
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import paddle
import PIL.Image
import PIL.ImageOps
import requests
from packaging import version

from .tokenizer_utils_base import ExplicitEnum

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def is_paddle_tensor(tensor):
    return paddle.is_tensor(tensor)


def to_numpy(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a Numpy array.
    """
    if isinstance(obj, (dict, UserDict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return np.array(obj)
    elif is_paddle_tensor(obj):
        return obj.detach().cpu().numpy()
    else:
        return obj


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PILImageResampling = PIL.Image.Resampling
else:
    PILImageResampling = PIL.Image


ImageInput = Union[
    "PIL.Image.Image", np.ndarray, "paddle.Tensor", List["PIL.Image.Image"], List[np.ndarray], List["paddle.Tensor"]
]  # noqa


class ChannelDimension(ExplicitEnum):
    FIRST = "channels_first"
    LAST = "channels_last"


def is_valid_image(img):
    return isinstance(img, PIL.Image.Image) or isinstance(img, np.ndarray) or is_paddle_tensor(img)


def valid_images(imgs):
    # If we have an list of images, make sure every image is valid
    if isinstance(imgs, (list, tuple)):
        for img in imgs:
            if not valid_images(img):
                return False
    # If not a list of tuple, we have been given a single image or batched tensor of images
    elif not is_valid_image(imgs):
        return False
    return True


def is_batched(img):
    if isinstance(img, (list, tuple)):
        return is_valid_image(img[0])
    return False


def to_numpy_array(img) -> np.ndarray:
    if not is_valid_image(img):
        raise ValueError(f"Invalid image type: {type(img)}")

    if isinstance(img, PIL.Image.Image):
        return np.array(img)
    return to_numpy(img)


def infer_channel_dimension_format(image: np.ndarray) -> ChannelDimension:
    """
    Infers the channel dimension format of `image`.

    Args:
        image (`np.ndarray`):
            The image to infer the channel dimension of.

    Returns:
        The channel dimension of the image.
    """
    if image.ndim == 3:
        first_dim, last_dim = 0, 2
    elif image.ndim == 4:
        first_dim, last_dim = 1, 3
    else:
        raise ValueError(f"Unsupported number of image dimensions: {image.ndim}")

    if image.shape[first_dim] in (1, 3):
        return ChannelDimension.FIRST
    elif image.shape[last_dim] in (1, 3):
        return ChannelDimension.LAST
    raise ValueError("Unable to infer channel dimension format")


def get_channel_dimension_axis(image: np.ndarray) -> int:
    """
    Returns the channel dimension axis of the image.

    Args:
        image (`np.ndarray`):
            The image to get the channel dimension axis of.

    Returns:
        The channel dimension axis of the image.
    """
    channel_dim = infer_channel_dimension_format(image)
    if channel_dim == ChannelDimension.FIRST:
        return image.ndim - 3
    elif channel_dim == ChannelDimension.LAST:
        return image.ndim - 1
    raise ValueError(f"Unsupported data format: {channel_dim}")


def get_image_size(image: np.ndarray, channel_dim: ChannelDimension = None) -> Tuple[int, int]:
    """
    Returns the (height, width) dimensions of the image.

    Args:
        image (`np.ndarray`):
            The image to get the dimensions of.
        channel_dim (`ChannelDimension`, *optional*):
            Which dimension the channel dimension is in. If `None`, will infer the channel dimension from the image.

    Returns:
        A tuple of the image's height and width.
    """
    if channel_dim is None:
        channel_dim = infer_channel_dimension_format(image)

    if channel_dim == ChannelDimension.FIRST:
        return image.shape[-2], image.shape[-1]
    elif channel_dim == ChannelDimension.LAST:
        return image.shape[-3], image.shape[-2]
    else:
        raise ValueError(f"Unsupported data format: {channel_dim}")


def is_valid_annotation_coco_detection(annotation: Dict[str, Union[List, Tuple]]) -> bool:
    if (
        isinstance(annotation, dict)
        and "image_id" in annotation
        and "annotations" in annotation
        and isinstance(annotation["annotations"], (list, tuple))
        and (
            # an image can have no annotations
            len(annotation["annotations"]) == 0
            or isinstance(annotation["annotations"][0], dict)
        )
    ):
        return True
    return False


def is_valid_annotation_coco_panoptic(annotation: Dict[str, Union[List, Tuple]]) -> bool:
    if (
        isinstance(annotation, dict)
        and "image_id" in annotation
        and "segments_info" in annotation
        and "file_name" in annotation
        and isinstance(annotation["segments_info"], (list, tuple))
        and (
            # an image can have no segments
            len(annotation["segments_info"]) == 0
            or isinstance(annotation["segments_info"][0], dict)
        )
    ):
        return True
    return False


def valid_coco_detection_annotations(annotations: Iterable[Dict[str, Union[List, Tuple]]]) -> bool:
    return all(is_valid_annotation_coco_detection(ann) for ann in annotations)


def valid_coco_panoptic_annotations(annotations: Iterable[Dict[str, Union[List, Tuple]]]) -> bool:
    return all(is_valid_annotation_coco_panoptic(ann) for ann in annotations)


def load_image(image: Union[str, "PIL.Image.Image"]) -> "PIL.Image.Image":
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.

    Returns:
        `PIL.Image.Image`: A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


class ImageFeatureExtractionMixin:
    """
    Mixin that contain utilities for preparing image features.
    """

    def _ensure_format_supported(self, image):
        if not isinstance(image, (PIL.Image.Image, np.ndarray)) and not is_paddle_tensor(image):
            raise ValueError(
                f"Got type {type(image)} which is not supported, only `PIL.Image.Image`, `np.array` and "
                "`paddle.Tensor` are."
            )

    def to_pil_image(self, image, rescale=None):
        """
        Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
        needed.

        Args:
            image (`PIL.Image.Image` or `numpy.ndarray` or `paddle.Tensor`):
                The image to convert to the PIL Image format.
            rescale (`bool`, *optional*):
                Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will
                default to `True` if the image type is a floating type, `False` otherwise.
        """
        self._ensure_format_supported(image)

        if is_paddle_tensor(image):
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if rescale is None:
                # rescale default to the array being of floating type.
                rescale = isinstance(image.flat[0], np.floating)
            # If the channel as been moved to first dim, we put it back at the end.
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            if rescale:
                image = image * 255
            image = image.astype(np.uint8)
            return PIL.Image.fromarray(image)
        return image

    def convert_rgb(self, image):
        """
        Converts `PIL.Image.Image` to RGB format.

        Args:
            image (`PIL.Image.Image`):
                The image to convert.
        """
        self._ensure_format_supported(image)
        if not isinstance(image, PIL.Image.Image):
            return image

        return image.convert("RGB")

    def rescale(self, image: np.ndarray, scale: Union[float, int]) -> np.ndarray:
        """
        Rescale a numpy image by scale amount
        """
        self._ensure_format_supported(image)
        return image * scale

    def to_numpy_array(self, image, rescale=None, channel_first=True):
        """
        Converts `image` to a numpy array. Optionally rescales it and puts the channel dimension as the first
        dimension.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `paddle.Tensor`):
                The image to convert to a NumPy array.
            rescale (`bool`, *optional*):
                Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Will
                default to `True` if the image is a PIL Image or an array/tensor of integers, `False` otherwise.
            channel_first (`bool`, *optional*, defaults to `True`):
                Whether or not to permute the dimensions of the image to put the channel dimension first.
        """
        self._ensure_format_supported(image)

        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        if is_paddle_tensor(image):
            image = image.numpy()

        rescale = isinstance(image.flat[0], np.integer) if rescale is None else rescale

        if rescale:
            image = self.rescale(image.astype(np.float32), 1 / 255.0)

        if channel_first and image.ndim == 3:
            image = image.transpose(2, 0, 1)

        return image

    def expand_dims(self, image):
        """
        Expands 2-dimensional `image` to 3 dimensions.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `paddle.Tensor`):
                The image to expand.
        """
        self._ensure_format_supported(image)

        # Do nothing if PIL image
        if isinstance(image, PIL.Image.Image):
            return image

        if is_paddle_tensor(image):
            image = image.unsqueeze(0)
        else:
            image = np.expand_dims(image, axis=0)
        return image

    def normalize(self, image, mean, std, rescale=False):
        """
        Normalizes `image` with `mean` and `std`. Note that this will trigger a conversion of `image` to a NumPy array
        if it's a PIL Image.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `paddle.Tensor`):
                The image to normalize.
            mean (`List[float]` or `np.ndarray` or `paddle.Tensor`):
                The mean (per channel) to use for normalization.
            std (`List[float]` or `np.ndarray` or `paddle.Tensor`):
                The standard deviation (per channel) to use for normalization.
            rescale (`bool`, *optional*, defaults to `False`):
                Whether or not to rescale the image to be between 0 and 1. If a PIL image is provided, scaling will
                happen automatically.
        """
        self._ensure_format_supported(image)

        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image, rescale=True)
        # If the input image is a PIL image, it automatically gets rescaled. If it's another
        # type it may need rescaling.
        elif rescale:
            if isinstance(image, np.ndarray):
                image = self.rescale(image.astype(np.float32), 1 / 255.0)
            elif is_paddle_tensor(image):
                image = self.rescale(image.astype("float32"), 1 / 255.0)

        if isinstance(image, np.ndarray):
            if not isinstance(mean, np.ndarray):
                mean = np.array(mean).astype(image.dtype)
            if not isinstance(std, np.ndarray):
                std = np.array(std).astype(image.dtype)
        elif is_paddle_tensor(image):

            if not isinstance(mean, paddle.Tensor):
                mean = paddle.to_tensor(mean).astype(image.dtype)
            if not isinstance(std, paddle.Tensor):
                std = paddle.to_tensor(std).astype(image.dtype)

        if image.ndim == 3 and image.shape[0] in [1, 3]:
            return (image - mean[:, None, None]) / std[:, None, None]
        else:
            return (image - mean) / std

    def resize(self, image, size, resample=None, default_to_square=True, max_size=None):
        """
        Resizes `image`. Enforces conversion of input to PIL.Image.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `paddle.Tensor`):
                The image to resize.
            size (`int` or `Tuple[int, int]`):
                The size to use for resizing the image. If `size` is a sequence like (h, w), output size will be
                matched to this.

                If `size` is an int and `default_to_square` is `True`, then image will be resized to (size, size). If
                `size` is an int and `default_to_square` is `False`, then smaller edge of the image will be matched to
                this number. i.e, if height > width, then image will be rescaled to (size * height / width, size).
            resample (`int`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                The filter to user for resampling.
            default_to_square (`bool`, *optional*, defaults to `True`):
                How to convert `size` when it is a single int. If set to `True`, the `size` will be converted to a
                square (`size`,`size`). If set to `False`, will replicate
                [`paddle.vision.transforms.Resize`](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/Resize_cn.html#resize)
                with support for resizing only the smallest edge and providing an optional `max_size`.
            max_size (`int`, *optional*, defaults to `None`):
                The maximum allowed for the longer edge of the resized image: if the longer edge of the image is
                greater than `max_size` after being resized according to `size`, then the image is resized again so
                that the longer edge is equal to `max_size`. As a result, `size` might be overruled, i.e the smaller
                edge may be shorter than `size`. Only used if `default_to_square` is `False`.

        Returns:
            image: A resized `PIL.Image.Image`.
        """
        resample = resample if resample is not None else PILImageResampling.BILINEAR

        self._ensure_format_supported(image)

        if not isinstance(image, PIL.Image.Image):
            image = self.to_pil_image(image)

        if isinstance(size, list):
            size = tuple(size)

        if isinstance(size, int) or len(size) == 1:
            if default_to_square:
                size = (size, size) if isinstance(size, int) else (size[0], size[0])
            else:
                width, height = image.size
                # specified size only for the smallest edge
                short, long = (width, height) if width <= height else (height, width)
                requested_new_short = size if isinstance(size, int) else size[0]

                if short == requested_new_short:
                    return image

                new_short, new_long = requested_new_short, int(requested_new_short * long / short)

                if max_size is not None:
                    if max_size <= requested_new_short:
                        raise ValueError(
                            f"max_size = {max_size} must be strictly greater than the requested "
                            f"size for the smaller edge size = {size}"
                        )
                    if new_long > max_size:
                        new_short, new_long = int(max_size * new_short / new_long), max_size

                size = (new_short, new_long) if width <= height else (new_long, new_short)

        return image.resize(size, resample=resample)

    def center_crop(self, image, size):
        """
        Crops `image` to the given size using a center crop. Note that if the image is too small to be cropped to the
        size given, it will be padded (so the returned result has the size asked).

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `paddle.Tensor` of shape (n_channels, height, width) or (height, width, n_channels)):
                The image to resize.
            size (`int` or `Tuple[int, int]`):
                The size to which crop the image.

        Returns:
            new_image: A center cropped `PIL.Image.Image` or `np.ndarray` or `paddle.Tensor` of shape: (n_channels,
            height, width).
        """
        self._ensure_format_supported(image)

        if not isinstance(size, tuple):
            size = (size, size)

        # PIL Image.size is (width, height) but NumPy array and paddle Tensors have (height, width)
        if is_paddle_tensor(image) or isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = self.expand_dims(image)
            image_shape = image.shape[1:] if image.shape[0] in [1, 3] else image.shape[:2]
        else:
            image_shape = (image.size[1], image.size[0])

        top = (image_shape[0] - size[0]) // 2
        bottom = top + size[0]  # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
        left = (image_shape[1] - size[1]) // 2
        right = left + size[1]  # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.

        # For PIL Images we have a method to crop directly.
        if isinstance(image, PIL.Image.Image):
            return image.crop((left, top, right, bottom))

        # Check if image is in (n_channels, height, width) or (height, width, n_channels) format
        channel_first = True if image.shape[0] in [1, 3] else False

        # Transpose (height, width, n_channels) format images
        if not channel_first:
            if isinstance(image, np.ndarray):
                image = image.transpose(2, 0, 1)
            if is_paddle_tensor(image):
                image = image.transpose([2, 0, 1])

        # Check if cropped area is within image boundaries
        if top >= 0 and bottom <= image_shape[0] and left >= 0 and right <= image_shape[1]:
            return image[..., top:bottom, left:right]

        # Otherwise, we may need to pad if the image is too small. Oh joy...
        new_shape = image.shape[:-2] + (max(size[0], image_shape[0]), max(size[1], image_shape[1]))
        if isinstance(image, np.ndarray):
            new_image = np.zeros_like(image, shape=new_shape)
        elif is_paddle_tensor(image):
            new_image = paddle.zeros(new_shape, dtype=image.dtype)

        top_pad = (new_shape[-2] - image_shape[0]) // 2
        bottom_pad = top_pad + image_shape[0]
        left_pad = (new_shape[-1] - image_shape[1]) // 2
        right_pad = left_pad + image_shape[1]
        new_image[..., top_pad:bottom_pad, left_pad:right_pad] = image

        top += top_pad
        bottom += top_pad
        left += left_pad
        right += left_pad

        new_image = new_image[
            ..., max(0, top) : min(new_image.shape[-2], bottom), max(0, left) : min(new_image.shape[-1], right)
        ]

        return new_image

    def flip_channel_order(self, image):
        """
        Flips the channel order of `image` from RGB to BGR, or vice versa. Note that this will trigger a conversion of
        `image` to a NumPy array if it's a PIL Image.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `paddle.Tensor`):
                The image whose color channels to flip. If `np.ndarray` or `paddle.Tensor`, the channel dimension should
                be first.
        """
        self._ensure_format_supported(image)

        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image)

        return image[::-1, :, :]

    def rotate(self, image, angle, resample=None, expand=0, center=None, translate=None, fillcolor=None):
        """
        Returns a rotated copy of `image`. This method returns a copy of `image`, rotated the given number of degrees
        counter clockwise around its centre.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `paddle.Tensor`):
                The image to rotate. If `np.ndarray` or `paddle.Tensor`, will be converted to `PIL.Image.Image` before
                rotating.

        Returns:
            image: A rotated `PIL.Image.Image`.
        """
        resample = resample if resample is not None else PIL.Image.NEAREST

        self._ensure_format_supported(image)

        if not isinstance(image, PIL.Image.Image):
            image = self.to_pil_image(image)

        return image.rotate(
            angle, resample=resample, expand=expand, center=center, translate=translate, fillcolor=fillcolor
        )
