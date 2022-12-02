# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for CLIP."""

from typing import List, Optional, Union

import paddle
import numpy as np
import PIL.Image
from PIL import Image

from ..feature_extraction_utils import BatchFeature, FeatureExtractionMixin

from ..tokenizer_utils_base import TensorType
from ..image_utils import ImageFeatureExtractionMixin

from ...utils.tools import compare_version

if compare_version(PIL.__version__, "9.1.0") >= 0:
    Resampling = PIL.Image.Resampling
else:
    Resampling = PIL.Image

__all__ = ["CLIPFeatureExtractor"]


class CLIPFeatureExtractor(
    FeatureExtractionMixin,
    ImageFeatureExtractionMixin,
):
    r"""
    Constructs a CLIP feature extractor.
    This feature extractor inherits from [`ImageFeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int`, *optional*, defaults to 224):
            Resize the input to the given size. Only has an effect if `do_resize` is set to `True`.
        resample (`int`, *optional*, defaults to `PIL.Image.[Resampling.]BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.[Resampling.]NEAREST`, `PIL.Image.[Resampling.]BOX`,
            `PIL.Image.[Resampling.]BILINEAR`, `PIL.Image.[Resampling.]HAMMING`, `PIL.Image.[Resampling.]BICUBIC` or
            `PIL.Image.[Resampling.]LANCZOS`. Only has an effect if `do_resize` is set to `True`.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the
            image is padded with 0's and then center cropped.
        crop_size (`int`, *optional*, defaults to 224):
            Desired output size when applying center-cropping. Only has an effect if `do_center_crop` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with `image_mean` and `image_std`.
        image_mean (`List[int]`, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
        convert_rgb (`bool`, defaults to `True`):
            Whether or not to convert `PIL.Image.Image` into `RGB` format
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=224,
        resample=Resampling.BICUBIC,
        do_center_crop=True,
        crop_size=224,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        do_convert_rgb=True,
        **kwargs
    ):
        super().__init__()
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std if image_std is not None else [0.26862954, 0.26130258, 0.27577711]
        self.do_convert_rgb = do_convert_rgb

    def __call__(
        self,
        images: Union[
            Image.Image,
            np.ndarray,
            "paddle.Tensor",
            List[Image.Image],
            List[np.ndarray],
            List["paddle.Tensor"],  # noqa
        ],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ):
        """
        Main method to prepare for the model one or several image(s).
        <Tip warning={true}>
        NumPy arrays and Paddle tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.
        </Tip>
        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `paddle.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[paddle.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or Paddle
                tensor. In case of a NumPy array/Paddle tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'pd'`: Return Paddle `paddle.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **pixel_values** -- Pixel values to be fed to a model.
        """
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or paddle.is_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or paddle.is_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `paddle.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[paddle.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or paddle.is_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (convert rgb + resizing + center cropping + normalization)
        if self.do_convert_rgb:
            images = [self.convert_rgb(image) for image in images]
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [
                self.resize(image=image, size=self.size, resample=self.resample, default_to_square=False)
                for image in images
            ]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
