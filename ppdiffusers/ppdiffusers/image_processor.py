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

import warnings
from typing import List, Optional, Union

import numpy as np
import paddle
import paddle.nn.functional as F
import PIL
from PIL import Image

from .configuration_utils import ConfigMixin, register_to_config
from .utils import CONFIG_NAME, PIL_INTERPOLATION, deprecate


class VaeImageProcessor(ConfigMixin):
    """
    Image Processor for VAE

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`. Can accept
            `height` and `width` arguments from `preprocess` method
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            VAE scale factor. If `do_resize` is True, the image will be automatically resized to multiples of this
            factor.
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use when resizing the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1]
        do_convert_rgb (`bool`, *optional*, defaults to be `False`):
            Whether to convert the images to RGB format.
    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        resample: str = "lanczos",
        do_normalize: bool = True,
        do_convert_rgb: bool = False,
    ):
        super().__init__()

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @staticmethod
    def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
        """
        Convert a PIL image or a list of PIL images to numpy arrays.
        """
        if not isinstance(images, list):
            images = [images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)

        return images

    @staticmethod
    def numpy_to_pd(images: np.ndarray) -> paddle.Tensor:
        """
        Convert a numpy image to a paddle tensor
        """
        if images.ndim == 3:
            images = images[..., None]

        images = paddle.to_tensor(images.transpose([0, 3, 1, 2]))
        return images

    @staticmethod
    def pd_to_numpy(images: paddle.Tensor) -> np.ndarray:
        """
        Convert a paddle tensor to a numpy image
        """
        images = images.cast("float32").transpose([0, 2, 3, 1]).numpy()
        return images

    @staticmethod
    def normalize(images):
        """
        Normalize an image array to [-1,1]
        """
        return 2.0 * images - 1.0

    @staticmethod
    def denormalize(images):
        """
        Denormalize an image array to [0,1]
        """
        return (images / 2 + 0.5).clip(0, 1)

    @staticmethod
    def convert_to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Converts an image to RGB format.
        """
        image = image.convert("RGB")
        return image

    def resize(
        self,
        image: PIL.Image.Image,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> PIL.Image.Image:
        """
        Resize a PIL image. Both height and width will be downscaled to the next integer multiple of `vae_scale_factor`
        """
        if height is None:
            height = image.height
        if width is None:
            width = image.width

        width, height = (
            x - x % self.config.vae_scale_factor for x in (width, height)
        )  # resize to integer multiple of vae_scale_factor
        image = image.resize((width, height), resample=PIL_INTERPOLATION[self.config.resample])
        return image

    def preprocess(
        self,
        image: Union[paddle.Tensor, PIL.Image.Image, np.ndarray],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> paddle.Tensor:
        """
        Preprocess the image input, accepted formats are PIL images, numpy arrays or paddle tensors"
        """
        supported_formats = (PIL.Image.Image, np.ndarray, paddle.Tensor)
        if isinstance(image, supported_formats):
            image = [image]
        elif not (isinstance(image, list) and all(isinstance(i, supported_formats) for i in image)):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support {', '.join(supported_formats)}"
            )

        if isinstance(image[0], PIL.Image.Image):
            if self.config.do_convert_rgb:
                image = [self.convert_to_rgb(i) for i in image]
            if self.config.do_resize:
                image = [self.resize(i, height, width) for i in image]
            image = self.pil_to_numpy(image)  # to np
            image = self.numpy_to_pd(image)  # to pd

        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            image = self.numpy_to_pd(image)
            _, _, height, width = image.shape
            if self.config.do_resize and (
                height % self.config.vae_scale_factor != 0 or width % self.config.vae_scale_factor != 0
            ):
                raise ValueError(
                    f"Currently we only support resizing for PIL image - please resize your numpy array to be divisible by {self.config.vae_scale_factor}"
                    f"currently the sizes are {height} and {width}. You can also pass a PIL image instead to use resize option in VAEImageProcessor"
                )

        elif isinstance(image[0], paddle.Tensor):
            image = paddle.concat(image, axis=0) if image[0].ndim == 4 else paddle.stack(image, axis=0)
            _, channel, height, width = image.shape

            # don't need any preprocess if the image is latents
            if channel == 4:
                return image

            if self.config.do_resize and (
                height % self.config.vae_scale_factor != 0 or width % self.config.vae_scale_factor != 0
            ):
                raise ValueError(
                    f"Currently we only support resizing for PIL image - please resize your paddle tensor to be divisible by {self.config.vae_scale_factor}"
                    f"currently the sizes are {height} and {width}. You can also pass a PIL image instead to use resize option in VAEImageProcessor"
                )

        # expected range [0,1], normalize to [-1,1]
        do_normalize = self.config.do_normalize
        if image.min() < 0:
            warnings.warn(
                "Passing `image` as paddle tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as paddle tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False

        if do_normalize:
            image = self.normalize(image)

        return image

    def preprocess_mask(self, mask, batch_size=1):
        if not isinstance(mask, paddle.Tensor):
            mask = mask.convert("L")
            w, h = mask.size
            w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
            mask = mask.resize(
                (w // self.config.vae_scale_factor, h // self.config.vae_scale_factor),
                resample=PIL_INTERPOLATION["nearest"],
            )
            mask = np.array(mask).astype(np.float32) / 255.0
            mask = np.tile(mask, (4, 1, 1))
            mask = np.vstack([mask[None]] * batch_size)
            mask = 1 - mask  # repaint white, keep black
            mask = paddle.to_tensor(mask)
        else:
            valid_mask_channel_sizes = [1, 3]
            # if mask channel is fourth tensor dimension, permute dimensions to pytorch standard (B, C, H, W)
            if mask.shape[3] in valid_mask_channel_sizes:
                mask = mask.transpose([0, 3, 1, 2])
            elif mask.shape[1] not in valid_mask_channel_sizes:
                raise ValueError(
                    f"Mask channel dimension of size in {valid_mask_channel_sizes} should be second or fourth dimension,"
                    f" but received mask of shape {tuple(mask.shape)}"
                )
            # (potentially) reduce mask channel dimension from 3 to 1 for broadcasting to latent shape
            mask = mask.mean(axis=1, keepdim=True)
            h, w = mask.shape[-2:]
            h, w = (x - x % 8 for x in (h, w))  # resize to integer multiple of 8
            mask = F.interpolate(mask, (h // self.config.vae_scale_factor, w // self.config.vae_scale_factor))
        return mask

    def postprocess(
        self,
        image: paddle.Tensor,
        output_type: str = "pil",
        do_denormalize: Optional[List[bool]] = None,
    ):
        if not isinstance(image, paddle.Tensor):
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support paddle tensor"
            )
        if output_type not in ["latent", "pd", "np", "pil"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
                "`pil`, `np`, `pd`, `latent`"
            )
            deprecate("Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False)
            output_type = "np"

        if output_type == "latent":
            return image

        if do_denormalize is None:
            do_denormalize = [self.config.do_normalize] * image.shape[0]

        image = paddle.stack(
            [self.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])]
        )

        if output_type == "pd":
            return image

        image = self.pd_to_numpy(image)

        if output_type == "np":
            return image

        if output_type == "pil":
            return self.numpy_to_pil(image)
