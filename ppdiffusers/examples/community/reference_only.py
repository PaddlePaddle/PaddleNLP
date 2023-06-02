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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import paddle
import PIL
from packaging import version
from PIL import Image

from paddlenlp.transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from ppdiffusers.configuration_utils import FrozenDict
from ppdiffusers.models import AutoencoderKL, UNet2DConditionModel
from ppdiffusers.models.cross_attention import CrossAttention
from ppdiffusers.models.transformer_2d import Transformer2DModelOutput
from ppdiffusers.models.unet_2d_blocks import (
    ResnetBlock2D,
    Transformer2DModel,
    Upsample2D,
)
from ppdiffusers.pipeline_utils import DiffusionPipeline
from ppdiffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from ppdiffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from ppdiffusers.schedulers import KarrasDiffusionSchedulers
from ppdiffusers.utils import (
    PIL_INTERPOLATION,
    check_min_version,
    deprecate,
    logging,
    randn_tensor,
    replace_example_docstring,
)

check_min_version("0.14.1")

EPS = 1e-6

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import paddle
        >>> from ppdiffusers import ReferenceOnlyPipeline
        >>> from ppdiffusers.utils import load_image
        >>> pipe = ReferenceOnlyPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", paddle_dtype=paddle.float16)
        >>> image = load_image("dog_rel.png").resize((512, 512))
        >>> prompt = "a dog running on grassland, best quality"
        >>> image = pipe(prompt,
        ...     image=image,
        ...     width=512,
        ...     height=512,
        ...     control_name="refernce_only", # "none", "reference_only", "reference_adain", "reference_adain+attn"
        ...     attention_auto_machine_weight=1.0,
        ...     gn_auto_machine_weight=1.0,
        ...     current_style_fidelity=1.0).images[0]
        >>> image.save("refernce_only_dog.png")
        ```
"""


def stable_var(x, axis=None, unbiased=True, keepdim=False, name=None):
    dtype = x.dtype
    u = paddle.mean(x, axis=axis, keepdim=True, name=name)
    n = paddle.cast(paddle.numel(x), paddle.int64) / paddle.cast(paddle.numel(u), paddle.int64)
    n = n.astype(dtype)
    if unbiased:
        one_const = paddle.ones([], x.dtype)
        n = paddle.where(n > one_const, n - 1.0, one_const)
    n = n**0.5
    n.stop_gradient = True
    out = paddle.sum(paddle.pow((x - u) / n, 2), axis=axis, keepdim=keepdim, name=name)
    return out


def var_mean(x, axis=-1, keepdim=True, unbiased=True, correction=None):
    if correction is not None:
        unbiased = correction
    var = stable_var(x, axis=axis, keepdim=keepdim, unbiased=unbiased)
    mean = paddle.mean(x, axis=axis, keepdim=keepdim)
    return var, mean


def self_attn_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
    attn_output = None

    if getattr(self, "enable_attn", False):
        assert attention_mask is None, "attention_mask must be None!"
        if self.attention_auto_machine_weight > self.attn_weight:
            do_classifier_free_guidance = len(self.current_uc_indices) > 0
            chunk_num = 2 if do_classifier_free_guidance else 1
            latent_hidden_states = hidden_states[:chunk_num]  # uc, c
            image_hidden_states = hidden_states[chunk_num:]  # uc, c

            image_self_attn1 = self.processor(
                self,
                hidden_states=image_hidden_states,
                encoder_hidden_states=image_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            latent_self_attn1_uc = self.processor(
                self,
                latent_hidden_states,
                encoder_hidden_states=paddle.concat(
                    [latent_hidden_states]
                    + image_hidden_states.split([chunk_num] * (image_hidden_states.shape[0] // chunk_num)),
                    axis=1,
                ),
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            if do_classifier_free_guidance and self.current_style_fidelity > 1e-5:
                latent_self_attn1_c = latent_self_attn1_uc.clone()
                latent_self_attn1_c[self.current_uc_indices] = self.processor(
                    self,
                    hidden_states=latent_hidden_states[self.current_uc_indices],
                    encoder_hidden_states=latent_hidden_states[self.current_uc_indices],
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
                latent_self_attn1 = (
                    self.current_style_fidelity * latent_self_attn1_c
                    + (1.0 - self.current_style_fidelity) * latent_self_attn1_uc
                )
            else:
                latent_self_attn1 = latent_self_attn1_uc

            attn_output = paddle.concat([latent_self_attn1, image_self_attn1])

    if attn_output is None:
        attn_output = self.processor(
            self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
    return attn_output


def transformer_2d_model_forward(
    self,
    hidden_states,
    encoder_hidden_states=None,
    timestep=None,
    class_labels=None,
    cross_attention_kwargs=None,
    return_dict: bool = True,
):
    x = self.original_forward(
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        class_labels=class_labels,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=return_dict,
    )[0]
    output = None
    if getattr(self, "enable_gn", False):
        if self.gn_auto_machine_weight > self.gn_weight:
            do_classifier_free_guidance = len(self.current_uc_indices) > 0
            chunk_num = 2 if do_classifier_free_guidance else 1

            latent_hidden_states = x[:chunk_num]  # uc, c
            image_hidden_states = x[chunk_num:]  # uc, c
            image_var, image_mean = var_mean(image_hidden_states, axis=(2, 3), keepdim=True, unbiased=False)
            var, mean = var_mean(latent_hidden_states, axis=(2, 3), keepdim=True, unbiased=False)
            std = paddle.maximum(var, paddle.zeros_like(var) + EPS) ** 0.5

            div_num = image_hidden_states.shape[0] // chunk_num
            mean_acc = sum(image_mean.split([chunk_num] * div_num)) / div_num
            var_acc = sum(image_var.split([chunk_num] * div_num)) / div_num

            std_acc = paddle.maximum(var_acc, paddle.zeros_like(var_acc) + EPS) ** 0.5
            y_uc = (((latent_hidden_states - mean) / std) * std_acc) + mean_acc
            if do_classifier_free_guidance and self.current_style_fidelity > 1e-5:
                y_c = y_uc.clone()
                y_c[self.current_uc_indices] = latent_hidden_states[self.current_uc_indices]
                latent_hidden_states = self.current_style_fidelity * y_c + (1.0 - self.current_style_fidelity) * y_uc
            else:
                latent_hidden_states = y_uc
            output = paddle.concat([latent_hidden_states, image_hidden_states])

    if output is None:
        output = x
    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def resnet_block_2d_forward(self, input_tensor, temb):
    x = self.original_forward(input_tensor, temb=temb)
    output = None
    if getattr(self, "enable_gn", False):
        if self.gn_auto_machine_weight > self.gn_weight:
            do_classifier_free_guidance = len(self.current_uc_indices) > 0
            chunk_num = 2 if do_classifier_free_guidance else 1

            latent_hidden_states = x[:chunk_num]  # uc, c
            image_hidden_states = x[chunk_num:]  # uc, c
            image_var, image_mean = var_mean(image_hidden_states, axis=(2, 3), keepdim=True, unbiased=False)
            var, mean = var_mean(latent_hidden_states, axis=(2, 3), keepdim=True, unbiased=False)
            std = paddle.maximum(var, paddle.zeros_like(var) + EPS) ** 0.5

            div_num = image_hidden_states.shape[0] // chunk_num
            mean_acc = sum(image_mean.split([chunk_num] * div_num)) / div_num
            var_acc = sum(image_var.split([chunk_num] * div_num)) / div_num

            std_acc = paddle.maximum(var_acc, paddle.zeros_like(var_acc) + EPS) ** 0.5
            y_uc = (((latent_hidden_states - mean) / std) * std_acc) + mean_acc
            if do_classifier_free_guidance and self.current_style_fidelity > 1e-5:
                y_c = y_uc.clone()
                y_c[self.current_uc_indices] = latent_hidden_states[self.current_uc_indices]
                latent_hidden_states = self.current_style_fidelity * y_c + (1.0 - self.current_style_fidelity) * y_uc
            else:
                latent_hidden_states = y_uc
            output = paddle.concat([latent_hidden_states, image_hidden_states])

    if output is None:
        output = x

    return output


def upsample_2d_forward(self, hidden_states, output_size=None):
    x = self.original_forward(hidden_states, output_size=output_size)
    output = None
    if getattr(self, "enable_gn", False):
        if self.gn_auto_machine_weight > self.gn_weight:
            do_classifier_free_guidance = len(self.current_uc_indices) > 0
            chunk_num = 2 if do_classifier_free_guidance else 1

            latent_hidden_states = x[:chunk_num]  # uc, c
            image_hidden_states = x[chunk_num:]  # uc, c
            image_var, image_mean = var_mean(image_hidden_states, axis=(2, 3), keepdim=True, unbiased=False)
            var, mean = var_mean(latent_hidden_states, axis=(2, 3), keepdim=True, unbiased=False)
            std = paddle.maximum(var, paddle.zeros_like(var) + EPS) ** 0.5

            div_num = image_hidden_states.shape[0] // chunk_num
            mean_acc = sum(image_mean.split([chunk_num] * div_num)) / div_num
            var_acc = sum(image_var.split([chunk_num] * div_num)) / div_num

            std_acc = paddle.maximum(var_acc, paddle.zeros_like(var_acc) + EPS) ** 0.5
            y_uc = (((latent_hidden_states - mean) / std) * std_acc) + mean_acc
            if do_classifier_free_guidance and self.current_style_fidelity > 1e-5:
                y_c = y_uc.clone()
                y_c[self.current_uc_indices] = latent_hidden_states[self.current_uc_indices]
                latent_hidden_states = self.current_style_fidelity * y_c + (1.0 - self.current_style_fidelity) * y_uc
            else:
                latent_hidden_states = y_uc
            output = paddle.concat([latent_hidden_states, image_hidden_states])

    if output is None:
        output = x

    return output


if not hasattr(CrossAttention, "original_forward"):
    CrossAttention.original_forward = CrossAttention.forward
if not hasattr(Transformer2DModel, "original_forward"):
    Transformer2DModel.original_forward = Transformer2DModel.forward
if not hasattr(ResnetBlock2D, "original_forward"):
    ResnetBlock2D.original_forward = ResnetBlock2D.forward
if not hasattr(Upsample2D, "original_forward"):
    Upsample2D.original_forward = Upsample2D.forward
CrossAttention.forward = self_attn_forward
Transformer2DModel.forward = transformer_2d_model_forward
ResnetBlock2D.forward = resnet_block_2d_forward
Upsample2D.forward = upsample_2d_forward


def preprocess(image, resize_mode, width, height):
    if isinstance(image, paddle.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = resize_image(resize_mode=resize_mode, im=image, width=width, height=height)
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        image = [resize_image(resize_mode=resize_mode, im=im, width=width, height=height) for im in image]

        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = paddle.to_tensor(image)
    elif isinstance(image[0], paddle.Tensor):
        image = paddle.concat(image, axis=0)
    return image


def resize_image(resize_mode, im, width, height, upscaler_name=None):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
           -1: do nothing.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.
    """
    # ["Just resize", "Crop and resize", "Resize and fill", "Do nothing"]
    #         0              1                   2               -1
    def resize(im, w, h):
        if upscaler_name is None or upscaler_name == "None" or im.mode == "L":
            return im.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])

    if resize_mode == -1:
        return im
    elif resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(
                resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                box=(0, fill_height + src_h),
            )
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(
                resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                box=(fill_width + src_w, 0),
            )

    return res


class ReferenceOnlyPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with refernce only.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`PNDMScheduler`], [`EulerDiscreteScheduler`], [`EulerAncestralDiscreteScheduler`]
            or [`DPMSolverMultistepScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. PaddleNLP team, diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                f"Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_ppdiffusers_version") and version.parse(
            version.parse(unet.config._ppdiffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        self.attn_modules = None
        self.gn_modules = None

    def set_reference_only(
        self,
        attention_auto_machine_weight=1.0,
        gn_auto_machine_weight=1.0,
        current_style_fidelity=0.5,
        enable_attn=True,
        enable_gn=True,
        do_classifier_free_guidance=True,
    ):
        assert 0.0 <= attention_auto_machine_weight <= 1.0
        assert 0.0 <= gn_auto_machine_weight <= 2.0
        assert 0.0 <= current_style_fidelity <= 2.0

        if self.attn_modules is not None:
            for module in self.attn_modules:
                module.enable_attn = enable_attn
                module.attention_auto_machine_weight = attention_auto_machine_weight
                module.current_style_fidelity = current_style_fidelity
                module.current_uc_indices = [0] if do_classifier_free_guidance else []

        if self.gn_modules is not None:
            for module in self.gn_modules:
                module.enable_gn = enable_gn
                module.gn_auto_machine_weight = gn_auto_machine_weight
                module.current_style_fidelity = current_style_fidelity
                module.current_uc_indices = [0] if do_classifier_free_guidance else []

        # init attn_modules
        if self.attn_modules is None:
            attn_modules = []
            self_attn_processors_keys = []
            for name in self.unet.attn_processors.keys():
                if not name.endswith("attn1.processor"):
                    continue
                name = name.replace(".processor", "")
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]
                self_attn_processors_keys.append([name, hidden_size])

            # sorted by (-hidden_size, name)，down -> mid -> up.
            for i, (name, _) in enumerate(sorted(self_attn_processors_keys, key=lambda x: (-x[1], x[0]))):
                module = self.unet.get_sublayer(name)
                module.attn_weight = float(i) / float(len(self_attn_processors_keys))

                module.enable_attn = enable_attn
                module.attention_auto_machine_weight = attention_auto_machine_weight
                module.current_style_fidelity = current_style_fidelity
                module.current_uc_indices = [0] if do_classifier_free_guidance else []

                attn_modules.append(module)
            self.attn_modules = attn_modules

        # init gn_modules
        if self.gn_modules is None:
            gn_modules = [
                self.unet.mid_block.attentions[-1],
            ]
            self.unet.mid_block.attentions[-1].gn_weight = 0.0  # mid             0.0

            input_block_names = [
                ("down_blocks.1.resnets.0", "down_blocks.1.attentions.0"),  # 4   2.0
                ("down_blocks.1.resnets.1", "down_blocks.1.attentions.1"),  # 5   1.66
                ("down_blocks.2.resnets.0", "down_blocks.2.attentions.0"),  # 7   1.33
                ("down_blocks.2.resnets.1", "down_blocks.2.attentions.1"),  # 8   1.0
                ("down_blocks.3.resnets.0",),  # 10                               0.66
                ("down_blocks.3.resnets.1",),  # 11                               0.33
            ]
            for w, block_names in enumerate(input_block_names):
                module = self.unet.get_sublayer(block_names[-1])
                module.gn_weight = 1.0 - float(w) / float(len(input_block_names))
                gn_modules.append(module)

            output_block_names = [
                ("up_blocks.0.resnets.0",),  # 0                                 0.0
                ("up_blocks.0.resnets.1",),  # 1                                 0.25
                ("up_blocks.0.resnets.2", "up_blocks.0.upsamplers.0"),  # 2      0.5
                ("up_blocks.1.resnets.0", "up_blocks.1.attentions.0"),  # 3      0.75
                ("up_blocks.1.resnets.1", "up_blocks.1.attentions.1"),  # 4      1.0
                ("up_blocks.1.resnets.2", "up_blocks.1.attentions.2"),  # 5      1.25
                ("up_blocks.2.resnets.0", "up_blocks.2.attentions.0"),  # 6      1.5
                ("up_blocks.2.resnets.1", "up_blocks.2.attentions.1"),  # 7      1.75
            ]
            for w, block_names in enumerate(output_block_names):
                module = self.unet.get_sublayer(block_names[-1])
                module.gn_weight = float(w) / float(len(output_block_names))
                gn_modules.append(module)

            for module in gn_modules:
                module.gn_weight *= 2
                module.enable_gn = enable_gn
                module.gn_auto_machine_weight = gn_auto_machine_weight
                module.current_style_fidelity = current_style_fidelity
                module.current_uc_indices = [0] if do_classifier_free_guidance else []

            self.gn_modules = gn_modules

    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pd",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pd").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not paddle.equal_all(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.cast(self.text_encoder.dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile([1, num_images_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape([bs_embed * num_images_per_prompt, seq_len, -1])

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pd",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids,
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.cast(self.text_encoder.dtype)

            negative_prompt_embeds = negative_prompt_embeds.tile([1, num_images_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape([batch_size * num_images_per_prompt, seq_len, -1])

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = paddle.concat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def run_safety_checker(self, image, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pd")
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.cast(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.transpose([0, 2, 3, 1]).cast("float32").numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
        shape = [batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor]
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_image_latents(self, image, batch_size, dtype, generator=None, do_classifier_free_guidance=False):
        if not isinstance(image, (paddle.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `paddle.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )
        image = image.cast(dtype)

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = paddle.concat(init_latents, axis=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = self.vae.config.scaling_factor * init_latents

        if do_classifier_free_guidance:
            init_latents = paddle.concat([init_latents] * 2)

        return init_latents

    @paddle.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], paddle.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # reference
        control_name: str = "reference_only",  # "none", "reference_only", "reference_adain", "reference_adain+attn"
        attention_auto_machine_weight: float = 1.0,
        gn_auto_machine_weight: float = 1.0,
        current_style_fidelity: float = 0.5,
        resize_mode: int = -1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `paddle.Tensor`):
                The image or images to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
                One or a list of paddle generator(s) to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: paddle.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in `ppdiffusers.models.cross_attention`.
        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        assert control_name in ["none", "reference_only", "reference_adain", "reference_adain+attn"]
        assert num_images_per_prompt == 1
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        dtype = prompt_embeds.dtype

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. reference_only
        enable_attn = (
            "only" in control_name
            or "attn" in control_name
            and image is not None
            and attention_auto_machine_weight > 0
        )
        enable_gn = "adain" in control_name and image is not None and gn_auto_machine_weight > 0
        self.set_reference_only(
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            current_style_fidelity,
            enable_attn,
            enable_gn,
            do_classifier_free_guidance,
        )

        if enable_attn or enable_gn:
            image = preprocess(image, resize_mode, width, height)
            image_latents = self.prepare_image_latents(
                image, batch_size, dtype, generator, do_classifier_free_guidance
            )
            prompt_embeds = prompt_embeds.tile([1 + image.shape[0], 1, 1])

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if enable_attn or enable_gn:
                    image_noise = randn_tensor(image_latents.shape, generator=generator, dtype=dtype)
                    image_latent_model_input = self.scheduler.scale_model_input(
                        self.scheduler.add_noise(image_latents, image_noise, t), t
                    )
                    chunk_num = 2 if do_classifier_free_guidance else 1
                    noise_pred = self.unet(
                        paddle.concat([latent_model_input, image_latent_model_input]),
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample[:chunk_num]
                else:
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 9. Post-processing
            image = self.decode_latents(latents)

            # 10. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, prompt_embeds.dtype)

            # 11. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 9. Post-processing
            image = self.decode_latents(latents)

            # 10. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, prompt_embeds.dtype)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
