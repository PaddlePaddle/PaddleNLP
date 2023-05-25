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

import html
import inspect
import re
import urllib.parse as ul
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import paddle
import paddle.nn as nn
import PIL

from paddlenlp.transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

from ...models import UNet2DConditionModel
from ...schedulers import DDPMScheduler
from ...utils import (
    BACKENDS_MAPPING,
    PIL_INTERPOLATION,
    is_bs4_available,
    is_ftfy_available,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from ..pipeline_utils import DiffusionPipeline
from . import IFPipelineOutput
from .safety_checker import IFSafetyChecker
from .watermark import IFWatermarker

if is_bs4_available():
    from bs4 import BeautifulSoup
if is_ftfy_available():
    import ftfy
logger = logging.get_logger(__name__)


def resize(images: PIL.Image.Image, img_size: int) -> PIL.Image.Image:
    w, h = images.size
    coef = w / h
    w, h = img_size, img_size
    if coef >= 1:
        w = int(round(img_size / 8 * coef) * 8)
    else:
        h = int(round(img_size / 8 / coef) * 8)
    images = images.resize((w, h), resample=PIL_INTERPOLATION["bicubic"], reducing_gap=None)
    return images


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from ppdiffusers import IFInpaintingPipeline, IFInpaintingSuperResolutionPipeline, DiffusionPipeline
        >>> from ppdiffusers.utils import pt_to_pil
        >>> import paddle
        >>> from PIL import Image
        >>> import requests
        >>> from io import BytesIO

        >>> url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/person.png"
        >>> response = requests.get(url)
        >>> original_image = Image.open(BytesIO(response.content)).convert("RGB")
        >>> original_image = original_image

        >>> url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/glasses_mask.png"
        >>> response = requests.get(url)
        >>> mask_image = Image.open(BytesIO(response.content))
        >>> mask_image = mask_image

        >>> pipe = IFInpaintingPipeline.from_pretrained(
        ...     "DeepFloyd/IF-I-XL-v1.0", variant="fp16", paddle_dtype=paddle.float16
        ... )

        >>> prompt = "blue sunglasses"

        >>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)
        >>> image = pipe(
        ...     image=original_image,
        ...     mask_image=mask_image,
        ...     prompt_embeds=prompt_embeds,
        ...     negative_prompt_embeds=negative_embeds,
        ...     output_type="pd",
        ... ).images

        >>> # save intermediate image
        >>> pil_image = pt_to_pil(image)
        >>> pil_image[0].save("./if_stage_I.png")

        >>> super_res_1_pipe = IFInpaintingSuperResolutionPipeline.from_pretrained(
        ...     "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", paddle_dtype=paddle.float16
        ... )

        >>> image = super_res_1_pipe(
        ...     image=image,
        ...     mask_image=mask_image,
        ...     original_image=original_image,
        ...     prompt_embeds=prompt_embeds,
        ...     negative_prompt_embeds=negative_embeds,
        ... ).images
        >>> image[0].save("./if_stage_II.png")
        ```
    """


class IFInpaintingSuperResolutionPipeline(DiffusionPipeline):
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel
    unet: UNet2DConditionModel
    scheduler: DDPMScheduler
    image_noising_scheduler: DDPMScheduler
    feature_extractor: Optional[CLIPImageProcessor]
    safety_checker: Optional[IFSafetyChecker]
    watermarker: Optional[IFWatermarker]
    bad_punct_regex = re.compile(
        "["
        + "#®•©™&@·º½¾¿¡§~"
        + "\\)"
        + "\\("
        + "\\]"
        + "\\["
        + "\\}"
        + "\\{"
        + "\\|"
        + "\\"
        + "\\/"
        + "\\*"
        + "]{1,}"
    )
    _optional_components = ["tokenizer", "text_encoder", "safety_checker", "feature_extractor", "watermarker"]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        image_noising_scheduler: DDPMScheduler,
        safety_checker: Optional[IFSafetyChecker],
        feature_extractor: Optional[CLIPImageProcessor],
        watermarker: Optional[IFWatermarker],
        requires_safety_checker: bool = True,
    ):
        super().__init__()
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure that you abide to the conditions of the IF license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )
        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
        if unet.config.in_channels != 6:
            logger.warn(
                "It seems like you have loaded a checkpoint that shall not be used for super resolution from {unet.config._name_or_path} as it accepts {unet.config.in_channels} input channels instead of 6. Please make sure to pass a super resolution checkpoint as the `'unet'`: IFSuperResolutionPipeline.from_pretrained(unet=super_resolution_unet, ...)`."
            )
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            image_noising_scheduler=image_noising_scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            watermarker=watermarker,
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def _text_preprocessing(self, text, clean_caption=False):
        if clean_caption and not is_bs4_available():
            logger.warn(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            logger.warn("Setting `clean_caption` to False...")
            clean_caption = False
        if clean_caption and not is_ftfy_available():
            logger.warn(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            logger.warn("Setting `clean_caption` to False...")
            clean_caption = False
        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                text = text.lower().strip()
            return text

        return [process(t) for t in text]

    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        caption = re.sub(
            "\\b((?:https?:(?:\\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\\w/-]*\\b\\/?(?!@)))",
            "",
            caption,
        )
        caption = re.sub(
            "\\b((?:www:(?:\\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\\w/-]*\\b\\/?(?!@)))",
            "",
            caption,
        )
        caption = BeautifulSoup(caption, features="html.parser").text
        caption = re.sub("@[\\w\\d]+\\b", "", caption)
        caption = re.sub("[\\u31c0-\\u31ef]+", "", caption)
        caption = re.sub("[\\u31f0-\\u31ff]+", "", caption)
        caption = re.sub("[\\u3200-\\u32ff]+", "", caption)
        caption = re.sub("[\\u3300-\\u33ff]+", "", caption)
        caption = re.sub("[\\u3400-\\u4dbf]+", "", caption)
        caption = re.sub("[\\u4dc0-\\u4dff]+", "", caption)
        caption = re.sub("[\\u4e00-\\u9fff]+", "", caption)
        caption = re.sub(
            "[\\u002D\\u058A\\u05BE\\u1400\\u1806\\u2010-\\u2015\\u2E17\\u2E1A\\u2E3A\\u2E3B\\u2E40\\u301C\\u3030\\u30A0\\uFE31\\uFE32\\uFE58\\uFE63\\uFF0D]+",
            "-",
            caption,
        )
        caption = re.sub("[`´«»“”¨]", '"', caption)
        caption = re.sub("[‘’]", "'", caption)
        caption = re.sub("&quot;?", "", caption)
        caption = re.sub("&amp", "", caption)
        caption = re.sub("\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}", " ", caption)
        caption = re.sub("\\d:\\d\\d\\s+$", "", caption)
        caption = re.sub("\\\\n", " ", caption)
        caption = re.sub("#\\d{1,3}\\b", "", caption)
        caption = re.sub("#\\d{5,}\\b", "", caption)
        caption = re.sub("\\b\\d{6,}\\b", "", caption)
        caption = re.sub("[\\S]+\\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)
        caption = re.sub("[\\\"\\']{2,}", '"', caption)
        caption = re.sub("[\\.]{2,}", " ", caption)
        caption = re.sub(self.bad_punct_regex, " ", caption)
        caption = re.sub("\\s+\\.\\s+", " ", caption)
        regex2 = re.compile("(?:\\-|\\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)
        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))
        caption = re.sub("\\b[a-zA-Z]{1,3}\\d{3,15}\\b", "", caption)
        caption = re.sub("\\b[a-zA-Z]+\\d+[a-zA-Z]+\\b", "", caption)
        caption = re.sub("\\b\\d+[a-zA-Z]+\\d+\\b", "", caption)
        caption = re.sub("(worldwide\\s+)?(free\\s+)?shipping", "", caption)
        caption = re.sub("(free\\s)?download(\\sfree)?", "", caption)
        caption = re.sub("\\bclick\\b\\s(?:for|on)\\s\\w+", "", caption)
        caption = re.sub("\\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\\simage[s]?)?", "", caption)
        caption = re.sub("\\bpage\\s+\\d+\\b", "", caption)
        caption = re.sub("\\b\\d*[a-zA-Z]+\\d+[a-zA-Z]+\\d+[a-zA-Z\\d]*\\b", " ", caption)
        caption = re.sub("\\b\\d+\\.?\\d*[xх×]\\d+\\.?\\d*\\b", "", caption)
        caption = re.sub("\\b\\s+\\:\\s+", ": ", caption)
        caption = re.sub("(\\D[,\\./])\\b", "\\1 ", caption)
        caption = re.sub("\\s+", " ", caption)
        caption.strip()
        caption = re.sub("^[\\\"\\']([\\w\\W]+)[\\\"\\']$", "\\1", caption)
        caption = re.sub("^[\\'\\_,\\-\\:;]", "", caption)
        caption = re.sub("[\\'\\_,\\-\\:\\-\\+]$", "", caption)
        caption = re.sub("^\\.\\S+$", "", caption)
        return caption.strip()

    @paddle.no_grad()
    def encode_prompt(
        self,
        prompt,
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
        negative_prompt=None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        clean_caption: bool = False,
    ):
        """
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
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
        if prompt is not None and negative_prompt is not None:
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}."
                )
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        max_length = 77
        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pd",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pd").input_ids
            if (
                untruncated_ids.shape[-1] >= text_input_ids.shape[-1]
                and not paddle.equal_all(x=text_input_ids, y=untruncated_ids).item()
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
                logger.warning(
                    f"The following part of your input was truncated because CLIP can only handle sequences up to {max_length} tokens: {removed_text}"
                )
            attention_mask = text_inputs.attention_mask
            prompt_embeds = self.text_encoder(text_input_ids, attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.unet is not None:
            dtype = self.unet.dtype
        else:
            dtype = None
        prompt_embeds = prompt_embeds.cast(dtype)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.tile(repeat_times=[1, num_images_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape([bs_embed * num_images_per_prompt, seq_len, -1])
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
            uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pd",
            )
            attention_mask = uncond_input.attention_mask
            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids, attention_mask=attention_mask)
            negative_prompt_embeds = negative_prompt_embeds[0]
        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.cast(dtype)
            negative_prompt_embeds = negative_prompt_embeds.tile(repeat_times=[1, num_images_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape([batch_size * num_images_per_prompt, seq_len, -1])
        else:
            negative_prompt_embeds = None
        return prompt_embeds, negative_prompt_embeds

    def run_safety_checker(self, image, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pd").cast(dtype)
            image, nsfw_detected, watermark_detected = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype=dtype)
            )
        else:
            nsfw_detected = None
            watermark_detected = None
            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()
        return image, nsfw_detected, watermark_detected

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        image,
        original_image,
        mask_image,
        batch_size,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if (
            callback_steps is None
            or callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}."
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    f"`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}."
                )
        if isinstance(image, list):
            check_image_type = image[0]
        else:
            check_image_type = image
        if (
            not isinstance(check_image_type, paddle.Tensor)
            and not isinstance(check_image_type, PIL.Image.Image)
            and not isinstance(check_image_type, np.ndarray)
        ):
            raise ValueError(
                f"`image` has to be of type `paddle.Tensor`, `PIL.Image.Image`, `np.ndarray`, or List[...] but is {type(check_image_type)}"
            )
        if isinstance(image, list):
            image_batch_size = len(image)
        elif isinstance(image, paddle.Tensor):
            image_batch_size = image.shape[0]
        elif isinstance(image, PIL.Image.Image):
            image_batch_size = 1
        elif isinstance(image, np.ndarray):
            image_batch_size = image.shape[0]
        else:
            assert False
        if batch_size != image_batch_size:
            raise ValueError(f"image batch size: {image_batch_size} must be same as prompt batch size {batch_size}")
        if isinstance(original_image, list):
            check_image_type = original_image[0]
        else:
            check_image_type = original_image
        if (
            not isinstance(check_image_type, paddle.Tensor)
            and not isinstance(check_image_type, PIL.Image.Image)
            and not isinstance(check_image_type, np.ndarray)
        ):
            raise ValueError(
                f"`original_image` has to be of type `paddle.Tensor`, `PIL.Image.Image`, `np.ndarray`, or List[...] but is {type(check_image_type)}"
            )
        if isinstance(original_image, list):
            image_batch_size = len(original_image)
        elif isinstance(original_image, paddle.Tensor):
            image_batch_size = original_image.shape[0]
        elif isinstance(original_image, PIL.Image.Image):
            image_batch_size = 1
        elif isinstance(original_image, np.ndarray):
            image_batch_size = original_image.shape[0]
        else:
            assert False
        if batch_size != image_batch_size:
            raise ValueError(
                f"original_image batch size: {image_batch_size} must be same as prompt batch size {batch_size}"
            )
        if isinstance(mask_image, list):
            check_image_type = mask_image[0]
        else:
            check_image_type = mask_image
        if (
            not isinstance(check_image_type, paddle.Tensor)
            and not isinstance(check_image_type, PIL.Image.Image)
            and not isinstance(check_image_type, np.ndarray)
        ):
            raise ValueError(
                f"`mask_image` has to be of type `paddle.Tensor`, `PIL.Image.Image`, `np.ndarray`, or List[...] but is {type(check_image_type)}"
            )
        if isinstance(mask_image, list):
            image_batch_size = len(mask_image)
        elif isinstance(mask_image, paddle.Tensor):
            image_batch_size = mask_image.shape[0]
        elif isinstance(mask_image, PIL.Image.Image):
            image_batch_size = 1
        elif isinstance(mask_image, np.ndarray):
            image_batch_size = mask_image.shape[0]
        else:
            assert False
        if image_batch_size != 1 and batch_size != image_batch_size:
            raise ValueError(
                f"mask_image batch size: {image_batch_size} must be `1` or the same as prompt batch size {batch_size}"
            )

    def preprocess_original_image(self, image: PIL.Image.Image) -> paddle.Tensor:
        if not isinstance(image, list):
            image = [image]

        def numpy_to_pt(images):
            if images.ndim == 3:
                images = images[..., None]
            images = paddle.to_tensor(data=images.transpose(0, 3, 1, 2))
            return images

        if isinstance(image[0], PIL.Image.Image):
            new_image = []
            for image_ in image:
                image_ = image_.convert("RGB")
                image_ = resize(image_, self.unet.sample_size)
                image_ = np.array(image_)
                image_ = image_.astype(np.float32)
                image_ = image_ / 127.5 - 1
                new_image.append(image_)
            image = new_image
            image = np.stack(image, axis=0)
            image = numpy_to_pt(image)
        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            image = numpy_to_pt(image)
        elif isinstance(image[0], paddle.Tensor):
            image = paddle.concat(x=image, axis=0) if image[0].ndim == 4 else paddle.stack(x=image, axis=0)
        return image

    def preprocess_image(self, image: PIL.Image.Image, num_images_per_prompt) -> paddle.Tensor:
        if not isinstance(image, paddle.Tensor) and not isinstance(image, list):
            image = [image]
        if isinstance(image[0], PIL.Image.Image):
            image = [(np.array(i).astype(np.float32) / 255.0) for i in image]
            image = np.stack(image, axis=0)
            paddle.to_tensor(data=image.transpose(0, 3, 1, 2))
        elif isinstance(image[0], np.ndarray):
            image = np.stack(image, axis=0)
            if image.ndim == 5:
                image = image[0]
            image = paddle.to_tensor(data=image.transpose(0, 3, 1, 2))
        elif isinstance(image, list) and isinstance(image[0], paddle.Tensor):
            dims = image[0].ndim
            if dims == 3:
                image = paddle.stack(x=image, axis=0)
            elif dims == 4:
                image = paddle.concat(x=image, axis=0)
            else:
                raise ValueError(f"Image must have 3 or 4 dimensions, instead got {dims}")
        image = image.cast(self.unet.dtype)
        image = image.repeat_interleave(repeats=num_images_per_prompt, axis=0)
        return image

    def preprocess_mask_image(self, mask_image) -> paddle.Tensor:
        if not isinstance(mask_image, list):
            mask_image = [mask_image]
        if isinstance(mask_image[0], paddle.Tensor):
            mask_image = (
                paddle.concat(x=mask_image, axis=0) if mask_image[0].ndim == 4 else paddle.stack(x=mask_image, axis=0)
            )
            if mask_image.ndim == 2:
                mask_image = mask_image.unsqueeze(axis=0).unsqueeze(axis=0)
            elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
                mask_image = mask_image.unsqueeze(axis=0)
            elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
                mask_image = mask_image.unsqueeze(axis=1)
            mask_image[mask_image < 0.5] = 0
            mask_image[mask_image >= 0.5] = 1
        elif isinstance(mask_image[0], PIL.Image.Image):
            new_mask_image = []
            for mask_image_ in mask_image:
                mask_image_ = mask_image_.convert("L")
                mask_image_ = resize(mask_image_, self.unet.sample_size)
                mask_image_ = np.array(mask_image_)
                mask_image_ = mask_image_[(None), (None), :]
                new_mask_image.append(mask_image_)
            mask_image = new_mask_image
            mask_image = np.concatenate(mask_image, axis=0)
            mask_image = mask_image.astype(np.float32) / 255.0
            mask_image[mask_image < 0.5] = 0
            mask_image[mask_image >= 0.5] = 1
            mask_image = paddle.to_tensor(data=mask_image)
        elif isinstance(mask_image[0], np.ndarray):
            mask_image = np.concatenate([m[(None), (None), :] for m in mask_image], axis=0)
            mask_image[mask_image < 0.5] = 0
            mask_image[mask_image >= 0.5] = 1
            mask_image = paddle.to_tensor(data=mask_image)
        return mask_image

    def get_timesteps(self, num_inference_steps, strength):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]
        return timesteps, num_inference_steps - t_start

    def prepare_intermediate_images(
        self, image, timestep, batch_size, num_images_per_prompt, dtype, mask_image, generator=None
    ):
        image_batch_size, channels, height, width = image.shape
        batch_size = batch_size * num_images_per_prompt
        shape = batch_size, channels, height, width
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        noise = randn_tensor(shape, generator=generator, dtype=dtype)
        image = image.repeat_interleave(repeats=num_images_per_prompt, axis=0)
        noised_image = self.scheduler.add_noise(image, noise, timestep)
        image = (1 - mask_image) * image + mask_image * noised_image
        return image

    @paddle.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray, paddle.Tensor],
        original_image: Union[
            PIL.Image.Image, paddle.Tensor, np.ndarray, List[PIL.Image.Image], List[paddle.Tensor], List[np.ndarray]
        ] = None,
        mask_image: Union[
            PIL.Image.Image, paddle.Tensor, np.ndarray, List[PIL.Image.Image], List[paddle.Tensor], List[np.ndarray]
        ] = None,
        strength: float = 0.8,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 100,
        timesteps: List[int] = None,
        guidance_scale: float = 4.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        noise_level: int = 0,
        clean_caption: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image (`paddle.Tensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            original_image (`paddle.Tensor` or `PIL.Image.Image`):
                The original image that `image` was varied from.
            mask_image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted
                to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, H, W, 1)`.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
                One or a list of paddle generator(s)
                to make generation deterministic.
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
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: paddle.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                ppdiffusers.cross_attention.
            noise_level (`int`, *optional*, defaults to 0):
                The amount of noise to add to the upscaled image. Must be in the range `[0, 1000)`
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.IFPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.IFPipelineOutput`] if `return_dict` is True, otherwise a `tuple. When
            returning a tuple, the first element is a list with the generated images, and the second element is a list
            of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw)
            or watermarked content, according to the `safety_checker`.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        self.check_inputs(
            prompt,
            image,
            original_image,
            mask_image,
            batch_size,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clean_caption=clean_caption,
        )
        if do_classifier_free_guidance:
            prompt_embeds = paddle.concat(x=[negative_prompt_embeds, prompt_embeds])
        dtype = prompt_embeds.dtype
        if timesteps is not None:
            self.scheduler.set_timesteps(
                timesteps=timesteps,
            )
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(
                num_inference_steps,
            )
            timesteps = self.scheduler.timesteps
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        original_image = self.preprocess_original_image(original_image)
        original_image = original_image.cast(dtype)
        mask_image = self.preprocess_mask_image(mask_image)
        mask_image = mask_image.cast(dtype)
        if mask_image.shape[0] == 1:
            mask_image = mask_image.repeat_interleave(repeats=batch_size * num_images_per_prompt, axis=0)
        else:
            mask_image = mask_image.repeat_interleave(repeats=num_images_per_prompt, axis=0)
        noise_timestep = timesteps[0:1]
        noise_timestep = noise_timestep.tile(repeat_times=[batch_size * num_images_per_prompt])
        intermediate_images = self.prepare_intermediate_images(
            original_image, noise_timestep, batch_size, num_images_per_prompt, dtype, mask_image, generator
        )
        _, _, height, width = original_image.shape
        image = self.preprocess_image(
            image,
            num_images_per_prompt,
        )
        upscaled = nn.functional.interpolate(x=image, size=(height, width), mode="bilinear", align_corners=True)
        noise_level = paddle.to_tensor(data=[noise_level] * upscaled.shape[0], place=upscaled.place)
        noise = randn_tensor(upscaled.shape, generator=generator, dtype=upscaled.dtype)
        upscaled = self.image_noising_scheduler.add_noise(upscaled, noise, timesteps=noise_level)
        if do_classifier_free_guidance:
            noise_level = paddle.concat(x=[noise_level] * 2)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_input = paddle.concat(x=[intermediate_images, upscaled], axis=1)
                model_input = paddle.concat(x=[model_input] * 2) if do_classifier_free_guidance else model_input
                model_input = self.scheduler.scale_model_input(model_input, t)
                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    class_labels=noise_level,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(chunks=2)
                    noise_pred_uncond, _ = noise_pred_uncond.split(
                        [model_input.shape[1] // 2, noise_pred_uncond.shape[1] - model_input.shape[1] // 2], axis=1
                    )
                    noise_pred_text, predicted_variance = noise_pred_text.split(
                        [model_input.shape[1] // 2, noise_pred_text.shape[1] - model_input.shape[1] // 2], axis=1
                    )
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = paddle.concat(x=[noise_pred, predicted_variance], axis=1)
                prev_intermediate_images = intermediate_images
                intermediate_images = self.scheduler.step(
                    noise_pred, t, intermediate_images, **extra_step_kwargs
                ).prev_sample
                intermediate_images = (1 - mask_image) * prev_intermediate_images + mask_image * intermediate_images
                if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, intermediate_images)
        image = intermediate_images
        if output_type == "pil":
            image = (image / 2 + 0.5).clip(min=0, max=1)
            image = image.cpu().transpose(perm=[0, 2, 3, 1]).astype(dtype="float32").numpy()
            image, nsfw_detected, watermark_detected = self.run_safety_checker(image, prompt_embeds.dtype)
            image = self.numpy_to_pil(image)
            if self.watermarker is not None:
                self.watermarker.apply_watermark(image, self.unet.config.sample_size)
        elif output_type == "pd":
            nsfw_detected = None
            watermark_detected = None
            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()
        else:
            image = (image / 2 + 0.5).clip(min=0, max=1)
            image = image.cpu().transpose(perm=[0, 2, 3, 1]).astype(dtype="float32").numpy()
            image, nsfw_detected, watermark_detected = self.run_safety_checker(image, prompt_embeds.dtype)
        if not return_dict:
            return image, nsfw_detected, watermark_detected
        return IFPipelineOutput(images=image, nsfw_detected=nsfw_detected, watermark_detected=watermark_detected)
