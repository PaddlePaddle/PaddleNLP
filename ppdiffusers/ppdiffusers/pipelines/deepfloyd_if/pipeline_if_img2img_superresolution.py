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
import paddle.nn.functional as F
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


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from ppdiffusers.pipelines.deepfloyd_if.pipeline_if_img2img.resize
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
        >>> from ppdiffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline, DiffusionPipeline
        >>> from ppdiffusers.utils import pd_to_pil
        >>> import paddle
        >>> from PIL import Image
        >>> import requests
        >>> from io import BytesIO

        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        >>> response = requests.get(url)
        >>> original_image = Image.open(BytesIO(response.content)).convert("RGB")
        >>> original_image = original_image.resize((768, 512))

        >>> pipe = IFImg2ImgPipeline.from_pretrained(
        ...     "DeepFloyd/IF-I-XL-v1.0",
        ...     variant="fp16",
        ...     paddle_dtype=paddle.float16,
        ... )

        >>> prompt = "A fantasy landscape in style minecraft"
        >>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

        >>> image = pipe(
        ...     image=original_image,
        ...     prompt_embeds=prompt_embeds,
        ...     negative_prompt_embeds=negative_embeds,
        ...     output_type="pd",
        ... ).images

        >>> # save intermediate image
        >>> pil_image = pd_to_pil(image)
        >>> pil_image[0].save("./if_stage_I.png")

        >>> super_res_1_pipe = IFImg2ImgSuperResolutionPipeline.from_pretrained(
        ...     "DeepFloyd/IF-II-L-v1.0",
        ...     text_encoder=None,
        ...     variant="fp16",
        ...     paddle_dtype=paddle.float16,
        ... )

        >>> image = super_res_1_pipe(
        ...     image=image,
        ...     original_image=original_image,
        ...     prompt_embeds=prompt_embeds,
        ...     negative_prompt_embeds=negative_embeds,
        ... ).images
        >>> image[0].save("./if_stage_II.png")
        ```
"""


class IFImg2ImgSuperResolutionPipeline(DiffusionPipeline):
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel

    unet: UNet2DConditionModel
    scheduler: DDPMScheduler
    image_noising_scheduler: DDPMScheduler

    feature_extractor: Optional[CLIPImageProcessor]
    safety_checker: Optional[IFSafetyChecker]

    watermarker: Optional[IFWatermarker]

    bad_punct_regex = re.compile(
        r"[" + "#®•©™&@·º½¾¿¡§~" + "\)" + "\(" + "\]" + "\[" + "\}" + "\{" + "\|" + "\\" + "\/" + "\*" + r"]{1,}"
    )  # noqa

    _optional_components = ["tokenizer", "text_encoder", "safety_checker", "feature_extractor"]

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
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the IF license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                f"Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
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

    # Copied from ppdiffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing
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

    # Copied from ppdiffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption
    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    @paddle.no_grad()
    # Copied from ppdiffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.encode_prompt
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
        r"""
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
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # while T5 can handle much longer input sequences than 77, the text encoder was trained with a max length of 77 for IF
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

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not paddle.equal_all(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )
            attention_mask = text_inputs.attention_mask
            prompt_embeds = self.text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.unet is not None:
            dtype = self.unet.dtype
        else:
            dtype = None

        if dtype is not None:
            prompt_embeds = prompt_embeds.cast(dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile([1, num_images_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape([bs_embed * num_images_per_prompt, seq_len, -1])

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
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

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids,
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            if dtype is not None:
                negative_prompt_embeds = negative_prompt_embeds.cast(dtype)

            negative_prompt_embeds = negative_prompt_embeds.tile([1, num_images_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape([batch_size * num_images_per_prompt, seq_len, -1])

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
        else:
            negative_prompt_embeds = None

        return prompt_embeds, negative_prompt_embeds

    # Copied from ppdiffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.run_safety_checker
    def run_safety_checker(self, image, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pd")
            image, nsfw_detected, watermark_detected = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.cast(dtype),
            )
        else:
            nsfw_detected = None
            watermark_detected = None

        return image, nsfw_detected, watermark_detected

    # Copied from ppdiffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.prepare_extra_step_kwargs
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
        image,
        original_image,
        batch_size,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
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

        # image

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
                "`image` has to be of type `paddle.Tensor`, `PIL.Image.Image`, `np.ndarray`, or List[...] but is"
                f" {type(check_image_type)}"
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

        # original_image

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
                "`original_image` has to be of type `paddle.Tensor`, `PIL.Image.Image`, `np.ndarray`, or List[...] but is"
                f" {type(check_image_type)}"
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

    # Copied from ppdiffusers.pipelines.deepfloyd_if.pipeline_if_img2img.IFImg2ImgPipeline.preprocess_image with preprocess_image -> preprocess_original_image
    def preprocess_original_image(self, image: PIL.Image.Image) -> paddle.Tensor:
        if not isinstance(image, list):
            image = [image]

        def numpy_to_pd(images):
            if images.ndim == 3:
                images = images[..., None]

            images = paddle.to_tensor(images.transpose(0, 3, 1, 2))
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

            image = np.stack(image, axis=0)  # to np
            image = numpy_to_pd(image)  # to pd

        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            image = numpy_to_pd(image)

        elif isinstance(image[0], paddle.Tensor):
            image = paddle.concat(image, axis=0) if image[0].ndim == 4 else paddle.stack(image, axis=0)

        return image

    # Copied from ppdiffusers.pipelines.deepfloyd_if.pipeline_if_superresolution.IFSuperResolutionPipeline.preprocess_image
    def preprocess_image(self, image: PIL.Image.Image, num_images_per_prompt) -> paddle.Tensor:
        if not isinstance(image, paddle.Tensor) and not isinstance(image, list):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            image = [np.array(i).astype(np.float32) / 255.0 for i in image]

            image = np.stack(image, axis=0)  # to np
            image = paddle.to_tensor(image.transpose(0, 3, 1, 2))
        elif isinstance(image[0], np.ndarray):
            image = np.stack(image, axis=0)  # to np
            if image.ndim == 5:
                image = image[0]

            image = paddle.to_tensor(image.transpose(0, 3, 1, 2))
        elif isinstance(image, list) and isinstance(image[0], paddle.Tensor):
            dims = image[0].ndim

            if dims == 3:
                image = paddle.stack(image, axis=0)
            elif dims == 4:
                image = paddle.concat(image, axis=0)
            else:
                raise ValueError(f"Image must have 3 or 4 dimensions, instead got {dims}")

        image = image.cast(self.unet.dtype)

        image = image.repeat_interleave(num_images_per_prompt, axis=0)

        return image

    # Copied from ppdiffusers.pipelines.deepfloyd_if.pipeline_if_img2img.IFImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    # Copied from ppdiffusers.pipelines.deepfloyd_if.pipeline_if_img2img.IFImg2ImgPipeline.prepare_intermediate_images
    def prepare_intermediate_images(self, image, timestep, batch_size, num_images_per_prompt, dtype, generator=None):
        _, channels, height, width = image.shape

        batch_size = batch_size * num_images_per_prompt

        shape = (batch_size, channels, height, width)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        noise = randn_tensor(shape, generator=generator, dtype=dtype)

        image = image.repeat_interleave(num_images_per_prompt, axis=0)
        image = self.scheduler.add_noise(image, noise, timestep)

        return image

    @paddle.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray, paddle.Tensor],
        original_image: Union[
            PIL.Image.Image, paddle.Tensor, np.ndarray, List[PIL.Image.Image], List[paddle.Tensor], List[np.ndarray]
        ] = None,
        strength: float = 0.8,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
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
        noise_level: int = 250,
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
                One or a list of [paddle generator(s)] to make generation deterministic.
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
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            noise_level (`int`, *optional*, defaults to 250):
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
        # 1. Check inputs. Raise error if not correct
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
            batch_size,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
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
            prompt_embeds = paddle.concat([negative_prompt_embeds, prompt_embeds])

        dtype = prompt_embeds.dtype

        # 4. Prepare timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps

        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)

        # 5. prepare original image
        original_image = self.preprocess_original_image(original_image)
        original_image = original_image.cast(dtype=dtype)

        # 6. Prepare intermediate images
        noise_timestep = timesteps[0:1]
        noise_timestep = noise_timestep.tile((batch_size * num_images_per_prompt,))

        intermediate_images = self.prepare_intermediate_images(
            original_image,
            noise_timestep,
            batch_size,
            num_images_per_prompt,
            dtype,
            generator,
        )

        # 7. Prepare upscaled image and noise level
        _, _, height, width = original_image.shape

        image = self.preprocess_image(image, num_images_per_prompt)

        upscaled = F.interpolate(image, (height, width), mode="bilinear", align_corners=True)

        noise_level = paddle.to_tensor([noise_level] * upscaled.shape[0])
        noise = randn_tensor(upscaled.shape, generator=generator, dtype=upscaled.dtype)
        upscaled = self.image_noising_scheduler.add_noise(upscaled, noise, timesteps=noise_level)

        if do_classifier_free_guidance:
            noise_level = paddle.concat([noise_level] * 2)

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_input = paddle.concat([intermediate_images, upscaled.cast(intermediate_images.dtype)], axis=1)

                model_input = paddle.concat([model_input] * 2) if do_classifier_free_guidance else model_input
                model_input = self.scheduler.scale_model_input(model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    class_labels=noise_level,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_uncond, _ = noise_pred_uncond.split(
                        [model_input.shape[1] // 2, noise_pred_uncond.shape[1] - model_input.shape[1] // 2], axis=1
                    )
                    noise_pred_text, predicted_variance = noise_pred_text.split(
                        [model_input.shape[1] // 2, noise_pred_text.shape[1] - model_input.shape[1] // 2], axis=1
                    )
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = paddle.concat([noise_pred, predicted_variance], axis=1)

                # compute the previous noisy sample x_t -> x_t-1
                intermediate_images = self.scheduler.step(
                    noise_pred, t, intermediate_images, **extra_step_kwargs
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, intermediate_images)

        image = intermediate_images

        if output_type == "pil":
            # 10. Post-processing
            image = (image / 2 + 0.5).clip(0, 1)
            image = image.cpu().transpose([0, 2, 3, 1]).cast("float32").numpy()

            # 11. Run safety checker
            image, nsfw_detected, watermark_detected = self.run_safety_checker(image, prompt_embeds.dtype)

            # 12. Convert to PIL
            image = self.numpy_to_pil(image)

            # 13. Apply watermark
            if self.watermarker is not None:
                self.watermarker.apply_watermark(image, self.unet.config.sample_size)
        elif output_type == "pd":
            nsfw_detected = None
            watermark_detected = None
        else:
            # 10. Post-processing
            image = (image / 2 + 0.5).clip(0, 1)
            image = image.cpu().transpose([0, 2, 3, 1]).cast("float32").numpy()

            # 11. Run safety checker
            image, nsfw_detected, watermark_detected = self.run_safety_checker(image, prompt_embeds.dtype)

        if not return_dict:
            return (image, nsfw_detected, watermark_detected)

        return IFPipelineOutput(images=image, nsfw_detected=nsfw_detected, watermark_detected=watermark_detected)
