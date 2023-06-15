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
#
# modified from https://github.com/AUTOMATIC1111/stable-diffusion-webui
# Here is the AGPL-3.0 license https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/LICENSE.txt

import inspect
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import paddle
import paddle.nn as nn

from paddlenlp.transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from ppdiffusers.models import AutoencoderKL, UNet2DConditionModel
from ppdiffusers.pipelines.pipeline_utils import DiffusionPipeline
from ppdiffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from ppdiffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from ppdiffusers.schedulers import KarrasDiffusionSchedulers
from ppdiffusers.utils import (
    PPDIFFUSERS_CACHE,
    logging,
    ppdiffusers_url_download,
    randn_tensor,
    safetensors_load,
    smart_load,
    torch_load,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


import copy
import os
import os.path

from huggingface_hub.file_download import _request_wrapper, hf_raise_for_status

# lark omegaconf


def get_civitai_download_url(display_url, url_prefix="https://civitai.com"):
    if "api/download" in display_url:
        return display_url
    import bs4
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE"
    }
    r = requests.get(display_url, headers=headers)
    soup = bs4.BeautifulSoup(r.text, "lxml")
    download_url = None
    for a in soup.find_all("a", href=True):
        if "Download" in str(a):
            download_url = url_prefix + a["href"].split("?")[0]
            break
    return download_url


def http_file_name(
    url: str,
    *,
    proxies=None,
    headers: Optional[Dict[str, str]] = None,
    timeout=10.0,
    max_retries=0,
):
    """
    Get a remote file name.
    """
    headers = copy.deepcopy(headers) or {}
    r = _request_wrapper(
        method="GET",
        url=url,
        stream=True,
        proxies=proxies,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
    )
    hf_raise_for_status(r)
    displayed_name = url.split("/")[-1]
    content_disposition = r.headers.get("Content-Disposition")
    if content_disposition is not None and "filename=" in content_disposition:
        # Means file is on CDN
        displayed_name = content_disposition.split("filename=")[-1]
    return displayed_name


@paddle.no_grad()
def load_lora(
    pipeline,
    state_dict: dict,
    LORA_PREFIX_UNET: str = "lora_unet",
    LORA_PREFIX_TEXT_ENCODER: str = "lora_te",
    ratio: float = 1.0,
):
    ratio = float(ratio)
    visited = []
    for key in state_dict:
        if ".alpha" in key or ".lora_up" in key or key in visited:
            continue

        if "text" in key:
            tmp_layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            hf_to_ppnlp = {
                "encoder": "transformer",
                "fc1": "linear1",
                "fc2": "linear2",
            }
            layer_infos = []
            for layer_info in tmp_layer_infos:
                if layer_info == "mlp":
                    continue
                layer_infos.append(hf_to_ppnlp.get(layer_info, layer_info))
            curr_layer: paddle.nn.Linear = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer: paddle.nn.Linear = pipeline.unet

        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                if temp_name == "to":
                    raise ValueError()
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        triplet_keys = [key, key.replace("lora_down", "lora_up"), key.replace("lora_down.weight", "alpha")]
        dtype: paddle.dtype = curr_layer.weight.dtype
        weight_down: paddle.Tensor = state_dict[triplet_keys[0]].cast(dtype)
        weight_up: paddle.Tensor = state_dict[triplet_keys[1]].cast(dtype)
        rank: float = float(weight_down.shape[0])
        if triplet_keys[2] in state_dict:
            alpha: float = state_dict[triplet_keys[2]].cast(dtype).item()
            scale: float = alpha / rank
        else:
            scale = 1.0

        if not hasattr(curr_layer, "backup_weights"):
            curr_layer.backup_weights = curr_layer.weight.clone()

        if len(weight_down.shape) == 4:
            if weight_down.shape[2:4] == [1, 1]:
                # conv2d 1x1
                curr_layer.weight.copy_(
                    curr_layer.weight
                    + ratio
                    * paddle.matmul(weight_up.squeeze([-1, -2]), weight_down.squeeze([-1, -2])).unsqueeze([-1, -2])
                    * scale,
                    True,
                )
            else:
                # conv2d 3x3
                curr_layer.weight.copy_(
                    curr_layer.weight
                    + ratio
                    * paddle.nn.functional.conv2d(weight_down.transpose([1, 0, 2, 3]), weight_up).transpose(
                        [1, 0, 2, 3]
                    )
                    * scale,
                    True,
                )
        else:
            # linear
            curr_layer.weight.copy_(curr_layer.weight + ratio * paddle.matmul(weight_up, weight_down).T * scale, True)

        # update visited list
        visited.extend(triplet_keys)
    return pipeline


class WebUIStableDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

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
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]
    enable_emphasis = True
    comma_padding_backtrack = 20
    LORA_DIR = os.path.join(PPDIFFUSERS_CACHE, "lora")
    TI_DIR = os.path.join(PPDIFFUSERS_CACHE, "textual_inversion")

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

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

        # custom data
        clip_model = FrozenCLIPEmbedder(text_encoder, tokenizer)
        self.sj = StableDiffusionModelHijack(clip_model)
        self.orginal_scheduler_config = self.scheduler.config
        self.supported_scheduler = [
            "pndm",
            "lms",
            "euler",
            "euler-ancestral",
            "dpm-multi",
            "dpm-single",
            "unipc-multi",
            "ddim",
            "ddpm",
            "deis-multi",
            "heun",
            "kdpm2-ancestral",
            "kdpm2",
        ]
        self.weights_has_changed = False

        # register_state_dict_hook to fix text_encoder, when we save_pretrained text model.
        def map_to(state_dict, *args, **kwargs):
            if "text_model.token_embedding.wrapped.weight" in state_dict:
                state_dict["text_model.token_embedding.weight"] = state_dict.pop(
                    "text_model.token_embedding.wrapped.weight"
                )
            return state_dict

        self.text_encoder.register_state_dict_hook(map_to)

    def add_ti_embedding_dir(self, embeddings_dir=None):
        self.sj.embedding_db.add_embedding_dir(embeddings_dir)
        self.sj.embedding_db.load_textual_inversion_embeddings()

    def clear_ti_embedding(self):
        self.sj.embedding_db.clear_embedding_dirs()
        self.sj.embedding_db.load_textual_inversion_embeddings(True)

    def download_civitai_lora_file(self, url):
        if os.path.isfile(url):
            dst = os.path.join(self.LORA_DIR, os.path.basename(url))
            shutil.copyfile(url, dst)
            return dst

        download_url = get_civitai_download_url(url) or url
        file_path = ppdiffusers_url_download(
            download_url, cache_dir=self.LORA_DIR, filename=http_file_name(download_url).strip('"')
        )
        return file_path

    def download_civitai_ti_file(self, url):
        if os.path.isfile(url):
            dst = os.path.join(self.TI_DIR, os.path.basename(url))
            shutil.copyfile(url, dst)
            return dst

        download_url = get_civitai_download_url(url) or url
        file_path = ppdiffusers_url_download(
            download_url, cache_dir=self.TI_DIR, filename=http_file_name(download_url).strip('"')
        )
        return file_path

    def change_scheduler(self, scheduler_type="ddim"):
        self.switch_scheduler(scheduler_type)

    def switch_scheduler(self, scheduler_type="ddim"):
        scheduler_type = scheduler_type.lower()
        from ppdiffusers import (
            DDIMScheduler,
            DDPMScheduler,
            DEISMultistepScheduler,
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            HeunDiscreteScheduler,
            KDPM2AncestralDiscreteScheduler,
            KDPM2DiscreteScheduler,
            LMSDiscreteScheduler,
            PNDMScheduler,
            UniPCMultistepScheduler,
        )

        if scheduler_type == "pndm":
            scheduler = PNDMScheduler.from_config(self.orginal_scheduler_config, skip_prk_steps=True)
        elif scheduler_type == "lms":
            scheduler = LMSDiscreteScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "heun":
            scheduler = HeunDiscreteScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "euler":
            scheduler = EulerDiscreteScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "euler-ancestral":
            scheduler = EulerAncestralDiscreteScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "dpm-multi":
            scheduler = DPMSolverMultistepScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "dpm-single":
            scheduler = DPMSolverSinglestepScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "kdpm2-ancestral":
            scheduler = KDPM2AncestralDiscreteScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "kdpm2":
            scheduler = KDPM2DiscreteScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "unipc-multi":
            scheduler = UniPCMultistepScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "ddim":
            scheduler = DDIMScheduler.from_config(
                self.orginal_scheduler_config,
                steps_offset=1,
                clip_sample=False,
                set_alpha_to_one=False,
            )
        elif scheduler_type == "ddpm":
            scheduler = DDPMScheduler.from_config(
                self.orginal_scheduler_config,
            )
        elif scheduler_type == "deis-multi":
            scheduler = DEISMultistepScheduler.from_config(
                self.orginal_scheduler_config,
            )
        else:
            raise ValueError(
                f"Scheduler of type {scheduler_type} doesn't exist! Please choose in {self.supported_scheduler}!"
            )
        self.scheduler = scheduler

    @paddle.no_grad()
    def _encode_prompt(
        self,
        prompt: str,
        do_classifier_free_guidance: float = 7.5,
        negative_prompt: str = None,
        num_inference_steps: int = 50,
    ):
        if do_classifier_free_guidance:
            assert isinstance(negative_prompt, str)
            negative_prompt = [negative_prompt]
            uc = get_learned_conditioning(self.sj.clip, negative_prompt, num_inference_steps)
        else:
            uc = None

        c = get_multicond_learned_conditioning(self.sj.clip, prompt, num_inference_steps)
        return c, uc

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

        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"`prompt` has to be of type `str` but is {type(prompt)}")

        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise ValueError(f"`negative_prompt` has to be of type `str` but is {type(negative_prompt)}")

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

    @paddle.no_grad()
    def __call__(
        self,
        prompt: str = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str = None,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = 1,
        enable_lora: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
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
            negative_prompt (`str`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
                One or a list of paddle generator(s) to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
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
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            clip_skip (`int`, *optional*, defaults to 1):
                CLIP_stop_at_last_layers, if clip_skip <= 1, we will use the last_hidden_state from text_encoder.
        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        self.add_ti_embedding_dir(self.TI_DIR)

        try:
            # 0. Default height and width to unet
            height = height or max(self.unet.config.sample_size * self.vae_scale_factor, 512)
            width = width or max(self.unet.config.sample_size * self.vae_scale_factor, 512)

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                negative_prompt,
            )

            batch_size = 1

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            prompts, extra_network_data = parse_prompts([prompt])

            if enable_lora and self.LORA_DIR is not None:
                if os.path.exists(self.LORA_DIR):
                    lora_mapping = {p.stem: p.absolute() for p in Path(self.LORA_DIR).glob("*.safetensors")}
                    for params in extra_network_data["lora"]:
                        assert len(params.items) > 0
                        name = params.items[0]
                        if name in lora_mapping:
                            ratio = float(params.items[1]) if len(params.items) > 1 else 1.0
                            lora_state_dict = smart_load(lora_mapping[name], map_location=paddle.get_device())
                            self.weights_has_changed = True
                            load_lora(self, state_dict=lora_state_dict, ratio=ratio)
                            del lora_state_dict
                        else:
                            print(f"We can't find lora weight: {name}! Please make sure that exists!")
                else:
                    if len(extra_network_data["lora"]) > 0:
                        print(f"{self.LORA_DIR} not exists, so we cant load loras!")

            self.sj.clip.CLIP_stop_at_last_layers = clip_skip
            # 3. Encode input prompt
            prompt_embeds, negative_prompt_embeds = self._encode_prompt(
                prompts,
                do_classifier_free_guidance,
                negative_prompt,
                num_inference_steps=num_inference_steps,
            )

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps

            # 5. Prepare latent variables
            num_channels_latents = self.unet.in_channels
            latents = self.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                self.unet.dtype,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    step = i // self.scheduler.order
                    do_batch = False
                    conds_list, cond_tensor = reconstruct_multicond_batch(prompt_embeds, step)
                    try:
                        weight = conds_list[0][0][1]
                    except Exception:
                        weight = 1.0
                    if do_classifier_free_guidance:
                        uncond_tensor = reconstruct_cond_batch(negative_prompt_embeds, step)
                        do_batch = cond_tensor.shape[1] == uncond_tensor.shape[1]

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = paddle.concat([latents] * 2) if do_batch else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if do_batch:
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=paddle.concat([uncond_tensor, cond_tensor]),
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + weight * guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    else:
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=cond_tensor,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample

                        if do_classifier_free_guidance:
                            noise_pred_uncond = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=uncond_tensor,
                                cross_attention_kwargs=cross_attention_kwargs,
                            ).sample
                            noise_pred = noise_pred_uncond + weight * guidance_scale * (noise_pred - noise_pred_uncond)

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
                # 8. Post-processing
                image = self.decode_latents(latents)

                # 9. Run safety checker
                image, has_nsfw_concept = self.run_safety_checker(image, self.unet.dtype)

                # 10. Convert to PIL
                image = self.numpy_to_pil(image)
            else:
                # 8. Post-processing
                image = self.decode_latents(latents)

                # 9. Run safety checker
                image, has_nsfw_concept = self.run_safety_checker(image, self.unet.dtype)

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        except Exception as e:
            raise ValueError(e)
        finally:
            if enable_lora and self.weights_has_changed:
                for sub_layer in self.text_encoder.sublayers(include_self=True):
                    if hasattr(sub_layer, "backup_weights"):
                        sub_layer.weight.copy_(sub_layer.backup_weights, True)
                for sub_layer in self.unet.sublayers(include_self=True):
                    if hasattr(sub_layer, "backup_weights"):
                        sub_layer.weight.copy_(sub_layer.backup_weights, True)
                self.weights_has_changed = False


# clip.py
import math
from collections import namedtuple


class PromptChunk:
    """
    This object contains token ids, weight (multipliers:1.4) and textual inversion embedding info for a chunk of prompt.
    If a prompt is short, it is represented by one PromptChunk, otherwise, multiple are necessary.
    Each PromptChunk contains an exact amount of tokens - 77, which includes one for start and end token,
    so just 75 tokens from prompt.
    """

    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []


PromptChunkFix = namedtuple("PromptChunkFix", ["offset", "embedding"])
"""An object of this type is a marker showing that textual inversion embedding's vectors have to placed at offset in the prompt
chunk. Thos objects are found in PromptChunk.fixes and, are placed into FrozenCLIPEmbedderWithCustomWordsBase.hijack.fixes, and finally
are applied by sd_hijack.EmbeddingsWithFixes's forward function."""


class FrozenCLIPEmbedder(nn.Layer):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(self, text_encoder, tokenizer, freeze=True, layer="last", layer_idx=None):
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.text_encoder.eval()
        for param in self.parameters():
            param.stop_gradient = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pd",
        )
        tokens = batch_encoding["input_ids"]
        outputs = self.text_encoder(input_ids=tokens, output_hidden_states=self.layer == "hidden", return_dict=True)
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedderWithCustomWordsBase(nn.Layer):
    """A pytorch module that is a wrapper for FrozenCLIPEmbedder module. it enhances FrozenCLIPEmbedder, making it possible to
    have unlimited prompt length and assign weights to tokens in prompt.
    """

    def __init__(self, wrapped, hijack):
        super().__init__()

        self.wrapped = wrapped
        """Original FrozenCLIPEmbedder module; can also be FrozenOpenCLIPEmbedder or xlmr.BertSeriesModelWithTransformation,
        depending on model."""

        self.hijack = hijack
        self.chunk_length = 75

    def empty_chunk(self):
        """creates an empty PromptChunk and returns it"""

        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def get_target_prompt_token_count(self, token_count):
        """returns the maximum number of tokens a prompt of a known length can have before it requires one more PromptChunk to be represented"""

        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def tokenize(self, texts):
        """Converts a batch of texts into a batch of token ids"""

        raise NotImplementedError

    def encode_with_text_encoder(self, tokens):
        """
        converts a batch of token ids (in python lists) into a single tensor with numeric respresentation of those tokens;
        All python lists with tokens are assumed to have same length, usually 77.
        if input is a list with B elements and each element has T tokens, expected output shape is (B, T, C), where C depends on
        model - can be 768 and 1024.
        Among other things, this call will read self.hijack.fixes, apply it to its inputs, and clear it (setting it to None).
        """

        raise NotImplementedError

    def encode_embedding_init_text(self, init_text, nvpt):
        """Converts text into a tensor with this text's tokens' embeddings. Note that those are embeddings before they are passed through
        transformers. nvpt is used as a maximum length in tokens. If text produces less teokens than nvpt, only this many is returned."""

        raise NotImplementedError

    def tokenize_line(self, line):
        """
        this transforms a single prompt into a list of PromptChunk objects - as many as needed to
        represent the prompt.
        Returns the list and the total number of tokens in the prompt.
        """

        if WebUIStableDiffusionPipeline.enable_emphasis:
            parsed = parse_prompt_attention(line)
        else:
            parsed = [[line, 1.0]]

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            """puts current chunk into the list of results and produces the next one - empty;
            if is_last is true, tokens <end-of-text> tokens at the end won't add to token_count"""
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk

            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == "BREAK" and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)

                # this is when we are at the end of alloted 75 tokens for the current chunk, and the current token is not a comma. opts.comma_padding_backtrack
                # is a setting that specifies that if there is a comma nearby, the text after the comma should be moved out of this chunk and into the next.
                elif (
                    WebUIStableDiffusionPipeline.comma_padding_backtrack != 0
                    and len(chunk.tokens) == self.chunk_length
                    and last_comma != -1
                    and len(chunk.tokens) - last_comma <= WebUIStableDiffusionPipeline.comma_padding_backtrack
                ):
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                embedding, embedding_length_in_tokens = self.hijack.embedding_db.find_embedding_at_position(
                    tokens, position
                )
                if embedding is None:
                    chunk.tokens.append(token)
                    chunk.multipliers.append(weight)
                    position += 1
                    continue

                emb_len = int(embedding.vec.shape[0])
                if len(chunk.tokens) + emb_len > self.chunk_length:
                    next_chunk()

                chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))

                chunk.tokens += [0] * emb_len
                chunk.multipliers += [weight] * emb_len
                position += embedding_length_in_tokens

        if len(chunk.tokens) > 0 or len(chunks) == 0:
            next_chunk(is_last=True)

        return chunks, token_count

    def process_texts(self, texts):
        """
        Accepts a list of texts and calls tokenize_line() on each, with cache. Returns the list of results and maximum
        length, in tokens, of all texts.
        """

        token_count = 0

        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)

                cache[line] = chunks

            batch_chunks.append(chunks)

        return batch_chunks, token_count

    def forward(self, texts):
        """
        Accepts an array of texts; Passes texts through transformers network to create a tensor with numerical representation of those texts.
        Returns a tensor with shape of (B, T, C), where B is length of the array; T is length, in tokens, of texts (including padding) - T will
        be a multiple of 77; and C is dimensionality of each token - for SD1 it's 768, and for SD2 it's 1024.
        An example shape returned by this function can be: (2, 77, 768).
        Webui usually sends just one text at a time through this function - the only time when texts is an array with more than one elemenet
        is when you do prompt editing: "a picture of a [cat:dog:0.4] eating ice cream"
        """

        batch_chunks, token_count = self.process_texts(texts)

        used_embeddings = {}
        chunk_count = max([len(x) for x in batch_chunks])

        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]

            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            self.hijack.fixes = [x.fixes for x in batch_chunk]

            for fixes in self.hijack.fixes:
                for position, embedding in fixes:
                    used_embeddings[embedding.name] = embedding

            z = self.process_tokens(tokens, multipliers)
            zs.append(z)

        if len(used_embeddings) > 0:
            embeddings_list = ", ".join(
                [f"{name} [{embedding.checksum()}]" for name, embedding in used_embeddings.items()]
            )
            self.hijack.comments.append(f"Used embeddings: {embeddings_list}")

        return paddle.concat(zs, axis=1)

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        """
        sends one single prompt chunk to be encoded by transformers neural network.
        remade_batch_tokens is a batch of tokens - a list, where every element is a list of tokens; usually
        there are exactly 77 tokens in the list. batch_multipliers is the same but for multipliers instead of tokens.
        Multipliers are used to give more or less weight to the outputs of transformers network. Each multiplier
        corresponds to one token.
        """
        tokens = paddle.to_tensor(remade_batch_tokens)

        # this is for SD2: SD1 uses the same token for padding and end of text, while SD2 uses different ones.
        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index + 1 : tokens.shape[1]] = self.id_pad

        z = self.encode_with_text_encoder(tokens)

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers = paddle.to_tensor(batch_multipliers)
        original_mean = z.mean()
        z = z * batch_multipliers.reshape(
            batch_multipliers.shape
            + [
                1,
            ]
        ).expand(z.shape)
        new_mean = z.mean()
        z = z * (original_mean / new_mean)

        return z


class FrozenCLIPEmbedderWithCustomWords(FrozenCLIPEmbedderWithCustomWordsBase):
    def __init__(self, wrapped, hijack, CLIP_stop_at_last_layers=-1):
        super().__init__(wrapped, hijack)
        self.CLIP_stop_at_last_layers = CLIP_stop_at_last_layers
        self.tokenizer = wrapped.tokenizer

        vocab = self.tokenizer.get_vocab()

        self.comma_token = vocab.get(",</w>", None)

        self.token_mults = {}
        tokens_with_parens = [(k, v) for k, v in vocab.items() if "(" in k or ")" in k or "[" in k or "]" in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == "[":
                    mult /= 1.1
                if c == "]":
                    mult *= 1.1
                if c == "(":
                    mult *= 1.1
                if c == ")":
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

        self.id_start = self.wrapped.tokenizer.bos_token_id
        self.id_end = self.wrapped.tokenizer.eos_token_id
        self.id_pad = self.id_end

    def tokenize(self, texts):
        tokenized = self.wrapped.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

        return tokenized

    def encode_with_text_encoder(self, tokens):
        output_hidden_states = self.CLIP_stop_at_last_layers > 1
        outputs = self.wrapped.text_encoder(
            input_ids=tokens, output_hidden_states=output_hidden_states, return_dict=True
        )

        if output_hidden_states:
            z = outputs.hidden_states[-self.CLIP_stop_at_last_layers]
            z = self.wrapped.text_encoder.text_model.ln_final(z)
        else:
            z = outputs.last_hidden_state

        return z

    def encode_embedding_init_text(self, init_text, nvpt):
        embedding_layer = self.wrapped.text_encoder.text_model
        ids = self.wrapped.tokenizer(init_text, max_length=nvpt, return_tensors="pd", add_special_tokens=False)[
            "input_ids"
        ]
        embedded = embedding_layer.token_embedding.wrapped(ids).squeeze(0)

        return embedded


# extra_networks.py
import re
from collections import defaultdict


class ExtraNetworkParams:
    def __init__(self, items=None):
        self.items = items or []


re_extra_net = re.compile(r"<(\w+):([^>]+)>")


def parse_prompt(prompt):
    res = defaultdict(list)

    def found(m):
        name = m.group(1)
        args = m.group(2)

        res[name].append(ExtraNetworkParams(items=args.split(":")))

        return ""

    prompt = re.sub(re_extra_net, found, prompt)

    return prompt, res


def parse_prompts(prompts):
    res = []
    extra_data = None

    for prompt in prompts:
        updated_prompt, parsed_extra_data = parse_prompt(prompt)

        if extra_data is None:
            extra_data = parsed_extra_data

        res.append(updated_prompt)

    return res, extra_data


# image_embeddings.py

import base64
import json
import zlib

import numpy as np
from PIL import Image


class EmbeddingDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, d):
        if "TORCHTENSOR" in d:
            return paddle.to_tensor(np.array(d["TORCHTENSOR"]))
        return d


def embedding_from_b64(data):
    d = base64.b64decode(data)
    return json.loads(d, cls=EmbeddingDecoder)


def lcg(m=2**32, a=1664525, c=1013904223, seed=0):
    while True:
        seed = (a * seed + c) % m
        yield seed % 255


def xor_block(block):
    g = lcg()
    randblock = np.array([next(g) for _ in range(np.product(block.shape))]).astype(np.uint8).reshape(block.shape)
    return np.bitwise_xor(block.astype(np.uint8), randblock & 0x0F)


def crop_black(img, tol=0):
    mask = (img > tol).all(2)
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), mask.shape[1] - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), mask.shape[0] - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def extract_image_data_embed(image):
    d = 3
    outarr = (
        crop_black(np.array(image.convert("RGB").getdata()).reshape(image.size[1], image.size[0], d).astype(np.uint8))
        & 0x0F
    )
    black_cols = np.where(np.sum(outarr, axis=(0, 2)) == 0)
    if black_cols[0].shape[0] < 2:
        print("No Image data blocks found.")
        return None

    data_block_lower = outarr[:, : black_cols[0].min(), :].astype(np.uint8)
    data_block_upper = outarr[:, black_cols[0].max() + 1 :, :].astype(np.uint8)

    data_block_lower = xor_block(data_block_lower)
    data_block_upper = xor_block(data_block_upper)

    data_block = (data_block_upper << 4) | (data_block_lower)
    data_block = data_block.flatten().tobytes()

    data = zlib.decompress(data_block)
    return json.loads(data, cls=EmbeddingDecoder)


# prompt_parser.py
import re
from collections import namedtuple
from typing import List

import lark

# a prompt like this: "fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][ in background:0.25] [shoddy:masterful:0.5]"
# will be represented with prompt_schedule like this (assuming steps=100):
# [25, 'fantasy landscape with a mountain and an oak in foreground shoddy']
# [50, 'fantasy landscape with a lake and an oak in foreground in background shoddy']
# [60, 'fantasy landscape with a lake and an oak in foreground in background masterful']
# [75, 'fantasy landscape with a lake and an oak in background masterful']
# [100, 'fantasy landscape with a lake and a christmas tree in background masterful']

schedule_parser = lark.Lark(
    r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER "]"
alternate: "[" prompt ("|" prompt)+ "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
"""
)


def get_learned_conditioning_prompt_schedules(prompts, steps):
    """
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b: 3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [[[b]]:2]")
    [[2, 'a '], [10, 'a [[b]]']]
    >>> g("[(a:2):3]")
    [[3, ''], [10, '(a:2)']]
    >>> g("a [b : c : 1] d")
    [[1, 'a b  d'], [10, 'a  c  d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("a [unbalanced")
    [[10, 'a [unbalanced']]
    >>> g("a [b:.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    >>> g("a [{b|d{:.5] c")  # not handling this right now
    [[5, 'a  c'], [10, 'a {b|d{ c']]
    >>> g("((a][:b:c [d:3]")
    [[3, '((a][:b:c '], [10, '((a][:b:c d']]
    >>> g("[a|(b:1.1)]")
    [[1, 'a'], [2, '(b:1.1)'], [3, 'a'], [4, '(b:1.1)'], [5, 'a'], [6, '(b:1.1)'], [7, 'a'], [8, '(b:1.1)'], [9, 'a'], [10, '(b:1.1)']]
    """

    def collect_steps(steps, tree):
        l = [steps]

        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                tree.children[-1] = float(tree.children[-1])
                if tree.children[-1] < 1:
                    tree.children[-1] *= steps
                tree.children[-1] = min(steps, int(tree.children[-1]))
                l.append(tree.children[-1])

            def alternate(self, tree):
                l.extend(range(1, steps + 1))

        CollectSteps().visit(tree)
        return sorted(set(l))

    def at_step(step, tree):
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, _, when = args
                yield before or () if step <= when else after

            def alternate(self, args):
                yield next(args[(step - 1) % len(args)])

            def start(self, args):
                def flatten(x):
                    if type(x) == str:
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)

                return "".join(flatten(args))

            def plain(self, args):
                yield args[0].value

            def __default__(self, data, children, meta):
                for child in children:
                    yield child

        return AtStep().transform(tree)

    def get_schedule(prompt):
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError:
            if 0:
                import traceback

                traceback.print_exc()
            return [[steps, prompt]]
        return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]


ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])


def get_learned_conditioning(model, prompts, steps):
    """converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning, specifying the comdition (cond),
    and the sampling step at which this condition is to be replaced by the next one.

    Input:
    (model, ['a red crown', 'a [blue:green:5] jeweled crown'], 20)

    Output:
    [
        [
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0523,  ..., -0.4901, -0.3066,  0.0674], ..., [ 0.3317, -0.5102, -0.4066,  ...,  0.4119, -0.7647, -1.0160]], device='cuda:0'))
        ],
        [
            ScheduledPromptConditioning(end_at_step=5, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.0192,  0.3867, -0.4644,  ...,  0.1135, -0.3696, -0.4625]], device='cuda:0')),
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.7352, -0.4356, -0.7888,  ...,  0.6994, -0.4312, -1.2593]], device='cuda:0'))
        ]
    ]
    """
    res = []

    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps)
    cache = {}

    for prompt, prompt_schedule in zip(prompts, prompt_schedules):

        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue

        texts = [x[1] for x in prompt_schedule]
        conds = model(texts)

        cond_schedule = []
        for i, (end_at_step, text) in enumerate(prompt_schedule):
            cond_schedule.append(ScheduledPromptConditioning(end_at_step, conds[i]))

        cache[prompt] = cond_schedule
        res.append(cond_schedule)

    return res


re_AND = re.compile(r"\bAND\b")
re_weight = re.compile(r"^(.*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")


def get_multicond_prompt_list(prompts):
    res_indexes = []

    prompt_flat_list = []
    prompt_indexes = {}

    for prompt in prompts:
        subprompts = re_AND.split(prompt)

        indexes = []
        for subprompt in subprompts:
            match = re_weight.search(subprompt)

            text, weight = match.groups() if match is not None else (subprompt, 1.0)

            weight = float(weight) if weight is not None else 1.0

            index = prompt_indexes.get(text, None)
            if index is None:
                index = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = index

            indexes.append((index, weight))

        res_indexes.append(indexes)

    return res_indexes, prompt_flat_list, prompt_indexes


class ComposableScheduledPromptConditioning:
    def __init__(self, schedules, weight=1.0):
        self.schedules: List[ScheduledPromptConditioning] = schedules
        self.weight: float = weight


class MulticondLearnedConditioning:
    def __init__(self, shape, batch):
        self.shape: tuple = shape  # the shape field is needed to send this object to DDIM/PLMS
        self.batch: List[List[ComposableScheduledPromptConditioning]] = batch


def get_multicond_learned_conditioning(model, prompts, steps) -> MulticondLearnedConditioning:
    """same as get_learned_conditioning, but returns a list of ScheduledPromptConditioning along with the weight objects for each prompt.
    For each prompt, the list is obtained by splitting the prompt using the AND separator.

    https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/
    """

    res_indexes, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)

    learned_conditioning = get_learned_conditioning(model, prompt_flat_list, steps)

    res = []
    for indexes in res_indexes:
        res.append([ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes])

    return MulticondLearnedConditioning(shape=(len(prompts),), batch=res)


def reconstruct_cond_batch(c: List[List[ScheduledPromptConditioning]], current_step):
    param = c[0][0].cond
    res = paddle.zeros(
        [
            len(c),
        ]
        + param.shape,
        dtype=param.dtype,
    )
    for i, cond_schedule in enumerate(c):
        target_index = 0
        for current, (end_at, cond) in enumerate(cond_schedule):
            if current_step <= end_at:
                target_index = current
                break
        res[i] = cond_schedule[target_index].cond

    return res


def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step):
    param = c.batch[0][0].schedules[0].cond

    tensors = []
    conds_list = []

    for batch_no, composable_prompts in enumerate(c.batch):
        conds_for_batch = []

        for cond_index, composable_prompt in enumerate(composable_prompts):
            target_index = 0
            for current, (end_at, cond) in enumerate(composable_prompt.schedules):
                if current_step <= end_at:
                    target_index = current
                    break

            conds_for_batch.append((len(tensors), composable_prompt.weight))
            tensors.append(composable_prompt.schedules[target_index].cond)

        conds_list.append(conds_for_batch)

    # if prompts have wildly different lengths above the limit we'll get tensors fo different shapes
    # and won't be able to torch.stack them. So this fixes that.
    token_count = max([x.shape[0] for x in tensors])
    for i in range(len(tensors)):
        if tensors[i].shape[0] != token_count:
            last_vector = tensors[i][-1:]
            last_vector_repeated = last_vector.tile([token_count - tensors[i].shape[0], 1])
            tensors[i] = paddle.concat([tensors[i], last_vector_repeated], axis=0)

    return conds_list, paddle.stack(tensors).cast(dtype=param.dtype)


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


# sd_hijack.py


class StableDiffusionModelHijack:
    fixes = None
    comments = []
    layers = None
    circular_enabled = False

    def __init__(self, clip_model, embeddings_dir=None, CLIP_stop_at_last_layers=-1):
        model_embeddings = clip_model.text_encoder.text_model
        model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
        clip_model = FrozenCLIPEmbedderWithCustomWords(
            clip_model, self, CLIP_stop_at_last_layers=CLIP_stop_at_last_layers
        )

        self.embedding_db = EmbeddingDatabase(clip_model)
        self.embedding_db.add_embedding_dir(embeddings_dir)

        # hack this!
        self.clip = clip_model

        def flatten(el):
            flattened = [flatten(children) for children in el.children()]
            res = [el]
            for c in flattened:
                res += c
            return res

        self.layers = flatten(clip_model)

    def clear_comments(self):
        self.comments = []

    def get_prompt_lengths(self, text):
        _, token_count = self.clip.process_texts([text])

        return token_count, self.clip.get_target_prompt_token_count(token_count)


class EmbeddingsWithFixes(nn.Layer):
    def __init__(self, wrapped, embeddings):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                emb = embedding.vec.cast(self.wrapped.dtype)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = paddle.concat([tensor[0 : offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len :]])

            vecs.append(tensor)

        return paddle.stack(vecs)


# textual_inversion.py

import os
import sys
import traceback


class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.shape = None
        self.vectors = 0
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.optimizer_state_dict = None
        self.filename = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        paddle.save(embedding_data, filename)

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f"{const_hash(self.vec.flatten() * 100) & 0xffff:04x}"
        return self.cached_checksum


class DirWithTextualInversionEmbeddings:
    def __init__(self, path):
        self.path = path
        self.mtime = None

    def has_changed(self):
        if not os.path.isdir(self.path):
            return False

        mt = os.path.getmtime(self.path)
        if self.mtime is None or mt > self.mtime:
            return True

    def update(self):
        if not os.path.isdir(self.path):
            return

        self.mtime = os.path.getmtime(self.path)


class EmbeddingDatabase:
    def __init__(self, clip):
        self.clip = clip
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.skipped_embeddings = {}
        self.expected_shape = -1
        self.embedding_dirs = {}
        self.previously_displayed_embeddings = ()

    def add_embedding_dir(self, path):
        if path is not None and path not in self.embedding_dirs:
            self.embedding_dirs[path] = DirWithTextualInversionEmbeddings(path)

    def clear_embedding_dirs(self):
        self.embedding_dirs.clear()

    def register_embedding(self, embedding, model):
        self.word_embeddings[embedding.name] = embedding

        ids = model.tokenize([embedding.name])[0]

        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []

        self.ids_lookup[first_id] = sorted(
            self.ids_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]), reverse=True
        )

        return embedding

    def get_expected_shape(self):
        vec = self.clip.encode_embedding_init_text(",", 1)
        return vec.shape[1]

    def load_from_file(self, path, filename):
        name, ext = os.path.splitext(filename)
        ext = ext.upper()

        if ext in [".PNG", ".WEBP", ".JXL", ".AVIF"]:
            _, second_ext = os.path.splitext(name)
            if second_ext.upper() == ".PREVIEW":
                return

            embed_image = Image.open(path)
            if hasattr(embed_image, "text") and "sd-ti-embedding" in embed_image.text:
                data = embedding_from_b64(embed_image.text["sd-ti-embedding"])
                name = data.get("name", name)
            else:
                data = extract_image_data_embed(embed_image)
                if data:
                    name = data.get("name", name)
                else:
                    # if data is None, means this is not an embeding, just a preview image
                    return
        elif ext in [".BIN", ".PT"]:
            data = torch_load(path)
        elif ext in [".SAFETENSORS"]:
            data = safetensors_load(path)
        else:
            return

        # textual inversion embeddings
        if "string_to_param" in data:
            param_dict = data["string_to_param"]
            if hasattr(param_dict, "_parameters"):
                param_dict = getattr(param_dict, "_parameters")
            assert len(param_dict) == 1, "embedding file has multiple terms in it"
            emb = next(iter(param_dict.items()))[1]
        # diffuser concepts
        elif type(data) == dict and type(next(iter(data.values()))) == paddle.Tensor:
            assert len(data.keys()) == 1, "embedding file has multiple terms in it"

            emb = next(iter(data.values()))
            if len(emb.shape) == 1:
                emb = emb.unsqueeze(0)
        else:
            raise Exception(
                f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept."
            )

        with paddle.no_grad():
            if hasattr(emb, "detach"):
                emb = emb.detach()
            if hasattr(emb, "cpu"):
                emb = emb.cpu()
            if hasattr(emb, "numpy"):
                emb = emb.numpy()
            emb = paddle.to_tensor(emb)
            vec = emb.detach().cast(paddle.float32)
        embedding = Embedding(vec, name)
        embedding.step = data.get("step", None)
        embedding.sd_checkpoint = data.get("sd_checkpoint", None)
        embedding.sd_checkpoint_name = data.get("sd_checkpoint_name", None)
        embedding.vectors = vec.shape[0]
        embedding.shape = vec.shape[-1]
        embedding.filename = path

        if self.expected_shape == -1 or self.expected_shape == embedding.shape:
            self.register_embedding(embedding, self.clip)
        else:
            self.skipped_embeddings[name] = embedding

    def load_from_dir(self, embdir):
        if not os.path.isdir(embdir.path):
            return

        for root, dirs, fns in os.walk(embdir.path, followlinks=True):
            for fn in fns:
                try:
                    fullfn = os.path.join(root, fn)

                    if os.stat(fullfn).st_size == 0:
                        continue

                    self.load_from_file(fullfn, fn)
                except Exception:
                    print(f"Error loading embedding {fn}:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    continue

    def load_textual_inversion_embeddings(self, force_reload=False):
        if not force_reload:
            need_reload = False
            for path, embdir in self.embedding_dirs.items():
                if embdir.has_changed():
                    need_reload = True
                    break

            if not need_reload:
                return

        self.ids_lookup.clear()
        self.word_embeddings.clear()
        self.skipped_embeddings.clear()
        self.expected_shape = self.get_expected_shape()

        for path, embdir in self.embedding_dirs.items():
            self.load_from_dir(embdir)
            embdir.update()

        displayed_embeddings = (tuple(self.word_embeddings.keys()), tuple(self.skipped_embeddings.keys()))
        if self.previously_displayed_embeddings != displayed_embeddings:
            self.previously_displayed_embeddings = displayed_embeddings
            print(
                f"Textual inversion embeddings loaded({len(self.word_embeddings)}): {', '.join(self.word_embeddings.keys())}"
            )
            if len(self.skipped_embeddings) > 0:
                print(
                    f"Textual inversion embeddings skipped({len(self.skipped_embeddings)}): {', '.join(self.skipped_embeddings.keys())}"
                )

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset : offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None
