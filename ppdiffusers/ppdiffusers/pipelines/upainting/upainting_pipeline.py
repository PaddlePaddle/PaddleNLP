# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import gc
import inspect
from typing import Callable, List, Optional, Union

import paddle
from paddle.distributed.fleet.utils import recompute
from paddle.vision import transforms

from ...models import AutoencoderKL, UNet2DConditionModel
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import (
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from ...utils import logging
from .disco_diffusion.functions import (
    MakeCutoutsDango,
    parse_prompt,
    range_loss,
    sat_loss,
    spherical_dist_loss,
    tv_loss,
)

try:
    from paddlenlp.transformers import (
        BertTokenizer,
        ChineseCLIPFeatureExtractor,
        ChineseCLIPModel,
        ChineseCLIPTokenizer,
        CLIPFeatureExtractor,
        CLIPModel,
        CLIPTokenizer,
        CMSIMLockFeatureExtractor,
        CMSIMLockModel,
        CMSIMLockTokenizer,
        ErnieViLFeatureExtractor,
        ErnieViLModel,
        ErnieViLTokenizer,
    )
except ImportError:
    raise ImportError("Please install the dependencies first, `pip install paddlenlp>=2.5.0`!")

from ..latent_diffusion.pipeline_latent_diffusion import LDMBertModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ERNIEVIL_PRETRAINED_MODEL_ARCHIVE_LIST = ["PaddlePaddle/ernie_vil-2.0-base-zh"]
CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "OFA-Sys/chinese-clip-vit-base-patch16",
    "OFA-Sys/chinese-clip-vit-huge-patch14",
    "OFA-Sys/chinese-clip-vit-large-patch14",
    "OFA-Sys/chinese-clip-vit-large-patch14-336px",
]
CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # vit model
    "openai/clip-vit-base-patch32",  # ViT-B/32
    "openai/clip-vit-base-patch16",  # ViT-B/16
    "openai/clip-vit-large-patch14",  # ViT-L/14
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    # resnet model
    "openai/clip-rn50",  # RN50
    "openai/clip-rn101",  # RN101
    "openai/clip-rn50x4",  # RN50x4
]
CMSIM_LOCK_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # vit model
    "PaddlePaddle/cmsim-lock-vit-base-patch32",  # ViT-B/32
    "PaddlePaddle/cmsim-lock-vit-base-patch16",  # ViT-B/16
    # resnet model
    "PaddlePaddle/cmsim-lock-rn50",  # RN50
]
model_type_to_cls = {
    "clip": (CLIPModel, CLIPTokenizer, CLIPFeatureExtractor),
    "chineseclip": (ChineseCLIPModel, ChineseCLIPTokenizer, ChineseCLIPFeatureExtractor),
    "ernievil": (ErnieViLModel, ErnieViLTokenizer, ErnieViLFeatureExtractor),
    "cmsim_lock": (CMSIMLockModel, CMSIMLockTokenizer, CMSIMLockFeatureExtractor),
}


def get_cls(model_name):
    if model_name in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST:
        model_type = "clip"
    elif model_name in CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST:
        model_type = "chineseclip"
    elif model_name in ERNIEVIL_PRETRAINED_MODEL_ARCHIVE_LIST:
        model_type = "ernievil"
    elif model_name in CMSIM_LOCK_PRETRAINED_MODEL_ARCHIVE_LIST:
        model_type = "cmsim_lock"
    else:
        raise ValueError(f"We donnot support {model_name} weights!")
    return model_type_to_cls[model_type]


def build_models(model_names):
    """
    args:
       model_names: a list of names.
    """
    all_models = []

    for name in model_names:
        model_cls, tokenizer_cls, feature_extractor_cls = get_cls(name)
        model = model_cls.from_pretrained(name)
        tokenizer = tokenizer_cls.from_pretrained(name)
        feature_extractor = feature_extractor_cls.from_pretrained(name)
        model.eval()
        set_stop_gradient(model, True)
        all_models.append(
            {
                "model": model,
                "tokenizer": tokenizer,
                "feature_extraction": feature_extractor,
            }
        )
    return all_models


def set_stop_gradient(model, value):
    for param in model.parameters():
        param.stop_gradient = value


class UPaintingPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using UPainting.
    Wei Li, Xue Xu, et al. "UPainting: Unified Text-to-Image Diffusion Generation with Cross-modal Guidance."
    https://arxiv.org/abs/2210.16031

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`LDMBertModel`]):
            Text-encoder model based on [BERT](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.bert.modeling.html#paddlenlp.transformers.bert.modeling.BertModel) architecture.
        tokenizer (`BertTokenizer`):
            Tokenizer of class
            [BertTokenizer](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.bert.tokenizer.html#paddlenlp.transformers.bert.tokenizer.BertTokenizer).
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`PNDMScheduler`], [`EulerDiscreteScheduler`] or [`EulerAncestralDiscreteScheduler`].
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: LDMBertModel,
        tokenizer: BertTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler],
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.clip_models = []

        # freeze model
        self.freeze_text_encoder()
        self.freeze_unet()
        self.freeze_vae()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.vae_enable_gradient_checkpointing = False
        self.cond_with_uncond_embeddings = True
        self.fp16 = True
        self.amp_level = "O2"

    def add_clip_models(self, model_names_or_model_lists=None):
        if model_names_or_model_lists is not None:
            if isinstance(model_names_or_model_lists[0], str):
                model_list = build_models(model_names_or_model_lists)
            elif isinstance(model_names_or_model_lists[0], dict) and "model" in model_names_or_model_lists[0]:
                model_list = model_names_or_model_lists
            self.clip_models.extend(model_list)
            self.freeze_clip_models()

    def remove_clip_models(self):
        self.clip_models = []
        gc.collect()

    def freeze_clip_models(self):
        for model_dict in self.clip_models:
            set_stop_gradient(model_dict["model"], True)

    def unfreeze_clip_models(self):
        for model_dict in self.clip_models:
            set_stop_gradient(model_dict["model"], False)

    def freeze_text_encoder(self):
        set_stop_gradient(self.text_encoder, True)

    def unfreeze_text_encoder(self):
        set_stop_gradient(self.text_encoder, False)

    def freeze_vae(self):
        set_stop_gradient(self.vae, True)

    def unfreeze_vae(self):
        set_stop_gradient(self.vae, False)

    def freeze_unet(self):
        set_stop_gradient(self.unet, True)

    def unfreeze_unet(self):
        set_stop_gradient(self.unet, False)

    @paddle.no_grad()
    def prepare_clip(self, clip_model, en_frame_prompt, zh_frame_prompt):
        """
        For one CLIP model, prepare embeds, cutouts, etc.
        """
        model = clip_model["model"]
        tokenizer = clip_model["tokenizer"]
        model_stat = {
            "clip_model": model,
            "target_embeds": [],
            "weights": [],
            "resolution": model.config.vision_config.image_size,
            "feature_extraction": clip_model["feature_extraction"],
        }
        frame_prompt = en_frame_prompt if model.config.model_type in ["clip"] else zh_frame_prompt
        if isinstance(frame_prompt, str):
            frame_prompt = [frame_prompt]
        for prompt in frame_prompt:
            prompt, weight = parse_prompt(prompt)
            text_input = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pd",
            )
            txt = model.get_text_features(**text_input)
            model_stat["target_embeds"].append(txt)
            model_stat["weights"].append(weight)

        model_stat["target_embeds"] = paddle.concat(model_stat["target_embeds"])
        model_stat["weights"] = paddle.to_tensor(model_stat["weights"])
        if model_stat["weights"].sum().abs() < 1e-3:
            raise RuntimeError("The weights must not sum to 0.")
        model_stat["weights"] /= model_stat["weights"].sum().abs()
        return model_stat

    def get_clip_guided_losses(
        self, x_in, t, model_stat, cut_overview, cut_innercut, cut_icgray_p, skip_augs, animation_mode, cut_ic_pow
    ):
        """
        Calculate CLIP guided losses
        """
        t_int = max(int(t.item()), 1)
        input_resolution = model_stat["resolution"]
        feature_extraction = model_stat["feature_extraction"]
        cut_overview = eval(cut_overview)
        cut_innercut = eval(cut_innercut)
        cut_icgray_p = eval(cut_icgray_p)
        cuts = MakeCutoutsDango(
            input_resolution,
            skip_augs=skip_augs,
            animation_mode=animation_mode,
            Overview=cut_overview[1000 - t_int],
            InnerCrop=cut_innercut[1000 - t_int],
            IC_Size_Pow=cut_ic_pow,
            IC_Grey_P=cut_icgray_p[1000 - t_int],
        )
        normalize = transforms.Normalize(mean=feature_extraction.image_mean, std=feature_extraction.image_std)
        clip_in = normalize(cuts((x_in.clip(-1, 1) + 1.0) / 2.0))
        with paddle.amp.auto_cast(self.fp16, level=self.amp_level):
            image_embeds = model_stat["clip_model"].get_image_features(clip_in)
        image_embeds = image_embeds.cast(clip_in.dtype)
        dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
        dists = dists.reshape([cut_overview[1000 - t_int] + cut_innercut[1000 - t_int], x_in.shape[0], -1])
        losses = dists.multiply(model_stat["weights"]).sum(2).mean(0)
        return losses

    @paddle.set_grad_enabled(True)
    def cond_fn(
        self,
        latents,
        timestep,
        noise_pred_original,
        model_stats,
        cut_overview,
        cut_innercut,
        cut_icgray_p,
        skip_augs,
        animation_mode,
        cut_ic_pow,
        cutn_batches,
        clip_guidance_scale,
        tv_scale,
        range_scale,
        sat_scale,
        clamp_grad,
        clamp_max,
        cond_weight,
        text_embeddings=None,
        extra_step_kwargs={},
    ):
        # not cond_with_uncond_embeddings
        if self.cond_with_uncond_embeddings:
            noise_pred = noise_pred_original
        else:
            assert text_embeddings is not None
            latents = latents.detach()
            latents.stop_gradient = False
            latent_model_input = self.scheduler.scale_model_input(latents, timestep)

            # step 1: predict the noise residual
            with paddle.amp.auto_cast(self.fp16, level=self.amp_level):
                noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        # step 2: predict the sample
        if isinstance(
            self.scheduler, (PNDMScheduler, DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler)
        ):
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            if isinstance(self.scheduler, (EulerDiscreteScheduler, EulerAncestralDiscreteScheduler)):
                pred_original_sample = self.scheduler.step(
                    noise_pred, timestep, latents, **extra_step_kwargs
                ).pred_original_sample
            else:
                pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
            fac = paddle.sqrt(beta_prod_t)
            sample = pred_original_sample * (fac) + latents * (1 - fac)
        else:
            raise ValueError(f"scheduler type {type(self.scheduler)} not supported")

        # step 3: vae decode
        sample = 1 / 0.18215 * sample
        orig_dtype = sample.dtype
        with paddle.amp.auto_cast(self.fp16, level=self.amp_level):
            if self.vae_enable_gradient_checkpointing:

                def create_custom_forward(module, return_dict=False):
                    def custom_forward(*inputs):
                        return module.decode(*inputs, return_dict=return_dict)[0]

                    return custom_forward

                image = recompute(create_custom_forward(self.vae), sample)
            else:
                image = self.vae.decode(sample).sample
        image = image.cast(orig_dtype)

        def grads_wrt_x(x, x_in):
            # 1. init x_in_grad
            x_in_grad = paddle.zeros_like(x_in)

            # 2. compute clip x_in_grad
            for model_stat in model_stats:
                for _ in range(cutn_batches):
                    losses = self.get_clip_guided_losses(
                        x_in,
                        timestep,
                        model_stat,
                        cut_overview,
                        cut_innercut,
                        cut_icgray_p,
                        skip_augs,
                        animation_mode,
                        cut_ic_pow,
                    )
                    x_in_grad += paddle.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches

            # 3. compute tv_loss + range_loss + sat_losses
            tv_losses = tv_loss(x_in).sum() if tv_scale > 0 else 0.0
            range_losses = range_loss(x_in).sum() if range_scale > 0 else 0.0
            sat_losses = sat_loss(x_in).sum() if sat_scale > 0 else 0.0
            loss = tv_losses * tv_scale + range_losses * range_scale + sat_losses * sat_scale

            x_in_grad += paddle.grad(loss, x_in)[0]

            isnan = paddle.isnan(x_in_grad).any()
            if not isnan:
                grads = -paddle.grad(x_in, x, x_in_grad)[0]
                isinf = paddle.isinf(grads).any()
                isnan = paddle.isnan(grads).any()
                if isinf or isnan:
                    grads = paddle.zeros_like(x)
            else:
                grads = paddle.zeros_like(x)

            # 4. clip grad
            if clamp_grad and not isnan:
                magnitude = grads.square().mean().sqrt()
                return grads * magnitude.clip(max=clamp_max) / magnitude
            return grads

        # step 4: compute grads score
        grads = grads_wrt_x(latents, image)

        # step 5: update noise_pred
        noise_pred = noise_pred_original - cond_weight * beta_prod_t.sqrt() * grads
        return noise_pred

    def _encode_prompt(self, prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        config = (
            self.text_encoder.config
            if isinstance(self.text_encoder.config, dict)
            else self.text_encoder.config.to_dict()
        )
        if config.get("use_attention_mask", None) is not None and config["use_attention_mask"]:
            return_attention_mask = True
        else:
            return_attention_mask = False

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pd",
            return_attention_mask=return_attention_mask,
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pd").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not paddle.equal_all(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because LDMBert can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
        if return_attention_mask:
            attention_mask = text_inputs.attention_mask
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(text_input_ids, attention_mask=attention_mask)[0]

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
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

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pd",
                return_attention_mask=return_attention_mask,
            )
            if return_attention_mask:
                attention_mask = text_inputs.attention_mask
            else:
                attention_mask = None
            uncond_embeddings = self.text_encoder(uncond_input.input_ids, attention_mask=attention_mask)[0]
            text_embeddings = paddle.concat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
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

    def check_inputs(self, prompt, height, width, callback_steps, en_prompt, use_clip_guidance):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if use_clip_guidance:
            if any(model["model"].config.model_type in ["clip"] for model in self.clip_models):
                if not isinstance(en_prompt, str) and not isinstance(en_prompt, list):
                    raise ValueError("`en_prompt` has to be of type `str` or `list` when we use but english clip.")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
        shape = [batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor]
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            if isinstance(generator, list):
                shape = [
                    1,
                ] + shape[1:]
                latents = [paddle.randn(shape, generator=generator[i], dtype=dtype) for i in range(batch_size)]
                latents = paddle.concat(latents, axis=0)
            else:
                latents = paddle.randn(shape, generator=generator, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def denoising_step(self, latents, t, text_embeddings, do_classifier_free_guidance, guidance_scale):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        with paddle.amp.auto_cast(self.fp16, level=self.amp_level):
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform classifier free guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred

    def denoising_step_twice(self, latents, t, text_embeddings, do_classifier_free_guidance, guidance_scale):
        latent_model_input = self.scheduler.scale_model_input(latents, t)

        if do_classifier_free_guidance:
            text_embeddings_for_guidance, text_embeddings_uncond = text_embeddings.chunk(2)
        else:
            text_embeddings_for_guidance = text_embeddings
            text_embeddings_uncond = None

        with paddle.amp.auto_cast(self.fp16, level=self.amp_level):
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings_for_guidance).sample

        if do_classifier_free_guidance:
            with paddle.amp.auto_cast(self.fp16, level=self.amp_level):
                noise_pred_uncond = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings_uncond
                ).sample
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
        return noise_pred

    @paddle.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        # upainting kwargs
        use_clip_guidance: bool = True,
        clip_guidance_skip_steps: Optional[int] = 10,
        en_prompt: Optional[Union[str, List[str]]] = None,
        cut_overview: Optional[str] = "[12]*400+[4]*600",
        cut_innercut: Optional[str] = "[4]*400+[12]*600",
        cut_icgray_p: Optional[str] = "[0.2]*400+[0]*600",
        skip_augs: Optional[bool] = False,
        animation_mode: Optional[str] = "None",
        cut_ic_pow: Optional[float] = 1.0,
        cutn_batches: Optional[int] = 4,
        clip_guidance_scale: Optional[float] = 20000.0,
        tv_scale: Optional[float] = 0.0,
        range_scale: Optional[float] = 150.0,
        sat_scale: Optional[float] = 0.0,
        clamp_grad: Optional[bool] = True,
        clamp_max: Optional[float] = 0.05,
        cond_weight: Optional[float] = 10.0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 256:
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 256:
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 1.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`paddle.Generator`, *optional*):
                One or a list of paddle generator(s) to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: paddle.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            use_clip_guidance (`bool`, *optional*, defaults to `True`):
                Whether or not to use clip guidance like DiscoDiffusion Model.
            clip_guidance_skip_steps (`int`, *optional*, defaults to 10):
                The number of skip steps without clip guidance.
            en_prompt (`str`, *optional*, defaults to `None`):
                The english prompt or english prompts to guide the image generation when we use English clip.
            cut_overview (`str`, *optional*, defaults to `[12]*400+[4]*600`):
                The schedule of overview cuts, which take a snapshot of the entire image and evaluate that
                against the prompt.
            cut_innercut (`str`, *optional*, defaults to `[4]*400+[12]*600`):
                The schedule of inner cuts, which are smaller cropped images from the interior of the image, helpful
                in tuning fine details.
            cut_icgray_p (`str`, *optional*, defaults to `[0.2]*400+[0]*600`):
                A portion of the cuts can be set to be grayscale instead of color.
                This may help with improved definition of shapes and edges, especially in the early diffusion steps.
            skip_augs (`bool`, *optional*, defaults to `False`):
                Whether or not to use paddle.vision argumentations. These augmentations are intended to help improve
                image quality, but can have a 'smoothing' effect on edges that you may not want.
            animation_mode (`str`, *optional*, defaults to `None`):
                The animation_mode from ['None', '2D', '3D', 'Video Input']. Currently we only support 'None'.
            cut_ic_pow (`float`, *optional*, defaults to 1.0):
                This value is in the range of [0.5~100.0]. This sets the size of the border used for inner cuts. High cut_ic_pow values have larger borders.
            cutn_batches (`int`, *optional*, defaults to 4):
                This value is in the range of [1~8]. The number of batches when evaluating prompt x 'cuts'.
                Increasing cutn_batches will increase render times.
            clip_guidance_scale (`float`, *optional*, defaults to 20000.0):
                This value in in the range of [1500.0~100000.0].
                It tells UPainting how strongly you want CLIP to move toward your prompt each timestep.
            tv_scale (`float`, *optional*, defaults to 0.0):
                This value in in the range of [0.0~1000.0].
                Total variance denoising. Optional, set to zero to turn off.
                Controls 'smoothness' of final output.
                If used, tv_scale will try to smooth out your final image to reduce overall noise.
            range_scale (`float`, *optional*, defaults to 150.0):
                This value in in the range of [0.0~1000.0].
                Set to zero to turn off.  Used for adjustment of color contrast.
                Lower range_scale will increase contrast. Very low numbers create a reduced color palette,
                resulting in more vibrant or poster-like images.
                Higher range_scale will reduce contrast, for more muted images.
            sat_scale (`float`, *optional*, defaults to 0.0):
                This value in in the range of [0.0~20000.0].
                Saturation scale. Optional, set to zero to turn off.
                If used, sat_scale will help mitigate oversaturation.
                If your image is too saturated, increase sat_scale to reduce the saturation.
            clamp_grad (`bool`, *optional*, defaults to `True`):
                Whether or not to clamp grads.
            clamp_max (`float`, *optional*, defaults to 0.05):
                The value of the clamp_grad limitation.
            cond_weight (`float`, *optional*, defaults to 0.05):
                The conditional weight for clip guidance. Optional, set to zero to turn off.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipeline_utils.ImagePipelineOutput `] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
        # 0. Check use_clip_guidance.
        if len(self.clip_models) == 0 or cond_weight <= 0:
            use_clip_guidance = False

        if use_clip_guidance:
            self.unet.enable_gradient_checkpointing()
            self.vae_enable_gradient_checkpointing = True
        else:
            self.unet.disable_gradient_checkpointing()
            self.vae_enable_gradient_checkpointing = False

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps, en_prompt, use_clip_guidance)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, do_classifier_free_guidance, negative_prompt)

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
            text_embeddings.dtype,
            generator,
            latents,
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6. Prepare clip_model
        if use_clip_guidance:
            model_stats = []
            for clip_model in self.clip_models:
                model_stat = self.prepare_clip(clip_model, en_frame_prompt=en_prompt, zh_frame_prompt=prompt)
                model_stats.append(model_stat)

        # 7. Do denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                with_cond = use_clip_guidance and (i + 1) > (clip_guidance_skip_steps + num_warmup_steps)

                if with_cond and self.cond_with_uncond_embeddings:
                    latents = latents.detach()
                    latents.stop_gradient = False

                # perform denoising_step
                if self.cond_with_uncond_embeddings:
                    # [0 ~ clip_guidance_skip_steps - 1] no grad, [clip_guidance_skip_steps, num_inference_steps - 1] with grad
                    with paddle.set_grad_enabled(with_cond):
                        if with_cond:
                            # do denoising_step_twice to save gpu memory
                            noise_pred = self.denoising_step_twice(
                                latents, t, text_embeddings, do_classifier_free_guidance, guidance_scale
                            )
                        else:
                            noise_pred = self.denoising_step(
                                latents, t, text_embeddings, do_classifier_free_guidance, guidance_scale
                            )
                else:
                    # no grad, we will do noise_pred without_uncond_embeddings in cond_fn
                    noise_pred = self.denoising_step(
                        latents, t, text_embeddings, do_classifier_free_guidance, guidance_scale
                    )

                # perform clip guidance
                if with_cond:
                    if self.cond_with_uncond_embeddings:
                        text_embeddings_for_guidance = None
                    else:
                        text_embeddings_for_guidance = (
                            text_embeddings.chunk(2)[1] if do_classifier_free_guidance else text_embeddings
                        )
                    noise_pred = self.cond_fn(
                        latents,
                        t,
                        noise_pred,
                        model_stats,
                        cut_overview=cut_overview,
                        cut_innercut=cut_innercut,
                        cut_icgray_p=cut_icgray_p,
                        skip_augs=skip_augs,
                        animation_mode=animation_mode,
                        cut_ic_pow=cut_ic_pow,
                        cutn_batches=cutn_batches,
                        clip_guidance_scale=clip_guidance_scale,
                        tv_scale=tv_scale,
                        range_scale=range_scale,
                        sat_scale=sat_scale,
                        clamp_grad=clamp_grad,
                        clamp_max=clamp_max,
                        cond_weight=cond_weight,
                        text_embeddings=text_embeddings_for_guidance,
                        extra_step_kwargs=extra_step_kwargs,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
