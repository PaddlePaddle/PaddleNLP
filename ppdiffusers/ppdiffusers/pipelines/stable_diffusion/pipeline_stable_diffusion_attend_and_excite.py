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
import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn import functional as F

from paddlenlp.transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from ...models import AutoencoderKL, UNet2DConditionModel
from ...models.cross_attention import CrossAttention
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import logging, randn_tensor, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import paddle
        >>> from ppdiffusers import StableDiffusionAttendAndExcitePipeline

        >>> pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
        ...     "CompVis/stable-diffusion-v1-4", paddle_dtype=paddle.float16
        ... )


        >>> prompt = "a cat and a frog"

        >>> # use get_indices function to find out indices of the tokens you want to alter
        >>> pipe.get_indices(prompt)
        {0: '<|startoftext|>', 1: 'a</w>', 2: 'cat</w>', 3: 'and</w>', 4: 'a</w>', 5: 'frog</w>', 6: '<|endoftext|>'}

        >>> token_indices = [2, 5]
        >>> seed = 6141
        >>> generator = paddle.Generator().manual_seed(seed)

        >>> images = pipe(
        ...     prompt=prompt,
        ...     token_indices=token_indices,
        ...     guidance_scale=7.5,
        ...     generator=generator,
        ...     num_inference_steps=50,
        ...     max_iter_to_alter=25,
        ... ).images

        >>> image = images[0]
        >>> image.save(f"../images/{prompt}_{seed}.png")
        ```
"""


class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down": [], "mid": [], "up": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0 and is_cross:
            if attn.shape[1] == self.attn_res**2:
                self.step_store[place_in_unet].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def aggregate_attention(self, from_where: List[str]) -> paddle.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[location]:
                cross_maps = item.reshape([-1, self.attn_res, self.attn_res, item.shape[-1]])
                out.append(cross_maps)
        out = paddle.concat(out, axis=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res=16):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res


class AttendExciteCrossAttnProcessor:
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # only need to store attention maps during the Attend and Excite process
        if not attention_probs.stop_gradient:
            # TODO must flatten （0, 1)
            # [bs, num_heads, q_len, k_len] -> [bs*num_heads, q_len, k_len]
            self.attnstore(attention_probs.flatten(0, 1), is_cross, self.place_in_unet)

        hidden_states = paddle.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class StableDiffusionAttendAndExcitePipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion and Attend and Excite.

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
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
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

    # Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
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

    # Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pd")
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.cast(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    # Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.transpose([0, 2, 3, 1]).cast("float32").numpy()
        return image

    # Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
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
        indices,
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

        if (indices is None) or (indices is not None and not isinstance(indices, List)):
            raise ValueError(f"`indices` has to be a list but is {type(indices)}")

    # Copied from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
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

    @staticmethod
    def _compute_max_attention_per_index(
        attention_maps: paddle.Tensor,
        indices: List[int],
    ) -> List[paddle.Tensor]:
        """Computes the maximum attention value for each of the tokens we wish to alter."""
        attention_for_text = attention_maps[:, :, 1:-1]
        attention_for_text *= 100
        attention_for_text = F.softmax(attention_for_text, axis=-1)

        # Shift indices since we removed the first token
        indices = [index - 1 for index in indices]

        # Extract the maximum values
        max_indices_list = []
        for i in indices:
            image = attention_for_text[:, :, i]
            smoothing = GaussianSmoothing()
            input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
            image = smoothing(input).squeeze(0).squeeze(0)
            # paddle.max donot support float16
            max_indices_list.append(image.max())
        return max_indices_list

    def _aggregate_and_get_max_attention_per_token(
        self,
        indices: List[int],
    ):
        """Aggregates the attention for each token and computes the max activation value for each token to alter."""
        attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"),
        )
        max_attention_per_index = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices=indices,
        )
        return max_attention_per_index

    @staticmethod
    def _compute_loss(max_attention_per_index: List[paddle.Tensor]) -> paddle.Tensor:
        """Computes the attend-and-excite loss using the maximum attention value for each token."""
        losses = [max(0, 1.0 - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        return loss

    @staticmethod
    def _update_latent(latents: paddle.Tensor, loss: paddle.Tensor, step_size: float) -> paddle.Tensor:
        """Update the latent according to the computed loss."""
        loss.stop_gradient = False
        grad_cond = paddle.autograd.grad(loss, [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step(
        self,
        latents: paddle.Tensor,
        indices: List[int],
        loss: paddle.Tensor,
        threshold: float,
        text_embeddings: paddle.Tensor,
        step_size: float,
        t: int,
        max_refinement_steps: int = 20,
    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent code
        according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_loss = max(0, 1.0 - threshold)
        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach()
            latents.stop_gradient = False
            self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
            self.unet.clear_gradients()

            # Get max activation value for each subject token
            max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                indices=indices,
            )

            loss = self._compute_loss(max_attention_per_index)

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            logger.info(f"\t Try {iteration}. loss: {loss}")

            if iteration >= max_refinement_steps:
                logger.info(f"\t Exceeded max number of iterations ({max_refinement_steps})! ")
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach()
        latents.stop_gradient = False

        _ = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
        self.unet.clear_gradients()

        # Get max activation value for each subject token
        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
            indices=indices,
        )
        loss = self._compute_loss(max_attention_per_index)
        logger.info(f"\t Finished with loss of: {loss}")
        return loss, latents, max_attention_per_index

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteCrossAttnProcessor(
                attnstore=self.attention_store, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    def get_indices(self, prompt: str) -> Dict[str, int]:
        """Utility function to list the indices of the tokens you wish to alte"""
        ids = self.tokenizer(prompt).input_ids
        indices = {i: tok for tok, i in zip(self.tokenizer.convert_ids_to_tokens(ids), range(len(ids)))}
        return indices

    @paddle.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        token_indices: List[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
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
        max_iter_to_alter: int = 25,
        thresholds: dict = {0: 0.05, 10: 0.5, 20: 0.8},
        scale_factor: int = 20,
        attn_res: int = 16,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            token_indices (`List[int]`):
                The token indices to alter with attend-and-excite.
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
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            max_iter_to_alter (`int`, *optional*, defaults to `25`):
                Number of denoising steps to apply attend-and-excite. The first <max_iter_to_alter> denoising steps are
                where the attend-and-excite is applied. I.e. if `max_iter_to_alter` is 25 and there are a total of `30`
                denoising steps, the first 25 denoising steps will apply attend-and-excite and the last 5 will not
                apply attend-and-excite.
            thresholds (`dict`, *optional*, defaults to `{0: 0.05, 10: 0.5, 20: 0.8}`):
                Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in.
            scale_factor (`int`, *optional*, default to 20):
                Scale factor that controls the step size of each Attend and Excite update.
            attn_res (`int`, *optional*, default to 16):
                The resolution of most semantic attention map.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`. :type attention_store: object
        """

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            token_indices,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
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

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self.attention_store = AttentionStore(attn_res=attn_res)
        self.register_attention_control()

        # default config for step size from original repo
        scale_range = np.linspace(1.0, 0.5, len(self.scheduler.timesteps))
        step_size = scale_factor * np.sqrt(scale_range)

        text_embeddings = (
            prompt_embeds[batch_size * num_images_per_prompt :] if do_classifier_free_guidance else prompt_embeds
        )

        if isinstance(token_indices[0], int):
            token_indices = [token_indices]
        indices = []
        for ind in token_indices:
            indices = indices + [ind] * num_images_per_prompt

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Attend and excite process
                with paddle.set_grad_enabled(True):
                    latents = latents.clone().detach()
                    latents.stop_gradient = False
                    updated_latents = []
                    for latent, index, text_embedding in zip(latents, indices, text_embeddings):
                        # Forward pass of denoising with text conditioning
                        latent = latent.unsqueeze(0)
                        text_embedding = text_embedding.unsqueeze(0)

                        self.unet(
                            latent,
                            t,
                            encoder_hidden_states=text_embedding,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample
                        self.unet.clear_gradients()

                        # Get max activation value for each subject token
                        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                            indices=index,
                        )

                        loss = self._compute_loss(max_attention_per_index=max_attention_per_index)

                        # If this is an iterative refinement step, verify we have reached the desired threshold for all
                        if i in thresholds.keys() and loss > 1.0 - thresholds[i]:
                            loss, latent, max_attention_per_index = self._perform_iterative_refinement_step(
                                latents=latent,
                                indices=index,
                                loss=loss,
                                threshold=thresholds[i],
                                text_embeddings=text_embedding,
                                step_size=step_size[i],
                                t=t,
                            )

                        # Perform gradient update
                        if i < max_iter_to_alter:
                            if loss != 0:
                                latent = self._update_latent(
                                    latents=latent,
                                    loss=loss,
                                    step_size=step_size[i],
                                )
                            logger.info(f"Iteration {i} | Loss: {loss.item():0.4f}")

                        updated_latents.append(latent)

                    latents = paddle.concat(updated_latents, axis=0)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
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

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, prompt_embeds.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


class GaussianSmoothing(nn.Layer):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed separately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = paddle.meshgrid([paddle.arange(size, dtype=paddle.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * paddle.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / paddle.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.reshape([1, 1, *kernel.shape])
        kernel = kernel.tile([channels, *[1] * (kernel.ndim - 1)])

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (paddle.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (paddle.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.cast(input.dtype), groups=self.groups)
