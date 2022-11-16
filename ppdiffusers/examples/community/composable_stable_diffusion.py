# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import inspect
from typing import Callable, List, Optional, Union

import paddle

from ppdiffusers.configuration_utils import FrozenDict
from ppdiffusers.models import AutoencoderKL, UNet2DConditionModel
from ppdiffusers.pipeline_utils import DiffusionPipeline
from ppdiffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from ppdiffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from ppdiffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from ppdiffusers.utils import deprecate, logging
from paddlenlp.transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ComposableStableDiffusionPipeline(DiffusionPipeline):
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
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/junnyu/stable-diffusion-v1-4-paddle) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        if hasattr(scheduler.config,
                   "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file")
            deprecate("steps_offset!=1",
                      "1.0.0",
                      deprecation_message,
                      standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None:
            logger.warn(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. PaddleNLP team, diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
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

    def enable_attention_slicing(self,
                                 slice_size: Optional[Union[str,
                                                            int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    @paddle.no_grad()
    def __call__(
        self,
        prompt: str,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: str = None,
        # num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        seed: Optional[int] = None,
        latents: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        weights: Optional[str] = "",
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reduce_memory: Optional[bool] = True,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        
        Args:
            prompt (`str``):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
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
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            seed (`int`, *optional*):
                Random number seed.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `seed`.
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
            reduce_memory (`bool`, *optional*, defaults to True):
                Whether or not reduce_memory when unet forward.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if isinstance(prompt, str):
            batch_size = 1
        else:
            raise ValueError(
                f"`prompt` has to be of type `str`but is {type(prompt)}")
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise ValueError(
                f"`negative_prompt` has to be of type `str`but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (callback_steps is not None and
                                        (not isinstance(callback_steps, int)
                                         or callback_steps <= 0)):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}.")

        if "|" in prompt:
            prompt = [x.strip() for x in prompt.split("|")]
            print(f"composing {prompt}...")

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pd",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(
                text_input_ids[:, self.tokenizer.model_max_length:])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}")
            text_input_ids = text_input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = paddle.ones_like(text_input_ids)
        text_embeddings = self.text_encoder(text_input_ids,
                                            attention_mask=attention_mask)[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        # bs_embed, seq_len, _ = text_embeddings.shape
        # text_embeddings = text_embeddings.tile([1, num_images_per_prompt, 1])
        # text_embeddings = text_embeddings.reshape(
        #     [bs_embed * num_images_per_prompt, seq_len, -1])

        if not weights:
            # specify weights for prompts (excluding the unconditional score)
            print("using equal weights for all prompts...")
            pos_weights = paddle.to_tensor(
                [1 / (text_embeddings.shape[0] - 1)] *
                (text_embeddings.shape[0] - 1)).reshape([-1, 1, 1, 1])
            neg_weights = paddle.to_tensor([1.0]).reshape([-1, 1, 1, 1])
            mask = paddle.to_tensor([False] + [True] * pos_weights.shape[0],
                                    dtype=paddle.bool)
        else:
            # set prompt weight for each
            num_prompts = len(prompt) if isinstance(prompt, list) else 1
            weights = [float(w.strip()) for w in weights.split("|")]
            if len(weights) < num_prompts:
                weights.append(1.0)
            assert len(weights) == text_embeddings.shape[
                0], "weights specified are not equal to the number of prompts"
            pos_weights = []
            neg_weights = []
            mask = []  # first one is unconditional score
            for w in weights:
                if w > 0:
                    pos_weights.append(w)
                    mask.append(True)
                else:
                    neg_weights.append(abs(w))
                    mask.append(False)
            # normalize the weights
            pos_weights = paddle.to_tensor(pos_weights).reshape([-1, 1, 1, 1])
            pos_weights = pos_weights / pos_weights.sum()
            if neg_weights:
                neg_weights = paddle.to_tensor(neg_weights).reshape(
                    [-1, 1, 1, 1])
                neg_weights = neg_weights / neg_weights.sum()
            mask = paddle.to_tensor(mask, dtype=paddle.bool)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            if paddle.all(mask):
                uncond_tokens: str
                if negative_prompt is None:
                    uncond_tokens = ""
                else:
                    uncond_tokens = negative_prompt

                max_length = text_input_ids.shape[-1]
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pd",
                )
                attention_mask = paddle.ones_like(uncond_input.input_ids)
                uncond_embeddings = self.text_encoder(
                    uncond_input.input_ids, attention_mask=attention_mask)[0]

                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                # seq_len = uncond_embeddings.shape[1]
                # uncond_embeddings = uncond_embeddings.tile(
                #     [batch_size, num_images_per_prompt, 1])
                # uncond_embeddings = uncond_embeddings.reshape(
                #     [batch_size * num_images_per_prompt, seq_len, -1])

                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                text_embeddings = paddle.concat(
                    [uncond_embeddings, text_embeddings])

                # update negative weights
                neg_weights = paddle.to_tensor([1.0]).reshape([-1, 1, 1, 1])
                mask = paddle.to_tensor([False] + mask.tolist(),
                                        dtype=paddle.bool)

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = [
            batch_size, self.unet.in_channels, height // 8, width // 8
        ]
        if latents is None:
            if seed is not None:
                paddle.seed(seed)
            latents = paddle.randn(latents_shape, dtype=text_embeddings.dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = paddle.concat(
                [latents] * text_embeddings.shape[0]
            ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            if reduce_memory:
                # reduce memory by predicting each score sequentially
                noise_preds = []
                # predict the noise residual
                for latent_in, text_embedding_in in zip(
                        latent_model_input.chunk(latent_model_input.shape[0],
                                                 axis=0),
                        text_embeddings.chunk(text_embeddings.shape[0], axis=0),
                ):
                    noise_preds.append(
                        self.unet(
                            latent_in,
                            t,
                            encoder_hidden_states=text_embedding_in).sample)
                noise_preds = paddle.concat(noise_preds, axis=0)
            else:
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond = (noise_preds[~mask] * neg_weights).sum(
                    axis=0, keepdim=True)
                noise_pred_text = (noise_preds[mask] * pos_weights).sum(
                    axis=0, keepdim=True)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents,
                                          **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clip(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.transpose([0, 2, 3, 1]).astype("float32").numpy()

        # run safety checker
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="pd")
            image, has_nsfw_concept = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.astype(
                    text_embeddings.dtype))
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept)
