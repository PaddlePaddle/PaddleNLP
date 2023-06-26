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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlenlp.transformers import (
    ClapTextModelWithProjection,
    RobertaTokenizer,
    SpeechT5HifiGan,
)

from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import logging, randn_tensor, replace_example_docstring
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline

logger = logging.get_logger(__name__)
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import paddle
        >>> from ppdiffusers import AudioLDMPipeline

        >>> pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm", paddle_dtype=paddle.float16)

        >>> prompt = "A hammer hitting a wooden surface"
        >>> audio = pipe(prompt).audio[0]
        ```
"""


class AudioLDMPipeline(DiffusionPipeline):
    """
    Pipeline for text-to-audio generation using AudioLDM.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode audios to and from latent representations.
        text_encoder ([`ClapTextModelWithProjection`]):
            Frozen text-encoder. AudioLDM uses the text portion of CLAP,
            specifically the RoBERTa HSTAT-unfused variant.
        tokenizer ([`PreTrainedTokenizer`]):
            Tokenizer of class RobertaTokenizer.
        unet ([`UNet2DConditionModel`]): U-Net architecture to denoise the encoded audio latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded audio latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        vocoder ([`SpeechT5HifiGan`]):
            Vocoder of class SpeechT5HifiGan.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: ClapTextModelWithProjection,
        tokenizer: RobertaTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vocoder: SpeechT5HifiGan,
    ):
        super().__init__()
        self.register_modules(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, vocoder=vocoder
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def _encode_prompt(
        self,
        prompt,
        num_waveforms_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
    ):
        """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_waveforms_per_prompt (`int`):
                number of waveforms that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if self.text_encoder.text_model.embeddings.token_type_ids.dtype not in [
            paddle.int16,
            paddle.int32,
            paddle.int64,
        ]:
            self.text_encoder.text_model.embeddings.token_type_ids = (
                self.text_encoder.text_model.embeddings.token_type_ids.cast("int32")
            )

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
                return_attention_mask=True,
                truncation=True,
                return_tensors="pd",
            )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pd", return_attention_mask=True
            ).input_ids
            if (
                untruncated_ids.shape[-1] >= text_input_ids.shape[-1]
                and not paddle.equal_all(x=text_input_ids, y=untruncated_ids).item()
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    f"The following part of your input was truncated because CLAP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
            prompt_embeds = self.text_encoder(text_input_ids.cast("int32"), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds.text_embeds
            # additional L_2 normalization over each hidden-state
            prompt_embeds = F.normalize(x=prompt_embeds, axis=-1)
        prompt_embeds = prompt_embeds.cast(self.text_encoder.dtype)
        bs_embed, seq_len = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile(repeat_times=[1, num_waveforms_per_prompt])
        prompt_embeds = prompt_embeds.reshape([bs_embed * num_waveforms_per_prompt, seq_len])

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`."
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
                return_attention_mask=True,
            )
            uncond_input_ids = uncond_input.input_ids
            attention_mask = uncond_input.attention_mask
            negative_prompt_embeds = self.text_encoder(uncond_input_ids.cast("int32"), attention_mask=attention_mask)
            negative_prompt_embeds = negative_prompt_embeds.text_embeds
            # additional L_2 normalization over each hidden-state
            negative_prompt_embeds = F.normalize(x=negative_prompt_embeds, axis=-1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.cast(self.text_encoder.dtype)

            negative_prompt_embeds = negative_prompt_embeds.tile(repeat_times=[1, num_waveforms_per_prompt])
            negative_prompt_embeds = negative_prompt_embeds.reshape([batch_size * num_waveforms_per_prompt, seq_len])

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = paddle.concat(x=[negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram = self.vae.decode(latents).sample
        return mel_spectrogram

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(axis=1)
        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.astype(dtype="float32").cpu()
        return waveform

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
        audio_length_in_s,
        vocoder_upsample_factor,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        min_audio_length_in_s = vocoder_upsample_factor * self.vae_scale_factor
        if audio_length_in_s < min_audio_length_in_s:
            raise ValueError(
                f"`audio_length_in_s` has to be a positive value greater than or equal to {min_audio_length_in_s}, but is {audio_length_in_s}."
            )
        if self.vocoder.config.model_in_dim % self.vae_scale_factor != 0:
            raise ValueError(
                f"The number of frequency bins in the vocoder's log-mel spectrogram has to be divisible by the VAE scale factor, but got {self.vocoder.config.model_in_dim} bins and a scale factor of {self.vae_scale_factor}."
            )
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

    def prepare_latents(self, batch_size, num_channels_latents, height, dtype, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            self.vocoder.config.model_in_dim // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)
        else:
            latents = latents

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @paddle.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 2.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = "np",
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the audio generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            audio_length_in_s (`int`, *optional*, defaults to 5.12):
                The length of the generated audio sample in seconds.
            num_inference_steps (`int`, *optional*, defaults to 10):
                The number of denoising steps. More denoising steps usually lead to a higher quality audio at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 2.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate audios that are closely linked to the text `prompt`,
                usually at the expense of lower sound quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_waveforms_per_prompt (`int`, *optional*, defaults to 1):
                The number of waveforms to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
                One or a list of paddle generator(s)
                to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for audio
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
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
                `self.processor` in ppdiffusers.cross_attention.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generate image. Choose between:
                - `"np"`: Return Numpy `np.ndarray` objects.
                - `"pt"`: Return `paddle.Tensor` objects.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated audios.
        """
        # 0. Convert audio input length from seconds to spectrogram height
        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate
        if audio_length_in_s is None:
            audio_length_in_s = self.unet.config.sample_size * self.vae_scale_factor * vocoder_upsample_factor
        height = int(audio_length_in_s / vocoder_upsample_factor)
        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} so that it can be handled by the model. It will be cut to {audio_length_in_s} after the denoising process."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            audio_length_in_s,
            vocoder_upsample_factor,
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
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_latents,
            height,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat(x=[latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=None,
                    class_labels=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(chunks=2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        mel_spectrogram = self.decode_latents(latents)

        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)

        audio = audio[:, :original_waveform_length]

        if output_type == "np":
            audio = audio.numpy()

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)
