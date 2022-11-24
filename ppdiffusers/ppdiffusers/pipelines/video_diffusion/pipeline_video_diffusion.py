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
from typing import Callable, List, Optional, Tuple, Union
import os

import paddle
import paddle.nn as nn
from paddlenlp.transformers import PretrainedModel, PretrainedTokenizer
from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLVid, UNet3DModel
from ...pipeline_utils import DiffusionPipeline, VideoPipelineOutput
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from ...utils import deprecate, logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VideoDiffusionPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vae ([`AutoencoderKLVid`]):
            Autoencoder Model to encode and decode videos to and from latent representations.
        bert ([`paddlenlp.transformers.BertModel`]):
            Text-encoder model based on [BERT](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.bert.modeling.html#paddlenlp.transformers.bert.modeling.BertModel) architecture.
        tokenizer (`paddlenlp.transformers.BertTokenizer`):
            Tokenizer of class
            [BertTokenizer](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.bert.tokenizer.html#paddlenlp.transformers.bert.tokenizer.BertTokenizer).
        unet ([`UNet3DModel`]): 3D conditional U-Net architecture to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: Union[AutoencoderKLVid],
        bert: PretrainedModel,
        tokenizer: PretrainedTokenizer,
        unet: Union[UNet3DModel],
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
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

        self.register_modules(vae=vae,
                              bert=bert,
                              tokenizer=tokenizer,
                              unet=unet,
                              scheduler=scheduler)

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
        prompt: Union[str, List[str]],
        num_frames: Optional[int] = 8,
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        std: Optional[Union[int, paddle.Tensor]] = 1,
        mean: Optional[Union[int, paddle.Tensor]] = 0,
        num_inference_steps: Optional[int] = 50,
        eta: Optional[float] = 0.0,
        seed: Optional[int] = None,
        latents: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ) -> Union[Tuple, VideoPipelineOutput]:
        r"""
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 256):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 256):
                The width in pixels of the generated image.
            std (`paddle.Tensor`, *optional*, defaults to 1):
                The std value during the calculation of latents decoding into pictures.
            mean (`paddle.Tensor`, *optional*, defaults to 0):
                The mean value during the calculation of latents decoding into pictures.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
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
                Whether or not to return a [`~pipeline_utils.VideoPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: paddle.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
                
        Returns:
            [`~pipeline_utils.VideoPipelineOutput`] or `tuple`: [`~pipeline_utils.VideoPipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
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

        text_embeddings = self.bert_embed(self.tokenize(prompt))

        # get the initial random noise unless the user supplied it
        latents_shape = [
            batch_size, self.unet.channels, num_frames, height // 8, width // 8
        ]  # (batch_size, C, N, H, W)

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
            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t,
                                   text_embeddings).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents,
                                          **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        latents = latents * std + mean

        sampled_videos = self.vae.decode(latents).sample

        videos_frames = []
        for idx in range(sampled_videos.shape[0]):
            video = sampled_videos[idx]
            video_frames = []
            for fidx in range(video.shape[1]):
                frame = video[:, fidx]
                frame = (frame / 2 + 0.5).clip(0, 1)
                frame = frame.transpose([1, 2, 0]).astype("float32").numpy()
                if output_type == "pil":
                    frame = self.numpy_to_pil(frame)
                video_frames.append(frame)
            videos_frames.append(video_frames)

        return VideoPipelineOutput(videos=videos_frames)

    def tokenize(self, texts, add_special_tokens=True):
        if not isinstance(texts, (list, tuple)):
            texts = [texts]  # ['aerial vire of lake and barren mountains']

        encoding = self.tokenizer.batch_encode(
            texts,
            add_special_tokens=add_special_tokens,
            padding=True,
            return_tensors='pd')

        token_ids = encoding.input_ids
        return token_ids

    @paddle.no_grad()
    def bert_embed(self, token_ids, return_cls_repr=False, eps=1e-8, pad_id=0.):
        mask = token_ids != pad_id

        outputs = self.bert(input_ids=token_ids,
                            attention_mask=mask,
                            output_hidden_states=True,
                            return_dict=True)

        hidden_state = outputs.hidden_states[-1]
        if return_cls_repr:
            return hidden_state[:, 0]

        mask = mask[:, 1:]
        mask = paddle.reshape(mask, [mask.shape[0], mask.shape[1], 1])

        numer = (hidden_state[:, 1:] * mask).sum(axis=1)
        denom = mask.sum(axis=1)
        masked_mean = numer / (denom + eps)
        return masked_mean
