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
"""
    modeled after the textual_inversion.py / train_dreambooth.py and the work
    of justinpinkney here: https://github.com/justinpinkney/stable-diffusion/blob/main/notebooks/imagic.ipynb
"""
import inspect
from typing import List, Optional, Union

import numpy as np
import paddle
import paddle.nn.functional as F

import PIL
from ppdiffusers.configuration_utils import FrozenDict
from ppdiffusers.models import AutoencoderKL, UNet2DConditionModel
from ppdiffusers.pipeline_utils import DiffusionPipeline
from ppdiffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from ppdiffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from ppdiffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from ppdiffusers.utils import deprecate, logging

from tqdm.auto import tqdm
from paddlenlp.transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from paddle.optimizer import Adam
from paddlenlp.utils.tools import compare_version
if compare_version(PIL.__version__, "9.1.0") >= 0:
    Resampling = PIL.Image.Resampling
else:
    Resampling = PIL.Image

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def set_stop_gradient(model, value):
    for param in model.parameters():
        param.stop_gradient = value


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = paddle.to_tensor(image)
    return 2.0 * image - 1.0


class ImagicStableDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for imagic image editing.
    See paper here: https://arxiv.org/pdf/2210.09276.pdf

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
            Classification module that estimates whether generated images could be considered offsensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
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

        if hasattr(scheduler.config,
                   "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set",
                      "1.0.0",
                      deprecation_message,
                      standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None:
            logger.warn(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
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

    def train(
        self,
        prompt: Union[str, List[str]],
        init_image: Union[paddle.Tensor, PIL.Image.Image],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        seed: Optional[int] = None,
        embedding_learning_rate: float = 0.001,
        diffusion_model_learning_rate: float = 2e-6,
        text_embedding_optimization_steps: int = 500,
        model_fine_tuning_optimization_steps: int = 1000,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
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
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            seed (`int`, *optional*):
                A random seed.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        # Freeze vae and unet
        set_stop_gradient(self.vae, True)
        set_stop_gradient(self.unet, True)
        set_stop_gradient(self.text_encoder, True)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

        # get text embeddings for prompt
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncaton=True,
            return_tensors="pd",
        )
        attention_mask = paddle.ones_like(text_input.input_ids)
        text_embeddings = self.text_encoder(text_input.input_ids,
                                            attention_mask=attention_mask)[0]
        text_embeddings = text_embeddings.detach()
        text_embeddings.stop_gradient = False
        text_embeddings_orig = text_embeddings.clone()

        # Initialize the optimizer
        optimizer = Adam(
            parameters=[text_embeddings],  # only optimize the embeddings
            learning_rate=embedding_learning_rate,
        )

        if isinstance(init_image, PIL.Image.Image):
            init_image = preprocess(init_image)

        latents_dtype = text_embeddings.dtype
        init_image = init_image.astype(dtype=latents_dtype)
        init_latent_image_dist = self.vae.encode(init_image).latent_dist
        init_image_latents = init_latent_image_dist.sample()
        init_image_latents = 0.18215 * init_image_latents

        progress_bar = tqdm(range(text_embedding_optimization_steps))
        progress_bar.set_description("Steps")

        global_step = 0

        logger.info(
            "First optimizing the text embedding to better reconstruct the init image"
        )
        for _ in range(text_embedding_optimization_steps):
            # Sample noise that we'll add to the latents
            noise = paddle.randn(init_image_latents.shape)
            timesteps = paddle.randint(1000, (1, )).astype("int64")

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.scheduler.add_noise(init_image_latents, noise,
                                                     timesteps)

            # Predict the noise residual
            noise_pred = self.unet(noisy_latents, timesteps,
                                   text_embeddings).sample

            loss = F.mse_loss(noise_pred, noise,
                              reduction="none").mean([1, 2, 3]).mean()
            loss.backward()

            optimizer.step()
            optimizer.clear_grad()

            progress_bar.update(1)
            global_step += 1
            logs = {"loss": loss.item()}
            progress_bar.set_postfix(**logs)

        text_embeddings.stop_gradient = True

        # Now we fine tune the unet to better reconstruct the image
        set_stop_gradient(self.unet, False)
        self.unet.train()
        optimizer = Adam(
            parameters=self.unet.parameters(),  # only optimize unet
            learning_rate=diffusion_model_learning_rate,
        )
        progress_bar = tqdm(range(model_fine_tuning_optimization_steps),
                            disable=not accelerator.is_local_main_process)

        logger.info(
            "Next fine tuning the entire model to better reconstruct the init image"
        )
        for _ in range(model_fine_tuning_optimization_steps):
            # Sample noise that we'll add to the latents
            noise = paddle.randn(init_image_latents.shape)
            timesteps = paddle.randint(1000, (1, )).astype("int64")

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.scheduler.add_noise(init_image_latents, noise,
                                                     timesteps)

            # Predict the noise residual
            noise_pred = self.unet(noisy_latents, timesteps,
                                   text_embeddings).sample

            loss = F.mse_loss(noise_pred, noise,
                              reduction="none").mean([1, 2, 3]).mean()
            loss.backward()

            optimizer.step()
            optimizer.clear_grad()

            progress_bar.update(1)
            global_step += 1

            logs = {"loss": loss.item()}
            progress_bar.set_postfix(**logs)

        self.text_embeddings_orig = text_embeddings_orig
        self.text_embeddings = text_embeddings

    @paddle.no_grad()
    def __call__(
        self,
        alpha: float = 1.2,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        seed: Optional[int] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
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
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            seed (`int`, *optional*):
                A random seed.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )
        if self.text_embeddings is None:
            raise ValueError(
                "Please run the pipe.train() before trying to generate an image."
            )
        if self.text_embeddings_orig is None:
            raise ValueError(
                "Please run the pipe.train() before trying to generate an image."
            )

        text_embeddings = alpha * self.text_embeddings_orig + (
            1 - alpha) * self.text_embeddings

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens = [""]
            max_length = self.tokenizer.model_max_length
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
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.reshape([1, seq_len, -1])

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = paddle.concat(
                [uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if seed is not None:
            paddle.seed(seed)
        latents = paddle.randn(latents_shape, dtype=latents_dtype)

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
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input,
                                   t,
                                   encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents,
                                          **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clip(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.transpose([0, 2, 3, 1]).astype("float32").numpy()

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
