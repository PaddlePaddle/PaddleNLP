# coding=utf-8
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from ..image_processor import VaeImageProcessor
from ..utils import (
    DIFFUSERS_CACHE,
    FASTDEPLOY_MODEL_NAME,
    FASTDEPLOY_WEIGHTS_NAME,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
    ONNX_EXTERNAL_WEIGHTS_NAME,
    ONNX_WEIGHTS_NAME,
    PPDIFFUSERS_CACHE,
    _add_variant,
    _get_model_file,
    is_fastdeploy_available,
    is_paddle_available,
    logging,
    randn_tensor,
)
from ..version import VERSION as __version__

__all__ = ["FastDeployRuntimeModel", "FastDeployDiffusionPipelineMixin"]

if is_paddle_available():
    import paddle

if is_fastdeploy_available():
    import fastdeploy as fd
    from fastdeploy import ModelFormat

    def fdtensor2pdtensor(fdtensor: "fd.C.FDTensor"):
        dltensor = fdtensor.to_dlpack()
        pdtensor = paddle.utils.dlpack.from_dlpack(dltensor)
        return pdtensor

    def pdtensor2fdtensor(pdtensor: paddle.Tensor, name: str = "", share_with_raw_ptr=False):
        if not share_with_raw_ptr:
            dltensor = paddle.utils.dlpack.to_dlpack(pdtensor)
            return fd.C.FDTensor.from_dlpack(name, dltensor)
        else:
            return fd.C.FDTensor.from_external_data(
                name,
                pdtensor.data_ptr(),
                pdtensor.shape,
                pdtensor.dtype.name,
                str(pdtensor.place),
                int(pdtensor.place.gpu_device_id()),
            )


logger = logging.get_logger(__name__)


class FastDeployDiffusionPipelineMixin:
    def post_init(self, vae_scaling_factor=0.18215, vae_scale_factor=8, dtype="float32"):
        self.vae_scaling_factor = vae_scaling_factor
        self.vae_scale_factor = vae_scale_factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        self.dtype = dtype

    def get_timesteps(self, num_inference_steps, strength=0.0):
        if strength <= 0:
            return self.scheduler.timesteps.cast(self.dtype), num_inference_steps

        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :].cast(self.dtype)

        return timesteps, num_inference_steps - t_start

    def check_inputs_img2img(
        self, prompt, strength, callback_steps, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

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

    def check_inputs_txt2img(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor} but are {height} and {width}."
            )

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

    def prepare_latents_txt2img(self, batch_size, num_channels_latents, height, width, generator, latents=None):
        shape = [batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor]
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=self.dtype)
        else:
            if str(latents.dtype).replace("paddle.", "") != self.dtype:
                latents = latents.cast(self.dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * float(self.scheduler.init_noise_sigma)
        return latents

    def prepare_latents_img2img(self, image, timestep, batch_size, num_images_per_prompt, generator=None, noise=None):
        if not isinstance(image, (paddle.Tensor, list)):
            raise ValueError(f"`image` has to be of type `paddle.Tensor` or list but is {type(image)}")

        image = image.cast(self.dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image_shape = image.shape
        init_latents = paddle.zeros(
            [image_shape[0], 4, image_shape[2] // self.vae_scale_factor, image_shape[3] // self.vae_scale_factor],
            dtype=self.dtype,
        )
        vae_input_name = self.vae_encoder.model.get_input_info(0).name
        vae_output_name = self.vae_encoder.model.get_output_info(0).name

        self.vae_encoder.zero_copy_infer(
            prebinded_inputs={vae_input_name: image},
            prebinded_outputs={vae_output_name: init_latents},
            share_with_raw_ptr=True,
        )

        init_latents = self.vae_scaling_factor * init_latents

        if noise is None:
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, dtype=self.dtype)
        else:
            if str(noise.dtype).replace("paddle.", "") != self.dtype:
                noise = noise.cast(self.dtype)

        clean_latents = init_latents
        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents, clean_latents

    def is_scheduler_support_step_index(self):
        kwargs_keys = set(inspect.signature(self.scheduler.step).parameters.keys())
        return "kwargs" in kwargs_keys or "step_index" in kwargs_keys

    def decode_latents(self, latents):
        latents_shape = latents.shape
        vae_output_shape = [
            latents_shape[0],
            3,
            latents_shape[2] * self.vae_scale_factor,
            latents_shape[3] * self.vae_scale_factor,
        ]
        images_vae = paddle.zeros(vae_output_shape, dtype=self.dtype)

        vae_input_name = self.vae_decoder.model.get_input_info(0).name
        vae_output_name = self.vae_decoder.model.get_output_info(0).name

        self.vae_decoder.zero_copy_infer(
            prebinded_inputs={vae_input_name: latents},
            prebinded_outputs={vae_output_name: images_vae},
            share_with_raw_ptr=True,
        )

        return images_vae

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
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # get prompt text embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="np").input_ids  # check

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not np.array_equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(input_ids=text_input_ids.astype(np.int64))[0]
            prompt_embeds = paddle.to_tensor(prompt_embeds, dtype=self.dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile([1, num_images_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape([bs_embed * num_images_per_prompt, seq_len, -1])

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                uncond_tokens = [""]
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
                return_tensors="np",
            )
            negative_prompt_embeds = self.text_encoder(
                input_ids=uncond_input.input_ids.astype(np.int64),
            )[0]
            negative_prompt_embeds = paddle.to_tensor(negative_prompt_embeds, dtype=self.dtype)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.tile([1, num_images_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape([batch_size * num_images_per_prompt, seq_len, -1])

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = paddle.concat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def run_safety_checker(self, image):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if paddle.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="np")
            image, has_nsfw_concept = self.safety_checker(
                images=image.numpy(), clip_input=safety_checker_input.pixel_values.astype(self.dtype)
            )
            image = paddle.to_tensor(image, dtype=self.dtype)
        return image, has_nsfw_concept

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


class FastDeployRuntimeModel:
    def __init__(self, model=None, **kwargs):
        logger.info("`ppdiffusers.FastDeployRuntimeModel` is experimental and might change in the future.")
        self.model = model
        self.model_save_dir = kwargs.get("model_save_dir", None)
        self.model_format = kwargs.get("model_format", None)
        self.latest_model_name = kwargs.get("latest_model_name", None)
        self.latest_params_name = kwargs.get("latest_params_name", None)

        if self.model_format in [ModelFormat.PADDLE, "PADDLE", None]:
            if self.latest_model_name is None:
                self.latest_model_name = FASTDEPLOY_MODEL_NAME
            if self.latest_params_name is None:
                self.latest_params_name = FASTDEPLOY_WEIGHTS_NAME
            self.model_format = ModelFormat.PADDLE
        if self.model_format in [ModelFormat.ONNX, "ONNX"]:
            if self.latest_model_name is None:
                self.latest_model_name = ONNX_WEIGHTS_NAME
            self.latest_params_name = None
            self.model_format = ModelFormat.ONNX

    def zero_copy_infer(self, prebinded_inputs: dict, prebinded_outputs: dict, share_with_raw_ptr=True, **kwargs):
        """
        Execute inference without copying data from cpu to gpu.

        Arguments:
            kwargs (`dict(name, paddle.Tensor)`):
                An input map from name to tensor.
        Return:
            List of output tensor.
        """
        for inputs_name, inputs_tensor in prebinded_inputs.items():
            input_fdtensor = pdtensor2fdtensor(inputs_tensor, inputs_name, share_with_raw_ptr=share_with_raw_ptr)
            self.model.bind_input_tensor(inputs_name, input_fdtensor)

        for outputs_name, outputs_tensor in prebinded_outputs.items():
            output_fdtensor = pdtensor2fdtensor(outputs_tensor, outputs_name, share_with_raw_ptr=share_with_raw_ptr)
            self.model.bind_output_tensor(outputs_name, output_fdtensor)

        self.model.zero_copy_infer()

    def __call__(self, **kwargs):
        inputs = {k: np.array(v) for k, v in kwargs.items()}
        return self.model.infer(inputs)

    @staticmethod
    def load_model(
        model_path: Union[str, Path],
        params_path: Union[str, Path] = None,
        runtime_options: Optional["fd.RuntimeOption"] = None,
    ):
        """
        Loads an FastDeploy Inference Model with fastdeploy.RuntimeOption

        Arguments:
            model_path (`str` or `Path`):
                Model path from which to load
            params_path (`str` or `Path`):
                Params path from which to load
            runtime_options (fd.RuntimeOption, *optional*):
                The RuntimeOption of fastdeploy to initialize the fastdeploy runtime. Default setting
                the device to cpu and the backend to paddle inference
        """
        option = runtime_options
        if option is None or not isinstance(runtime_options, fd.RuntimeOption):
            logger.info("No fastdeploy.RuntimeOption specified, using CPU device and paddle inference backend.")
            option = fd.RuntimeOption()
            option.use_paddle_backend()
            option.use_cpu()

        if params_path is None or model_path.endswith(".onnx"):
            option.use_ort_backend()
            option.set_model_path(model_path, model_format=ModelFormat.ONNX)
        else:
            option.set_model_path(model_path, params_path)
        return fd.Runtime(option)

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        model_file_name: Optional[str] = None,
        params_file_name: Optional[str] = None,
        **kwargs
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~FastDeployRuntimeModel.from_pretrained`] class method. It will always save the
        latest_model_name.

        Arguments:
            save_directory (`str` or `Path`):
                Directory where to save the model file.
            model_file_name(`str`, *optional*):
                Overwrites the default model file name from `"inference.pdmodel"` to `model_file_name`. This allows you to save the
                model with a different name.
            params_file_name(`str`, *optional*):
                Overwrites the default model file name from `"inference.pdiparams"` to `params_file_name`. This allows you to save the
                model with a different name.
        """
        is_onnx_model = self.model_format == ModelFormat.ONNX
        model_file_name = (
            model_file_name
            if model_file_name is not None
            else FASTDEPLOY_MODEL_NAME
            if not is_onnx_model
            else ONNX_WEIGHTS_NAME
        )
        params_file_name = params_file_name if params_file_name is not None else FASTDEPLOY_WEIGHTS_NAME

        src_model_path = self.model_save_dir.joinpath(self.latest_model_name)
        dst_model_path = Path(save_directory).joinpath(model_file_name)

        try:
            shutil.copyfile(src_model_path, dst_model_path)
        except shutil.SameFileError:
            pass

        if is_onnx_model:
            # copy external weights (for models >2GB)
            src_model_path = self.model_save_dir.joinpath(ONNX_EXTERNAL_WEIGHTS_NAME)
            if src_model_path.exists():
                dst_model_path = Path(save_directory).joinpath(ONNX_EXTERNAL_WEIGHTS_NAME)
                try:
                    shutil.copyfile(src_model_path, dst_model_path)
                except shutil.SameFileError:
                    pass

        if not is_onnx_model:
            src_params_path = self.model_save_dir.joinpath(self.latest_params_name)
            dst_params_path = Path(save_directory).joinpath(params_file_name)
            try:
                shutil.copyfile(src_params_path, dst_params_path)
            except shutil.SameFileError:
                pass

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        **kwargs,
    ):
        """
        Save a model to a directory, so that it can be re-loaded using the [`~FastDeployRuntimeModel.from_pretrained`] class
        method.:

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # saving model weights/files
        self._save_pretrained(save_directory, **kwargs)

    @classmethod
    def _from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        model_file_name: Optional[str] = None,
        params_file_name: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[str] = None,
        subfolder: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        runtime_options: Optional["fd.RuntimeOption"] = None,
        from_hf_hub: Optional[bool] = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        user_agent: Union[Dict, str, None] = None,
        is_onnx_model: bool = False,
        **kwargs,
    ):
        """
        Load a model from a directory or the HF Hub.

        Arguments:
            pretrained_model_name_or_path (`str` or `Path`):
                Directory from which to load
            model_file_name (`str`):
                Overwrites the default model file name from `"inference.pdmodel"` to `file_name`. This allows you to load
                different model files from the same repository or directory.
            params_file_name (`str`):
                Overwrites the default params file name from `"inference.pdiparams"` to `file_name`. This allows you to load
                different model files from the same repository or directory.
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private or gated repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            cache_dir (`Union[str, Path]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            runtime_options (`fastdeploy.RuntimeOption`, *optional*):
                The RuntimeOption of fastdeploy.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """

        model_file_name = (
            model_file_name
            if model_file_name is not None
            else FASTDEPLOY_MODEL_NAME
            if not is_onnx_model
            else ONNX_WEIGHTS_NAME
        )
        params_file_name = params_file_name if params_file_name is not None else FASTDEPLOY_WEIGHTS_NAME
        kwargs["model_format"] = "ONNX" if is_onnx_model else "PADDLE"

        # load model from local directory
        if os.path.isdir(pretrained_model_name_or_path):
            model_path = os.path.join(pretrained_model_name_or_path, model_file_name)
            params_path = None if is_onnx_model else os.path.join(pretrained_model_name_or_path, params_file_name)
            model = FastDeployRuntimeModel.load_model(
                model_path,
                params_path,
                runtime_options=runtime_options,
            )
            kwargs["model_save_dir"] = Path(pretrained_model_name_or_path)
        # load model from hub or paddle bos
        else:
            model_cache_path = _get_model_file(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                weights_name=model_file_name,
                subfolder=subfolder,
                cache_dir=cache_dir,
                force_download=force_download,
                revision=revision,
                from_hf_hub=from_hf_hub,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
            )
            if is_onnx_model:
                params_cache_path = None
                kwargs["latest_params_name"] = None
            else:
                params_cache_path = _get_model_file(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    weights_name=params_file_name,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    revision=revision,
                    from_hf_hub=from_hf_hub,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                )
                kwargs["latest_params_name"] = Path(params_cache_path).name
            kwargs["model_save_dir"] = Path(model_cache_path).parent
            kwargs["latest_model_name"] = Path(model_cache_path).name

            model = FastDeployRuntimeModel.load_model(
                model_cache_path,
                params_cache_path,
                runtime_options=runtime_options,
            )
        return cls(model=model, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        model_file_name: Optional[str] = None,
        params_file_name: Optional[str] = None,
        runtime_options: Optional["fd.RuntimeOption"] = None,
        is_onnx_model: bool = False,
        **kwargs,
    ):
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        variant = kwargs.pop("variant", None)

        user_agent = {
            "ppdiffusers": __version__,
            "file_type": "model",
            "framework": "fastdeploy",
        }

        return cls._from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_file_name=_add_variant(model_file_name, variant),
            params_file_name=_add_variant(params_file_name, variant),
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            force_download=force_download,
            cache_dir=cache_dir,
            runtime_options=runtime_options,
            from_hf_hub=from_hf_hub,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            user_agent=user_agent,
            is_onnx_model=is_onnx_model,
            **kwargs,
        )
