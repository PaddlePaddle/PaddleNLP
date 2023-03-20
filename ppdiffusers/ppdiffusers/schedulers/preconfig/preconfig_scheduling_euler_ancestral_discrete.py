# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 Katherine Crowson and The HuggingFace Team. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import paddle

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput, logging, randn_tensor
from ..scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
# Copied from ppdiffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->EulerAncestralDiscrete
class PreconfigEulerAncestralDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: paddle.Tensor
    pred_original_sample: Optional[paddle.Tensor] = None


# Copied from ppdiffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> paddle.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return paddle.to_tensor(betas, dtype=paddle.float32)


class PreconfigEulerAncestralDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Ancestral sampling with Euler method steps. Based on the original k-diffusion implementation by Katherine Crowson:
    https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L72

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://imagen.research.google/video/paper.pdf)

    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        preconfig: bool = True,
    ):
        if trained_betas is not None:
            self.betas = paddle.to_tensor(trained_betas, dtype=paddle.float32)
        elif beta_schedule == "linear":
            self.betas = paddle.linspace(beta_start, beta_end, num_train_timesteps, dtype=paddle.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                paddle.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=paddle.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = paddle.cumprod(self.alphas, 0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = paddle.to_tensor(sigmas)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = self.sigmas.max()

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        self.timesteps = paddle.to_tensor(timesteps, dtype=paddle.float32)
        self.is_scale_input_called = False
        self.preconfig = preconfig

    def scale_model_input(
        self, sample: paddle.Tensor, timestep: Union[float, paddle.Tensor], **kwargs
    ) -> paddle.Tensor:
        """
        Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`paddle.Tensor`): input sample
            timestep (`float` or `paddle.Tensor`): the current timestep in the diffusion chain

        Returns:
            `paddle.Tensor`: scaled input sample
        """
        self.is_scale_input_called = True
        if kwargs.get("step_index") is not None:
            step_index = kwargs["step_index"]
        else:
            step_index = (self.timesteps == timestep).nonzero().item()

        if not self.preconfig:
            sigma = self.sigmas[step_index]
            sample = sample / ((sigma**2 + 1) ** 0.5)
            return sample
        else:
            return sample * self.latent_scales[step_index]

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps

        timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=float)[::-1].copy()
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = paddle.to_tensor(sigmas)
        self.timesteps = paddle.to_tensor(timesteps, dtype=paddle.float32)
        if self.preconfig:
            self.sigma_up = []
            self.sigma_down = []
            for step_index_i in range(len(self.timesteps)):
                sigma_from = self.sigmas[step_index_i]
                sigma_to = self.sigmas[step_index_i + 1]
                sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
                sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
                self.sigma_up.append(sigma_up)
                self.sigma_down.append(sigma_down)
            self.latent_scales = 1 / ((self.sigmas**2 + 1) ** 0.5)

    def step(
        self,
        model_output: paddle.Tensor,
        timestep: Union[float, paddle.Tensor],
        sample: paddle.Tensor,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[PreconfigEulerAncestralDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`paddle.Tensor`): direct output from learned diffusion model.
            timestep (`float`): current timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                current instance of sample being created by diffusion process.
            generator (`paddle.Generator`, optional): Random number generator.
            return_dict (`bool`): option for returning tuple rather than PreconfigEulerAncestralDiscreteSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.PreconfigEulerAncestralDiscreteSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.PreconfigEulerAncestralDiscreteSchedulerOutput`] if `return_dict` is True, otherwise
            a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )
        if kwargs.get("return_pred_original_sample") is not None:
            return_pred_original_sample = kwargs["return_pred_original_sample"]
        else:
            return_pred_original_sample = True
        if kwargs.get("step_index") is not None:
            step_index = kwargs["step_index"]
        else:
            step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]
        if self.config.prediction_type == "epsilon" and not return_pred_original_sample:
            derivative = model_output
            pred_original_sample = None
        else:
            # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            if self.config.prediction_type == "epsilon":
                pred_original_sample = sample - sigma * model_output
            elif self.config.prediction_type == "v_prediction":
                # * c_out + input * c_skip
                pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
            elif self.config.prediction_type == "sample":
                raise NotImplementedError("prediction_type not implemented yet: sample")
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
                )
            derivative = (sample - pred_original_sample) / sigma
        if not self.preconfig:
            sigma_from = self.sigmas[step_index]
            sigma_to = self.sigmas[step_index + 1]
            sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
            sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
        else:
            sigma_up = self.sigma_up[step_index]
            sigma_down = self.sigma_down[step_index]
        # 2. Convert to an ODE derivative
        dt = sigma_down - sigma
        prev_sample = sample + derivative * dt
        noise = randn_tensor(model_output.shape, dtype=model_output.dtype, generator=generator)
        prev_sample = prev_sample + noise * sigma_up
        if not return_dict:
            if not return_pred_original_sample:
                return (prev_sample,)
            else:
                return (prev_sample, pred_original_sample)

        return PreconfigEulerAncestralDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )

    def add_noise(
        self,
        original_samples: paddle.Tensor,
        noise: paddle.Tensor,
        timesteps: paddle.Tensor,
    ) -> paddle.Tensor:
        # Make sure sigmas and timesteps have the same dtype as original_samples
        self.sigmas = self.sigmas.cast(original_samples.dtype)

        schedule_timesteps = self.timesteps
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = self.sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
