# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 Kakao Brain and The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple, Union

import numpy as np
import paddle

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, randn_tensor
from .scheduling_utils import SchedulerMixin


@dataclass
# Copied from ppdiffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->UnCLIP
class UnCLIPSchedulerOutput(BaseOutput):
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
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
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


class UnCLIPScheduler(SchedulerMixin, ConfigMixin):
    """
    This is a modified DDPM Scheduler specifically for the karlo unCLIP model.

    This scheduler has some minor variations in how it calculates the learned range variance and dynamically
    re-calculates betas based off the timesteps it is skipping.

    The scheduler also uses a slightly different step ratio when computing timesteps to use for inference.

    See [`~DDPMScheduler`] for more information on DDPM scheduling

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        variance_type (`str`):
            options to clip the variance used when adding noise to the denoised sample. Choose from `fixed_small_log`
            or `learned_range`.
        clip_sample (`bool`, default `True`):
            option to clip predicted sample between `-clip_sample_range` and `clip_sample_range` for numerical
            stability.
        clip_sample_range (`float`, default `1.0`):
            The range to clip the sample between. See `clip_sample`.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion process)
            or `sample` (directly predicting the noisy sample`)
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        variance_type: str = "fixed_small_log",
        clip_sample: bool = True,
        clip_sample_range: Optional[float] = 1.0,
        prediction_type: str = "epsilon",
        beta_schedule: str = "squaredcos_cap_v2",
    ):
        if beta_schedule != "squaredcos_cap_v2":
            raise ValueError("UnCLIPScheduler only supports `beta_schedule`: 'squaredcos_cap_v2'")

        self.betas = betas_for_alpha_bar(num_train_timesteps)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = paddle.cumprod(self.alphas, 0)
        self.one = paddle.to_tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = paddle.to_tensor(np.arange(0, num_train_timesteps)[::-1].copy())

        self.variance_type = variance_type

    def scale_model_input(self, sample: paddle.Tensor, timestep: Optional[int] = None) -> paddle.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`paddle.Tensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `paddle.Tensor`: scaled input sample
        """
        return sample

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Note that this scheduler uses a slightly different step ratio than the other diffusers schedulers. The
        different step ratio is to mimic the original karlo implementation and does not affect the quality or accuracy
        of the results.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = (self.config.num_train_timesteps - 1) / (self.num_inference_steps - 1)
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = paddle.to_tensor(timesteps)

    def _get_variance(self, t, prev_timestep=None, predicted_variance=None, variance_type=None):
        if prev_timestep is None:
            prev_timestep = t - 1

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        if prev_timestep == t - 1:
            beta = self.betas[t]
        else:
            beta = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = beta_prod_t_prev / beta_prod_t * beta

        if variance_type is None:
            variance_type = self.config.variance_type

        # hacks - were probably added for training stability
        if variance_type == "fixed_small_log":
            variance = paddle.log(paddle.clip(variance, min=1e-20))
            variance = paddle.exp(0.5 * variance)
        elif variance_type == "learned_range":
            # NOTE difference with DDPM scheduler
            min_log = variance.log()
            max_log = beta.log()

            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def step(
        self,
        model_output: paddle.Tensor,
        timestep: int,
        sample: paddle.Tensor,
        prev_timestep: Optional[int] = None,
        generator=None,
        return_dict: bool = True,
    ) -> Union[UnCLIPSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`paddle.Tensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                current instance of sample being created by diffusion process.
            prev_timestep (`int`, *optional*): The previous timestep to predict the previous sample at.
                Used to dynamically compute beta. If not given, `t-1` is used and the pre-computed beta is used.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than UnCLIPSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.UnCLIPSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.UnCLIPSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        t = timestep

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type == "learned_range":
            # must split like this, 3 -> split 2 -> [2, 1]
            model_output, predicted_variance = model_output.split(
                [sample.shape[1], model_output.shape[1] - sample.shape[1]], axis=1
            )
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        if prev_timestep is None:
            prev_timestep = t - 1

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        if prev_timestep == t - 1:
            beta = self.betas[t]
            alpha = self.alphas[t]
        else:
            beta = 1 - alpha_prod_t / alpha_prod_t_prev
            alpha = 1 - beta

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon` or `sample`"
                " for the UnCLIPScheduler."
            )

        # 3. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = paddle.clip(
                pred_original_sample, -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * beta) / beta_prod_t
        current_sample_coeff = alpha ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance = 0
        if t > 0:
            variance_noise = randn_tensor(
                model_output.shape,
                dtype=model_output.dtype,
                generator=generator,
            )

            variance = self._get_variance(
                t,
                predicted_variance=predicted_variance,
                prev_timestep=prev_timestep,
            )

            if self.variance_type == "fixed_small_log":
                variance = variance
            elif self.variance_type == "learned_range":
                variance = (0.5 * variance).exp()
            else:
                raise ValueError(
                    f"variance_type given as {self.variance_type} must be one of `fixed_small_log` or `learned_range`"
                    " for the UnCLIPScheduler."
                )

            variance = variance * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return UnCLIPSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
