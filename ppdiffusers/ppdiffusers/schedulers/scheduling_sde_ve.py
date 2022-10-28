# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 Google Brain and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/yang-song/score_sde_pytorch

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import paddle

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin, SchedulerOutput


@dataclass
class SdeVeOutput(BaseOutput):
    """
    Output class for the ScoreSdeVeScheduler's step function output.

    Args:
        prev_sample (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        prev_sample_mean (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Mean averaged `prev_sample`. Same as `prev_sample`, only mean-averaged over previous timesteps.
    """

    prev_sample: paddle.Tensor
    prev_sample_mean: paddle.Tensor


class ScoreSdeVeScheduler(SchedulerMixin, ConfigMixin):
    """
    The variance exploding stochastic differential equation (SDE) scheduler.

    For more information, see the original paper: https://arxiv.org/abs/2011.13456

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functions.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        snr (`float`):
            coefficient weighting the step from the model_output sample (from the network) to the random noise.
        sigma_min (`float`):
                initial noise scale for sigma sequence in sampling procedure. The minimum sigma should mirror the
                distribution of the data.
        sigma_max (`float`): maximum value used for the range of continuous timesteps passed into the model.
        sampling_eps (`float`): the end value of sampling, where timesteps decrease progressively from 1 to
        epsilon.
        correct_steps (`int`): number of correction steps performed on a produced sample.
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 2000,
        snr: float = 0.15,
        sigma_min: float = 0.01,
        sigma_max: float = 1348.0,
        sampling_eps: float = 1e-5,
        correct_steps: int = 1,
    ):

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

        # setable values
        self.timesteps = None

        self.set_sigmas(num_train_timesteps, sigma_min, sigma_max, sampling_eps)

    def scale_model_input(self,
                          sample: paddle.Tensor,
                          timestep: Optional[int] = None) -> paddle.Tensor:
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

    def set_timesteps(self,
                      num_inference_steps: int,
                      sampling_eps: float = None):
        """
        Sets the continuous timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            sampling_eps (`float`, optional): final timestep value (overrides value given at Scheduler instantiation).

        """
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps

        self.timesteps = paddle.linspace(1, sampling_eps, num_inference_steps)

    def set_sigmas(self,
                   num_inference_steps: int,
                   sigma_min: float = None,
                   sigma_max: float = None,
                   sampling_eps: float = None):
        """
        Sets the noise scales used for the diffusion chain. Supporting function to be run before inference.

        The sigmas control the weight of the `drift` and `diffusion` components of sample update.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            sigma_min (`float`, optional):
                initial noise scale value (overrides value given at Scheduler instantiation).
            sigma_max (`float`, optional): final noise scale value (overrides value given at Scheduler instantiation).
            sampling_eps (`float`, optional): final timestep value (overrides value given at Scheduler instantiation).

        """
        sigma_min = sigma_min if sigma_min is not None else self.config.sigma_min
        sigma_max = sigma_max if sigma_max is not None else self.config.sigma_max
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps
        if self.timesteps is None:
            self.set_timesteps(num_inference_steps, sampling_eps)

        self.sigmas = sigma_min * (sigma_max / sigma_min)**(self.timesteps /
                                                            sampling_eps)
        self.discrete_sigmas = paddle.exp(
            paddle.linspace(math.log(sigma_min), math.log(sigma_max),
                            num_inference_steps))
        self.sigmas = paddle.to_tensor(
            [sigma_min * (sigma_max / sigma_min)**t for t in self.timesteps])

    def get_adjacent_sigma(self, timesteps, t):
        return paddle.where(
            timesteps == 0,
            paddle.zeros_like(t),
            self.discrete_sigmas[timesteps - 1],
        )

    def step_pred(
        self,
        model_output: paddle.Tensor,
        timestep: int,
        sample: paddle.Tensor,
        return_dict: bool = True,
    ) -> Union[SdeVeOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`paddle.Tensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`: [`~schedulers.scheduling_sde_ve.SdeVeOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        timestep = timestep * paddle.ones((sample.shape[0], ))
        timesteps = (timestep * (len(self.timesteps) - 1)).astype("int64")

        sigma = self.discrete_sigmas[timesteps]
        adjacent_sigma = self.get_adjacent_sigma(timesteps, timestep)
        drift = paddle.zeros_like(sample)
        diffusion = (sigma**2 - adjacent_sigma**2)**0.5

        # equation 6 in the paper: the model_output modeled by the network is grad_x log pt(x)
        # also equation 47 shows the analog from SDE models to ancestral sampling methods
        diffusion = diffusion.flatten()
        while len(diffusion.shape) < len(sample.shape):
            diffusion = diffusion.unsqueeze(-1)
        drift = drift - diffusion**2 * model_output

        #  equation 6: sample noise for the diffusion term of
        noise = paddle.randn(sample.shape)
        prev_sample_mean = sample - drift  # subtract because `dt` is a small negative timestep
        # TODO is the variable diffusion the correct scaling term for the noise?
        prev_sample = prev_sample_mean + diffusion * noise  # add impact of diffusion field g

        if not return_dict:
            return (prev_sample, prev_sample_mean)

        return SdeVeOutput(prev_sample=prev_sample,
                           prev_sample_mean=prev_sample_mean)

    def step_correct(
        self,
        model_output: paddle.Tensor,
        sample: paddle.Tensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Correct the predicted sample based on the output model_output of the network. This is often run repeatedly
        after making the prediction for the previous timestep.

        Args:
            model_output (`paddle.Tensor`): direct output from learned diffusion model.
            sample (`paddle.Tensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`: [`~schedulers.scheduling_sde_ve.SdeVeOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # For small batch sizes, the paper "suggest replacing norm(z) with sqrt(d), where d is the dim. of z"
        # sample noise for correction
        noise = paddle.randn(sample.shape)

        # compute step size from the model_output, the noise, and the snr
        grad_norm = paddle.norm(model_output.reshape(
            [model_output.shape[0], -1]),
                                axis=-1).mean()
        noise_norm = paddle.norm(noise.reshape([noise.shape[0], -1]),
                                 axis=-1).mean()
        step_size = (self.config.snr * noise_norm / grad_norm)**2 * 2
        step_size = step_size * paddle.ones((sample.shape[0], ))
        # self.repeat_scalar(step_size, sample.shape[0])

        # compute corrected sample: model_output term and noise term
        step_size = step_size.flatten()
        while len(step_size.shape) < len(sample.shape):
            step_size = step_size.unsqueeze(-1)
        prev_sample_mean = sample + step_size * model_output
        prev_sample = prev_sample_mean + ((step_size * 2)**0.5) * noise

        if not return_dict:
            return (prev_sample, )

        return SchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
