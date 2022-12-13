# Copyright 2022 Katherine Crowson, The HuggingFace Team and hlky. All rights reserved.
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

from typing import List, Optional, Tuple, Union

import numpy as np
import paddle

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import _COMPATIBLE_STABLE_DIFFUSION_SCHEDULERS
from .scheduling_utils import SchedulerMixin, SchedulerOutput


class KDPM2DiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler created by @crowsonkb in [k_diffusion](https://github.com/crowsonkb/k-diffusion), see:
    https://github.com/crowsonkb/k-diffusion/blob/5b3af030dd83e0297272d861c19477735d0317ec/k_diffusion/sampling.py#L188

    Scheduler inspired by DPM-Solver-2 and Algorthim 2 from Karras et al. (2022).

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

    _compatibles = _COMPATIBLE_STABLE_DIFFUSION_SCHEDULERS.copy()
    order = 2

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,  # sensible defaults
        beta_end: float = 0.012,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
    ):
        if trained_betas is not None:
            self.betas = paddle.to_tensor(trained_betas, dtype="float32")
        elif beta_schedule == "linear":
            self.betas = paddle.linspace(beta_start, beta_end, num_train_timesteps, dtype="float32")
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = paddle.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype="float32") ** 2
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = paddle.cumprod(self.alphas, 0)

        #  set all values
        self.set_timesteps(num_train_timesteps, num_train_timesteps)

    def index_for_timestep(self, timestep):
        indices = (self.timesteps == timestep).nonzero()
        if self.state_in_first_order:
            pos = -1
        else:
            pos = 0
        return indices[pos].item()

    def scale_model_input(
        self,
        sample: paddle.Tensor,
        timestep: Union[float, paddle.Tensor],
    ) -> paddle.Tensor:
        """
        Args:
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
            sample (`paddle.Tensor`): input sample timestep (`int`, optional): current timestep
        Returns:
            `paddle.Tensor`: scaled input sample
        """
        step_index = self.index_for_timestep(timestep)

        if self.state_in_first_order:
            sigma = self.sigmas[step_index]
        else:
            sigma = self.sigmas_interpol[step_index]

        sample = sample / ((sigma**2 + 1) ** 0.5)
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int,
        num_train_timesteps: Optional[int] = None,
    ):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps

        num_train_timesteps = num_train_timesteps or self.config.num_train_timesteps

        timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps, dtype=np.float32)[::-1].copy()

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        self.log_sigmas = paddle.to_tensor(np.log(sigmas), dtype="float32")

        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        sigmas = paddle.to_tensor(sigmas)

        # interpolate sigmas
        sigmas_interpol = sigmas.log().lerp(sigmas.roll(1).log(), 0.5).exp()
        # must set to 0.0
        sigmas_interpol[-1] = 0.0

        self.sigmas = paddle.concat([sigmas[:1], sigmas[1:].repeat_interleave(2), sigmas[-1:]])
        self.sigmas_interpol = paddle.concat(
            [sigmas_interpol[:1], sigmas_interpol[1:].repeat_interleave(2), sigmas_interpol[-1:]]
        )

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = self.sigmas.max()

        timesteps = paddle.to_tensor(timesteps)

        # interpolate timesteps
        timesteps_interpol = self.sigma_to_t(sigmas_interpol)
        interleaved_timesteps = paddle.stack((timesteps_interpol[1:-1, None], timesteps[1:, None]), axis=-1).flatten()
        timesteps = paddle.concat([timesteps[:1], interleaved_timesteps])

        self.timesteps = timesteps

        self.sample = None

    def sigma_to_t(self, sigma):
        # get log sigma
        log_sigma = sigma.log()

        # get distribution
        dists = log_sigma - self.log_sigmas[:, None]

        # get sigmas range
        low_idx = (dists >= 0).cast("int64").cumsum(axis=0).argmax(axis=0).clip(max=self.log_sigmas.shape[0] - 2)

        high_idx = low_idx + 1

        low = self.log_sigmas[low_idx]
        high = self.log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = w.clip(0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    @property
    def state_in_first_order(self):
        return self.sample is None

    def step(
        self,
        model_output: Union[paddle.Tensor, np.ndarray],
        timestep: Union[float, paddle.Tensor],
        sample: Union[paddle.Tensor, np.ndarray],
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Args:
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
            model_output (`paddle.Tensor` or `np.ndarray`): direct output from learned diffusion model. timestep
            (`int`): current discrete timestep in the diffusion chain. sample (`paddle.Tensor` or `np.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class
        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        step_index = self.index_for_timestep(timestep)

        if self.state_in_first_order:
            sigma = self.sigmas[step_index]
            sigma_interpol = self.sigmas_interpol[step_index + 1]
            sigma_next = self.sigmas[step_index + 1]
        else:
            # 2nd order / KDPM2's method
            sigma = self.sigmas[step_index - 1]
            sigma_interpol = self.sigmas_interpol[step_index]
            sigma_next = self.sigmas[step_index]

        # currently only gamma=0 is supported. This usually works best anyways.
        # We can support gamma in the future but then need to scale the timestep before
        # passing it to the model which requires a change in API
        gamma = 0
        sigma_hat = sigma * (gamma + 1)  # Note: sigma_hat == sigma for now

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            sigma_input = sigma_hat if self.state_in_first_order else sigma_interpol
            pred_original_sample = sample - sigma_input * model_output
        elif self.config.prediction_type == "v_prediction":
            sigma_input = sigma_hat if self.state_in_first_order else sigma_interpol
            pred_original_sample = model_output * (-sigma_input / (sigma_input**2 + 1) ** 0.5) + (
                sample / (sigma_input**2 + 1)
            )
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        if self.state_in_first_order:
            # 2. Convert to an ODE derivative for 1st order
            derivative = (sample - pred_original_sample) / sigma_hat
            # 3. delta timestep
            dt = sigma_interpol - sigma_hat

            # store for 2nd order step
            self.sample = sample
        else:
            # DPM-Solver-2
            # 2. Convert to an ODE derivative for 2nd order
            derivative = (sample - pred_original_sample) / sigma_interpol

            # 3. delta timestep
            dt = sigma_next - sigma_hat

            sample = self.sample
            self.sample = None

        prev_sample = sample + derivative * dt

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: paddle.Tensor,
        noise: paddle.Tensor,
        timesteps: paddle.Tensor,
    ) -> paddle.Tensor:
        # Make sure sigmas and timesteps have the same dtype as original_samples
        self.sigmas = self.sigmas.cast(original_samples.dtype)

        step_indices = [self.index_for_timestep(t) for t in timesteps]

        sigma = self.sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
