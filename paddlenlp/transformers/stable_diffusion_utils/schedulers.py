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

from typing import Union
import math
import numpy as np
import paddle
from scipy import integrate

__all__ = [
    "SchedulerMixin",
    "DDIMScheduler",
    "LMSDiscreteScheduler",
    "PNDMScheduler",
]


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce. :param alpha_bar: a lambda that takes an argument t
    from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float32)


class SchedulerMixin:
    def set_format(self, tensor_format="pd"):
        self.tensor_format = tensor_format
        if tensor_format == "pd":
            for key, value in vars(self).items():
                if isinstance(value, np.ndarray):
                    setattr(self, key, paddle.to_tensor(value))

        return self

    def clip(self, tensor, min_value=None, max_value=None):
        tensor_format = getattr(self, "tensor_format", "pd")

        if tensor_format == "np":
            return np.clip(tensor, min_value, max_value)
        elif tensor_format == "pd":
            return paddle.clip(tensor, min_value, max_value)

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def log(self, tensor):
        tensor_format = getattr(self, "tensor_format", "pd")

        if tensor_format == "np":
            return np.log(tensor)
        elif tensor_format == "pd":
            return paddle.log(tensor)

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def match_shape(
        self,
        values: Union[np.ndarray, paddle.Tensor],
        broadcast_array: Union[np.ndarray, paddle.Tensor],
    ):
        """
        Turns a 1-D array into an array or tensor with len(broadcast_array.shape) dims.

        Args:
            values: an array or tensor of values to extract.
            broadcast_array: an array with a larger shape of K dimensions with the batch
                dimension equal to the length of timesteps.
        Returns:
            a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """

        tensor_format = getattr(self, "tensor_format", "pd")
        values = values.flatten()

        while len(values.shape) < len(broadcast_array.shape):
            values = values[..., None]

        return values

    def norm(self, tensor):
        tensor_format = getattr(self, "tensor_format", "pd")
        if tensor_format == "np":
            return np.linalg.norm(tensor)
        elif tensor_format == "pd":
            return paddle.norm(tensor.reshape([tensor.shape[0], -1]), axis=-1).mean()

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def randn_like(self, tensor):
        tensor_format = getattr(self, "tensor_format", "pd")
        if tensor_format == "np":
            return np.random.randn(np.shape(tensor))
        elif tensor_format == "pd":
            return paddle.randn(tensor.shape)

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def zeros_like(self, tensor):
        tensor_format = getattr(self, "tensor_format", "pd")
        if tensor_format == "np":
            return np.zeros_like(tensor)
        elif tensor_format == "pd":
            return paddle.zeros_like(tensor)

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")


class LMSDiscreteScheduler(SchedulerMixin):
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        trained_betas=None,
        timestep_values=None,
        tensor_format="pd",
    ):
        """
        Linear Multistep Scheduler for discrete beta schedules. Based on the original k-diffusion implementation by
        Katherine Crowson:
        https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L181
        """
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.trained_betas = trained_betas
        self.timestep_values = timestep_values
        self.tensor_format = tensor_format

        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                np.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=np.float32,
                )
                ** 2
            )
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        self.derivatives = []
        self.set_format(tensor_format=tensor_format)

    def get_lms_coefficient(self, order, t, current_order):
        """
        Compute a linear multistep coefficient
        """

        def lms_derivative(tau):
            prod = 1.0
            for k in range(order):
                if current_order == k:
                    continue
                prod *= (tau - self.sigmas[t - k]) / (self.sigmas[t - current_order] - self.sigmas[t - k])
            return prod

        integrated_coeff = integrate.quad(lms_derivative, self.sigmas[t], self.sigmas[t + 1], epsrel=1e-4)[0]

        return integrated_coeff

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=float)

        low_idx = np.floor(self.timesteps).astype(int)
        high_idx = np.ceil(self.timesteps).astype(int)
        frac = np.mod(self.timesteps, 1.0)
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = (1 - frac) * sigmas[low_idx] + frac * sigmas[high_idx]
        self.sigmas = np.concatenate([sigmas, [0.0]])

        self.derivatives = []
        self.set_format(tensor_format=self.tensor_format)

    def step(
        self,
        model_output: Union[paddle.Tensor, np.ndarray],
        timestep: int,
        sample: Union[paddle.Tensor, np.ndarray],
        order: int = 4,
    ):
        sigma = self.sigmas[timestep]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        pred_original_sample = sample - sigma * model_output

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma
        self.derivatives.append(derivative)
        if len(self.derivatives) > order:
            self.derivatives.pop(0)

        # 3. Compute linear multistep coefficients
        order = min(timestep + 1, order)
        lms_coeffs = [self.get_lms_coefficient(order, timestep, curr_order) for curr_order in range(order)]

        # 4. Compute previous sample based on the derivatives path
        prev_sample = sample + sum(
            coeff * derivative for coeff, derivative in zip(lms_coeffs, reversed(self.derivatives))
        )

        return {"prev_sample": prev_sample}

    def add_noise(self, original_samples, noise, timesteps):
        sigmas = self.match_shape(self.sigmas[timesteps], noise)
        noisy_samples = original_samples + noise * sigmas
        return noisy_samples

    def __len__(self):
        return self.num_train_timesteps


class PNDMScheduler(SchedulerMixin):
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        skip_prk_steps=False,
        tensor_format="pd",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.skip_prk_steps = skip_prk_steps
        self.tensor_format = tensor_format

        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                np.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=np.float32,
                )
                ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.one = np.array(1.0)

        # For now we only support F-PNDM, i.e. the runge-kutta method
        # For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
        # mainly at formula (9), (12), (13) and the Algorithm 2.
        self.pndm_order = 4

        # running values
        self.cur_model_output = 0
        self.counter = 0
        self.cur_sample = None
        self.ets = []

        # setable values
        self.num_inference_steps = None
        self._timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        self._offset = 0
        self.prk_timesteps = None
        self.plms_timesteps = None
        self.timesteps = None
        self.set_format(tensor_format=tensor_format)

    def set_timesteps(self, num_inference_steps, offset=0):
        self.num_inference_steps = num_inference_steps
        self._timesteps = list(
            range(
                0,
                self.num_train_timesteps,
                self.num_train_timesteps // num_inference_steps,
            )
        )
        self._offset = offset
        self._timesteps = np.array([t + self._offset for t in self._timesteps])

        if self.skip_prk_steps:
            # for some models like stable diffusion the prk steps can/should be skipped to
            # produce better results. When using PNDM with `self.skip_prk_steps` the implementation
            # is based on crowsonkb's PLMS sampler implementation: https://github.com/CompVis/latent-diffusion/pull/51
            self.prk_timesteps = np.array([])
            self.plms_timesteps = np.concatenate([self._timesteps[:-1], self._timesteps[-2:-1], self._timesteps[-1:]])[
                ::-1
            ].copy()
        else:
            prk_timesteps = np.array(self._timesteps[-self.pndm_order :]).repeat(2) + np.tile(
                np.array([0, self.num_train_timesteps // num_inference_steps // 2]),
                self.pndm_order,
            )
            self.prk_timesteps = (prk_timesteps[:-1].repeat(2)[1:-1])[::-1].copy()
            self.plms_timesteps = self._timesteps[:-3][
                ::-1
            ].copy()  # we copy to avoid having negative strides which are not supported by paddle.to_tensor

        self.timesteps = np.concatenate([self.prk_timesteps, self.plms_timesteps]).astype(np.int64)

        self.ets = []
        self.counter = 0
        self.set_format(tensor_format=self.tensor_format)

    def step(
        self,
        model_output: Union[paddle.Tensor, np.ndarray],
        timestep: int,
        sample: Union[paddle.Tensor, np.ndarray],
    ):
        if self.counter < len(self.prk_timesteps) and not self.skip_prk_steps:
            return self.step_prk(model_output=model_output, timestep=timestep, sample=sample)
        else:
            return self.step_plms(model_output=model_output, timestep=timestep, sample=sample)

    def step_prk(
        self,
        model_output: Union[paddle.Tensor, np.ndarray],
        timestep: int,
        sample: Union[paddle.Tensor, np.ndarray],
    ):
        """
        Step function propagating the sample with the Runge-Kutta method. RK takes 4 forward passes to approximate the
        solution to the differential equation.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        diff_to_prev = 0 if self.counter % 2 else self.num_train_timesteps // self.num_inference_steps // 2
        prev_timestep = max(timestep - diff_to_prev, self.prk_timesteps[-1])
        timestep = self.prk_timesteps[self.counter // 4 * 4]

        if self.counter % 4 == 0:
            self.cur_model_output += 1 / 6 * model_output
            self.ets.append(model_output)
            self.cur_sample = sample
        elif (self.counter - 1) % 4 == 0:
            self.cur_model_output += 1 / 3 * model_output
        elif (self.counter - 2) % 4 == 0:
            self.cur_model_output += 1 / 3 * model_output
        elif (self.counter - 3) % 4 == 0:
            model_output = self.cur_model_output + 1 / 6 * model_output
            self.cur_model_output = 0

        # cur_sample should not be `None`
        cur_sample = self.cur_sample if self.cur_sample is not None else sample

        prev_sample = self._get_prev_sample(cur_sample, timestep, prev_timestep, model_output)
        self.counter += 1

        return {"prev_sample": prev_sample}

    def step_plms(
        self,
        model_output: Union[paddle.Tensor, np.ndarray],
        timestep: int,
        sample: Union[paddle.Tensor, np.ndarray],
    ):
        """
        Step function propagating the sample with the linear multi-step method. This has one forward pass with multiple
        times to approximate the solution.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        if not self.skip_prk_steps and len(self.ets) < 3:
            raise ValueError(
                f"{self.__class__} can only be run AFTER scheduler has been run "
                "in 'prk' mode for at least 12 iterations "
                "See: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_pndm.py "
                "for more information."
            )

        prev_timestep = max(timestep - self.num_train_timesteps // self.num_inference_steps, 0)

        if self.counter != 1:
            self.ets.append(model_output)
        else:
            prev_timestep = timestep
            timestep = timestep + self.num_train_timesteps // self.num_inference_steps

        if len(self.ets) == 1 and self.counter == 0:
            model_output = model_output
            self.cur_sample = sample
        elif len(self.ets) == 1 and self.counter == 1:
            model_output = (model_output + self.ets[-1]) / 2
            sample = self.cur_sample
            self.cur_sample = None
        elif len(self.ets) == 2:
            model_output = (3 * self.ets[-1] - self.ets[-2]) / 2
        elif len(self.ets) == 3:
            model_output = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
        else:
            model_output = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])

        prev_sample = self._get_prev_sample(sample, timestep, prev_timestep, model_output)
        self.counter += 1

        return {"prev_sample": prev_sample}

    def _get_prev_sample(self, sample, timestep, timestep_prev, model_output):
        # See formula (9) of PNDM paper https://arxiv.org/pdf/2202.09778.pdf
        # this function computes x_(t−δ) using the formula of (9)
        # Note that x_t needs to be added to both sides of the equation

        # Notation (<variable name> -> <name in paper>
        # alpha_prod_t -> α_t
        # alpha_prod_t_prev -> α_(t−δ)
        # beta_prod_t -> (1 - α_t)
        # beta_prod_t_prev -> (1 - α_(t−δ))
        # sample -> x_t
        # model_output -> e_θ(x_t, t)
        # prev_sample -> x_(t−δ)
        alpha_prod_t = self.alphas_cumprod[timestep + 1 - self._offset]
        alpha_prod_t_prev = self.alphas_cumprod[timestep_prev + 1 - self._offset]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # corresponds to (α_(t−δ) - α_t) divided by
        # denominator of x_t in formula (9) and plus 1
        # Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
        # sqrt(α_(t−δ)) / sqrt(α_t))
        sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)

        # corresponds to denominator of e_θ(x_t, t) in formula (9)
        model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** (0.5) + (
            alpha_prod_t * beta_prod_t * alpha_prod_t_prev
        ) ** (0.5)

        # full formula (9)
        prev_sample = (
            sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff
        )

        return prev_sample

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = self.match_shape(sqrt_alpha_prod, original_samples)
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = self.match_shape(sqrt_one_minus_alpha_prod, original_samples)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.num_train_timesteps


class DDIMScheduler(
    SchedulerMixin,
):
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        trained_betas=None,
        timestep_values=None,
        clip_sample=True,
        set_alpha_to_one=True,
        tensor_format="pd",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.trained_betas = trained_betas
        self.timestep_values = timestep_values
        self.clip_sample = clip_sample
        self.set_alpha_to_one = set_alpha_to_one
        self.tensor_format = tensor_format

        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                np.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=np.float32,
                )
                ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this paratemer simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = np.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()

        self.tensor_format = tensor_format
        self.set_format(tensor_format=tensor_format)

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def set_timesteps(self, num_inference_steps, offset=0):
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.arange(
            0,
            self.num_train_timesteps,
            self.num_train_timesteps // self.num_inference_steps,
        )[::-1].copy()
        self.timesteps += offset
        self.set_format(tensor_format=self.tensor_format)

    def step(
        self,
        model_output: Union[paddle.Tensor, np.ndarray],
        timestep: int,
        sample: Union[paddle.Tensor, np.ndarray],
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
    ):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointingc to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 4. Clip "predicted x_0"
        if self.clip_sample:
            pred_original_sample = self.clip(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the model_output is always re-derived from the clipped x_0 in Glide
            model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            noise = paddle.randn(model_output.shape)
            variance = self._get_variance(timestep, prev_timestep) ** (0.5) * eta * noise

            if not paddle.is_tensor(model_output):
                variance = variance.numpy()

            prev_sample = prev_sample + variance

        return {"prev_sample": prev_sample}

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = self.match_shape(sqrt_alpha_prod, original_samples)
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = self.match_shape(sqrt_one_minus_alpha_prod, original_samples)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.num_train_timesteps
