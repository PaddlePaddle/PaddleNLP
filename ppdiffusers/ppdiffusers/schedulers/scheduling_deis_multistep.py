# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 FLAIR Lab and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: check https://arxiv.org/abs/2204.13902 and https://github.com/qsh-zh/deis for more info
# The codebase is modified based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import paddle

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


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


class DEISMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    DEIS (https://arxiv.org/abs/2204.13902) is a fast high order solver for diffusion ODEs. We slightly modify the
    polynomial fitting formula in log-rho space instead of the original linear t space in DEIS paper. The modification
    enjoys closed-form coefficients for exponential multistep update instead of replying on the numerical solver. More
    variants of DEIS can be found in https://github.com/qsh-zh/deis.

    Currently, we support the log-rho multistep DEIS. We recommend to use `solver_order=2 / 3` while `solver_order=1`
    reduces to DDIM.

    We also support the "dynamic thresholding" method in Imagen (https://arxiv.org/abs/2205.11487). For pixel-space
    diffusion models, you can set `thresholding=True` to use the dynamic thresholding.

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
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        solver_order (`int`, default `2`):
            the order of DEIS; can be `1` or `2` or `3`. We recommend to use `solver_order=2` for guided sampling, and
            `solver_order=3` for unconditional sampling.
        prediction_type (`str`, default `epsilon`):
            indicates whether the model predicts the noise (epsilon), or the data / `x0`. One of `epsilon`, `sample`,
            or `v-prediction`.
        thresholding (`bool`, default `False`):
            whether to use the "dynamic thresholding" method (introduced by Imagen, https://arxiv.org/abs/2205.11487).
            Note that the thresholding method is unsuitable for latent-space diffusion models (such as
            stable-diffusion).
        dynamic_thresholding_ratio (`float`, default `0.995`):
            the ratio for the dynamic thresholding method. Default is `0.995`, the same as Imagen
            (https://arxiv.org/abs/2205.11487).
        sample_max_value (`float`, default `1.0`):
            the threshold value for dynamic thresholding. Valid woks when `thresholding=True`
        algorithm_type (`str`, default `deis`):
            the algorithm type for the solver. current we support multistep deis, we will add other variants of DEIS in
            the future
        lower_order_final (`bool`, default `True`):
            whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. We empirically
            find this trick can stabilize the sampling of DEIS for steps < 15, especially for steps <= 10.

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
        trained_betas: Optional[np.ndarray] = None,
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "deis",
        solver_type: str = "logrho",
        lower_order_final: bool = True,
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
        # Currently we only support VP-type noise schedule
        self.alpha_t = paddle.sqrt(self.alphas_cumprod)
        self.sigma_t = paddle.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = paddle.log(self.alpha_t) - paddle.log(self.sigma_t)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # settings for DEIS
        if algorithm_type not in ["deis"]:
            if algorithm_type in ["dpmsolver", "dpmsolver++"]:
                algorithm_type = "deis"
            else:
                raise NotImplementedError(f"{algorithm_type} does is not implemented for {self.__class__}")

        if solver_type not in ["logrho"]:
            if solver_type in ["midpoint", "heun", "bh1", "bh2"]:
                solver_type = "logrho"
            else:
                raise NotImplementedError(f"solver type {solver_type} does is not implemented for {self.__class__}")

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = paddle.to_tensor(timesteps)
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps
        timesteps = (
            np.linspace(0, self.num_train_timesteps - 1, num_inference_steps + 1)
            .round()[::-1][:-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = paddle.to_tensor(timesteps)
        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0

    def convert_model_output(self, model_output: paddle.Tensor, timestep: int, sample: paddle.Tensor) -> paddle.Tensor:
        """
        Convert the model output to the corresponding type that the algorithm DEIS needs.

        Args:
            model_output (`paddle.Tensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `paddle.Tensor`: the converted model output.
        """
        if self.config.prediction_type == "epsilon":
            alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
            x0_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.config.prediction_type == "sample":
            x0_pred = model_output
        elif self.config.prediction_type == "v_prediction":
            alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
            x0_pred = alpha_t * sample - sigma_t * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction` for the DEISMultistepScheduler."
            )

        if self.config.thresholding:
            # Dynamic thresholding in https://arxiv.org/abs/2205.11487
            orig_dtype = x0_pred.dtype
            if orig_dtype not in [paddle.float32, paddle.float64]:
                x0_pred = x0_pred.cast("float32")
            dynamic_max_val = paddle.quantile(
                paddle.abs(x0_pred).reshape((x0_pred.shape[0], -1)), self.config.dynamic_thresholding_ratio, axis=1
            )
            dynamic_max_val = paddle.maximum(
                dynamic_max_val,
                self.config.sample_max_value * paddle.ones_like(dynamic_max_val),
            )[(...,) + (None,) * (x0_pred.ndim - 1)]
            x0_pred = paddle.clip(x0_pred, -dynamic_max_val, dynamic_max_val) / dynamic_max_val
            x0_pred = x0_pred.cast(orig_dtype)

        if self.config.algorithm_type == "deis":
            alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
            return (sample - alpha_t * x0_pred) / sigma_t
        else:
            raise NotImplementedError("only support log-rho multistep deis now")

    def deis_first_order_update(
        self,
        model_output: paddle.Tensor,
        timestep: int,
        prev_timestep: int,
        sample: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        One step for the first-order DEIS (equivalent to DDIM).

        Args:
            model_output (`paddle.Tensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `paddle.Tensor`: the sample tensor at the previous timestep.
        """
        lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
        alpha_t, alpha_s = self.alpha_t[prev_timestep], self.alpha_t[timestep]
        sigma_t, _ = self.sigma_t[prev_timestep], self.sigma_t[timestep]
        h = lambda_t - lambda_s
        if self.config.algorithm_type == "deis":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (paddle.exp(h) - 1.0)) * model_output
        else:
            raise NotImplementedError("only support log-rho multistep deis now")
        return x_t

    def multistep_deis_second_order_update(
        self,
        model_output_list: List[paddle.Tensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        One step for the second-order multistep DEIS.

        Args:
            model_output_list (`List[paddle.Tensor]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `paddle.Tensor`: the sample tensor at the previous timestep.
        """
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        alpha_t, alpha_s0, alpha_s1 = self.alpha_t[t], self.alpha_t[s0], self.alpha_t[s1]
        sigma_t, sigma_s0, sigma_s1 = self.sigma_t[t], self.sigma_t[s0], self.sigma_t[s1]

        rho_t, rho_s0, rho_s1 = sigma_t / alpha_t, sigma_s0 / alpha_s0, sigma_s1 / alpha_s1

        if self.config.algorithm_type == "deis":

            def ind_fn(t, b, c):
                # Integrate[(log(t) - log(c)) / (log(b) - log(c)), {t}]
                return t * (-paddle.log(c) + paddle.log(t) - 1) / (paddle.log(b) - paddle.log(c))

            coef1 = ind_fn(rho_t, rho_s0, rho_s1) - ind_fn(rho_s0, rho_s0, rho_s1)
            coef2 = ind_fn(rho_t, rho_s1, rho_s0) - ind_fn(rho_s0, rho_s1, rho_s0)

            x_t = alpha_t * (sample / alpha_s0 + coef1 * m0 + coef2 * m1)
            return x_t
        else:
            raise NotImplementedError("only support log-rho multistep deis now")

    def multistep_deis_third_order_update(
        self,
        model_output_list: List[paddle.Tensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        One step for the third-order multistep DEIS.

        Args:
            model_output_list (`List[paddle.Tensor]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `paddle.Tensor`: the sample tensor at the previous timestep.
        """
        t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        alpha_t, alpha_s0, alpha_s1, alpha_s2 = self.alpha_t[t], self.alpha_t[s0], self.alpha_t[s1], self.alpha_t[s2]
        sigma_t, sigma_s0, sigma_s1, simga_s2 = self.sigma_t[t], self.sigma_t[s0], self.sigma_t[s1], self.sigma_t[s2]
        rho_t, rho_s0, rho_s1, rho_s2 = (
            sigma_t / alpha_t,
            sigma_s0 / alpha_s0,
            sigma_s1 / alpha_s1,
            simga_s2 / alpha_s2,
        )

        if self.config.algorithm_type == "deis":

            def ind_fn(t, b, c, d):
                # Integrate[(log(t) - log(c))(log(t) - log(d)) / (log(b) - log(c))(log(b) - log(d)), {t}]
                numerator = t * (
                    paddle.log(c) * (paddle.log(d) - paddle.log(t) + 1)
                    - paddle.log(d) * paddle.log(t)
                    + paddle.log(d)
                    + paddle.log(t) ** 2
                    - 2 * paddle.log(t)
                    + 2
                )
                denominator = (paddle.log(b) - paddle.log(c)) * (paddle.log(b) - paddle.log(d))
                return numerator / denominator

            coef1 = ind_fn(rho_t, rho_s0, rho_s1, rho_s2) - ind_fn(rho_s0, rho_s0, rho_s1, rho_s2)
            coef2 = ind_fn(rho_t, rho_s1, rho_s2, rho_s0) - ind_fn(rho_s0, rho_s1, rho_s2, rho_s0)
            coef3 = ind_fn(rho_t, rho_s2, rho_s0, rho_s1) - ind_fn(rho_s0, rho_s2, rho_s0, rho_s1)

            x_t = alpha_t * (sample / alpha_s0 + coef1 * m0 + coef2 * m1 + coef3 * m2)

            return x_t
        else:
            raise NotImplementedError("only support log-rho multistep deis now")

    def step(
        self,
        model_output: paddle.Tensor,
        timestep: int,
        sample: paddle.Tensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the multistep DEIS.

        Args:
            model_output (`paddle.Tensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = len(self.timesteps) - 1
        else:
            step_index = step_index.item()
        prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1]
        lower_order_final = (
            (step_index == len(self.timesteps) - 1) and self.config.lower_order_final and len(self.timesteps) < 15
        )
        lower_order_second = (
            (step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, timestep, sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.deis_first_order_update(model_output, timestep, prev_timestep, sample)
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            timestep_list = [self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_deis_second_order_update(
                self.model_outputs, timestep_list, prev_timestep, sample
            )
        else:
            timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_deis_third_order_update(
                self.model_outputs, timestep_list, prev_timestep, sample
            )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: paddle.Tensor, *args, **kwargs) -> paddle.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`paddle.Tensor`): input sample

        Returns:
            `paddle.Tensor`: scaled input sample
        """
        return sample

    def add_noise(
        self,
        original_samples: paddle.Tensor,
        noise: paddle.Tensor,
        timesteps: paddle.Tensor,
    ) -> paddle.Tensor:
        # Make sure alphas_cumprod and timestep have same dtype as original_samples
        self.alphas_cumprod = self.alphas_cumprod.cast(original_samples.dtype)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
