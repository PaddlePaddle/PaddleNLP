# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 TSAIL Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: check https://arxiv.org/abs/2302.04867 and https://github.com/wl-zhao/UniPC for more info
# The codebase is modified based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import paddle

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


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


class UniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    UniPC is a training-free framework designed for the fast sampling of diffusion models, which consists of a
    corrector (UniC) and a predictor (UniP) that share a unified analytical form and support arbitrary orders. UniPC is
    by desinged model-agnostic, supporting pixel-space/latent-space DPMs on unconditional/conditional sampling. It can
    also be applied to both noise prediction model and data prediction model. The corrector UniC can be also applied
    after any off-the-shelf solvers to increase the order of accuracy.

    For more details, see the original paper: https://arxiv.org/abs/2302.04867

    Currently, we support the multistep UniPC for both noise prediction models and data prediction models. We recommend
    to use `solver_order=2` for guided sampling, and `solver_order=3` for unconditional sampling.

    We also support the "dynamic thresholding" method in Imagen (https://arxiv.org/abs/2205.11487). For pixel-space
    diffusion models, you can set both `predict_x0=True` and `thresholding=True` to use the dynamic thresholding. Note
    that the thresholding method is unsuitable for latent-space diffusion models (such as stable-diffusion).

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
            the order of UniPC, also the p in UniPC-p; can be any positive integer. Note that the effective order of
            accuracy is `solver_order + 1` due to the UniC. We recommend to use `solver_order=2` for guided sampling,
            and `solver_order=3` for unconditional sampling.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://imagen.research.google/video/paper.pdf)
        thresholding (`bool`, default `False`):
            whether to use the "dynamic thresholding" method (introduced by Imagen, https://arxiv.org/abs/2205.11487).
            For pixel-space diffusion models, you can set both `predict_x0=True` and `thresholding=True` to use the
            dynamic thresholding. Note that the thresholding method is unsuitable for latent-space diffusion models
            (such as stable-diffusion).
        dynamic_thresholding_ratio (`float`, default `0.995`):
            the ratio for the dynamic thresholding method. Default is `0.995`, the same as Imagen
            (https://arxiv.org/abs/2205.11487).
        sample_max_value (`float`, default `1.0`):
            the threshold value for dynamic thresholding. Valid only when `thresholding=True` and `predict_x0=True`.
        predict_x0 (`bool`, default `True`):
            whether to use the updating algrithm on the predicted x0. See https://arxiv.org/abs/2211.01095 for details
        solver_type (`str`, default `bh2`):
            the solver type of UniPC. We recommend use `bh1` for unconditional sampling when steps < 10, and use `bh2`
            otherwise.
        lower_order_final (`bool`, default `True`):
            whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. We empirically
            find this trick can stabilize the sampling of DPM-Solver for steps < 15, especially for steps <= 10.
        disable_corrector (`list`, default `[]`):
            decide which step to disable the corrector. For large guidance scale, the misalignment between the
            `epsilon_theta(x_t, c)`and `epsilon_theta(x_t^c, c)` might influence the convergence. This can be mitigated
            by disable the corrector at the first few steps (e.g., disable_corrector=[0])
        solver_p (`SchedulerMixin`, default `None`):
            can be any other scheduler. If specified, the algorithm will become solver_p + UniC.
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
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: List[int] = [],
        solver_p: SchedulerMixin = None,
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

        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                solver_type = "bh1"
            else:
                raise NotImplementedError(f"{solver_type} does is not implemented for {self.__class__}")

        self.predict_x0 = predict_x0
        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = paddle.to_tensor(timesteps)
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = disable_corrector
        self.solver_p = solver_p
        self.last_sample = None

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
        self.last_sample = None
        if self.solver_p:
            self.solver_p.set_timesteps(num_inference_steps)

    def convert_model_output(self, model_output: paddle.Tensor, timestep: int, sample: paddle.Tensor) -> paddle.Tensor:
        r"""
        Convert the model output to the corresponding type that the algorithm PC needs.

        Args:
            model_output (`paddle.Tensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `paddle.Tensor`: the converted model output.
        """
        if self.predict_x0:
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
                    " `v_prediction` for the UniPCMultistepScheduler."
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
            return x0_pred
        else:
            if self.config.prediction_type == "epsilon":
                return model_output
            elif self.config.prediction_type == "sample":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = (sample - alpha_t * model_output) / sigma_t
                return epsilon
            elif self.config.prediction_type == "v_prediction":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = alpha_t * model_output + sigma_t * sample
                return epsilon
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the UniPCMultistepScheduler."
                )

    def multistep_uni_p_bh_update(
        self,
        model_output: paddle.Tensor,
        prev_timestep: int,
        sample: paddle.Tensor,
        order: int,
    ) -> paddle.Tensor:
        """
        One step for the UniP (B(h) version). Alternatively, `self.solver_p` is used if is specified.

        Args:
            model_output (`paddle.Tensor`):
                direct outputs from learned diffusion model at the current timestep.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                current instance of sample being created by diffusion process.
            order (`int`): the order of UniP at this step, also the p in UniPC-p.

        Returns:
            `paddle.Tensor`: the sample tensor at the previous timestep.
        """
        timestep_list = self.timestep_list
        model_output_list = self.model_outputs

        s0, t = self.timestep_list[-1], prev_timestep
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            x_t = self.solver_p.step(model_output, s0, x).prev_sample
            return x_t

        lambda_t, lambda_s0 = self.lambda_t[t], self.lambda_t[s0]
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]

        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = timestep_list[-(i + 1)]
            mi = model_output_list[-(i + 1)]
            lambda_si = self.lambda_t[si]
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk.item())
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = paddle.to_tensor(rks)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = paddle.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = paddle.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(paddle.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = paddle.stack(R)
        b = paddle.to_tensor(b)

        if len(D1s) > 0:
            D1s = paddle.stack(D1s, axis=1).cast(paddle.float32)  # (B, K)
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = paddle.to_tensor([0.5])
            else:
                rhos_p = paddle.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                if rhos_p.shape[0] == 2:
                    rhos_p = rhos_p.squeeze(1)
                pred_res = paddle.einsum("k,bkchw->bchw", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                if rhos_p.shape[0] == 2:
                    rhos_p = rhos_p.squeeze(1)
                pred_res = paddle.einsum("k,bkchw->bchw", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        x_t = x_t.cast(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: paddle.Tensor,
        this_timestep: int,
        last_sample: paddle.Tensor,
        this_sample: paddle.Tensor,
        order: int,
    ) -> paddle.Tensor:
        """
        One step for the UniC (B(h) version).

        Args:
            this_model_output (`paddle.Tensor`): the model outputs at `x_t`
            this_timestep (`int`): the current timestep `t`
            last_sample (`paddle.Tensor`): the generated sample before the last predictor: `x_{t-1}`
            this_sample (`paddle.Tensor`): the generated sample after the last predictor: `x_{t}`
            order (`int`): the `p` of UniC-p at this step. Note that the effective order of accuracy
                should be order + 1

        Returns:
            `paddle.Tensor`: the corrected sample tensor at the current timestep.
        """
        timestep_list = self.timestep_list
        model_output_list = self.model_outputs

        s0, t = timestep_list[-1], this_timestep
        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        lambda_t, lambda_s0 = self.lambda_t[t], self.lambda_t[s0]
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]

        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = timestep_list[-(i + 1)]
            mi = model_output_list[-(i + 1)]
            lambda_si = self.lambda_t[si]
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk.item())
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = paddle.to_tensor(rks)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = paddle.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = paddle.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(paddle.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = paddle.stack(R)
        b = paddle.to_tensor(b)

        if len(D1s) > 0:
            # cast this  to float32
            D1s = paddle.stack(D1s, axis=1).cast("float32")
        else:
            D1s = None

        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = paddle.to_tensor([0.5])
        else:
            rhos_c = paddle.linalg.solve(R, b)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = paddle.einsum("k,bkchw->bchw", rhos_c[:-1].squeeze(1), D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = paddle.einsum("k,bkchw->bchw", rhos_c[:-1].squeeze(1), D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        x_t = x_t.cast(x.dtype)
        return x_t

    def step(
        self,
        model_output: paddle.Tensor,
        timestep: int,
        sample: paddle.Tensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the multistep UniPC.

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

        use_corrector = (
            step_index > 0 and step_index - 1 not in self.disable_corrector and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, timestep, sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                this_timestep=timestep,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        # now prepare to run the predictor
        prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1]

        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.config.lower_order_final:
            this_order = min(self.config.solver_order, len(self.timesteps) - step_index)
        else:
            this_order = self.config.solver_order

        self.this_order = min(this_order, self.lower_order_nums + 1)  # warmup for multistep
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,  # pass the original non-converted model output, in case solver-p is used
            prev_timestep=prev_timestep,
            sample=sample,
            order=self.this_order,
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
