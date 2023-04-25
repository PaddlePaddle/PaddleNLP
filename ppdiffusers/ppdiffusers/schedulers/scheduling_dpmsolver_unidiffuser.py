# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


def logaddexp(x, y):
    return paddle.log(1 + paddle.exp(paddle.minimum(x, y) - paddle.maximum(x, y))) + paddle.maximum(x, y)


def interpolate_fn(x: paddle.Tensor, xp: paddle.Tensor, yp: paddle.Tensor) -> paddle.Tensor:
    """Performs piecewise linear interpolation for x, using xp and yp keypoints (knots).
    Performs separate interpolation for each channel.
    Args:
        x: [N, C] points to be calibrated (interpolated). Batch with C channels.
        xp: [C, K] x coordinates of the PWL knots. C is the number of channels, K is the number of knots.
        yp: [C, K] y coordinates of the PWL knots. C is the number of channels, K is the number of knots.
    Returns:
        Interpolated points of the shape [N, C].
    The piecewise linear function extends for the whole x axis (the outermost keypoints define the outermost
    infinite lines).
    For example:
    >>> calibrate1d(paddle.to_tensor([[0.5]]), paddle.to_tensor([[0.0, 1.0]]), paddle.to_tensor([[0.0, 2.0]]))
    tensor([[1.0000]])
    >>> calibrate1d(paddle.to_tensor([[-10]]), paddle.to_tensor([[0.0, 1.0]]), paddle.to_tensor([[0.0, 2.0]]))
    tensor([[-20.0000]])
    """
    x_breakpoints = paddle.concat([x.unsqueeze(2), xp.unsqueeze(0).tile((x.shape[0], 1, 1))], axis=2)
    num_x_points = xp.shape[1]
    sorted_x_breakpoints = paddle.sort(x_breakpoints, axis=2)
    x_indices = paddle.argsort(x_breakpoints, axis=2)
    x_idx = paddle.argmin(x_indices, axis=2)
    cand_start_idx = x_idx - 1
    start_idx = paddle.where(
        paddle.equal(x_idx, 0),
        paddle.to_tensor([1]),
        paddle.where(
            paddle.equal(x_idx, num_x_points),
            paddle.to_tensor([num_x_points - 2]),
            cand_start_idx,
        ),
    )
    end_idx = paddle.where(paddle.equal(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = paddle.take_along_axis(arr=sorted_x_breakpoints, axis=2, indices=start_idx.unsqueeze(axis=2)).squeeze(
        axis=2
    )
    end_x = paddle.take_along_axis(arr=sorted_x_breakpoints, axis=2, indices=end_idx.unsqueeze(axis=2)).squeeze(axis=2)
    start_idx2 = paddle.where(
        paddle.equal(x_idx, 0),
        paddle.to_tensor([0]),
        paddle.where(
            paddle.equal(x_idx, num_x_points),
            paddle.to_tensor([num_x_points - 2]),
            cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand([x.shape[0], -1, -1])
    start_y = paddle.take_along_axis(y_positions_expanded, axis=2, indices=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = paddle.take_along_axis(y_positions_expanded, axis=2, indices=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


class DPMSolverUniDiffuserScheduler(SchedulerMixin, ConfigMixin):
    """
    DPM-Solver (and the improved version DPM-Solver++) is a fast dedicated high-order solver for diffusion ODEs with
    the convergence order guarantee. Empirically, sampling by DPM-Solver with only 20 steps can generate high-quality
    samples, and it can generate quite good samples even in only 10 steps.

    For more details, see the original paper: https://arxiv.org/abs/2206.00927 and https://arxiv.org/abs/2211.01095

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        method (`str`): the update method, Choose from `multistep` or `fast`.
        schedule (`str`): the schedule of NoiseScheduleVP. Default is `discrete`.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Default `scaled_linear`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://imagen.research.google/video/paper.pdf)
        algorithm_type (`str`, default `dpmsolver++`):
            the algorithm type for the solver. Either `dpmsolver` or `dpmsolver++`. The `dpmsolver` type implements the
            algorithms in https://arxiv.org/abs/2206.00927, and the `dpmsolver++` type implements the algorithms in
            https://arxiv.org/abs/2211.01095. We recommend to use `dpmsolver++` with `solver_order=2` for guided
            sampling (e.g. stable-diffusion).
        solver_type (`str`, default `midpoint`):
            the solver type for the second-order solver. Either `midpoint` or `heun`. The solver type slightly affects
            the sample quality, especially for small number of steps. We empirically find that `midpoint` solvers are
            slightly better, so we recommend to use the `midpoint` type.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        method="multistep",
        schedule: str = "discrete",
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
    ):
        if trained_betas is not None:
            self.betas = paddle.to_tensor(trained_betas, dtype=paddle.float32)
        if beta_schedule == "scaled_linear":
            # this schedule is very specific to the unidiffuser model.
            self.betas = (
                paddle.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=paddle.float32) ** 2
            )
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        if schedule == "discrete":
            log_alphas = 0.5 * paddle.log(1 - self.betas).cumsum(axis=0)
            self.total_N = len(log_alphas)
            self.t_discrete = paddle.linspace(1.0 / self.total_N, 1.0, self.total_N).reshape([1, -1])
            self.log_alpha_discrete = log_alphas.reshape((1, -1))
        else:
            raise ValueError

        self.method = method
        self.schedule = schedule
        self.prediction_type = prediction_type
        self.algorithm_type = algorithm_type
        self.solver_type = solver_type

        # settings for DPM-Solver
        if algorithm_type not in ["dpmsolver++"]:
            if algorithm_type == "deis":
                algorithm_type = "dpmsolver++"
            else:
                raise NotImplementedError(f"{algorithm_type} does is not implemented for {self.__class__}")
        if solver_type not in ["midpoint"]:
            if solver_type in ["logrho", "bh1", "bh2"]:
                solver_type = "midpoint"
            else:
                raise NotImplementedError(f"{solver_type} does is not implemented for {self.__class__}")

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0
        self.noise_prev_list = []
        self.t_prev_list = []

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == "discrete":
            return interpolate_fn(
                t.reshape((-1, 1)), self.t_discrete.clone(), self.log_alpha_discrete.clone()
            ).reshape((-1,))
        else:
            raise ValueError

    def marginal_alpha(self, t):
        return paddle.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return paddle.sqrt(1.0 - paddle.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * paddle.log(1.0 - paddle.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        if self.schedule == "discrete":
            log_alpha = -0.5 * logaddexp(paddle.zeros((1,)), -2.0 * lamb)
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                paddle.flip(self.log_alpha_discrete.clone(), [1]),
                paddle.flip(self.t_discrete.clone(), [1]),
            )
            return t.reshape((-1,))
        else:
            raise ValueError

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = paddle.linspace(1.0, 0.001, num_inference_steps + 1)

        self.noise_prev_list = []
        self.t_prev_list = []

    def convert_model_output(self, model_output: paddle.Tensor, timestep: int, sample: paddle.Tensor) -> paddle.Tensor:
        """
        Convert the model output to the corresponding type that the algorithm (DPM-Solver / DPM-Solver++) needs.

        DPM-Solver is designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to
        discretize an integral of the data prediction model. So we need to first convert the model output to the
        corresponding type to match the algorithm.

        Note that the algorithm type and the model type is decoupled. That is to say, we can use either DPM-Solver or
        DPM-Solver++ for both noise prediction model and data prediction model.

        Args:
            model_output (`paddle.Tensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `paddle.Tensor`: the converted model output.
        """
        # DPM-Solver++ needs to solve an integral of the data prediction model.
        alpha_t, sigma_t = self.marginal_alpha(timestep), self.marginal_std(timestep)
        x0_pred = (sample - sigma_t * model_output) / alpha_t
        return x0_pred

    def dpm_solver_first_order_update(
        self,
        model_output: paddle.Tensor,
        timestep: int,
        prev_timestep: int,
        sample: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        One step for the first-order DPM-Solver (equivalent to DDIM).

        See https://arxiv.org/abs/2206.00927 for the detailed derivation.

        Args:
            model_output (`paddle.Tensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `paddle.Tensor`: the sample tensor at the previous timestep.
        """
        lambda_t, lambda_s = self.marginal_lambda(timestep), self.marginal_lambda(prev_timestep)
        alpha_t = self.marginal_log_mean_coeff(timestep)
        sigma_t, sigma_s = self.marginal_std(timestep), self.marginal_std(prev_timestep)

        alpha_t = paddle.exp(alpha_t)
        h = lambda_t - lambda_s
        if self.config.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (paddle.exp(-h) - 1.0)) * model_output
        else:
            raise ValueError
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[paddle.Tensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        One step for the second-order multistep DPM-Solver.

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
        lambda_t, lambda_s0, lambda_s1 = self.marginal_lambda(t), self.marginal_lambda(s0), self.marginal_lambda(s1)
        log_alpha_t = self.marginal_log_mean_coeff(t)
        sigma_t, sigma_s0 = self.marginal_std(t), self.marginal_std(s0)
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        alpha_t = paddle.exp(log_alpha_t)  # Note: diff
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (paddle.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (paddle.exp(-h) - 1.0)) * D1
                )
            else:
                raise ValueError
        return x_t

    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: List[paddle.Tensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        One step for the third-order multistep DPM-Solver.

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
        lambda_t, lambda_s0, lambda_s1, lambda_s2 = (
            self.marginal_lambda(t),
            self.marginal_lambda(s0),
            self.marginal_lambda(s1),
            self.marginal_lambda(s2),
        )
        alpha_t = self.marginal_log_mean_coeff(t)
        alpha_t = paddle.exp(alpha_t)
        sigma_t, sigma_s0 = self.marginal_std(t), self.marginal_std(s0)
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (paddle.exp(-h) - 1.0)) * D0
                + (alpha_t * ((paddle.exp(-h) - 1.0) / h + 1.0)) * D1
                - (alpha_t * ((paddle.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
            )
        else:
            raise ValueError
        return x_t

    def step(
        self,
        model_output: paddle.Tensor,
        timestep: int,
        sample: paddle.Tensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the multistep DPM-Solver.

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

        order = 3
        if self.method == "multistep":
            if step_index == 0:
                vec_t = timestep.expand([sample.shape[0]])
                model_output = self.convert_model_output(model_output, vec_t, sample)
                self.noise_prev_list.append(model_output)
                self.t_prev_list.append(vec_t)

            if step_index > 0 and step_index < order:
                vec_t = timestep.expand([sample.shape[0]])
                sample = self.dpm_multistep_update(sample, self.noise_prev_list, self.t_prev_list, vec_t, step_index)
                model_output = self.convert_model_output(model_output, vec_t, sample)
                self.noise_prev_list.append(model_output)
                self.t_prev_list.append(vec_t)

            if step_index >= order and step_index < len(self.timesteps):
                vec_t = timestep.expand([sample.shape[0]])
                sample = self.dpm_multistep_update(sample, self.noise_prev_list, self.t_prev_list, vec_t, order)
                for i in range(order - 1):
                    self.t_prev_list[i] = self.t_prev_list[i + 1]
                    self.noise_prev_list[i] = self.noise_prev_list[i + 1]
                self.t_prev_list[-1] = vec_t
                if step_index < len(self.timesteps) - 1:
                    self.noise_prev_list[-1] = self.convert_model_output(model_output, vec_t, sample)
        else:
            raise ValueError

        prev_sample = sample

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def dpm_multistep_update(self, x, noise_prev_list, t_prev_list, t, order):
        if order == 1:
            return self.dpm_solver_first_order_update(noise_prev_list[-1], t, t_prev_list[-1], x)
        elif order == 2:
            return self.multistep_dpm_solver_second_order_update(noise_prev_list, t_prev_list, t, x)
        elif order == 3:
            return self.multistep_dpm_solver_third_order_update(noise_prev_list, t_prev_list, t, x)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

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

    def __len__(self):
        return self.config.num_train_timesteps
