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

import math

import paddle
import paddle.nn.functional as F


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


class NoiseScheduleVP:
    def __init__(self, schedule="discrete", beta_0=0.0001, beta_1=0.02, total_N=1000, betas=None, alphas_cumprod=None):
        """Create a wrapper class for the forward SDE (VP type).

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:


        ===============================================================

        We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
        schedule are the default settings in DDPM and improved-DDPM:

            beta_min: A `float` number. The smallest beta for the linear schedule.
            beta_max: A `float` number. The largest beta for the linear schedule.
            cosine_s: A `float` number. The hyperparameter in the cosine schedule.
            cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
            T: A `float` number. The ending time of the forward process.

        Note that the original DDPM (linear schedule) used the discrete-time label (0 to 999). We convert the discrete-time
        label to the continuous-time time (followed Song et al., 2021), so the beta here is 1000x larger than those in DDPM.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE ('linear' or 'cosine').

        Returns:
            A wrapper object of the forward SDE (VP type).
        """
        if schedule not in ["linear", "discrete", "cosine"]:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'linear' or 'cosine'".format(schedule)
            )
        self.total_N = total_N
        self.beta_0 = beta_0 * 1000.0
        self.beta_1 = beta_1 * 1000.0

        if schedule == "discrete":
            if betas is not None:
                log_alphas = 0.5 * paddle.log(1 - betas).cumsum(axis=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * paddle.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.t_discrete = paddle.linspace(1.0 / self.total_N, 1.0, self.total_N).reshape((1, -1))
            self.log_alpha_discrete = log_alphas.reshape((1, -1)).astype("float32")

        self.cosine_s = 0.008
        self.cosine_beta_max = 999.0
        self.cosine_t_max = (
            math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi) * 2.0 * (1.0 + self.cosine_s) / math.pi
            - self.cosine_s
        )
        self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0))
        self.schedule = schedule
        if schedule == "cosine":
            # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
            # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
            self.T = 0.9946
        else:
            self.T = 1.0

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == "linear":
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == "discrete":
            return interpolate_fn(
                t.reshape((-1, 1)), self.t_discrete.clone(), self.log_alpha_discrete.clone()
            ).reshape((-1,))
        elif self.schedule == "cosine":
            log_alpha_fn = lambda s: paddle.log(
                paddle.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0)
            )
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t
        else:
            raise ValueError("Unsupported ")

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
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == "linear":
            tmp = 2.0 * (self.beta_1 - self.beta_0) * logaddexp(-2.0 * lamb, paddle.zeros((1,)))
            Delta = self.beta_0**2 + tmp
            return tmp / (paddle.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        if self.schedule == "discrete":
            log_alpha = -0.5 * logaddexp(paddle.zeros((1,)), -2.0 * lamb)
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                paddle.flip(self.log_alpha_discrete.clone(), [1]),
                paddle.flip(self.t_discrete.clone(), [1]),
            )
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * logaddexp(-2.0 * lamb, paddle.zeros((1,)))
            t_fn = (
                lambda log_alpha_t: paddle.arccos(paddle.exp(log_alpha_t + self.cosine_log_alpha_0))
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            t = t_fn(log_alpha)
            return t


def model_wrapper(
    model,
    noise_schedule=None,
    is_cond_classifier=False,
    classifier_fn=None,
    classifier_scale=1.0,
    time_input_type="1",
    total_N=1000,
    model_kwargs={},
    is_deis=False,
):
    """Create a wrapper function for the noise prediction model.

    DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to
    firstly wrap the model function to a function that accepts the continuous time as the input.

    The input `model` has the following format:

    ``
        model(x, t_input, **model_kwargs) -> noise
    ``

    where `x` and `noise` have the same shape, and `t_input` is the time label of the model.
    (may be discrete-time labels (i.e. 0 to 999) or continuous-time labels (i.e. epsilon to T).)

    We wrap the model function to the following format:

    ``
        def model_fn(x, t_continuous) -> noise:
            t_input = get_model_input_time(t_continuous)
            return model(x, t_input, **model_kwargs)
    ``

    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.

    For DPMs with classifier guidance, we also combine the model output with the classifier gradient as used in [1].

    [1] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis," in Advances in Neural
    Information Processing Systems, vol. 34, 2021, pp. 8780-8794.

    ===============================================================

    Args:
        model: A noise prediction model with the following format:
            ``
                def model(x, t_input, **model_kwargs):
                    return noise
            ``
        noise_schedule: A noise schedule object, such as NoiseScheduleVP. Only used for the classifier guidance.
        is_cond_classifier: A `bool`. Whether to use the classifier guidance.
        classifier_fn: A classifier function. Only used for the classifier guidance. The format is:
            ``
                def classifier_fn(x, t_input):
                    return logits
            ``
        classifier_scale: A `float`. The scale for the classifier guidance.
        time_input_type: A `str`. The type for the time input of the model. We support three types:
            - '0': The continuous-time type. In this case, the model is trained on the continuous time,
                so `t_input` = `t_continuous`.
            - '1': The Type-1 discrete type described in the Appendix of DPM-Solver paper.
                **For discrete-time DPMs, we recommend to use this type for DPM-Solver**.
            - '2': The Type-2 discrete type described in the Appendix of DPM-Solver paper.
        total_N: A `int`. The total number of the discrete-time DPMs (default is 1000), used when `time_input_type`
            is '1' or '2'.
        model_kwargs: A `dict`. A dict for the other inputs of the model function.
    Returns:
        A function that accepts the continuous time as the input, with the following format:
            ``
                def model_fn(x, t_continuous):
                    t_input = get_model_input_time(t_continuous)
                    return model(x, t_input, **model_kwargs)
            ``
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        """
        if time_input_type == "0":
            # discrete_type == '0' means that the model is continuous-time model.
            # For continuous-time DPMs, the continuous time equals to the discrete time.
            return t_continuous
        elif time_input_type == "1":
            # Type-1 discrete label, as detailed in the Appendix of DPM-Solver.
            return 1000.0 * paddle.maximum(t_continuous - 1.0 / total_N, paddle.zeros_like(t_continuous))
        elif time_input_type == "2":
            # Type-2 discrete label, as detailed in the Appendix of DPM-Solver.
            max_N = (total_N - 1) / total_N * 1000.0
            return max_N * t_continuous
        else:
            raise ValueError("Unsupported time input type {}, must be '0' or '1' or '2'".format(time_input_type))

    def cond_fn(x, t_discrete, y):
        """
        Compute the gradient of the classifier, multiplied with the sclae of the classifier guidance.
        """
        assert y is not None
        with paddle.enable_grad():
            x_in = x.detach().stop_gradient = False
            logits = classifier_fn(x_in, t_discrete)
            log_probs = F.log_softmax(logits, axis=-1)
            selected = log_probs[range(len(logits)), y.reshape([-1])]
            return classifier_scale * paddle.autograd.grad(selected.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = paddle.ones((x.shape[0],)) * t_continuous
        if is_cond_classifier:
            y = model_kwargs.get("y", None)
            if y is None:
                raise ValueError("For classifier guidance, the label y has to be in the input.")
            t_discrete = get_model_input_time(t_continuous)
            noise_uncond = model(x, t_discrete, **model_kwargs)
            cond_grad = cond_fn(x, t_discrete, y)
            if is_deis:
                sigma_t = noise_schedule.marginal_std(t_continuous / 1000.0)
            else:
                sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = len(cond_grad.shape) - 1
            return noise_uncond - sigma_t[(...,) + (None,) * dims] * cond_grad
        else:
            t_discrete = get_model_input_time(t_continuous)
            return model(x, t_discrete, **model_kwargs)

    return model_fn


class DPM_Solver:
    def __init__(self, model_fn, noise_schedule, predict_x0=False, thresholding=False, max_val=1.0):
        """Construct a DPM-Solver.

        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input
                (t in [epsilon, T]):
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        """
        self.model = model_fn
        self.noise_schedule = noise_schedule
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.max_val = max_val

    def model_fn(self, x, t):
        if self.predict_x0:
            alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
            noise = self.model(x, t)
            dims = len(x.shape) - 1
            x0 = (x - sigma_t[(...,) + (None,) * dims] * noise) / alpha_t[(...,) + (None,) * dims]
            if self.thresholding:
                p = 0.995
                s = paddle.quantile(paddle.abs(x0).reshape((x0.shape[0], -1)), p, axis=1)
                s = paddle.maximum(s, paddle.ones_like(s))[(...,) + (None,) * dims]
                x0 = paddle.clip(x0, -s, s) / (s / self.max_val)
            return x0
        else:
            return self.model(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps, **recommended for DPM-Solver**.
                - 'time_uniform': uniform time for the time steps. (Used in DDIM and DDPM.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == "time_uniform":
            return paddle.linspace(t_T, t_0, N + 1)
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type)
            )

    def get_time_steps_for_dpm_solver_fast(self, skip_type, t_T, t_0, steps, order):
        """
        Compute the intermediate time steps and the order of each step for sampling by DPM-Solver-fast.

        We recommend DPM-Solver-fast for fast sampling of DPMs. Given a fixed number of function evaluations by `steps`,
        the sampling procedure by DPM-Solver-fast is:
            - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
            - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
            - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            steps: A `int`. The total number of function evaluations (NFE).
        Returns:
            orders: A list of the solver order of each step.
            timesteps: A pytorch tensor of the time steps, with the shape of (K + 1,).
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3] * (K - 1) + [1]
            else:
                orders = [3] * (K - 1) + [2]
            timesteps = self.get_time_steps(skip_type, t_T, t_0, K)
            return orders, timesteps
        elif order == 2:
            K = steps // 2
            if steps % 2 == 0:
                orders = [2] * K
            else:
                orders = [2] * K + [1]
            timesteps = self.get_time_steps(skip_type, t_T, t_0, K)
            return orders, timesteps
        else:
            raise ValueError("order must >= 2")

    def denoise_fn(self, x, s, noise_s=None):
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        sigma_s = ns.marginal_std(s)

        if noise_s is None:
            noise_s = self.model_fn(x, s)
        x_0 = (x - sigma_s[(...,) + (None,) * dims] * noise_s) / paddle.exp(log_alpha_s)[(...,) + (None,) * dims]
        return x_0

    def dpm_solver_first_update(self, x, s, t, noise_s=None, return_noise=False):
        """
        A single step for DPM-Solver-1.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            return_noise: A `bool`. If true, also return the predicted noise at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = paddle.exp(log_alpha_t)

        if self.predict_x0:
            phi_1 = (paddle.exp(-h) - 1.0) / -1.0
            if noise_s is None:
                noise_s = self.model_fn(x, s)
            x_t = (sigma_t / sigma_s)[(...,) + (None,) * dims] * x + (alpha_t * phi_1)[
                (...,) + (None,) * dims
            ] * noise_s
            if return_noise:
                return x_t, {"noise_s": noise_s}
            else:
                return x_t
        else:
            phi_1 = paddle.expm1(h)
            if noise_s is None:
                noise_s = self.model_fn(x, s)
            x_t = (
                paddle.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * dims] * x
                - (sigma_t * phi_1)[(...,) + (None,) * dims] * noise_s
            )
            if return_noise:
                return x_t, {"noise_s": noise_s}
            else:
                return x_t

    def dpm_solver_second_update(self, x, s, t, r1=0.5, noise_s=None, return_noise=False, solver_type="dpm_solver"):
        """
        A single step for DPM-Solver-2.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the second-order solver. We recommend the default setting `0.5`.
            noise_s: A pytorch tensor. The predicted noise at time `s`.
                If `noise_s` is None, we compute the predicted noise by `x` and `s`; otherwise we directly use it.
            return_noise: A `bool`. If true, also return the predicted noise at time `s` and `s1` (the intermediate time).
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = (
            ns.marginal_log_mean_coeff(s),
            ns.marginal_log_mean_coeff(s1),
            ns.marginal_log_mean_coeff(t),
        )
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = paddle.exp(log_alpha_s1), paddle.exp(log_alpha_t)

        if self.predict_x0:
            phi_11 = paddle.expm1(-r1 * h)
            phi_1 = paddle.expm1(-h)

            if noise_s is None:
                noise_s = self.model_fn(x, s)
            x_s1 = (sigma_s1 / sigma_s)[(...,) + (None,) * dims] * x - (alpha_s1 * phi_11)[
                (...,) + (None,) * dims
            ] * noise_s
            noise_s1 = self.model_fn(x_s1, s1)
            if solver_type == "dpm_solver":
                x_t = (
                    (sigma_t / sigma_s)[(...,) + (None,) * dims] * x
                    - (alpha_t * phi_1)[(...,) + (None,) * dims] * noise_s
                    - 0.5 / r1 * (alpha_t * phi_1)[(...,) + (None,) * dims] * (noise_s1 - noise_s)
                )
            elif solver_type == "taylor":
                x_t = (
                    (sigma_t / sigma_s)[(...,) + (None,) * dims] * x
                    - (alpha_t * phi_1)[(...,) + (None,) * dims] * noise_s
                    + 1.0
                    / r1
                    * (alpha_t * ((paddle.exp(-h) - 1.0) / h + 1.0))[(...,) + (None,) * dims]
                    * (noise_s1 - noise_s)
                )
            else:
                raise ValueError("solver_type must be either dpm_solver or taylor, got {}".format(solver_type))
        else:
            phi_11 = paddle.expm1(r1 * h)
            phi_1 = paddle.expm1(h)

            if noise_s is None:
                noise_s = self.model_fn(x, s)
            x_s1 = (
                paddle.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,) * dims] * x
                - (sigma_s1 * phi_11)[(...,) + (None,) * dims] * noise_s
            )
            noise_s1 = self.model_fn(x_s1, s1)
            if solver_type == "dpm_solver":
                x_t = (
                    paddle.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * dims] * x
                    - (sigma_t * phi_1)[(...,) + (None,) * dims] * noise_s
                    - 0.5 / r1 * (sigma_t * phi_1)[(...,) + (None,) * dims] * (noise_s1 - noise_s)
                )
            elif solver_type == "taylor":
                x_t = (
                    paddle.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * dims] * x
                    - (sigma_t * phi_1)[(...,) + (None,) * dims] * noise_s
                    - 1.0
                    / r1
                    * (sigma_t * ((paddle.exp(h) - 1.0) / h - 1.0))[(...,) + (None,) * dims]
                    * (noise_s1 - noise_s)
                )
            else:
                raise ValueError("solver_type must be either dpm_solver or taylor, got {}".format(solver_type))
        if return_noise:
            return x_t, {"noise_s": noise_s, "noise_s1": noise_s1}
        else:
            return x_t

    def dpm_solver_third_update(
        self,
        x,
        s,
        t,
        r1=1.0 / 3.0,
        r2=2.0 / 3.0,
        noise_s=None,
        noise_s1=None,
        noise_s2=None,
        return_noise=False,
        solver_type="dpm_solver",
    ):
        """
        A single step for DPM-Solver-3.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the third-order solver. We recommend the default setting `1 / 3`.
            r2: A `float`. The hyperparameter of the third-order solver. We recommend the default setting `2 / 3`.
            noise_s: A pytorch tensor. The predicted noise at time `s`.
                If `noise_s` is None, we compute the predicted noise by `x` and `s`; otherwise we directly use it.
            noise_s1: A pytorch tensor. The predicted noise at time `s1` (the intermediate time given by `r1`).
                If `noise_s1` is None, we compute the predicted noise by `s1`; otherwise we directly use it.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if r1 is None:
            r1 = 1.0 / 3.0
        if r2 is None:
            r2 = 2.0 / 3.0
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = (
            ns.marginal_log_mean_coeff(s),
            ns.marginal_log_mean_coeff(s1),
            ns.marginal_log_mean_coeff(s2),
            ns.marginal_log_mean_coeff(t),
        )
        sigma_s, sigma_s1, sigma_s2, sigma_t = (
            ns.marginal_std(s),
            ns.marginal_std(s1),
            ns.marginal_std(s2),
            ns.marginal_std(t),
        )
        alpha_s1, alpha_s2, alpha_t = paddle.exp(log_alpha_s1), paddle.exp(log_alpha_s2), paddle.exp(log_alpha_t)

        if self.predict_x0:
            phi_11 = paddle.expm1(-r1 * h)
            phi_12 = paddle.expm1(-r2 * h)
            phi_1 = paddle.expm1(-h)
            phi_22 = paddle.expm1(-r2 * h) / (r2 * h) + 1.0
            phi_2 = phi_1 / h + 1.0
            phi_3 = phi_2 / h - 0.5

            if noise_s is None:
                noise_s = self.model_fn(x, s)
            if noise_s1 is None:
                x_s1 = (sigma_s1 / sigma_s)[(...,) + (None,) * dims] * x - (alpha_s1 * phi_11)[
                    (...,) + (None,) * dims
                ] * noise_s
                noise_s1 = self.model_fn(x_s1, s1)
            if noise_s2 is None:
                x_s2 = (
                    (sigma_s2 / sigma_s)[(...,) + (None,) * dims] * x
                    - (alpha_s2 * phi_12)[(...,) + (None,) * dims] * noise_s
                    + r2 / r1 * (alpha_s2 * phi_22)[(...,) + (None,) * dims] * (noise_s1 - noise_s)
                )
                noise_s2 = self.model_fn(x_s2, s2)
            if solver_type == "dpm_solver":
                x_t = (
                    (sigma_t / sigma_s)[(...,) + (None,) * dims] * x
                    - (alpha_t * phi_1)[(...,) + (None,) * dims] * noise_s
                    + (1.0 / r2) * (alpha_t * phi_2)[(...,) + (None,) * dims] * (noise_s2 - noise_s)
                )
            elif solver_type == "taylor":
                D1_0 = (1.0 / r1) * (noise_s1 - noise_s)
                D1_1 = (1.0 / r2) * (noise_s2 - noise_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (sigma_t / sigma_s)[(...,) + (None,) * dims] * x
                    - (alpha_t * phi_1)[(...,) + (None,) * dims] * noise_s
                    + (alpha_t * phi_2)[(...,) + (None,) * dims] * D1
                    - (alpha_t * phi_3)[(...,) + (None,) * dims] * D2
                )
            else:
                raise ValueError("solver_type must be either dpm_solver or dpm_solver++, got {}".format(solver_type))
        else:
            phi_11 = paddle.expm1(r1 * h)
            phi_12 = paddle.expm1(r2 * h)
            phi_1 = paddle.expm1(h)
            phi_22 = paddle.expm1(r2 * h) / (r2 * h) - 1.0
            phi_2 = phi_1 / h - 1.0
            phi_3 = phi_2 / h - 0.5

            if noise_s is None:
                noise_s = self.model_fn(x, s)
            if noise_s1 is None:
                x_s1 = (
                    paddle.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,) * dims] * x
                    - (sigma_s1 * phi_11)[(...,) + (None,) * dims] * noise_s
                )
                noise_s1 = self.model_fn(x_s1, s1)
            if noise_s2 is None:
                x_s2 = (
                    paddle.exp(log_alpha_s2 - log_alpha_s)[(...,) + (None,) * dims] * x
                    - (sigma_s2 * phi_12)[(...,) + (None,) * dims] * noise_s
                    - r2 / r1 * (sigma_s2 * phi_22)[(...,) + (None,) * dims] * (noise_s1 - noise_s)
                )
                noise_s2 = self.model_fn(x_s2, s2)
            if solver_type == "dpm_solver":
                x_t = (
                    paddle.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * dims] * x
                    - (sigma_t * phi_1)[(...,) + (None,) * dims] * noise_s
                    - 1.0 / r2 * (sigma_t * phi_2)[(...,) + (None,) * dims] * (noise_s2 - noise_s)
                )
            elif solver_type == "taylor":
                D1_0 = (1.0 / r1) * (noise_s1 - noise_s)
                D1_1 = (1.0 / r2) * (noise_s2 - noise_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    paddle.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * dims] * x
                    - (sigma_t * phi_1)[(...,) + (None,) * dims] * noise_s
                    - (sigma_t * phi_2)[(...,) + (None,) * dims] * D1
                    - (sigma_t * phi_3)[(...,) + (None,) * dims] * D2
                )
            else:
                raise ValueError("solver_type must be either dpm_solver or dpm_solver++, got {}".format(solver_type))

        if return_noise:
            return x_t, {"noise_s": noise_s, "noise_s1": noise_s1, "noise_s2": noise_s2}
        else:
            return x_t

    def dpm_solver_update(self, x, s, t, order, return_noise=False, solver_type="dpm_solver", r1=None, r2=None):
        """
        A single step for DPM-Solver of the given order `order`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, return_noise=return_noise)
        elif order == 2:
            return self.dpm_solver_second_update(x, s, t, return_noise=return_noise, solver_type=solver_type, r1=r1)
        elif order == 3:
            return self.dpm_solver_third_update(
                x, s, t, return_noise=return_noise, solver_type=solver_type, r1=r1, r2=r2
            )
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def sample(
        self,
        x,
        steps=10,
        eps=1e-4,
        T=None,
        order=3,
        skip_type="time_uniform",
        denoise=False,
        method="fast",
        solver_type="dpm_solver",
        atol=0.0078,
        rtol=0.05,
    ):
        """
        Compute the sample at time `eps` by DPM-Solver, given the initial `x` at time `T`.

        We support the following algorithms:

            - Adaptive step size DPM-Solver (i.e. DPM-Solver-12 and DPM-Solver-23)

            - Fixed order DPM-Solver (i.e. DPM-Solver-1, DPM-Solver-2 and DPM-Solver-3).

            - Fast version of DPM-Solver (i.e. DPM-Solver-fast), which uses uniform logSNR steps and combine
                different orders of DPM-Solver.

        **We recommend DPM-Solver-fast for both fast sampling in few steps (<=20) and fast convergence in many steps (50 to 100).**

        Choosing the algorithms:

            - If `adaptive_step_size` is True:
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                If `order`=2, we use DPM-Solver-12 which combines DPM-Solver-1 and DPM-Solver-2.
                If `order`=3, we use DPM-Solver-23 which combines DPM-Solver-2 and DPM-Solver-3.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.

            - If `adaptive_step_size` is False and `fast_version` is True:
                We ignore `order` and use DPM-Solver-fast with number of function evaluations (NFE) = `steps`.
                We ignore `skip_type` and use uniform logSNR steps for DPM-Solver-fast.
                Given a fixed NFE=`steps`, the sampling procedure by DPM-Solver-fast is:
                    - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                    - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                    - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

            - If `adaptive_step_size` is False and `fast_version` is False:
                We use DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
                We support three types of `skip_type`:
                    - 'logSNR': uniform logSNR for the time steps, **recommended for DPM-Solver**.
                    - 'time_uniform': uniform time for the time steps. (Used in DDIM and DDPM.)
                    - 'time_quadratic': quadratic time for the time steps. (Used in DDIM.)

        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `T` (a sample from the normal distribution).
            steps: A `int`. The total number of function evaluations (NFE).
            eps: A `float`. The ending time of the sampling.
                We recommend `eps`=1e-3 when `steps` <= 15; and `eps`=1e-4 when `steps` > 15.
            T: A `float`. The starting time of the sampling. Default is `None`.
                If `T` is None, we use self.noise_schedule.T.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. Default is 'logSNR'.
            adaptive_step_size: A `bool`. If true, use the adaptive step size DPM-Solver.
            fast_version: A `bool`. If true, use DPM-Solver-fast (recommended).
            atol: A `float`. The absolute tolerance of the adaptive step size solver.
            rtol: A `float`. The relative tolerance of the adaptive step size solver.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        t_0 = eps
        t_T = self.noise_schedule.T if T is None else T

        if method == "fast":
            orders, _ = self.get_time_steps_for_dpm_solver_fast(
                skip_type=skip_type, t_T=t_T, t_0=t_0, steps=steps, order=order
            )
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps)
            with paddle.no_grad():
                i = 0
                for order in orders:
                    vec_s, vec_t = (
                        paddle.ones((x.shape[0],)) * timesteps[i],
                        paddle.ones((x.shape[0],)) * timesteps[i + order],
                    )
                    h = self.noise_schedule.marginal_lambda(
                        timesteps[i + order]
                    ) - self.noise_schedule.marginal_lambda(timesteps[i])
                    r1 = (
                        None
                        if order <= 1
                        else (
                            self.noise_schedule.marginal_lambda(timesteps[i + 1])
                            - self.noise_schedule.marginal_lambda(timesteps[i])
                        )
                        / h
                    )
                    r2 = (
                        None
                        if order <= 2
                        else (
                            self.noise_schedule.marginal_lambda(timesteps[i + 2])
                            - self.noise_schedule.marginal_lambda(timesteps[i])
                        )
                        / h
                    )
                    x = self.dpm_solver_update(x, vec_s, vec_t, order, solver_type=solver_type, r1=r1, r2=r2)
                    i += order
        if denoise:
            x = self.denoise_fn(x, paddle.ones((x.shape[0],)) * t_0)
        return x
