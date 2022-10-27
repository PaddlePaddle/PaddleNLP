# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Inc. team.
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

import contextlib
import copy
import os
import random

import numpy as np
import paddle

from .utils import logging

logger = logging.get_logger(__name__)


def enable_full_determinism(seed: int):
    """
    Helper function for reproducible behavior during distributed training.
    """
    # set seed first
    set_seed(seed)

    #  Enable Paddle deterministic mode. This potentially requires either the environment
    #  variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["FLAGS_cudnn_deterministic"] = "True"
    os.environ["FLAGS_benchmark"] = "True"


def set_seed(seed: int = None):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)


class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self,
                 model,
                 update_after_step=0,
                 inv_gamma=1.0,
                 power=2 / 3,
                 min_value=0.0,
                 max_value=0.9999):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        self.averaged_model = copy.deepcopy(model).eval()
        for params in self.averaged_model.parameters():
            params.stop_gradient = True

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma)**-self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @paddle.no_grad()
    def step(self, new_model):
        ema_state_dict = {}
        ema_params = self.averaged_model.state_dict()

        self.decay = self.get_decay(self.optimization_step)

        for key, param in new_model.named_parameters():
            if isinstance(param, dict):
                continue
            try:
                ema_param = ema_params[key]
            except KeyError:
                ema_param = param.astype("float32").clone(
                ) if param.ndim == 1 else copy.deepcopy(param)
                ema_params[key] = ema_param

            if not param.stop_gradient:
                ema_params[key].copy_(param.astype(ema_param.dtype), True)
                ema_param = ema_params[key]
            else:
                ema_param = ema_param.multiply(self.decay)
                ema_param.add_(param.astype(ema_param.dtype) * (1 - self.decay))

            ema_state_dict[key] = ema_param

        for key, param in new_model.named_buffers():
            ema_state_dict[key] = param

        self.averaged_model.load_dict(ema_state_dict)
        self.optimization_step += 1


@contextlib.contextmanager
def main_process_first(desc="work"):
    if paddle.distributed.get_world_size() > 1:
        rank = paddle.distributed.get_rank()
        is_main_process = rank == 0
        main_process_desc = "main local process"

        try:
            if not is_main_process:
                # tell all replicas to wait
                logger.debug(
                    f"{rank}: waiting for the {main_process_desc} to perform {desc}"
                )
                paddle.distributed.barrier()
            yield
        finally:
            if is_main_process:
                # the wait is over
                logger.debug(
                    f"{rank}: {main_process_desc} completed {desc}, releasing all replicas"
                )
                paddle.distributed.barrier()
    else:
        yield
