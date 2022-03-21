# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
"""
Utilities for the Trainer class. 
"""

import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from enum import Enum
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Union[np.ndarray, Tuple[np.ndarray]]


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: Dict[str, float]


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(
            os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(
        folder,
        max(checkpoints,
            key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


class IntervalStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class EvaluationStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """

    ADAMW = "adamw"
    ADAFACTOR = "adafactor"


class BestRun(NamedTuple):
    """
    The best run found by an hyperparameter search (see [`~Trainer.hyperparameter_search`]).

    Parameters:
        run_id (`str`):
            The id of the best run (if models were saved, the corresponding checkpoint will be in the folder ending
            with run-{run_id}).
        objective (`float`):
            The objective that was obtained for this run.
        hyperparameters (`Dict[str, Any]`):
            The hyperparameters picked to get this run.
    """

    run_id: str
    objective: float
    hyperparameters: Dict[str, Any]


def default_compute_objective(metrics: Dict[str, float]) -> float:
    """
    The default objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
    metrics are provided to the [`Trainer`], the sum of all metrics otherwise.

    Args:
        metrics (`Dict[str, float]`): The metrics returned by the evaluate method.

    Return:
        `float`: The objective to minimize or maximize
    """
    metrics = copy.deepcopy(metrics)
    loss = metrics.pop("eval_loss", None)
    _ = metrics.pop("epoch", None)
    # Remove speed metrics
    speed_metrics = [
        m for m in metrics.keys()
        if m.endswith("_runtime") or m.endswith("_per_second")
    ]
    for sm in speed_metrics:
        _ = metrics.pop(sm, None)
    return loss if len(metrics) == 0 else sum(metrics.values())


def is_main_process(local_rank):
    """
    Whether or not the current process is the local process, based on `xm.get_ordinal()` (for TPUs) first, then on
    `local_rank`.
    """

    return local_rank in [-1, 0]


def total_processes_number(local_rank):
    """
    Return the number of processes launched in parallel. Works with `paddle.distributed` and TPUs.
    """
    if local_rank != -1:
        import paddle

        return paddle.distributed.get_world_size()
    return 1


def speed_metrics(split, start_time, num_samples=None, num_steps=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:

    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
    return result


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
