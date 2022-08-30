# Copyright 2020-present the HuggingFace Inc. team.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

# This file is modified from
#  https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py

import contextlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
import types
from typing import Any, Dict, List, Optional

import paddle

from ..utils.log import logger
from .trainer_utils import (
    SchedulerType,
    IntervalStrategy,
    OptimizerNames,
)

__all__ = [
    "default_logdir",
    "TrainingArguments",
]


def default_logdir() -> str:
    """
    Same default
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
    itself**.

    Using [`PdArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        output_dir (`str`):
            The output directory where the model predictions and checkpoints will be written.
        overwrite_output_dir (`bool`, *optional*, defaults to `False`):
            If `True`, overwrite the content of the output directory. Use this to continue training if `output_dir`
            points to a checkpoint directory.
        do_train (`bool`, *optional*, defaults to `False`):
            Whether to run training or not. This argument is not directly used by [`Trainer`], it's intended to be used
            by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples) for more details.
        do_eval (`bool`, *optional*):
            Whether to run evaluation on the validation set or not. Will be set to `True` if `evaluation_strategy` is
            different from `"no"`. This argument is not directly used by [`Trainer`], it's intended to be used by your
            training/evaluation scripts instead. See the [example
            scripts](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples) for more details.
        do_predict (`bool`, *optional*, defaults to `False`):
            Whether to run predictions on the test set or not. This argument is not directly used by [`Trainer`], it's
            intended to be used by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples) for more details.
        do_export (`bool`, *optional*, defaults to `False`):
            Whether to export inference model or not. This argument is not directly used by [`Trainer`], it's
            intended to be used by your training/evaluation scripts instead.
        evaluation_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
            The evaluation strategy to adopt during training. Possible values are:

                - `"no"`: No evaluation is done during training.
                - `"steps"`: Evaluation is done (and logged) every `eval_steps`.
                - `"epoch"`: Evaluation is done at the end of each epoch.

        prediction_loss_only (`bool`, *optional*, defaults to `False`):
            When performing evaluation and generating predictions, only returns the loss.
        per_device_train_batch_size (`int`, *optional*, defaults to 8):
            The batch size per GPU core/CPU for training.
        per_device_eval_batch_size (`int`, *optional*, defaults to 8):
            The batch size per GPU core/CPU for evaluation.
        gradient_accumulation_steps (`int`, *optional*, defaults to 1):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

            <Tip warning={true}>

            When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging,
            evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training examples.

            </Tip>

        learning_rate (`float`, *optional*, defaults to 5e-5):
            The initial learning rate for [`AdamW`] optimizer.
        weight_decay (`float`, *optional*, defaults to 0):
            The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`]
            optimizer.
        adam_beta1 (`float`, *optional*, defaults to 0.9):
            The beta1 hyperparameter for the [`AdamW`] optimizer.
        adam_beta2 (`float`, *optional*, defaults to 0.999):
            The beta2 hyperparameter for the [`AdamW`] optimizer.
        adam_epsilon (`float`, *optional*, defaults to 1e-8):
            The epsilon hyperparameter for the [`AdamW`] optimizer.
        max_grad_norm (`float`, *optional*, defaults to 1.0):
            Maximum gradient norm (for gradient clipping).
        num_train_epochs(`float`, *optional*, defaults to 3.0):
            Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
            the last epoch before stopping training).
        max_steps (`int`, *optional*, defaults to -1):
            If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
            In case of using a finite iterable dataset the training may stop before reaching the set number of steps
            when all data is exhausted
        lr_scheduler_type (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
            The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values.
        warmup_ratio (`float`, *optional*, defaults to 0.0):
            Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
        warmup_steps (`int`, *optional*, defaults to 0):
            Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of `warmup_ratio`.
        log_on_each_node (`bool`, *optional*, defaults to `True`):
            In multinode distributed training, whether to log using `log_level` once per node, or only on the main
            node.
        logging_dir (`str`, *optional*):
            log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
        logging_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The logging strategy to adopt during training. Possible values are:

                - `"no"`: No logging is done during training.
                - `"epoch"`: Logging is done at the end of each epoch.
                - `"steps"`: Logging is done every `logging_steps`.

        logging_first_step (`bool`, *optional*, defaults to `False`):
            Whether to log and evaluate the first `global_step` or not.
        logging_steps (`int`, *optional*, defaults to 500):
            Number of update steps between two logs if `logging_strategy="steps"`.
        save_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The checkpoint save strategy to adopt during training. Possible values are:

                - `"no"`: No save is done during training.
                - `"epoch"`: Save is done at the end of each epoch.
                - `"steps"`: Save is done every `save_steps`.
        save_steps (`int`, *optional*, defaults to 500):
            Number of updates steps before two checkpoint saves if `save_strategy="steps"`.
        save_total_limit (`int`, *optional*):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            `output_dir`.
        save_on_each_node (`bool`, *optional*, defaults to `False`):
            When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on
            the main one.

            This should not be activated when the different nodes use the same storage as the files will be saved with
            the same names for each node.
        no_cuda (`bool`, *optional*, defaults to `False`):
            Whether to not use CUDA even when it is available or not.
        seed (`int`, *optional*, defaults to 42):
            Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the
            [`~Trainer.model_init`] function to instantiate the model if it has some randomly initialized parameters.
        fp16 (`bool`, *optional*, defaults to `False`):
            Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
        fp16_opt_level (`str`, *optional*, defaults to 'O1'):
            For `fp16` training,  AMP optimization level selected in ['O0', 'O1', 'O2']. See details at 
            https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amp/auto_cast_cn.html
        scale_loss (`float`,  *optional*, defaults to 32768):
            The value of initial scale_loss for fp16. (default: 32768)
        local_rank (`int`, *optional*, defaults to -1):
            Rank of the process during distributed training.
        dataloader_drop_last (`bool`, *optional*, defaults to `False`):
            Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
            or not.
        eval_steps (`int`, *optional*):
            Number of update steps between two evaluations if `evaluation_strategy="steps"`. Will default to the same
            value as `logging_steps` if not set.
        dataloader_num_workers (`int`, *optional*, defaults to 0):
            Number of subprocesses to use for data loading. 0 means that the data will be loaded in the
            main process.
        past_index (`int`, *optional*, defaults to -1):
            Some models like TransformerXL or XLNet can make use of the past hidden states for their predictions. 
            If this argument is set to a positive int, the `Trainer` will use the corresponding output (usually index 2) as
            the past state and feed it to the model at the next training step under the keyword argument `mems`.
        run_name (`str`, *optional*):
            A descriptor for the run. Typically used for logging.
        disable_tqdm (`bool`, *optional*):
            Whether or not to disable the tqdm progress bars and table of metrics. Will default to `True` if the logging
            level is set to warn or lower (default), `False` otherwise.
        remove_unused_columns (`bool`, *optional*, defaults to `True`):
            If using `datasets.Dataset` datasets, whether or not to automatically remove the columns unused by the
            model forward method.
        label_names (`List[str]`, *optional*):
            The list of keys in your dictionary of inputs that correspond to the labels.
            Will eventually default to `["labels"]` except if the model used is one of the `XxxForQuestionAnswering` in
            which case it will default to `["start_positions", "end_positions"]`.
        load_best_model_at_end (`bool`, *optional*, defaults to `False`):
            Whether or not to load the best model found during training at the end of training.

            <Tip>

            When set to `True`, the parameters `save_strategy` needs to be the same as `eval_strategy`, and in the case
            it is "steps", `save_steps` must be a round multiple of `eval_steps`.

            </Tip>

        metric_for_best_model (`str`, *optional*):
            Use in conjunction with `load_best_model_at_end` to specify the metric to use to compare two different
            models. Must be the name of a metric returned by the evaluation with or without the prefix `"eval_"`. Will
            default to `"loss"` if unspecified and `load_best_model_at_end=True` (to use the evaluation loss).

            If you set this value, `greater_is_better` will default to `True`. Don't forget to set it to `False` if
            your metric is better when lower.
        greater_is_better (`bool`, *optional*):
            Use in conjunction with `load_best_model_at_end` and `metric_for_best_model` to specify if better models
            should have a greater metric or not. Will default to:

            - `True` if `metric_for_best_model` is set to a value that isn't `"loss"` or `"eval_loss"`.
            - `False` if `metric_for_best_model` is not set, or set to `"loss"` or `"eval_loss"`.
        ignore_data_skip (`bool`, *optional*, defaults to `False`):
            When resuming training, whether or not to skip the epochs and batches to get the data loading at the same
            stage as in the previous training. If set to `True`, the training will begin faster (as that skipping step
            can take a long time) but will not yield the same results as the interrupted training would have.
        optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw"`):
            The optimizer to use: adamw, or adafactor.
        length_column_name (`str`, *optional*, defaults to `"length"`):
            Column name for precomputed lengths. If the column exists, grouping by length will use these values rather
            than computing them on train startup. Ignored unless `group_by_length` is `True` and the dataset is an
            instance of `Dataset`.
        report_to (`str` or `List[str]`, *optional*, defaults to `"visualdl"`):
            The list of integrations to report the results and logs to. Supported platforms is `"visualdl"`.
            `"none"` for no integrations.
        resume_from_checkpoint (`str`, *optional*):
            The path to a folder with a valid checkpoint for your model. This argument is not directly used by
            [`Trainer`], it's intended to be used by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples) for more details.
    """

    output_dir: str = field(metadata={
        "help":
        "The output directory where the model predictions and checkpoints will be written."
    }, )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help":
            ("Overwrite the content of the output directory. "
             "Use this to continue training if output_dir points to a checkpoint directory."
             )
        },
    )

    do_train: bool = field(default=False,
                           metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(
        default=False,
        metadata={"help": "Whether to run predictions on the test set."})
    do_export: bool = field(
        default=False, metadata={"help": "Whether to export infernece model."})
    evaluation_strategy: IntervalStrategy = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={
            "help":
            "When performing evaluation and predictions, only returns the loss."
        },
    )

    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU core/CPU for training."})
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU core/CPU for evaluation."})

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help":
            "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )

    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9,
                              metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999,
                              metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0,
                                 metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={
            "help":
            "If > 0: set total number of training steps to perform. Override num_train_epochs."
        },
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help":
            "The scheduler type to use. suppor linear, cosine, constant, constant_with_warmup"
        },
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Linear warmup over warmup_ratio fraction of total steps."
        })
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."})

    log_on_each_node: bool = field(
        default=True,
        metadata={
            "help":
            "When doing a multinode distributed training, whether to log once per node or just once on the main node."
        },
    )
    logging_dir: Optional[str] = field(default=None,
                                       metadata={"help": "VisualDL log dir."})
    logging_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_first_step: bool = field(
        default=False, metadata={"help": "Log the first global_step"})
    logging_steps: int = field(default=500,
                               metadata={"help": "Log every X updates steps."})

    save_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help":
            ("Limit the total amount of checkpoints. "
             "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
             )
        },
    )
    save_on_each_node: bool = field(
        default=False,
        metadata={
            "help":
            "When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on the main one"
        },
    )
    no_cuda: bool = field(
        default=False,
        metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(
        default=42,
        metadata={
            "help": "Random seed that will be set at the beginning of training."
        })

    fp16: bool = field(
        default=False,
        metadata={
            "help": "Whether to use fp16 (mixed) precision instead of 32-bit"
        },
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help":
            ("For fp16: AMP optimization level selected in ['O0', 'O1', and 'O2']. "
             "See details at https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amp/auto_cast_cn.html"
             )
        },
    )

    scale_loss: float = field(
        default=2**15,
        metadata={"help": "The value of initial scale_loss for fp16."})

    minimum_eval_times: int = field(
        default=None,
        metadata={
            "help":
            "If under eval_steps, the valid time is less then minimum_eval_times, the config of override eval_steps."
        })

    local_rank: int = field(
        default=-1, metadata={"help": "For distributed training: local_rank"})

    dataloader_drop_last: bool = field(
        default=False,
        metadata={
            "help":
            "Drop the last incomplete batch if it is not divisible by the batch size."
        })
    eval_steps: int = field(
        default=None, metadata={"help": "Run an evaluation every X steps."})
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help":
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        },
    )

    past_index: int = field(
        default=-1,
        metadata={
            "help":
            "If >=0, uses the corresponding part of the output as the past state for next step."
        },
    )

    run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run."})

    device: Optional[str] = field(
        default="gpu", metadata={"help": "select cpu, gpu, xpu devices."})

    disable_tqdm: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether or not to disable the tqdm progress bars."})

    remove_unused_columns: Optional[bool] = field(
        default=True,
        metadata={
            "help":
            "Remove columns not required by the model when using an nlp.Dataset."
        })

    label_names: Optional[List[str]] = field(
        default=None,
        metadata={
            "help":
            "The list of keys in your dictionary of inputs that correspond to the labels."
        })

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={
            "help":
            "Whether or not to load the best model found during training at the end of training."
        },
    )
    metric_for_best_model: Optional[str] = field(
        default=None,
        metadata={"help": "The metric to use to compare two different models."})
    greater_is_better: Optional[bool] = field(
        default=None,
        metadata={
            "help":
            "Whether the `metric_for_best_model` should be maximized or not."
        })
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help":
            "When resuming training, whether or not to skip the first epochs and batches to get to the same training data."
        },
    )
    optim: str = field(
        default="adamw",
        metadata={"help": "The optimizer to use."},
    )
    report_to: Optional[List[str]] = field(
        default=None,
        metadata={
            "help":
            "The list of integrations to report the results and logs to."
        })
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The path to a folder with a valid checkpoint for your model."
        },
    )

    def __post_init__(self):
        env_local_rank = int(os.environ.get("PADDLE_RANK_IN_NODE", -1))
        if env_local_rank != -1 and env_local_rank != self.local_rank and paddle.distributed.get_world_size(
        ) > 1:
            self.local_rank = env_local_rank

        # convert to int
        self.log_level = -1
        self.log_level_replica = -1

        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = False  # logger.getEffectiveLevel() > logging.WARN

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True

        if self.do_eval and self.evaluation_strategy == IntervalStrategy.NO:
            logger.warning(
                "evaluation_strategy reset to IntervalStrategy.STEPS for do_eval is True. you can also set evaluation_strategy='epoch'."
            )
            self.evaluation_strategy = IntervalStrategy.STEPS

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.evaluation_strategy == IntervalStrategy.STEPS and (
                self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(
                    f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}"
                )
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.evaluation_strategy} requires either non-zero --eval_steps or --logging_steps"
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(
                f"logging strategy {self.logging_strategy} requires non-zero --logging_steps"
            )

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.evaluation_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.evaluation_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                raise ValueError(
                    "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation "
                    f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                )

        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in [
                "loss", "eval_loss"
            ]
        if self.run_name is None:
            self.run_name = self.output_dir

        self.optim = OptimizerNames(self.optim)

        if self.report_to is None:
            logger.info(
                "The default value for the training argument `--report_to` will change in v5 (from all installed "
                "integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as "
                "now. You should start updating your code and make this info disappear :-)."
            )
            self.report_to = "all"
        if self.report_to == "all" or self.report_to == ["all"]:
            # Import at runtime to avoid a circular import.
            from .integrations import get_available_reporting_integrations

            self.report_to = get_available_reporting_integrations()
        elif self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio during training"
            )

    def __str__(self):
        self_as_dict = asdict(self)
        self_as_dict = {
            k: f"<{k.upper()}>" if k.endswith("_token") else v
            for k, v in self_as_dict.items()
        }

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training.
        """
        train_batch_size = self.per_device_train_batch_size
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation.
        """
        eval_batch_size = self.per_device_eval_batch_size
        return eval_batch_size

    @property
    def current_device(self) -> "paddle.device":
        """
        The device used by this process.
        """
        return paddle.device.get_device()

    @property
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        if self.local_rank != -1:
            return paddle.distributed.get_world_size()
        return 1

    @property
    def process_index(self):
        """
        The index of the current process used.
        """
        if self.local_rank != -1:
            return paddle.distributed.get_rank()
        return 0

    @property
    def local_process_index(self):
        """
        The index of the local process used.
        """
        if self.local_rank != -1:
            return self.local_rank
        return 0

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        if self.log_on_each_node:
            return self.local_process_index == 0
        else:
            return self.process_index == 0

    @property
    def should_save(self):
        """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.
        """
        if self.save_on_each_node:
            return self.local_process_index == 0
        else:
            return self.process_index == 0

    @property
    def _no_sync_in_gradient_accumulation(self):
        """
        Whether or not to use no_sync for the gradients when doing gradient accumulation.
        """
        return True

    @contextlib.contextmanager
    def main_process_first(self, local=True, desc="work"):
        """
        A context manager for paddle distributed environment where on needs to do something on the main process, while
        blocking replicas, and when it's finished releasing the replicas.

        One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
        which upon completion saves a cached version of results and which then automatically gets loaded by the
        replicas.

        Args:
            local (`bool`, *optional*, defaults to `True`):
                if `True` first means process of rank 0 of each node if `False` first means process of rank 0 of node
                rank 0 In multi-node environment with a shared filesystem you most likely will want to use
                `local=False` so that only the main process of the first node will do the processing. If however, the
                filesystem is not shared, then the main process of each node will need to do the processing, which is
                the default behavior.
            desc (`str`, *optional*, defaults to `"work"`):
                a work description to be used in debug logs

        """
        if self.world_size > 1:
            if local:
                is_main_process = self.local_process_index == 0
                main_process_desc = "main local process"
            else:
                is_main_process = self.process_index == 0
                main_process_desc = "main process"

            try:
                if not is_main_process:
                    # tell all replicas to wait
                    logger.debug(
                        f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}"
                    )
                    paddle.distributed.barrier()
                yield
            finally:
                if is_main_process:
                    # the wait is over
                    logger.debug(
                        f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas"
                    )
                    paddle.distributed.barrier()
        else:
            yield

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (self.warmup_steps if self.warmup_steps > 0 else
                        math.ceil(num_training_steps * self.warmup_ratio))
        return warmup_steps

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization
        """
        d = self.to_dict()
        d = {
            **d,
            **{
                "train_batch_size": self.train_batch_size,
                "eval_batch_size": self.eval_batch_size
            }
        }

        valid_types = [bool, int, float, str]
        valid_types.append(paddle.Tensor)

        return {
            k: v if type(v) in valid_types else str(v)
            for k, v in d.items()
        }

    def print_config(self, args=None, key=""):
        """
        print all config values.
        """
        logger.info("=" * 60)
        if args is None:
            args = self
            key = "Training"

        logger.info('{:^40}'.format("{} Configuration Arguments".format(key)))
        logger.info('{:30}:{}'.format("paddle commit id",
                                      paddle.version.commit))

        for a in dir(args):
            if a[:2] != "__":  #don't print double underscore methods
                v = getattr(args, a)
                if not isinstance(v, types.MethodType):
                    logger.info('{:30}:{}'.format(a, v))

        logger.info("")
