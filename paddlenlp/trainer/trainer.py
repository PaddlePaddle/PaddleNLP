# Copyright 2020-present the HuggingFace Inc. team.
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#  https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py

import collections
import contextlib
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.amp.auto_cast as autocast
import paddle.distributed as dist
import paddle.nn as nn
from packaging import version
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)
from paddle.io import DataLoader, Dataset, DistributedBatchSampler
from tqdm.auto import tqdm

from ..data import DataCollator, DataCollatorWithPadding, default_data_collator
from ..peft import LoRAModel, PrefixModelForCausalLM
from ..transformers.model_utils import PretrainedModel, _add_variant, unwrap_model
from ..transformers.tokenizer_utils import PretrainedTokenizer
from ..utils import device_guard
from ..utils.batch_sampler import DistributedBatchSampler as NlpDistributedBatchSampler
from ..utils.env import (
    LORA_WEIGHT_FILE_NAME,
    PADDLE_WEIGHT_FILE_NAME,
    PREFIX_WEIGHT_FILE_NAME,
)
from ..utils.import_utils import is_datasets_available
from ..utils.log import logger
from .integrations import get_reporting_integration_callbacks
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_utils import (  # set_hyrbid_parallel_seed,
    PREFIX_CHECKPOINT_DIR,
    EvalLoopOutput,
    EvalPrediction,
    IterableDatasetShard,
    OptimizerNames,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardingOption,
    TrainerMemoryTracker,
    TrainOutput,
    find_batch_size,
    get_last_checkpoint,
    get_scheduler,
    has_length,
    set_seed,
    speed_metrics,
)
from .training_args import TrainingArguments
from .utils.helper import (  # nested_truncate,
    distributed_concat,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
)

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"

OPTIMIZER_NAME = "optimizer.pdopt"
SCHEDULER_NAME = "scheduler.pdparams"
SCALER_NAME = "scaler.pdparams"


if is_datasets_available():
    import datasets


try:
    from paddle.distributed.fleet.utils import mix_precision_utils
except:
    mix_precision_utils = None

try:
    from paddle.io.dataloader.dataloader_iter import _DataLoaderIterBase
except:
    from paddle.fluid.dataloader.dataloader_iter import _DataLoaderIterBase


def paddlenlp_load(path, return_numpy=False):
    if return_numpy:
        with device_guard():
            return paddle.load(path)
    else:
        return paddle.load(path, return_numpy=return_numpy)


def is_dp_group_support_in_group_sharded_parallel():
    return "dp_group" in set(inspect.signature(paddle.distributed.sharding.group_sharded_parallel).parameters.keys())


__all__ = ["Trainer"]


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PaddlePaddle, optimized for PaddleNLP.

    Args:
        model ([`PretrainedModel`] or `paddle.nn.Layer`, *optional*):
            The model to train, evaluate or use for predictions.

            [`Trainer`] is optimized to work with the [`PretrainedModel`] provided by the library. You can still use
            your own models defined as `paddle.nn.Layer` as long as they work the same way as the PaddleNLP
            models.
        criterion(`paddle.nn.Layer`, *optional*):
            The model may only output the loggit, if you want do more computation for the output of model, you can
            add the criterion Layer.
        args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`default_data_collator`] if no `tokenizer` is provided, an instance of
            [`DataCollatorWithPadding`] otherwise.
        train_dataset (`paddle.io.Dataset` or `paddle.io.IterableDataset`, *optional*):
            The dataset to use for training. If it is an `datasets.Dataset`, columns not accepted by the
            `model.forward()` method are automatically removed.
        eval_dataset (Union[`paddle.io.Dataset`, Dict[str, `paddle.io.Dataset`]],  *optional*):
             The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
             `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
             dataset prepending the dictionary key to the metric name.
        tokenizer ([`PretrainedTokenizer`], *optional*):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values.
        callbacks (List of [`TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks.
            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        optimizers (`Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler]`, *optional*): A tuple
            containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your model
            and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        preprocess_logits_for_metrics (`Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PretrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, the inner model is
          wrapped in `paddle.DataParallel`. If model hasn't been wrapped, then `self.model_wrapped` is the same
          as `self.model`.

    """

    from .trainer_utils import log_metrics, metrics_format, save_metrics, save_state

    def __init__(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor] = None,
    ):

        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)

        self.args = args
        self.is_in_train = False
        # self.do_grad_scaling = args.fp16

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # Seed must be set before instantiating the model when using model
        set_seed(args=self.args)

        if model is None:
            raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")

        if self.args.should_save or self.args.should_save_model_state:
            os.makedirs(self.args.output_dir, exist_ok=True)

        self.sharding = None
        if len(args.sharding) > 0:
            if args.local_rank == -1:
                raise ValueError("Using sharding only works in distributed training.")
            self.sharding = True

        # init parallel env
        if paddle.distributed.get_world_size() > 1:
            if self.args.use_hybrid_parallel:
                self.hcg = fleet.get_hybrid_communicate_group()
                self.dp_group = self.hcg.get_data_parallel_group()
                self.sharding_group = self.hcg.get_sharding_parallel_group()

        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)

        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.model_wrapped = model
        self.model = model
        self.criterion = criterion

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers
        # Label smoothing
        # if self.args.label_smoothing_factor != 0:
        #     self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        # else:
        self.label_smoother = None
        self.state = TrainerState()
        self.control = TrainerControl()
        self._signature_columns = None
        self.optimizer_grouped_parameters = None

        if self.sharding is not None and self.optimizer is not None:
            raise RuntimeError(
                "Passing `optimizers` is not allowed if sharding is enabled."
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        if self.args.pipeline_parallel_degree > 1:
            from paddle.distributed.fleet.meta_parallel import PipelineLayer

            assert isinstance(
                model, PipelineLayer
            ), "Only support pipeline parallel mode when model is PipelineLayer!!!"

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        if args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        if train_dataset is not None and not isinstance(train_dataset, collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError("train_dataset does not implement __len__, max_steps has to be specified")

        self.do_grad_scaling = False
        self.enable_autocast_context_manager = False

        if args.fp16 or args.bf16:
            logger.info("Using half precision")
            self.enable_autocast_context_manager = True
            self.do_grad_scaling = True if args.fp16 else False
            self.amp_dtype = "float16" if args.fp16 else "bfloat16"
            # fix for load saved fp16 or bf16 ckpt, decorate model first.
            if self.args.fp16_opt_level == "O2":
                if self.amp_dtype == "bfloat16":
                    # fix for paddlepaddle < 2.4.1, not support for bf16
                    paddle.amp.decorate(models=model, level=self.args.fp16_opt_level, dtype=self.amp_dtype)
                else:
                    paddle.amp.decorate(models=model, level=self.args.fp16_opt_level)
            # for pipeline mode and pure tensor parallel
            if self.args.pipeline_parallel_degree > 1 or (
                self.args.tensor_parallel_degree > 1 and self.sharding is None
            ):
                self.scaler = paddle.amp.GradScaler(init_loss_scaling=self.args.scale_loss)
                if self.args.amp_master_grad:
                    mix_precision_utils.MixPrecisionScaler(self.scaler)  # retun value has no use
                self.scaler = fleet.distributed_scaler(self.scaler)
            elif self.sharding is not None:
                self.scaler = paddle.amp.GradScaler(init_loss_scaling=self.args.scale_loss)
                if self.amp_dtype == "float16" or self.amp_dtype == "bfloat16":
                    if ShardingOption.SHARD_OP in self.args.sharding:
                        self.scaler = fleet.distributed_scaler(self.scaler)
                    else:
                        # scaler for stage2 and stage3
                        if paddle.framework.in_dygraph_mode():
                            from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import (
                                GroupShardedScaler,
                            )

                            self.scaler = GroupShardedScaler(self.scaler)
                        else:
                            from paddle.distributed.fleet.meta_parallel.sharding.sharding_utils import (
                                ShardingScaler,
                            )

                            self.scaler = ShardingScaler(self.scaler)
                else:
                    self.do_grad_scaling = False
                    self.use_cuda_amp = False
                    self.amp_dtype = None

            else:
                self.scaler = paddle.amp.GradScaler(init_loss_scaling=self.args.scale_loss)

        if args.recompute:

            def fn(layer):
                if hasattr(layer, "enable_recompute") and (
                    layer.enable_recompute is False or layer.enable_recompute == 0
                ):
                    layer.enable_recompute = True

            model.apply(fn)

        default_label_names = (
            ["start_positions", "end_positions"]
            if "QusetionAnswering" in type(self.model).__name__ or "UIE" in type(self.model).__name__
            else ["labels"]
        )
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names

        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)
        self.print_config()

        # very last
        self._memory_tracker.stop_and_update_metrics()

    def add_callback(self, callback):
        """
        Add a callback to the current list of [`~TrainerCallback`].

        Args:
           callback (`type` or [`~TrainerCallback`]):
               A [`~TrainerCallback`] class or an instance of a [`~TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        """
        Remove a callback from the current list of [`~TrainerCallback`] and returns it.
        If the callback is not found, returns `None` (and no error is raised).
        Args:
           callback (`type` or [`~TrainerCallback`]):
               A [`~TrainerCallback`] class or an instance of a [`~TrainerCallback`]. In the
               first case, will pop the first member of that class found in the list of callbacks.
        Returns:
            [`~TrainerCallback`]: The callback removed, if found.
        """
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback):
        """
        Remove a callback from the current list of [`~TrainerCallback`].
        Args:
           callback (`type` or [`~TrainerCallback`]):
               A [`~TrainerCallback`] class or an instance of a [`~TrainerCallback`]. In the
               first case, will remove the first member of that class found in the list of callbacks.
        """
        self.callback_handler.remove_callback(callback)

    def load_state_dict_from_checkpoint(self, resume_from_checkpoint=None):
        """load state_dict from_checkpoint, Only load model state dict.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. Only load model state dict.
        """
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")

        if resume_from_checkpoint is not None:
            if isinstance(self.model, LoRAModel):
                weight_name = LORA_WEIGHT_FILE_NAME
            elif isinstance(self.model, PrefixModelForCausalLM):
                weight_name = PREFIX_WEIGHT_FILE_NAME
            else:
                weight_name = PADDLE_WEIGHT_FILE_NAME

            if not os.path.isfile(
                os.path.join(resume_from_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix))
            ):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint} .")

            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = paddle.load(
                os.path.join(resume_from_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix)),
                return_numpy=True,
            )
            # If the model is on the GPU, it still works!
            self._set_state_dict_in_model(state_dict)

            # release memory
            del state_dict

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
        """
        args = self.args
        self.is_in_train = True
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if isinstance(self.model, LoRAModel):
                weight_name = LORA_WEIGHT_FILE_NAME
            elif isinstance(self.model, PrefixModelForCausalLM):
                weight_name = PREFIX_WEIGHT_FILE_NAME
            else:
                weight_name = PADDLE_WEIGHT_FILE_NAME
            if not os.path.isfile(
                os.path.join(resume_from_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix))
            ):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint} .")

            # TODO: Need to load the model state dict on the CPU to avoid an OOM error.
            state_dict = paddle.load(
                os.path.join(resume_from_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix)),
                return_numpy=True,
            )
            # If the model is on the GPU, it still works!
            self._set_state_dict_in_model(state_dict)

            # release memory
            del state_dict

        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.dataset_world_size
        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = len(self.train_dataset)

            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = int(len(self.train_dataset) * args.num_train_epochs)

            if args.minimum_eval_times is not None and args.minimum_eval_times > 0:
                if max_steps // args.eval_steps < args.minimum_eval_times:
                    exp_step = max_steps / args.minimum_eval_times
                    exp_step = max(int(exp_step - exp_step % 10), 10)
                    logger.info("Reset eval step by minimum_eval_times to %d" % exp_step)
                    args.eval_steps = exp_step
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                f"args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}"
            )

        # delay_optimizer_creation = (
        #     self.sharding is not None
        #     and ShardingOption.SHARD_OP in self.args.sharding
        # )
        delay_optimizer_creation = False

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Total num train samples = {num_train_samples}")
        # per_device_trainable_numel = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
        # TODO: Temporary fix since Tensor.numel() not supported in distributed mode
        per_device_trainable_numel = sum(np.prod(p.shape) for p in model.parameters() if not p.stop_gradient)
        logger.info(f"  Number of trainable parameters = {per_device_trainable_numel} (per device)")
        if self.args.use_hybrid_parallel:
            # todo fix for pipeline_parallel_degree
            parts_num = max(self.args.tensor_parallel_degree, 1) * max(self.args.pipeline_parallel_degree, 1)
            if parts_num > 1:
                trainable_numel_tensor = paddle.to_tensor(per_device_trainable_numel, dtype="int64")
                paddle.distributed.all_reduce(trainable_numel_tensor)
                trainable_numel = trainable_numel_tensor.item() // self.args.dataset_world_size
                # the numel is roughly, because the tensor parallel still hold own bias or layer_norm weight without splited
                # so, the trainable numel is a little bigger than real.
                logger.info(f"  Number of trainable parameters = {trainable_numel} (all devices, roughly)")

        start_time = time.time()
        self._globalstep_last_start_time = time.time()
        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")
            if not args.ignore_data_skip:
                if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                    train_dataloader.batch_sampler, NlpDistributedBatchSampler
                ):
                    consumed_samples = (
                        self.state.global_step
                        * args.train_batch_size
                        * args.gradient_accumulation_steps
                        * args.dataset_world_size
                    )
                    train_dataloader.batch_sampler.set_epoch(consumed_samples=consumed_samples)
                    logger.info(f"Set DistributedBatchSampler consumed_samples to {consumed_samples}")

        epoch_iterator = train_dataloader
        # steps_in_epoch = len(epoch_iterator)
        steps_in_epoch = (
            len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
        )

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        self.state.max_steps = int(max_steps)
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        tr_loss = paddle.to_tensor(0.0)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        if self.args.device == "npu" and self.args.flatten_param_grads:
            from .plugins.npu_plugin import npu_accelerate_plugin

            npu_accelerate_plugin(self.optimizer)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                train_dataloader.batch_sampler, DistributedBatchSampler
            ):
                train_dataloader.batch_sampler.set_epoch(epoch)

            step = -1

            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                # for paddlenlp.utils.batch_sampler.DistributedBatchSampler
                # We use consumed_samples to reset the status
                if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                    train_dataloader.batch_sampler, NlpDistributedBatchSampler
                ):
                    if step == 0:
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(steps_trained_in_current_epoch)
                            steps_trained_progress_bar.close()
                            steps_trained_progress_bar = None
                        self._load_rng_state(resume_from_checkpoint)
                    step += steps_trained_in_current_epoch
                elif steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                dp_enabled = (
                    self.args.data_parallel_degree > 1 if self.args.use_hybrid_parallel else args.local_rank != -1
                )
                forbidden_no_sync = False
                # stage2 and stage3 should not no_sync, because the is no DDP wrapper and  no_sync API
                if self.sharding and (ShardingOption.SHARD_OP not in self.args.sharding):
                    forbidden_no_sync = True
                # hybrid_parallel (tp or pp) should not no_sync
                if self.args.use_hybrid_parallel and (
                    self.args.tensor_parallel_degree > 1 or self.args.pipeline_parallel_degree > 1
                ):
                    forbidden_no_sync = True

                availiable_no_sync = dp_enabled and not forbidden_no_sync

                is_no_sync = (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and availiable_no_sync
                    and args._no_sync_in_gradient_accumulation
                ) or (args.recompute and availiable_no_sync)
                # sharding
                # stage1. the same as ddp
                # stage2. manualy collect gradient on dp group

                if is_no_sync:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                tr_loss += tr_loss_step

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Maunally collect gradients when group_sharded_parallel can't accept dp_group
                    # Case 1: Use sharding stage 2/3 with dp
                    # Case 2: Use recompute and dp
                    # local_rank != -1 don't means dp in networks.
                    if self.sharding and ShardingOption.SHARD_OP not in self.args.sharding:
                        if self.args.data_parallel_degree > 1 and not is_dp_group_support_in_group_sharded_parallel():
                            fused_allreduce_gradients(model.parameters(), fleet.get_hybrid_communicate_group())
                            if ShardingOption.FULL_SHARD in self.args.sharding:
                                # Why need sync on parm again ?
                                # TODO: fix this.
                                for p in model.parameters():
                                    if hasattr(p, "bw_storage"):
                                        assert p.grad is None, "This case shouldn't happen."
                                        p.bw_storage.scale_(1.0 / self.dp_group.nranks)
                                        paddle.distributed.all_reduce(p.bw_storage, group=self.dp_group)

                    # Case 2: Use recompute and dp / sharding stage1,
                    # manualy collect gradient for dp.
                    elif args.recompute and availiable_no_sync:
                        fused_allreduce_gradients(list(model.parameters()), None)

                    # pipeline parallel mode,  handle gradient merge here
                    if args.pipeline_parallel_degree > 1 and getattr(model, "_delay_scale_loss", False):
                        for p in model._layers.parameters():
                            if hasattr(p, "main_grad") and p.main_grad is not None:
                                assert p.grad is None
                                p.main_grad = p.main_grad.scale(1.0 / self.args.gradient_accumulation_steps)
                            elif p.grad is not None:
                                p.grad = p.grad.scale(1.0 / self.args.gradient_accumulation_steps)

                    # Optimizer step
                    optimizer_was_run = True
                    if self.do_grad_scaling:
                        scale_before = self.scaler._scale.numpy()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler._scale.numpy()
                        optimizer_was_run = not self.scaler._cache_founf_inf
                        if not optimizer_was_run:
                            logger.warning(
                                f"optimizer not run, scale_before: {scale_before[0]}, scale_after: {scale_after[0]}"
                            )
                    else:
                        self.optimizer.step()

                    if optimizer_was_run:
                        self.lr_scheduler.step()

                    self.optimizer.clear_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch

                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({self.state.max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\nTraining completed. \n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            if args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, LoRAModel):
                weight_name = LORA_WEIGHT_FILE_NAME
            elif isinstance(self.model, PrefixModelForCausalLM):
                weight_name = PREFIX_WEIGHT_FILE_NAME
            else:
                weight_name = PADDLE_WEIGHT_FILE_NAME
            best_model_path = os.path.join(
                self.state.best_model_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix)
            )
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = paddle.load(best_model_path, return_numpy=True)
                # If the model is on the GPU, it still works!
                self._set_state_dict_in_model(state_dict)
            else:
                logger.warning(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)

        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.world_size <= 1:
            return paddle.io.BatchSampler(
                dataset=self.train_dataset,
                shuffle=True,
                batch_size=self.args.per_device_train_batch_size,
                drop_last=self.args.dataloader_drop_last,
            )

        return DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_replicas=self.args.dataset_world_size,
            rank=self.args.dataset_rank,
            drop_last=self.args.dataloader_drop_last,
        )

    def _set_state_dict_in_model(self, state_dict):
        # TODO  @ZHUI paddle need return the results of set_state_dict.
        self.model.set_state_dict(state_dict)

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss.subtract_(tr_loss)

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 8)
            logs["learning_rate"] = float("{0:.3e}".format(self._get_learning_rate()))
            logs["global_step"] = int(self.state.global_step)

            total_train_batch_size = (
                self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.dataset_world_size
            )
            num_steps = self.state.global_step - self._globalstep_last_logged
            logs.update(
                speed_metrics(
                    "interval",
                    self._globalstep_last_start_time,
                    num_samples=total_train_batch_size * num_steps,
                    num_steps=num_steps,
                )
            )

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self._globalstep_last_start_time = time.time()

            self.log(logs, **kwargs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _get_learning_rate(self):
        return self.optimizer.get_lr()

    def get_train_dataloader(self):
        """
        Returns the training [`~paddle.io.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if self._is_iterable_dataset(train_dataset):
            if self.args.dataset_world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.dataset_world_size,
                    process_index=self.args.dataset_rank,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset):
        if self.args.world_size <= 1:
            return paddle.io.BatchSampler(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                drop_last=False,
            )
        else:
            drop_last = False
            if self.args.pipeline_parallel_degree > 1:
                drop_last = True
                logger.warning(
                    "In parallel mode, the bacth_size is strictly checked. set DistributedBatchSampler drop_last=True."
                )

            return DistributedBatchSampler(
                eval_dataset,
                num_replicas=self.args.dataset_world_size,
                rank=self.args.dataset_rank,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                drop_last=drop_last,
            )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~paddle.io.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`paddle.io.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not accepted by
                the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        if self._is_iterable_dataset(eval_dataset):
            if self.args.dataset_world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.dataset_world_size,
                    process_index=self.args.dataset_rank,
                )

            return DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~paddle.io.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`paddle.io.Dataset`, *optional*):
                The test dataset to use. If it is an `datasets.Dataset`, columns not accepted by the `model.forward()`
                method are automatically removed. It must implement `__len__`.
        """
        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")

        if self._is_iterable_dataset(test_dataset):
            if self.args.dataset_world_size > 1:
                test_dataset = IterableDatasetShard(
                    test_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.dataset_world_size,
                    process_index=self.args.dataset_rank,
                )

            return DataLoader(
                test_dataset,
                batch_size=self.args.per_device_eval_batch_size * self.world_size,
                collate_fn=self.data_collator,  # _get_collator_with_removed_columns
                num_workers=self.args.dataloader_num_workers,
            )

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_scheduler(num_training_steps=num_training_steps)
        self.create_optimizer(self.lr_scheduler)

    def create_optimizer(self, lr_scheduler=None):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            if self.optimizer_grouped_parameters is not None:
                params = self.optimizer_grouped_parameters
                apply_decay_param_fun = None
            else:
                params = self.model.parameters()
                decay_parameters = [
                    p.name for n, p in self.model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])
                ]

                def apply_decay_param_fun(x):
                    return x in decay_parameters

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            if hasattr(optimizer_cls, "_create_master_weight") and self.args.fp16_opt_level == "O2":
                optimizer_kwargs["multi_precision"] = True

            if ShardingOption.SHARD_OP in self.args.sharding:
                from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
                    DygraphShardingOptimizer,
                )

                self.optimizer = DygraphShardingOptimizer(
                    hcg=fleet.get_hybrid_communicate_group(),
                    user_defined_strategy=None,
                    params=params,
                    inner_optimizer_class=optimizer_cls,
                    learning_rate=self.lr_scheduler if lr_scheduler is None else lr_scheduler,
                    apply_decay_param_fun=apply_decay_param_fun,
                    weight_decay=self.args.weight_decay,
                    grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm)
                    if self.args.max_grad_norm > 0
                    else None,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(
                    learning_rate=self.lr_scheduler if lr_scheduler is None else lr_scheduler,
                    apply_decay_param_fun=apply_decay_param_fun,
                    parameters=params,
                    weight_decay=self.args.weight_decay,
                    grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm)
                    if self.args.max_grad_norm > 0
                    else None,
                    **optimizer_kwargs,
                )

        return self.optimizer

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        # if use distributed training
        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        checkpoint_rng_state = paddle.load(rng_file, return_numpy=True)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])

        core = paddle.framework.core

        core.default_cpu_generator().manual_seed(checkpoint_rng_state["cpu"])
        if core.is_compiled_with_cuda():
            if not len(checkpoint_rng_state["cuda"]) == core.get_cuda_device_count():
                raise ValueError("Length of gpu state list shoule be equal to the gpu device count")
            for i in range(core.get_cuda_device_count()):
                core.default_cuda_generator(i).manual_seed(checkpoint_rng_state["cuda"][i])

        if self.args.use_hybrid_parallel:
            fleet.meta_parallel.get_rng_state_tracker().set_states_tracker(
                checkpoint_rng_state["hybrid_parallel_rng_state_tracker"]
            )

    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArguments) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`paddlenlp.training_args.TrainingArguments`):
                The training arguments for the training session.

        """
        # optimizer_kwargs = {"lr": args.learning_rate}
        optimizer_kwargs = {}
        adam_kwargs = {
            "beta1": args.adam_beta1,
            "beta2": args.adam_beta2,
            "epsilon": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAMW:
            from paddle.optimizer import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs

    def create_scheduler(self, num_training_steps: int):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        warmup = (
            self.args.warmup_steps if self.args.warmup_steps > 0 else int(self.args.warmup_ratio * num_training_steps)
        )

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                learning_rate=self.args.learning_rate,
                num_warmup_steps=warmup,
                num_training_steps=num_training_steps,
            )

        return self.lr_scheduler

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~paddle.io.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            dataset = dataloader.dataset
            # Special case for IterableDatasetShard, we need to dig deeper
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_train_batch_size

    def _wrap_model(self, model, training=True):

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Note: in paddle.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Mixed precision training
        if training and self.do_grad_scaling:  # self.args.fp16_opt_level=="O2":
            # model, self.optimizer
            if self.amp_dtype == "bfloat16":
                # fix for paddlepaddle < 2.4.1, not support for bf16
                decorated = paddle.amp.decorate(
                    models=model, optimizers=self.optimizer, level=self.args.fp16_opt_level, dtype=self.amp_dtype
                )
            else:
                decorated = paddle.amp.decorate(
                    models=model, optimizers=self.optimizer, level=self.args.fp16_opt_level
                )

            if self.optimizer is None:
                model = decorated
            else:
                model, self.optimizer = decorated

        # Multi-gpu training
        if self.args.world_size > 1 and not self.args.use_hybrid_parallel:
            model = paddle.DataParallel(model)
            # Distributed training (should be after fp16 initialization)

        in_pipeline_parallel_mode = self.args.pipeline_parallel_degree > 1
        in_sharding_parallel_mode = self.sharding is not None
        in_tensor_parallel_model = self.args.tensor_parallel_degree > 1

        # Pipeline mode
        if in_pipeline_parallel_mode:
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use
            # hack for pipeline model mini batch to batch
            # need batter solution @ZHUI
            # make batch_fn compatible for fleet.distributed_model decorate.
            prepare_pipeline_inputs_func = (
                model._prepare_pipeline_inputs_func if hasattr(model, "_prepare_pipeline_inputs_func") else None
            )
            model = fleet.distributed_model(model)
            if prepare_pipeline_inputs_func is not None:
                model._prepare_pipeline_inputs_func = prepare_pipeline_inputs_func
            else:

                def _prepare_pipeline_inputs_func(inputs):
                    first_stage_keys = ["input_ids", "attention_mask", "position_ids"]
                    last_stage_keys = ["labels"]

                    def get_expected_keys(inputs, keys):
                        ret = tuple([inputs.pop(k) for k in keys if k in inputs])
                        if len(ret) == 1:
                            ret = ret[0]
                        return ret

                    if type(inputs) is dict:
                        return [
                            get_expected_keys(inputs, first_stage_keys),
                            get_expected_keys(inputs, last_stage_keys),
                        ]

                    keys = list(inputs[0].keys())
                    inputs_batch = {key: [data.pop(key) for data in inputs] for key in keys}
                    return [
                        get_expected_keys(inputs_batch, first_stage_keys),
                        get_expected_keys(inputs_batch, last_stage_keys),
                    ]

                logger.warning(
                    "Using default prepare pipeline inputs func, only support input_ids and labels as inputs."
                )
                model._prepare_pipeline_inputs_func = _prepare_pipeline_inputs_func

            assert self.optimizer is not None, "Pipeline mode need decorate optimizer, pelease init optimizer."
            if self.args.amp_master_grad:
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)

        # No pipeline mode, sharding only
        if not in_pipeline_parallel_mode and in_sharding_parallel_mode:
            # Sharded DDP!
            if self.args.tensor_parallel_degree > 1:
                hcg = fleet.get_hybrid_communicate_group()
                assert (
                    ShardingOption.SHARD_GRAD_OP in self.args.sharding or ShardingOption.SHARD_OP in self.args.sharding
                ), "Only support tensor parallel + sharding stage1/stage2 hybrid parallel now."
                model = paddle.distributed.fleet.meta_parallel.TensorParallel(model, hcg, strategy=None)

            if ShardingOption.SHARD_OP in self.args.sharding:
                model = fleet.distributed_model(model)
                self.optimizer = fleet.distributed_optimizer(self.optimizer)
            else:
                # sync params (broadcast) buffers in dp group
                if not is_dp_group_support_in_group_sharded_parallel() and self.args.data_parallel_degree > 1:
                    try:
                        from paddle.fluid.dygraph.parallel import sync_params_buffers
                    except ImportError:
                        # fix for new api in paddlepaddle v2.5
                        from paddle.distributed.parallel import sync_params_buffers

                    hcg = fleet.get_hybrid_communicate_group()
                    dp_group = hcg.get_data_parallel_group()
                    sync_params_buffers(model, comm_group=dp_group, src_rank=dp_group.ranks[0])

                cpu_offload = ShardingOption.OFFLOAD in self.args.sharding
                assert self.optimizer is not None, "optimizer is empty!"
                level = None
                if ShardingOption.SHARD_GRAD_OP in self.args.sharding:
                    level = "os_g"
                if ShardingOption.FULL_SHARD in self.args.sharding:
                    level = "p_g_os"

                from paddle.distributed.sharding import group_sharded_parallel

                # add dp_group and exclude_layer params
                # https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/sharding/group_sharded_parallel_cn.html#group-sharded-parallel
                extra_kwargs = {}
                if is_dp_group_support_in_group_sharded_parallel():
                    extra_kwargs["dp_group"] = self.dp_group
                    extra_kwargs["exclude_layer"] = ["GroupNorm"]

                model, optimizer, _ = group_sharded_parallel(
                    model,
                    self.optimizer,
                    level=level,
                    scaler=None,
                    group=self.sharding_group,
                    offload=cpu_offload,
                    **extra_kwargs,
                )
                self.optimizer = optimizer

        # pure tesnor parallel mode, no pipeline_parallel, no sharding.
        if not in_pipeline_parallel_mode and not in_sharding_parallel_mode and in_tensor_parallel_model:
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use

            model = fleet.distributed_model(model)
            assert self.optimizer is not None, "Tensor parallel mode need decorate optimizer, pelease init optimizer."
            if self.args.amp_master_grad:
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)

        return model

    def _prepare_input(self, data: Union[paddle.Tensor, Any]) -> Union[paddle.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, paddle.Tensor):
            # kwargs = dict(device=self.args.current_device)
            # update data type for pure fp16
            if data.place.is_cuda_pinned_place():
                return data.cuda()
            return data
            # return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> Dict[str, Union[paddle.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def autocast_smart_context_manager(self):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        """
        if self.enable_autocast_context_manager:
            custom_black_list = ["reduce_sum", "c_softmax_with_cross_entropy"]
            custom_white_list = []
            if self.args.fp16_opt_level == "O2":
                # https://github.com/PaddlePaddle/Paddle/blob/eb97f4f0adca40b16a309b927e480178beb8ae96/python/paddle/amp/amp_lists.py#L85-L86
                # the lookup_table is in black_list, but in O2, we need it return fp16
                custom_white_list.extend(["lookup_table", "lookup_table_v2"])

            if self.args.bf16 and self.args.fp16_opt_level == "O2":
                # c_embedding not support bf16 yet
                custom_black_list.append("c_embedding")

            ctx_manager = autocast(
                True,
                custom_black_list=custom_black_list,
                custom_white_list=custom_white_list,
                level=self.args.fp16_opt_level,
                dtype=self.amp_dtype,
            )
        else:
            ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()

        return ctx_manager

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.criterion is not None:
            if "labels" in inputs:
                labels = inputs.pop("labels")
            elif "start_positions" in inputs and "end_positions" in inputs:
                labels = (inputs.pop("start_positions"), inputs.pop("end_positions"))
            elif self.args.label_names is not None:
                labels = []
                for label in self.label_names:
                    labels.append(inputs.pop(label))
                labels = tuple(labels)
            elif "generator_labels" in inputs:
                labels = inputs["generator_labels"]
        else:
            labels = None

        outputs = model(**inputs)

        if self.criterion is not None:
            loss = self.criterion(outputs, labels)
            outputs = (loss, outputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to train.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `paddle.Tensor`: The tensor with training loss on this batch.
        """
        if self.args.pipeline_parallel_degree > 1:
            return self.training_pipeline_step(model, inputs)

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach()

    def training_pipeline_step(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to train.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `paddle.Tensor`: The tensor with training loss on this batch.
        """
        # accumulation data
        if not hasattr(self, "_pp_data_buffer"):
            self._pp_data_buffer = []
        self._pp_data_buffer.append(inputs)
        if len(self._pp_data_buffer) != self.args.gradient_accumulation_steps:
            return paddle.zeros([])

        for v in self._pp_data_buffer[0].values():
            assert isinstance(v, paddle.Tensor), f"Only support tensor as pipeline mode input, got type {type(v)}"

        inputs = model._prepare_pipeline_inputs_func(self._pp_data_buffer)
        self._pp_data_buffer = []

        # hack _prepare_training, remove additional optimizer or scheduler check
        # https://github.com/PaddlePaddle/Paddle/blob/4695122492eee3cc9e9c585e33429c0f98dbdbb0/python/paddle/distributed/fleet/meta_parallel/pipeline_parallel.py#L241

        def _prepare_training(self, data):
            from paddle import framework

            # reset the virtual pp rank for each run
            self.set_virtual_pipeline_rank(0)
            assert framework._dygraph_tracer()._has_grad, "Please enable the generation of gradients."
            if self.is_pipeline_first_stage(ignore_virtual=True) or self.is_pipeline_last_stage(ignore_virtual=True):
                assert data is not None, "For the first and the last stage, the data must be set."
            else:
                data = None

            self._layers.train()

            return data

        model.train()

        # hack pipeline-layers
        # since the pipeline layer will check input is valid every iter.
        # in same case,  for example, batch size warmup, we need dynamic change gradient_accumulation_steps to implement.
        config_backup = model.micro_batch_size, model.accumulate_steps
        model.micro_batch_size = self.args.per_device_train_batch_size
        model.accumulate_steps = self.args.gradient_accumulation_steps

        inputs = _prepare_training(model, inputs)

        with self.autocast_smart_context_manager():
            loss = model.forward_backward_pipeline(inputs, self.scaler if self.do_grad_scaling else None)

        model.micro_batch_size, model.accumulate_steps = config_backup

        return loss.detach()

    def save_model(self, output_dir: Optional[str] = None, merge_tensor_parallel: Optional[bool] = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if self.args.should_save_model_state:
            self._save(output_dir=output_dir, merge_tensor_parallel=merge_tensor_parallel)

    def _save_checkpoint(self, model, metrics=None):
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self.args.output_dir

        output_dir = os.path.join(run_dir, checkpoint_folder)

        self.save_model(output_dir)

        optimizer_name = _add_variant(OPTIMIZER_NAME, self.args.optimizer_name_suffix)

        if self.args.use_hybrid_parallel:
            if self.dp_group.rank <= 0:
                os.makedirs(output_dir, exist_ok=True)
                paddle.save(
                    self.optimizer.state_dict(),
                    os.path.join(output_dir, optimizer_name),
                )

        if self.args.should_save:
            if not self.args.use_hybrid_parallel:
                paddle.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

            # FIXME: manybe only save one copy
            paddle.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

            if self.do_grad_scaling:
                paddle.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": [k.current_seed() for k in paddle.get_cuda_rng_state()],
            "cpu": paddle.framework.core.default_cpu_generator().get_state().current_seed(),
        }
        if self.args.use_hybrid_parallel:
            rng_states[
                "hybrid_parallel_rng_state_tracker"
            ] = fleet.meta_parallel.get_rng_state_tracker().get_states_tracker()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size > 1:
            # use global process_index to save
            process_index = self.args.process_index
            paddle.save(rng_states, os.path.join(output_dir, f"rng_state_{process_index}.pth"))
        else:
            paddle.save(rng_states, os.path.join(output_dir, "rng_state.pth"))

        # Maybe delete some older checkpoints.
        if self.args.should_save and (True if not self.args.use_hybrid_parallel else self.args.local_rank == 0):
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def set_optimizer_grouped_parameters(self, optimizer_grouped_parameters=None):
        """
        set optimizer grouped parameters:

        you can set optimizer_grouped_parameters with whatever argments on whatever parameters to train.
        """
        self.optimizer_grouped_parameters = optimizer_grouped_parameters

    def disable_autocast_context_manager(self):
        """
        For pure fp16 or pure bf16 training, the paddle.amp.autocast is annoy for always cast fp32 to fp16.
        if you networks cast fp16 to fp32 manually to get higher precision, autocast make it not work, since it cast fp32 to fp16 back.

        """
        assert self.args.fp16_opt_level == "O2", "disable_autocast_context_manager should only work for pure fp16/bf16"
        self.enable_autocast_context_manager = False

    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if self.state.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint)

    def _save(self, output_dir: Optional[str] = None, state_dict=None, merge_tensor_parallel=False):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        merge_tensor_parallel = merge_tensor_parallel and self.args.use_hybrid_parallel

        if (
            not isinstance(self.model, PretrainedModel)
            and not isinstance(self.model, LoRAModel)
            and not isinstance(self.model, PrefixModelForCausalLM)
        ):
            if isinstance(unwrap_model(self.model), PretrainedModel):
                unwrap_model(self.model).save_pretrained(
                    output_dir,
                    merge_tensor_parallel=merge_tensor_parallel,
                    variant=self.args.weight_name_suffix,
                    is_main_process=self.args.should_save,
                )
            else:
                logger.info("Trainer.model is not a `PretrainedModel`, only saving its state dict.")
                if merge_tensor_parallel:
                    logger.warning("Trainer.model is not a `PretrainedModel`, not suppor for merge_tensor_parallel.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                paddle.save(
                    state_dict,
                    os.path.join(output_dir, _add_variant(PADDLE_WEIGHT_FILE_NAME, self.args.weight_name_suffix)),
                )
        else:
            self.model.save_pretrained(
                output_dir,
                merge_tensor_parallel=merge_tensor_parallel,
                variant=self.args.weight_name_suffix,
                is_main_process=self.args.should_save,
            )

        if self.args.should_save:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            paddle.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        optimizer_name = _add_variant(OPTIMIZER_NAME, self.args.optimizer_name_suffix)

        if os.path.isfile(os.path.join(checkpoint, optimizer_name)) and os.path.isfile(
            os.path.join(checkpoint, SCHEDULER_NAME)
        ):
            # Load in optimizer and scheduler states
            self.optimizer.set_state_dict(paddlenlp_load(os.path.join(checkpoint, optimizer_name), return_numpy=True))

            self.lr_scheduler.set_state_dict(paddle.load(os.path.join(checkpoint, SCHEDULER_NAME)))
            if self.do_grad_scaling and os.path.isfile(os.path.join(checkpoint, SCALER_NAME)):
                self.scaler.load_state_dict(paddle.load(os.path.join(checkpoint, SCALER_NAME), return_numpy=True))

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 4)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.dataset_world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_eval_iters: Optional[int] = -1,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        if self.args.pipeline_parallel_degree > 1:
            # Only accept wrapped model for pipeline_parallel mode
            model = self.model_wrapped
        else:
            model = self.model

        if isinstance(dataloader, paddle.io.DataLoader):
            batch_size = dataloader.batch_sampler.batch_size
        elif isinstance(dataloader, _DataLoaderIterBase):
            # support for inner dataloader
            batch_size = dataloader._batch_sampler.batch_size
            # alias for inner dataloader
            dataloader.dataset = dataloader._dataset
        else:
            raise ValueError("Only support for paddle.io.DataLoader")

        num_samples = None
        if max_eval_iters > 0:
            # on eval limit steps
            num_samples = batch_size * self.args.dataset_world_size * max_eval_iters
            if isinstance(dataloader, _DataLoaderIterBase) and isinstance(
                dataloader._batch_sampler, NlpDistributedBatchSampler
            ):
                consumed_samples = (
                    ((self.state.global_step) // args.eval_steps)
                    * max_eval_iters
                    * args.per_device_eval_batch_size
                    * args.dataset_world_size
                )
                dataloader._batch_sampler.set_epoch(consumed_samples=consumed_samples)

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            if max_eval_iters > 0:
                logger.info(f"  Total prediction steps = {max_eval_iters}")
            else:
                logger.info(f"  Total prediction steps = {len(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
            if max_eval_iters > 0:
                logger.info(f"  Total prediction steps = {max_eval_iters}")

        logger.info(f"  Pre device batch size = {batch_size}")
        logger.info(f"  Total Batch size = {batch_size * self.args.dataset_world_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        losses = []
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                # losses = self._nested_gather(loss.repeat(batch_size))
                losses = self._nested_gather(paddle.tile(loss, repeat_times=[batch_size, 1]))
                losses_host = losses if losses_host is None else paddle.concat((losses_host, losses), axis=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)

                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

            if max_eval_iters > 0 and step >= max_eval_iters - 1:
                break

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if num_samples is not None:
            pass
        elif has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        model.train()

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.
        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)
        <Tip>
        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.
        </Tip>
        Returns: *NamedTuple* A namedtuple with the following keys:
            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.per_device_eval_batch_size * self.args.dataset_world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def prediction_pipeline_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        """
        prediction_step function for pipeline parallel mode.
        """
        if hasattr(model, "_prepare_pipeline_inputs_func"):
            inputs, labels = model._prepare_pipeline_inputs_func(inputs)
            has_labels = labels is not None
        else:
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            inputs = self._prepare_inputs(inputs)
            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            if has_labels:
                labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None
            inputs = inputs.pop("input_ids")

        with paddle.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    loss = model.eval_batch([inputs, labels], compute_loss=True)
                    # loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
            else:
                raise ValueError("pipeline mode eval need label!")

        return (loss, None, labels)

    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to evaluate.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        if self.args.pipeline_parallel_degree > 1:
            # hack for pipeline mode
            inputs = self._prepare_inputs(inputs)
            return self.prediction_pipeline_step(model, inputs, prediction_loss_only, ignore_keys)

        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with paddle.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if isinstance(logits, (list, tuple)) and len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        return self.args.process_index == 0

    def _nested_gather(self, tensors):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        if self.args.local_rank != -1:
            tensors = distributed_concat(tensors)
        return tensors

        # Copied from Accelerate.

    def _pad_across_processes(self, tensor, pad_index=-100):
        """
        Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
        they can safely be gathered.
        """
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(self._pad_across_processes(t, pad_index=pad_index) for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({k: self._pad_across_processes(v, pad_index=pad_index) for k, v in tensor.items()})
        elif not isinstance(tensor, paddle.Tensor):
            raise TypeError(
                f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
            )

        if len(tensor.shape) < 2:
            return tensor
        # Gather all sizes
        size = paddle.to_tensor(tensor.shape)[None]
        sizes = self._nested_gather(size).cpu()

        max_size = max(s[1] for s in sizes)
        if tensor.shape[1] == max_size:
            return tensor

        # Then pad to the maximum size
        old_size = tensor.shape
        new_size = list(old_size)
        new_size[1] = max_size
        # new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
        new_tensor = paddle.zeros(tuple(new_size), dtype=tensor.dtype) + pad_index
        new_tensor[:, : old_size[1]] = tensor
        return new_tensor

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids", "labels", "start_positions", "end_positions"]

        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                f" you can safely ignore this message."
            )

        columns = [k for k in self._signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def _is_iterable_dataset(self, dataset):
        return isinstance(dataset, paddle.io.IterableDataset)

    def print_config(self, args=None, key=""):
        """
        print config values
        """
        logger.info("=" * 60)
        if args is None:
            args = self.args
            key = "Training"

        logger.info("{:^40}".format("{} Configuration Arguments".format(key)))
        logger.info("{:30}: {}".format("paddle commit id", paddle.version.commit))

        for a in dir(args):
            if a[:2] != "__":  # don't print double underscore methods
                v = getattr(args, a)
                if not isinstance(v, types.MethodType):
                    logger.info("{:30}: {}".format(a, v))

        logger.info("")
