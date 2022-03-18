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
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from paddle.io import DataLoader, DistributedBatchSampler

import numpy as np

from trainer_args import TrainingArguments
# from trainer_callback import TrainerState, TrainerControl

from trainer_utils import (
    IntervalStrategy,
    EvaluationStrategy,
    EvalPrediction,
    PredictionOutput,
    EvalLoopOutput,
    speed_metrics, )

from trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState, )

DEFAULT_CALLBACKS = [DefaultFlowCallback]

from utils import logging

from paddlenlp.transformers.model_utils import PretrainedModel, unwrap_model
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer

# logger = logging.get_logger(__name__)

from paddle.io import Dataset


class DataCollator:
    pass


class DataCollatorWithPadding:
    def __init__(self, *args, **kwargs):
        pass


def paddle_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return paddle.concat((tensor1, tensor2), axis=0)

    raise ValueError("pass")
    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(
        tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[:tensor1.shape[0], :tensor1.shape[1]] = tensor1
    result[tensor1.shape[0]:, :tensor2.shape[1]] = tensor2
    return result


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(
            t, n, padding_index=padding_index)
                             for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, paddle.Tensor):
        return paddle_pad_and_concatenate(
            tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(
            tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(
            f"Unsupported type for concatenation: got {type(tensors)}")


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pdparams"
SCHEDULER_NAME = "scheduler.pdparams"
SCALER_NAME = "scaler.pdparams"

PREFIX_CHECKPOINT_DIR = "training"


def set_seed(seed):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(seed)
    np.random.seed(seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(seed)


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

    Args:
        model ([`PretrainedModel`] or `paddle.nn.Module`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.

            <Tip>

            [`Trainer`] is optimized to work with the [`PretrainedModel`] provided by the library. You can still use
            your own models defined as `paddle.nn.Module` as long as they work the same way as the 🤗 Transformers
            models.

            </Tip>

        args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`default_data_collator`] if no `tokenizer` is provided, an instance of
            [`DataCollatorWithPadding`] otherwise.
        train_dataset (`paddle.utils.data.Dataset` or `paddle.utils.data.IterableDataset`, *optional*):
            The dataset to use for training. If it is an `datasets.Dataset`, columns not accepted by the
            `model.forward()` method are automatically removed.

            Note that if it's a `paddle.utils.data.IterableDataset` with some randomization and you are training in a
            distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
            `paddle.Generator` for the randomization that must be identical on all processes (and the Trainer will
            manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
            sets the seed of the RNGs used.
        eval_dataset (`paddle.utils.data.Dataset`, *optional*):
             The dataset to use for evaluation. If it is an `datasets.Dataset`, columns not accepted by the
             `model.forward()` method are automatically removed.
        tokenizer ([`PretrainedTokenizer`], *optional*):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values.
        optimizers (`Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler]`, *optional*): A tuple
            containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your model
            and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PretrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
          the inner model is wrapped in `DeepSpeed` and then again in `paddle.nn.DistributedDataParallel`. If the inner
          model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to `False` if model parallel or deepspeed is used, or if the default
          `TrainingArguments.place_model_on_device` is overridden to return `False` .
        - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
          in `train`)

    """

    def __init__(
            self,
            model: Union[PretrainedModel, nn.Layer]=None,
            criterion: Union[nn.Layer]=None,
            args: TrainingArguments=None,
            data_collator: Optional[DataCollator]=None,
            train_dataset: Optional[Dataset]=None,
            eval_dataset: Optional[Dataset]=None,
            tokenizer: Optional[PretrainedTokenizer]=None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]]=None,
            optimizers: Tuple[paddle.optimizer.Optimizer,
                              paddle.optimizer.lr.LRScheduler]=(None, None), ):
        logger.info("init!!!!")

        if args is None:
            output_dir = "tmp_trainer"
            logger.info(
                f"No `TrainingArguments` passed, using `output_dir={output_dir}`."
            )
            args = TrainingArguments(output_dir=output_dir)
        args.world_size = 1
        args.fp16 = args.use_amp
        args.do_grad_scaling = args.use_amp
        self.do_grad_scaling = args.do_grad_scaling
        args.train_batch_size = args.batch_size
        args.eval_batch_size = args.batch_size

        args.dataloader_drop_last = True
        args.dataloader_num_workers = 0
        args.dataloader_pin_memory = True
        args.n_gpu = 1
        args.lr_scheduler_type = "linear"
        args.adam_beta1 = 0.9
        args.adam_beta2 = 0.999
        args.optim = "OptimizerNames.ADAMW"
        args.past_index = -1
        args.per_device_train_batch_size = args.batch_size
        args.per_device_eval_batch_size = args.batch_size
        args.logging_first_step = True
        args.logging_strategy = IntervalStrategy.STEPS
        args.evaluation_strategy = IntervalStrategy.STEPS
        args.save_strategy = IntervalStrategy.STEPS
        args.eval_steps = 500
        args.save_steps = 500
        args.label_names = None
        args.prediction_loss_only = False
        args.output_dir = "./out"
        args.should_save = True
        args.local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
        args.save_total_limit = 3
        args.metric_for_best_model = "accuracy"
        args.greater_is_better = True

        self.args = args
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)
        if model is None:
            raise RuntimeError(
                "`Trainer` requires either a `model` or `model_init` argument")

        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(
            tokenizer)

        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.model_wrapped = model
        self.model = model
        self.criterion = criterion

        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers

        self.state = TrainerState()
        self.control = TrainerControl()
        callbacks = DEFAULT_CALLBACKS
        self.callback_handler = CallbackHandler(callbacks, self.model,
                                                self.tokenizer, self.optimizer,
                                                self.lr_scheduler)

        self.add_callback(ProgressCallback)

        if args.max_steps > 0:
            logger.info(
                "max_steps is given, it will override any value given in num_train_epochs"
            )

        if train_dataset is not None and not isinstance(
                train_dataset, collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError(
                "train_dataset does not implement __len__, max_steps has to be specified"
            )

        if args.fp16:
            logger.info(f"Using  half precision")

        default_label_names = (["start_positions", "end_positions"] if
                               "QusetionAnswering" in type(self.model).__name__
                               else ["labels"])
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names

    def add_callback(self, callback):
        """
        Add a callback to the current list of [`~transformer.TrainerCallback`].

        Args:
           callback (`type` or [`~transformer.TrainerCallback`]):
               A [`~transformer.TrainerCallback`] class or an instance of a [`~transformer.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]]=None,
            ignore_keys_for_eval: Optional[List[str]]=None,
            **kwargs, ):
        print("training!!!!")
        logger.info("training!!!!")
        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model_wrapped)

        self.state = TrainerState()

        if self.args.max_steps > 0:
            self.args.num_training_steps = self.args.max_steps
            self.args.num_train_epochs = math.ceil(
                self.args.num_training_steps / len(train_dataloader))

        else:
            self.args.num_training_steps = len(
                train_dataloader) * self.args.num_train_epochs
            self.args.num_train_epochs = self.args.num_train_epochs

        if self.args.num_training_steps // self.args.valid_steps < self.args.minimum_valid_times:
            exp_step = self.args.num_training_steps / self.args.minimum_valid_times
            exp_step = max(int(exp_step - exp_step % 10), 10)
            logger.info("Set eval step to %d" % exp_step)
            self.args.valid_steps = exp_step

        args = self.args

        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_training_steps)

        num_examples = len(self.train_dataset)
        total_train_batch_size = self.args.per_device_train_batch_size * paddle.distributed.get_world_size(
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
        )
        logger.info(f"  Gradient Accumulation steps = {1}")
        logger.info(
            f"  Total optimization steps = {self.args.num_training_steps}")

        self.state.epoch = 0
        self.state.max_steps = int(self.args.num_training_steps)
        self.state.num_train_epochs = int(self.args.num_train_epochs)
        self.state.is_local_process_zero = 0
        self.state.is_world_process_zero = 0

        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        self.training_bar = tqdm(total=self.state.max_steps)

        epoch_iterator = train_dataloader
        steps_in_epoch = len(epoch_iterator)

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        self.control = self.callback_handler.on_train_begin(args, self.state,
                                                            self.control)

        tr_loss = paddle.to_tensor(0.0)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        for epoch in range(epochs_trained, args.num_train_epochs):
            step = -1

            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                # print(inputs)
                # print("=="*20)
                tr_loss_step = self.training_step(model, inputs)
                # self.scaler.step(self.optimizer)
                # self.scaler.update()
                tr_loss += tr_loss_step
                self.training_bar.update(1)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.clear_grad()

                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch

                self.control = self.callback_handler.on_step_end(
                    args, self.state, self.control)

                self._maybe_log_save_evaluate(tr_loss, model, epoch,
                                              ignore_keys_for_eval)

    def training_step(
            self, model: nn.Layer,
            inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        loss.backward()

        return loss.detach()

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        if self.args.world_size <= 1:
            # return RandomSampler(self.train_dataset)
            return DistributedBatchSampler(
                self.train_dataset,
                # num_replicas=self.args.world_size,
                # rank=self.args.process_index,
                batch_size=self.args.batch_size,
                shuffle=True,
                # seed=self.args.seed,
            )
        else:
            return DistributedBatchSampler(
                self.train_dataset,
                # num_replicas=self.args.world_size,
                # rank=self.args.process_index,
                # seed=self.args.seed,
            )

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch,
                                 ignore_keys_for_eval):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            # tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss_scalar = tr_loss.mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (
                self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state,
                                                         self.control)

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

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            # batch_size=self.args.train_batch_size,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            # drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            # pin_memory=self.args.dataloader_pin_memory,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset):
        if self.args.world_size <= 1:
            return DistributedBatchSampler(
                eval_dataset,
                # num_replicas=self.args.world_size,
                # rank=self.args.process_index,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                # seed=self.args.seed,
            )
        else:
            return DistributedBatchSampler(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False)

    def get_eval_dataloader(self,
                            eval_dataset: Optional[Dataset]=None) -> DataLoader:
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

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            # batch_size=self.args.train_batch_size,
            batch_sampler=eval_sampler,
            collate_fn=self.data_collator,
            # drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            # pin_memory=self.args.dataloader_pin_memory,
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

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory, )

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
            decay_parameters = [
                p.name for n, p in self.model.named_parameters()
                if not any(nd in n for nd in ["bias", "norm"])
            ]
            apply_decay_param_fun = lambda x: x in decay_parameters

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args)

            self.optimizer = optimizer_cls(
                learning_rate=self.lr_scheduler
                if lr_scheduler is None else lr_scheduler,
                apply_decay_param_fun=apply_decay_param_fun,
                parameters=self.model.parameters(),
                weight_decay=self.args.weight_decay,
                grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm),
                **optimizer_kwargs)

        return self.optimizer

    @staticmethod
    def get_optimizer_cls_and_kwargs(
            args: TrainingArguments) -> Tuple[Any, Any]:
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
        if args.optim == "OptimizerNames.ADAMW":
            from paddle.optimizer import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        else:
            raise ValueError(
                f"Trainer cannot instantiate unsupported optimizer: {args.optim}"
            )
        return optimizer_cls, optimizer_kwargs

    def create_scheduler(self,
                         num_training_steps: int,
                         optimizer: paddle.optimizer.Optimizer=None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """

        def get_scheduler(lr_scheduler_type, learning_rate, num_warmup_steps,
                          num_training_steps):
            # TODO  @ZHUI support others
            return LinearDecayWithWarmup(learning_rate, num_training_steps,
                                         num_warmup_steps)

        warmup = self.args.warmup_steps if self.args.warmup_steps > 0 else self.args.warmup_proportion

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                learning_rate=self.args.learning_rate,
                num_warmup_steps=warmup,
                num_training_steps=num_training_steps, )

        return self.lr_scheduler

    def _wrap_model(self, model, training=True):
        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        if self.args.n_gpu > 1:
            model = nn.DistributedDataParallel(model)

        # Note: in paddle.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        return model

    def _prepare_input(
            self, data: Union[paddle.Tensor, Any]) -> Union[paddle.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)(
                {k: self._prepare_input(v)
                 for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, paddle.Tensor):
            kwargs = dict(device=self.args.device)
            # update data type for pure fp16
            return data
            # return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[paddle.Tensor, Any]]
                        ) -> Dict[str, Union[paddle.Tensor, Any]]:
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
        if self.args.use_amp:
            ctx_manager = autocast()
        else:
            ctx_manager = contextlib.nullcontext() if sys.version_info >= (
                3, 7) else contextlib.suppress()

        return ctx_manager

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.criterion is not None:
            labels = inputs.pop("labels")
        else:
            labels = None

        # print(inputs)

        outputs = model(**inputs)

        # outputs = model(*inputs)

        if self.criterion is not None:
            # print(outputs)
            loss = self.criterion(outputs, labels)
            outputs = (loss, outputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(
            self, model: nn.Layer,
            inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `paddle.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean(
            )  # mean() to average on multi-gpu parallel training

        # if self.args.gradient_accumulation_steps > 1:
        #     # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
        #     loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        # print(loss)
        return loss.detach()

    def save_model(self, output_dir: Optional[str]=None):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if self.args.should_save:
            self._save(output_dir)

    def _save_checkpoint(self, model, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self.args.output_dir

        output_dir = os.path.join(run_dir, checkpoint_folder)

        self.save_model(output_dir)

        if self.args.should_save:
            # deepspeed.save_checkpoint above saves model/optim/sched
            paddle.save(self.optimizer.state_dict(),
                        os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                paddle.save(self.lr_scheduler.state_dict(),
                            os.path.join(output_dir, SCHEDULER_NAME))
            if self.do_grad_scaling:
                paddle.save(self.scaler.state_dict(),
                            os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (self.state.best_metric is None or
                    self.state.best_model_checkpoint is None or
                    operator(metric_value, self.state.best_metric)):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(
                os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)
        local_rank = self.args.local_rank

        if local_rank == -1:
            paddle.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            paddle.save(rng_states,
                        os.path.join(output_dir, f"rng_state_{local_rank}.pth"))

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _sorted_checkpoints(self,
                            output_dir=None,
                            checkpoint_prefix=PREFIX_CHECKPOINT_DIR,
                            use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [
            str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")
        ]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append(
                    (os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append(
                        (int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [
            checkpoint[1] for checkpoint in checkpoints_sorted
        ]
        # Make sure we don't delete the best model.
        if self.state.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(
                str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[
                    i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (self.state.best_model_checkpoint is not None and
                self.args.save_total_limit == 1 and
                checkpoints_sorted[-1] != self.state.best_model_checkpoint):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(
            0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:
                                                       number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(
                f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
            )
            shutil.rmtree(checkpoint)

    def _save(self, output_dir: Optional[str]=None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PretrainedModel):
            if isinstance(unwrap_model(self.model), PretrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict)
            else:
                logger.info(
                    "Trainer.model is not a `PretrainedModel`, only saving its state dict."
                )
                if state_dict is None:
                    state_dict = self.model.state_dict()
                paddle.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        paddle.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if os.path.isfile(os.path.join(
                checkpoint, OPTIMIZER_NAME)) and os.path.isfile(
                    os.path.join(checkpoint, SCHEDULER_NAME)):
            # Load in optimizer and scheduler states
            map_location = self.args.device
            self.optimizer.load_state_dict(
                paddle.load(
                    os.path.join(checkpoint, OPTIMIZER_NAME),
                    map_location=map_location))
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(
                    paddle.load(os.path.join(checkpoint, SCHEDULER_NAME)))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling and os.path.isfile(
                    os.path.join(checkpoint, SCALER_NAME)):
                self.scaler.load_state_dict(
                    paddle.load(os.path.join(checkpoint, SCALER_NAME)))

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = { ** logs, ** {"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state,
                                                    self.control, logs)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset]=None,
            ignore_keys: Optional[List[str]]=None,
            metric_key_prefix: str="eval", ) -> Dict[str, float]:
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
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix, )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size), ))

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics)

        return output.metrics

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool]=None,
            ignore_keys: Optional[List[str]]=None,
            metric_key_prefix: str="eval", ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        prediction_loss_only = False

        model = self._wrap_model(self.model, training=False)

        batch_size = dataloader.batch_size
        num_samples = self.num_examples(dataloader)
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {num_samples}")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
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
            # Prediction step
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            losses.append(loss.numpy())

            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)

        model.train()

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(
                EvalPrediction(
                    predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        metrics["eval_loss"] = float(np.mean(losses))

        print(metrics)

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples)

    def predict(self,
                test_dataset: Dataset,
                ignore_keys: Optional[List[str]]=None,
                metric_key_prefix: str="test") -> PredictionOutput:
        pass

    def prediction_step(
            self,
            model: nn.Layer,
            inputs: Dict[str, Union[paddle.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]]=None, ) -> Tuple[Optional[
                paddle.Tensor], Optional[paddle.Tensor], Optional[
                    paddle.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config,
                                      "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(
                tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with paddle.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items()
                                   if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items()
                                   if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset.

        Will raise an exception if the underlying dataset does not implement method `__len__`
        """
        return len(dataloader.dataset)

    def create_dataloader(self,
                          dataset,
                          mode='train',
                          batch_size=16,
                          batchify_fn=None,
                          trans_fn=None,
                          batched=False):
        """
        """
        if trans_fn:
            dataset = dataset.map(trans_fn, batched=batched)

        shuffle = True if mode == 'train' else False
        if mode == 'train':
            batch_sampler = paddle.io.DistributedBatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            batch_sampler = paddle.io.BatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle)

        return paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)

    def eval(self, *args, **kwargs):
        """
        """
        pass

    def prepare_train_config(self):
        """
        """
        if self.args.max_steps > 0:
            self.args.num_training_steps = self.args.max_steps
            self.args.num_train_epochs = math.ceil(
                self.args.num_training_steps / len(self.train_dl))

        else:
            self.args.num_training_steps = len(
                self.train_dl) * self.args.num_train_epochs
            self.args.num_train_epochs = self.args.num_train_epochs

        if self.args.num_training_steps // self.args.valid_steps < self.args.minimum_valid_times:
            exp_step = self.args.num_training_steps / self.args.minimum_valid_times
            exp_step = max(int(exp_step - exp_step % 10), 10)
            logger.info("Set eval step to %d" % exp_step)
            self.args.valid_steps = exp_step

        warmup = self.args.warmup_steps if self.args.warmup_steps > 0 else self.args.warmup_proportion

        self.lr_scheduler = LinearDecayWithWarmup(
            self.args.learning_rate, self.args.num_training_steps, warmup)

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        self.optimizer = paddle.optimizer.AdamW(
            learning_rate=self.lr_scheduler,
            beta1=0.9,
            beta2=0.999,
            epsilon=self.args.adam_epsilon,
            parameters=self.model.parameters(),
            weight_decay=self.args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params,
            grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm))

    def print_config(self):
        """
        """
        logger.info('{:^40}'.format("Configuration Arguments"))
        logger.info('{:20}:{}'.format("paddle commit id",
                                      paddle.version.commit))
        for arg in vars(self.args):
            logger.info('{:20}:{}'.format(arg, getattr(self.args, arg)))


class TrainerBase(object):
    """
    """

    def create_dataloader(self,
                          dataset,
                          mode='train',
                          batch_size=16,
                          batchify_fn=None,
                          trans_fn=None,
                          batched=False):
        """
        """
        if trans_fn:
            dataset = dataset.map(trans_fn, batched=batched)

        shuffle = True if mode == 'train' else False
        if mode == 'train':
            batch_sampler = paddle.io.DistributedBatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            batch_sampler = paddle.io.BatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle)

        return paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)

    def train(self, *args, **kwargs):
        """
        """
        pass

    def eval(self, *args, **kwargs):
        """
        """
        pass

    def prepare_train_config(self):
        """
        """
        if self.args.max_steps > 0:
            self.args.num_training_steps = self.args.max_steps
            self.args.num_train_epochs = math.ceil(
                self.args.num_training_steps / len(self.train_dl))

        else:
            self.args.num_training_steps = len(
                self.train_dl) * self.args.num_train_epochs
            self.args.num_train_epochs = self.args.num_train_epochs

        if self.args.num_training_steps // self.args.valid_steps < self.args.minimum_valid_times:
            exp_step = self.args.num_training_steps / self.args.minimum_valid_times
            exp_step = max(int(exp_step - exp_step % 10), 10)
            logger.info("Set eval step to %d" % exp_step)
            self.args.valid_steps = exp_step

        warmup = self.args.warmup_steps if self.args.warmup_steps > 0 else self.args.warmup_proportion

        self.lr_scheduler = LinearDecayWithWarmup(
            self.args.learning_rate, self.args.num_training_steps, warmup)

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        self.optimizer = paddle.optimizer.AdamW(
            learning_rate=self.lr_scheduler,
            beta1=0.9,
            beta2=0.999,
            epsilon=self.args.adam_epsilon,
            parameters=self.model.parameters(),
            weight_decay=self.args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params,
            grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm))

    def print_config(self):
        """
        """
        logger.info('{:^40}'.format("Configuration Arguments"))
        logger.info('{:20}:{}'.format("paddle commit id",
                                      paddle.version.commit))
        for arg in vars(self.args):
            logger.info('{:20}:{}'.format(arg, getattr(self.args, arg)))
