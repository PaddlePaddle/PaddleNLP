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

import copy
import itertools
import math
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
import tqdm
from data import DummyDataset, PromptOnlyBatch
from paddle.io import DataLoader, Dataset, DistributedBatchSampler
from paddle.utils import map_structure

from paddlenlp.data import DataCollator
from paddlenlp.generation import GenerationConfig
from paddlenlp.generation.utils import print
from paddlenlp.trainer.trainer import (
    DEFAULT_CALLBACKS,
    DEFAULT_PROGRESS_CALLBACK,
    PADDLE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    CallbackHandler,
    DygraphShardingOptimizer,
    EvalPrediction,
    HybridParallelOptimizer,
    LoRAModel,
    NlpDistributedBatchSampler,
    PrefixModelForCausalLM,
    PrinterCallback,
    ShardingOption,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    TrainOutput,
    _add_variant,
    _obtain_optimizer_parameters_list,
    distributed_file,
    distributed_isfile,
    fleet,
    fused_allreduce_gradients,
    get_reporting_integration_callbacks,
    get_timers,
    has_length,
    is_dp_group_support_in_group_sharded_parallel,
    logger,
    set_seed,
    set_timers,
    speed_metrics,
)

# from paddlenlp.trainer.utils import nested_detach
from paddlenlp.transformers import BatchEncoding, PretrainedModel, PretrainedTokenizer
from paddlenlp.transformers.tokenizer_utils_base import (
    PaddingStrategy,
    TruncationStrategy,
)


def batch_retokenize(
    input_ids: paddle.Tensor,
    src_tokenizer: PretrainedTokenizer,
    dest_tokenizer: PretrainedTokenizer,
    *,
    padding: bool | str | PaddingStrategy = PaddingStrategy.LONGEST,
    truncation: bool | str | TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    skip_special_tokens: bool = True,
) -> BatchEncoding:
    """Re-tokenize a batch of input ids from one tokenizer to another."""
    output = dest_tokenizer(
        [
            text + dest_tokenizer.eos_token
            for text in src_tokenizer.batch_decode(
                input_ids,
                skip_special_tokens=skip_special_tokens,
            )
        ],
        padding=padding,
        truncation=truncation,
        return_tensors="pd",
    )
    return output


def gather_log_probabilities(logits: paddle.Tensor, labels: paddle.Tensor) -> paddle.Tensor:
    """Gather log probabilities of the given labels from the logits."""
    log_probs = F.log_softmax(logits, axis=-1)
    log_probs_labels = paddle.take_along_axis(log_probs, axis=-1, indices=labels.unsqueeze(axis=-1))
    return log_probs_labels.squeeze(axis=-1)


def init_train_model_opt(self: Trainer, max_steps: int, resume_from_checkpoint: bool = False) -> PretrainedModel:
    if not self.args.should_load_sharding_stage1_model:
        self._load_from_checkpoint(resume_from_checkpoint)

    # delay_optimizer_creation = (
    #     self.sharding is not None
    #     and ShardingOption.SHARD_OP in self.args.sharding
    # )
    delay_optimizer_creation = False

    if not delay_optimizer_creation:
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    if self.args.should_load_sharding_stage1_model:
        model = self._wrap_model_and_load_sharded_checkpoint(resume_from_checkpoint)
    elif self.args.should_save_sharding_stage1_model:
        # In the non-sharded mode, should invoke _load_from_checkpoint before _wrap_model.
        # In this mode, the rank0 load all params and the _wrap_model implicitly broadcast params from rank0 to the other ranks.
        model = self._wrap_model(self.model_wrapped)
        if self.sharding_io is not None:
            assert delay_optimizer_creation is False, "delay_optimizer_creation should be False"
            # the self.optimizer should be wrapped and it is done in _wrap_model
            self.sharding_io.set_optimizer(self.optimizer)
        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model
        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
    else:
        model = self._wrap_model(self.model_wrapped)
        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model
        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

    if self.args.device == "npu" and self.args.flatten_param_grads:
        from .plugins.npu_plugin import npu_accelerate_plugin

        npu_accelerate_plugin(self.optimizer)

    return model


def init_train_num(self: Trainer, train_dataloader: DataLoader):
    args = self.args

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

    return (
        total_train_batch_size,
        len_dataloader,
        max_steps,
        num_train_epochs,
        num_update_steps_per_epoch,
        num_examples,
        num_train_samples,
    )


def init_train_state(
    self: Trainer,
    resume_from_checkpoint: bool,
    train_dataloader: DataLoader,
    max_steps: int,
    num_train_epochs: int,
    num_update_steps_per_epoch: int,
):
    args = self.args

    self.state = TrainerState()
    self.state.epoch = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    steps_trained_progress_bar = None

    # Check if continuing training from a checkpoint
    if resume_from_checkpoint is not None and distributed_isfile(
        os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    ):
        self.state = TrainerState.load_from_json(
            distributed_file(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        )
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

    self.state.max_steps = int(max_steps)
    self.state.num_train_epochs = num_train_epochs
    self.state.is_local_process_zero = self.is_local_process_zero()
    self.state.is_world_process_zero = self.is_world_process_zero()

    return epochs_trained, steps_trained_in_current_epoch, steps_trained_progress_bar


def init_train_log(
    self: Trainer,
    num_examples: int,
    num_train_epochs: int,
    total_train_batch_size: int,
    max_steps: int,
    num_train_samples: int,
    model: PretrainedModel,
):
    args = self.args

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples:,}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps:,}")
    logger.info(f"  Total num train samples = {num_train_samples:,}")
    # per_device_trainable_numel = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
    # TODO: Temporary fix since Tensor.numel() not supported in distributed mode
    per_device_trainable_numel = sum(np.prod(p.shape) for p in model.parameters() if not p.stop_gradient)
    logger.info(f"  Number of trainable parameters = {per_device_trainable_numel:,} (per device)")
    if self.args.use_hybrid_parallel:
        # todo fix for pipeline_parallel_degree
        parts_num = max(self.args.tensor_parallel_degree, 1) * max(self.args.pipeline_parallel_degree, 1)
        if parts_num > 1:
            all_reduce_dtype = "int64"
            if paddle.get_device().split(":")[0] in ["npu", "xpu"]:
                # TODO(duanyanhui): fix when NPU all_reduce supports int64
                all_reduce_dtype = "float32"
            trainable_numel_tensor = paddle.to_tensor(per_device_trainable_numel, dtype=all_reduce_dtype)
            paddle.distributed.all_reduce(trainable_numel_tensor)
            trainable_numel = int(trainable_numel_tensor.item()) // self.args.dataset_world_size
            # the numel is roughly, because the tensor parallel still hold own bias or layer_norm weight without splited
            # so, the trainable numel is a little bigger than real.
            logger.info(f"  Number of trainable parameters = {trainable_numel:,} (all devices, roughly)")


def init_train(
    self: Trainer,
    resume_from_checkpoint: Optional[Union[str, bool]] = None,
    ignore_keys_for_eval: Optional[List[str]] = None,
):
    args = self.args
    self.is_in_train = True

    # memory metrics - must set up as early as possible
    self._memory_tracker.start()

    # ##### trainging related num setting #####
    train_dataloader = self.get_train_dataloader()
    (
        total_train_batch_size,
        len_dataloader,
        max_steps,
        num_train_epochs,
        num_update_steps_per_epoch,
        num_examples,
        num_train_samples,
    ) = self.init_train_num(train_dataloader)

    # ##### model and optimizer related setting #####
    model = self.init_train_model_opt(max_steps, resume_from_checkpoint)

    # ##### traing statistic logging #####
    self.init_train_log(num_examples, num_train_epochs, total_train_batch_size, max_steps, num_train_samples, model)

    # ##### set training state and resume #####
    epochs_trained, steps_trained_in_current_epoch, steps_trained_progress_bar = self.init_train_state(
        resume_from_checkpoint, train_dataloader, max_steps, num_train_epochs, num_update_steps_per_epoch
    )

    # ##### training track vars #####
    epoch_iterator = train_dataloader
    # steps_in_epoch = len(epoch_iterator)
    steps_in_epoch = (
        len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
    )
    if len_dataloader is not None:
        if self.args.gradient_accumulation_steps > len(epoch_iterator):
            logger.warning(
                f"changing accumulation step from `{self.args.gradient_accumulation_steps}` to `{len(epoch_iterator)}` to avoid, cross epoch accumulate"
            )
            self.args.gradient_accumulation_steps = len(epoch_iterator)

    self.callback_handler.model = self.model
    self.callback_handler.optimizer = self.optimizer
    self.callback_handler.lr_scheduler = self.lr_scheduler
    self.callback_handler.train_dataloader = train_dataloader

    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    tr_loss = paddle.to_tensor(0.0)
    self._total_loss_scalar = 0.0
    self._globalstep_last_logged = self.state.global_step

    # return local vars or move local vars to outter space
    return {
        "train_dataloader": train_dataloader,
        "num_train_samples": num_train_samples,
        "num_train_epochs": num_train_epochs,
        "epochs_trained": epochs_trained,
        "steps_trained_in_current_epoch": steps_trained_in_current_epoch,
        "steps_trained_progress_bar": steps_trained_progress_bar,
        "steps_in_epoch": steps_in_epoch,
        "tr_loss": tr_loss,
        "model": model,
    }


# def full_training_step(self: Trainer,
#                        epoch: int,
#                        step: int,
#                        steps_in_epoch: int,
#                        inputs: Dict[str, paddle.Tensor],
#                        model: PretrainedModel,
#                        train_dataloader: DataLoader,
#                        resume_from_checkpoint: bool = False,
#                        steps_trained_in_current_epoch: int = 0,
#                        steps_trained_progress_bar=None,
#                        ignore_keys_for_eval: Optional[List[str]] = None):
def full_training_step(self: Trainer, inputs: Dict[str, paddle.Tensor], **kwargs):
    # maybe should be abstracted in Trainer

    # TODO(guosheng): step, steps_trained_in_current_epoch and steps_trained_progress_bar
    # should use reference since they would be overwrite.
    # for state update
    epoch = kwargs.get("epoch", 0)
    step = kwargs.get("step", 0)
    steps_in_epoch = kwargs.get("steps_in_epoch", 0)
    step_control = kwargs.get("step_control", 0)
    # for step and progress update when resuming data
    train_dataloader = kwargs.get("train_dataloader", None)
    resume_from_checkpoint = kwargs.get("resume_from_checkpoint", None)
    steps_trained_in_current_epoch = kwargs.get("steps_trained_in_current_epoch", 0)
    steps_trained_progress_bar = kwargs.get("steps_trained_progress_bar", None)
    # for eval output ignore to gather
    ignore_keys_for_eval = kwargs.get("ignore_keys_for_eval", None)
    tr_loss = kwargs.get("tr_loss", 0.0)
    model = kwargs.get("model", self.model_wrapped)

    args = self.args

    # self.timers and self.timers("read-data").stop()
    # os.environ["TRAINER_GLOBAL_STEP"] = str(self.state.global_step)
    # self.callback_handler.on_load_data_end(args,
    #                                        self.state,
    #                                        self.control,
    #                                        inputs=inputs)

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
        # continue
        return
    elif steps_trained_progress_bar is not None:
        steps_trained_progress_bar.close()
        steps_trained_progress_bar = None

    if step_control % args.gradient_accumulation_steps == 0:
        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
        self.timers and self.timers("forward-backward").start()

    dp_enabled = self.args.data_parallel_degree > 1 if self.args.use_hybrid_parallel else args.local_rank != -1
    forbidden_no_sync = False
    # stage2 and stage3 should not no_sync, because the is no DDP wrapper and no_sync API
    # hybrid_parallel (tp or pp or sharding stage 1) should not no_sync
    if self.args.use_hybrid_parallel:
        forbidden_no_sync = True

    availiable_no_sync = dp_enabled and not forbidden_no_sync

    is_no_sync = (
        ((step_control + 1) % args.gradient_accumulation_steps != 0)
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
    print("=" * 20, "after training_step")

    tr_loss += tr_loss_step

    if (step_control + 1) % args.gradient_accumulation_steps == 0 or (
        # last step in epoch but step is always smaller than gradient_accumulation_steps
        steps_in_epoch <= args.gradient_accumulation_steps
        and (step + 1) == steps_in_epoch
    ):
        self.timers and self.timers("forward-backward").stop()
        # Maunally collect gradients when group_sharded_parallel can't accept dp_group
        # Case 1: Use sharding stage 2/3 with dp
        # Case 2: Use recompute and dp
        # local_rank != -1 don't means dp in networks.
        self.timers and self.timers("all-reduce").start()

        if self.sharding and ShardingOption.SHARD_OP not in self.args.sharding:
            if self.args.data_parallel_degree > 1 and not is_dp_group_support_in_group_sharded_parallel():
                print("=" * 20, "before fused_allreduce_gradients")
                fused_allreduce_gradients(model.parameters(), fleet.get_hybrid_communicate_group())
                print("=" * 20, "after fused_allreduce_gradients")
                if ShardingOption.FULL_SHARD in self.args.sharding:
                    print("=" * 20, "before all_reduce bw_storage")
                    # Why need sync on parm again ?
                    # TODO: fix this.
                    for p in model.parameters():
                        if hasattr(p, "bw_storage"):
                            assert p.grad is None, "This case shouldn't happen."
                            p.bw_storage.scale_(1.0 / self.dp_group.nranks)
                            paddle.distributed.all_reduce(p.bw_storage, group=self.dp_group)
                    print("=" * 20, "after all_reduce bw_storage")

        # Case 2: Use recompute and dp / sharding stage1,
        # manualy collect gradient for dp.
        elif args.recompute and availiable_no_sync:
            fused_allreduce_gradients(list(model.parameters()), None)

        pipeline_parallel_config = set(args.pipeline_parallel_config.split(" "))
        enable_delay_scale_loss = "enable_delay_scale_loss" in pipeline_parallel_config
        enable_dp_comm_overlap = "enable_dp_comm_overlap" in pipeline_parallel_config

        if isinstance(self.optimizer, HybridParallelOptimizer) and not self.do_grad_scaling:
            parameters_list = _obtain_optimizer_parameters_list(self.optimizer._inner_opt)

            if not enable_dp_comm_overlap:
                if self.optimizer._sharding_enable:
                    assert isinstance(self.optimizer._inner_opt, DygraphShardingOptimizer)
                    self.optimizer._inner_opt.reduce_gradients(list(parameters_list), self.optimizer._hcg)

                if self.optimizer._dp_enable:
                    fused_allreduce_gradients(list(parameters_list), self.optimizer._hcg)
        self.timers and self.timers("all-reduce").stop()
        self.timers and self.timers("optimizer-step").start()

        # pipeline parallel mode,  handle gradient merge here
        if args.pipeline_parallel_degree > 1 and enable_delay_scale_loss:
            for p in model._layers.parameters():
                with paddle.no_grad():
                    if hasattr(p, "main_grad") and p.main_grad is not None:
                        assert p.grad is None
                        p.main_grad.scale_(1.0 / self.args.gradient_accumulation_steps)
                    elif p.grad is not None:
                        p.grad.scale_(1.0 / self.args.gradient_accumulation_steps)

        # Optimizer step
        print("=" * 20, "on_optimizer_begin")
        self.callback_handler.on_optimizer_begin(
            args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
        )
        optimizer_was_run = True
        if self.do_grad_scaling:
            scale_before = self.scaler._scale.cpu().numpy()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler._scale.cpu().numpy()
            optimizer_was_run = not self.scaler._cache_founf_inf
            if not optimizer_was_run:
                logger.warning(f"optimizer not run, scale_before: {scale_before[0]}, scale_after: {scale_after[0]}")
        elif isinstance(self.optimizer, HybridParallelOptimizer):
            self.optimizer._step(parameters_list)
        else:
            self.optimizer.step()

        self.timers and self.timers("optimizer-step").stop()

        if optimizer_was_run:
            self.lr_scheduler.step()

        self.optimizer.clear_grad()
        self.callback_handler.on_optimizer_end(
            args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
        )
        print("=" * 20, "on_optimizer_end")

        self.state.global_step += 1
        self.state.epoch = epoch + (step + 1) / steps_in_epoch
        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
        self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
        print("=" * 20, "after _maybe_log_save_evaluate")
        # if self.state.global_step == 2:
        #     exit(0)
        self._print_timer()
        step_control = 0
    else:
        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
        step_control += 1

    # if self.control.should_epoch_stop or self.control.should_training_stop:
    #     break
    # self.timers and self.timers("read-data").start()

    final_local_vars = locals()
    for k in kwargs.keys():
        if k in final_local_vars:
            kwargs[k] = final_local_vars[k]
    return kwargs


Trainer.init_train_model_opt = init_train_model_opt
Trainer.init_train_log = init_train_log
Trainer.init_train_state = init_train_state
Trainer.init_train = init_train
Trainer.full_training_step = full_training_step


def train(
    self: Trainer,
    resume_from_checkpoint: Optional[Union[str, bool]] = None,
    ignore_keys_for_eval: Optional[List[str]] = None,
):
    args = self.args

    init_train_dict = self.init_train(resume_from_checkpoint, ignore_keys_for_eval)
    train_dataloader = epoch_iterator = init_train_dict["train_dataloader"]
    epochs_trained = init_train_dict["epochs_trained"]
    num_train_epochs = init_train_dict["num_train_epochs"]
    num_train_samples = init_train_dict["num_train_samples"]
    tr_loss = init_train_dict["tr_loss"]
    model = init_train_dict["model"]

    # train_step_kwargs = {
    #     "resume_from_checkpoint": resume_from_checkpoint,
    #     "ignore_keys_for_eval": ignore_keys_for_eval,
    #     "train_dataloader": train_dataloader,
    #     "steps_in_epoch": steps_in_epoch,
    #     "steps_trained_in_current_epoch": steps_trained_in_current_epoch,
    #     "steps_trained_progress_bar": steps_trained_progress_bar,
    #     "tr_loss": tr_loss,
    #     "model": model
    # }

    start_time = time.time()
    self._globalstep_last_start_time = time.time()
    self.timers and self.timers("read-data").start()

    for epoch in range(epochs_trained, num_train_epochs):
        if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
            train_dataloader.batch_sampler, DistributedBatchSampler
        ):
            train_dataloader.batch_sampler.set_epoch(epoch)

        self.step_control = 0  # used in loop control, reset to 0 after every step
        self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

        for step, inputs in enumerate(epoch_iterator):
            self.timers and self.timers("read-data").stop()
            os.environ["TRAINER_GLOBAL_STEP"] = str(self.state.global_step)
            self.callback_handler.on_load_data_end(args, self.state, self.control, inputs=inputs)
            init_train_dict.update({"epoch": epoch, "step": step})
            init_train_dict = self.full_training_step(inputs, **init_train_dict)
            if self.control.should_epoch_stop or self.control.should_training_stop:
                break
            self.timers and self.timers("read-data").start()
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

        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        if isinstance(self.model, LoRAModel) or isinstance(self.model, PrefixModelForCausalLM):
            self._load_best_model_from_peft_checkpoint()
        else:
            weight_name = PADDLE_WEIGHTS_NAME
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


Trainer.train = train


class PolicyTrainer(Trainer):
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

        super().__init__(
            model,
            criterion,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def _prepare_inputs(self, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> Dict[str, Union[paddle.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        # start = inputs.get("start", 0)
        inputs = super()._prepare_inputs(inputs)
        return inputs

    def actor_loss_fn(
        self,
        log_probs: paddle.Tensor,
        old_log_probs: paddle.Tensor,
        advantages: paddle.Tensor,
        mask: paddle.Tensor,
    ) -> paddle.Tensor:
        # policy gradient loss
        ratio = paddle.exp(log_probs - old_log_probs)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * paddle.clip(
            ratio,
            1.0 - self.clip_range_ratio,
            1.0 + self.clip_range_ratio,
        )
        return paddle.sum(paddle.maximum(pg_loss1, pg_loss2) * mask) / mask.sum()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        labels = inputs.get("labels", None)
        if labels is not None:
            labels = inputs.get("labels", None)
            outputs = model(**inputs)
            ptx_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return ptx_loss

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        old_log_probs = inputs["old_log_probs"]
        reward_advantages = inputs["reward_advantages"]
        sequence_mask = inputs["sequence_mask"]
        start = inputs["start"]
        use_cache = inputs["use_cache"]
        return_dict = inputs["return_dict"]

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache, return_dict=return_dict
        )

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        elif isinstance(outputs, tuple):
            logits = outputs[1]

        log_probs = gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
        actor_loss = self.actor_loss_fn(
            log_probs[:, start:],
            old_log_probs[:, start:],
            reward_advantages,
            sequence_mask[:, start:],
        )
        print("=" * 20, "log_probs", log_probs)
        print("=" * 20, "actor_loss", actor_loss)

        return actor_loss

    def full_training_step(self: Trainer, inputs: Dict[str, paddle.Tensor], **kwargs):
        labels = inputs.get("labels", None)
        if labels is not None:  # use ptx
            loss_name = "ptx_loss"
        else:
            loss_name = "actor_loss"
        kwargs["model"] = kwargs.pop("policy_model")
        kwargs["step_control"] = kwargs.pop("policy_step_control")
        kwargs["tr_loss"] = kwargs.pop(loss_name)
        kwargs = super().full_training_step(inputs, **kwargs)
        kwargs["policy_model"] = kwargs.pop("model")
        kwargs["policy_step_control"] = kwargs.pop("step_control")
        kwargs[loss_name] = kwargs.pop("tr_loss")
        return kwargs


class ValueTrainer(Trainer):
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

        super().__init__(
            model,
            criterion,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def critic_loss_fn(
        self,
        values: paddle.Tensor,
        old_values: paddle.Tensor,
        returns: paddle.Tensor,
        mask: paddle.Tensor,
    ) -> paddle.Tensor:
        """Compute critic loss."""
        values_clipped = paddle.clip(
            values,
            old_values - self.clip_range_value,
            old_values + self.clip_range_value,
        )
        vf_loss1 = paddle.square(values - returns)
        vf_loss2 = paddle.square(values_clipped - returns)
        return 0.5 * paddle.sum(paddle.maximum(vf_loss1, vf_loss2) * mask) / mask.sum()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        old_reward_values = inputs["old_reward_values"]
        reward_returns = inputs["reward_returns"]
        sequence_mask = inputs["sequence_mask"]
        start = inputs["start"]
        use_cache = inputs["use_cache"]
        return_dict = inputs["return_dict"]

        print("=" * 20, "before critic model call")
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache, return_dict=return_dict
        )
        print("=" * 20, "after critic model call")

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        reward_values = outputs["scores"] if isinstance(outputs, dict) else outputs
        if isinstance(outputs, dict):
            reward_values = outputs["scores"]
        elif isinstance(outputs, tuple):
            reward_values = outputs[0]

        reward_values = reward_values.squeeze(axis=-1)[:, :-1]
        reward_critic_loss = self.critic_loss_fn(
            reward_values[:, start:],
            old_reward_values[:, start:],
            reward_returns,
            sequence_mask[:, start:],
        )
        print("=" * 20, "after critic_loss_fn")
        print("=" * 20, "reward_values", reward_values)
        print("=" * 20, "reward_critic_loss", reward_critic_loss)

        return reward_critic_loss

    def full_training_step(self: Trainer, inputs: Dict[str, paddle.Tensor], **kwargs):
        kwargs["model"] = kwargs.pop("value_model")
        kwargs["step_control"] = kwargs.pop("value_step_control")
        kwargs["tr_loss"] = kwargs.pop("reward_critic_loss")
        kwargs = super().full_training_step(inputs, **kwargs)
        kwargs["value_model"] = kwargs.pop("model")
        kwargs["value_step_control"] = kwargs.pop("step_control")
        kwargs["reward_critic_loss"] = kwargs.pop("tr_loss")
        return kwargs


@contextmanager
def guard_set_args(args, arg_name_values):
    for k, v in arg_name_values.items():
        old_value = getattr(args, k, None)
        setattr(args, k, v)
        arg_name_values[k] = old_value
    yield
    for k, v in arg_name_values.items():
        old_value = getattr(args, k)
        setattr(args, k, v)
        arg_name_values[k] = old_value


class MuteDefaultFlowCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_save = False
        control.should_evaluate = False
        control.should_log = False
        return control


def is_same_tokenizer(
    tokenizer: PretrainedTokenizer,
    other_tokenizer: PretrainedTokenizer,
) -> bool:
    """Check if two tokenizers are the same."""
    return tokenizer is other_tokenizer or (
        tokenizer.__class__ == other_tokenizer.__class__ and tokenizer.get_vocab() == other_tokenizer.get_vocab()
    )


class PPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        ptx_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor] = None,
    ):
        self.args = args
        self.is_in_train = False

        # Seed must be set before instantiating the model when using model
        set_seed(args=self.args)

        if self.args.should_save or self.args.should_save_model_state:
            os.makedirs(self.args.output_dir, exist_ok=True)

        self.train_dataset = train_dataset
        self.ptx_dataset = ptx_dataset
        self.eval_dataset = eval_dataset

        (policy_model, reference_model, reward_model, value_model) = model
        # policy_tokenizer and value_model should be same
        (policy_tokenizer, reference_tokenizer, reward_tokenizer, value_tokenizer) = tokenizer
        self.reference_model = reference_model
        self.reference_model.eval()
        if False:
            # NOTE: reference_model/reward_model should be guaranteed to be BF16
            # by model provider since they would not be wrapped by Trainer.
            logger.warning()
        self.reward_model = reward_model
        self.reward_model.eval()
        self.reward_tokenizer = reward_tokenizer
        self.tokenizer = policy_tokenizer
        if is_same_tokenizer(self.tokenizer, self.reward_tokenizer):
            self.reward_tokenizer = self.tokenizer
        policy_training_args = copy.deepcopy(args)
        self.use_ptx = self.ptx_dataset is not None
        if self.use_ptx:
            policy_training_args.gradient_accumulation_steps *= 2
        self.policy_trainer = PolicyTrainer(
            policy_model,
            criterion,
            policy_training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            policy_tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        value_training_args = copy.deepcopy(args)
        for attr_name in [
            "critic_learning_rate",
            "critic_weight_decay",
            "critic_lr_scheduler_type",
            "critic_warmup_ratio",
            "critic_recompute",
        ]:
            if getattr(value_training_args, attr_name, None) is not None:
                setattr(value_training_args, attr_name[len("critic_") :], getattr(value_training_args, attr_name))
        self.value_trainer = ValueTrainer(
            value_model,
            criterion,
            value_training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            value_tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

        if not args.skip_profile_timer:
            set_timers()
        self.timers = get_timers()
        self.state = TrainerState()
        self.control = TrainerControl()

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(callbacks, None, self.tokenizer, None, None)
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        self.generation_config = GenerationConfig(
            max_length=self.args.max_length,
            num_return_sequences=self.args.num_return_sequences,
            temperature=self.args.temperature,
            # top_p=self.args.top_p,
            top_k=self.args.top_k,
            repetition_penalty=self.args.repetition_penalty,
            do_sample=True,
            trunc_input=False,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Those value can be changed
        self.kl_coeff = self.args.kl_coeff
        self.policy_trainer.clip_range_ratio = self.clip_range_ratio = self.args.clip_range_ratio
        self.clip_range_score = self.args.clip_range_score
        self.value_trainer.clip_range_value = self.clip_range_value = self.args.clip_range_value
        self.ptx_coeff = self.args.ptx_coeff
        self.gamma = 1.0
        self.gae_lambda = 0.95

    @property
    def actor_model(self):
        if self.training:
            return self.policy_trainer.model_wrapped
        if self.policy_trainer.args.pipeline_parallel_degree > 1:
            # Only accept wrapped model for pipeline_parallel mode
            model = self.policy_trainer.model_wrapped
        else:
            model = self.policy_trainer.model
        return model

    @property
    def reward_critic_model(self):
        if self.training:
            return self.value_trainer.model_wrapped
        if self.value_trainer.args.pipeline_parallel_degree > 1:
            # Only accept wrapped model for pipeline_parallel mode
            model = self.value_trainer.model_wrapped
        else:
            # model = self.value_trainer.model
            model = self.value_trainer.model
        return model

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for all models."""
        if mode:
            # self.is_in_train = True
            self.training = True
            self.actor_model.train()
            self.reward_critic_model.train()
        else:
            self.training = False
            self.actor_model.eval()
            self.reward_critic_model.eval()

    def set_eval(self) -> None:
        """Set model to evaluation mode."""
        self.set_train(mode=False)

    def get_epoch_iterator(self):
        # TODO(guosheng): support iter dataset
        num_prompt_only_batches = len(self.prompt_only_dataloader)
        num_ptx_batches = len(self.ptx_dataloader)
        num_ptx_replicas = (num_prompt_only_batches + num_ptx_batches - 1) // num_ptx_batches

        def gen_epoch_data():
            for prompt_only_batch, ptx_batch in zip(
                self.prompt_only_dataloader,
                itertools.chain.from_iterable([self.ptx_dataloader] * num_ptx_replicas),
            ):
                # generate batches
                self.set_eval()
                print("=" * 20, self.args.local_rank, prompt_only_batch)
                rl_batches = self.split_rl_micro_batches(prompt_only_batch)
                if self.use_ptx:
                    ptx_batches = self.split_ptx_micro_batches(ptx_batch)
                else:
                    ptx_batches = [None for _ in range(len(rl_batches))]
                paddle.device.cuda.empty_cache()

                self.set_train()
                for _ in range(self.args.update_iters):
                    for rl_batch, ptx_batch in zip(rl_batches, ptx_batches):
                        yield rl_batch, ptx_batch

        class EpochIterator:
            def __iter__(self):
                return gen_epoch_data()

        return EpochIterator()

    def init_train_num(self: Trainer, train_dataloader: DataLoader):
        args = self.args

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.dataset_world_size

        len_dataloader = len(train_dataloader)
        num_train_sub_steps = (
            len_dataloader
            * self.args.update_iters
            * self.args.per_device_prompt_batch_size
            * self.args.num_return_sequences
            // self.args.per_device_train_batch_size
        )
        num_update_steps_per_epoch = num_train_sub_steps // args.gradient_accumulation_steps
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
            num_train_epochs = math.ceil(args.num_train_epochs)
        num_examples = num_train_samples = total_train_batch_size * max_steps

        return (
            total_train_batch_size,
            len_dataloader,
            max_steps,
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
        )

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
    ) -> None:
        args = self.args
        self.is_in_train = True

        # ##### trainging related num setting #####
        with guard_set_args(
            args, {"per_device_train_batch_size": self.args.per_device_prompt_batch_size}
        ), guard_set_args(
            self, {"train_dataset": self.train_dataset, "data_collator": self.train_dataset.get_collator()}
        ):
            train_dataloader = self.prompt_only_dataloader = self.get_train_dataloader()

        if self.use_ptx:
            with guard_set_args(
                args,
                {
                    "per_device_train_batch_size": self.args.per_device_prompt_batch_size
                    * self.args.num_return_sequences
                },
            ), guard_set_args(
                self, {"train_dataset": self.ptx_dataset, "data_collator": self.ptx_dataset.get_collator()}
            ):
                self.ptx_dataloader = self.get_train_dataloader()
        else:
            self.ptx_dataloader = DataLoader(DummyDataset(len(self.prompt_only_dataloader)))
        (
            total_train_batch_size,
            len_dataloader,
            max_steps,
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
        ) = self.init_train_num(train_dataloader)

        # ##### model and optimizer related setting #####
        # policy_trainer/value_trainer only init train with init_train_model_opt,
        # maybe more training setting used in full_training_step should be set here,
        # such as trainer.control and trainer.state
        policy_model = self.policy_trainer.init_train_model_opt(
            max_steps * 2 if self.use_ptx else max_steps, resume_from_checkpoint
        )
        value_model = self.value_trainer.init_train_model_opt(max_steps, resume_from_checkpoint)
        # disable inner trainers' callback/state/control
        self.policy_trainer.add_callback(MuteDefaultFlowCallback)
        self.value_trainer.add_callback(MuteDefaultFlowCallback)

        # ##### traing statistic logging #####
        # Number of trainable parameters only account for policy_model
        self.init_train_log(
            num_examples, num_train_epochs, total_train_batch_size, max_steps, num_train_samples, policy_model
        )

        # ##### set training state and resume #####
        # consumed_samples used to set train_dataloader.batch_sampler may not be
        # correct. Thus, data cannot be resumed perfectly when not breaking at epoch end.
        epochs_trained, steps_trained_in_current_epoch, steps_trained_progress_bar = self.init_train_state(
            resume_from_checkpoint, train_dataloader, max_steps, num_train_epochs, num_update_steps_per_epoch
        )

        epoch_iterator = self.get_epoch_iterator()
        steps_in_epoch = num_update_steps_per_epoch * args.gradient_accumulation_steps

        # self.callback_handler.model = self.model
        # self.callback_handler.optimizer = self.optimizer
        # self.callback_handler.lr_scheduler = self.lr_scheduler
        # self.callback_handler.train_dataloader = train_dataloader
        self.state.max_steps = int(max_steps)
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        actor_loss = paddle.to_tensor(0.0)
        reward_critic_loss = paddle.to_tensor(0.0)
        ptx_loss = paddle.to_tensor(0.0)
        # used when logging and last step
        self._total_actor_loss_scalar = 0.0
        self._total_reward_critic_loss_scalar = 0.0
        self._total_ptx_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        train_step_kwargs = {
            "resume_from_checkpoint": None,  # resume_from_checkpoint,
            "ignore_keys_for_eval": None,
            "train_dataloader": None,  # train_dataloader,
            # "num_train_samples": num_train_samples,
            # "num_train_epochs": num_train_epochs,
            # "epochs_trained": epochs_trained,
            # "steps_trained_in_current_epoch": steps_trained_in_current_epoch,
            # "steps_trained_progress_bar": steps_trained_progress_bar,
            "steps_in_epoch": steps_in_epoch,
            "actor_loss": actor_loss,
            "reward_critic_loss": reward_critic_loss,
            "ptx_loss": ptx_loss,
            "policy_model": policy_model,
            "value_model": value_model,
        }

        start_time = time.time()
        self._globalstep_last_start_time = start_time  # time.time()
        # self.timers and self.timers("read-data").start()

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                train_dataloader.batch_sampler, DistributedBatchSampler
            ):
                train_dataloader.batch_sampler.set_epoch(epoch)

            step_control = 0  # used in loop control, reset to 0 after every step
            train_step_kwargs.update({"policy_step_control": step_control, "value_step_control": step_control})
            # self.policy_trainer.step_control = 0
            # self.value_trainer.step_control = 0
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                # self.timers and self.timers("read-data").stop()
                os.environ["TRAINER_GLOBAL_STEP"] = str(self.state.global_step)
                self.callback_handler.on_load_data_end(args, self.state, self.control, inputs=inputs)
                # epoch, step and steps_in_epoch only mostly used in train_step by
                # `self.state.epoch = epoch + (step + 1) / steps_in_epoch` if not
                # resume data
                train_step_kwargs.update({"epoch": epoch, "step": step})
                rl_batch, ptx_batch = inputs
                print("=" * 20, "rl_batch", rl_batch)
                rl_info, train_step_kwargs = self.rl_step(rl_batch, **train_step_kwargs)
                paddle.device.cuda.empty_cache()
                if self.use_ptx:
                    print("=" * 20, "ptx_batch", ptx_batch)
                    ptx_info, train_step_kwargs = self.ptx_step(ptx_batch, **train_step_kwargs)
                    rl_info.update(ptx_info)
                    paddle.device.cuda.empty_cache()

                self.state.global_step = self.value_trainer.state.global_step
                self.state.epoch = self.value_trainer.state.epoch
                if train_step_kwargs["value_step_control"] == 0:
                    # on_step_end
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                else:
                    # on_sub_step_end
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(rl_info, None, epoch, ignore_keys_for_eval, inputs=inputs)
                # exit(0)

            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({self.state.max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(rl_info, None, epoch, ignore_keys_for_eval, inputs=inputs)

            if self.control.should_training_stop:
                break

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            for k, v in tr_loss.items():
                if isinstance(v, paddle.Tensor) and "lr" not in k:
                    v_scalar = self._nested_gather(v).mean().item()
                    logs[k] = round(v_scalar / (self.state.global_step - self._globalstep_last_logged), 8)
                    v.subtract_(v)
                    attr_name = "_total_" + k.split("/")[-1] + "_scalar"
                    attr_value = getattr(self, attr_name, 0)
                    setattr(self, attr_name, attr_value + v_scalar)
                else:
                    logs[k] = float("{0:.3e}".format(v))
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

            self._globalstep_last_logged = self.state.global_step
            self._globalstep_last_start_time = time.time()

            self.log(logs, **kwargs)

        with guard_set_args(self.control, {"should_log": False}):  # avoid log again
            super()._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval)

    def add_kl_divergence_regularization(
        self,
        prompt: paddle.Tensor,  # size = (B, S) # pylint: disable=unused-argument
        log_probs: paddle.Tensor,  # size = (B, L)
        ref_log_probs: paddle.Tensor,  # size = (B, L)
        reward_score: paddle.Tensor,  # size = (B,)
        sequence_mask: paddle.Tensor,  # size = (B, L)
    ) -> paddle.Tensor:
        kl_divergence_estimate = -self.kl_coeff * (log_probs - ref_log_probs)  # size = (B, L)
        rewards = kl_divergence_estimate  # size = (B, L)
        reward_clip = paddle.clip(  # size = (B,)
            reward_score,
            min=-self.clip_range_score,
            max=self.clip_range_score,
        )
        batch_size = log_probs.shape[0]
        for i in range(batch_size):
            end_index = sequence_mask[i].nonzero()[-1]
            # rewards[i, end_index] += reward_clip[i]
            rewards[i, end_index] = rewards[i, end_index] + reward_clip[i]

        return rewards

    def get_advantages_and_returns(
        self,
        values: paddle.Tensor,
        rewards: paddle.Tensor,
        sequence_mask: paddle.Tensor,
        start: int,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute advantages and returns using Generalized Advantage Estimation (GAE)."""
        # Modified from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        last_gae_lambda = 0.0
        advantages_reversed = []
        values = values * sequence_mask
        rewards = rewards * sequence_mask
        length = rewards.shape[-1]
        for t in reversed(range(start, length)):  # pylint: disable=invalid-name
            next_values = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * next_values - values[:, t]
            last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda
            advantages_reversed.append(last_gae_lambda)
        advantages = paddle.stack(advantages_reversed[::-1], axis=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def rl_step(self, rl_batch: Dict[str, paddle.Tensor], **kwargs) -> Dict[str, Any]:
        prompt = rl_batch["prompt"]
        old_log_probs = rl_batch["log_probs"]
        ref_log_probs = rl_batch["ref_log_probs"]
        rewards = rl_batch["rewards"]
        old_reward_values = rl_batch["reward_values"]
        input_ids = rl_batch["input_ids"]
        attention_mask = rl_batch["attention_mask"]

        start = prompt.shape[-1] - 1
        sequence_mask = attention_mask[:, 1:]

        with paddle.no_grad():
            old_rewards = self.add_kl_divergence_regularization(
                prompt,
                old_log_probs,
                ref_log_probs,
                rewards,
                sequence_mask,
            )
            reward_advantages, reward_returns = self.get_advantages_and_returns(
                old_reward_values,
                old_rewards,
                sequence_mask,
                start,
            )
        print("=" * 20, "after reward_advantages")
        print("=" * 20, "old_rewards", old_rewards)
        print("=" * 20, "reward_advantages", reward_advantages)
        print("=" * 20, "reward_returns", reward_returns)

        # value_trainer_inputs = {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "old_reward_values": old_reward_values,
        #     "reward_returns": reward_returns,
        #     "sequence_mask": sequence_mask,
        #     "start": start,
        #     "use_cache": False,
        #     "return_dict": True
        # }
        # # print(value_trainer_inputs)
        # # reward_critic_loss = self.value_trainer.training_step(
        # #     value_trainer_inputs)
        # # reward_critic_loss = self.value_trainer.full_training_step(
        # #     epoch, step, steps_in_epoch, value_trainer_inputs, model,
        # #     train_dataloader, resume_from_checkpoint,
        # #     steps_trained_in_current_epoch, steps_trained_progress_bar,
        # #     ignore_keys_for_eval)
        # print("=" * 20, "before value_trainer.full_training_step")
        # kwargs = self.value_trainer.full_training_step(value_trainer_inputs,
        #                                                **kwargs)
        # print("=" * 20, "after value_trainer.full_training_step")

        policy_trainer_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "old_log_probs": old_log_probs,
            "reward_advantages": reward_advantages,
            "sequence_mask": sequence_mask,
            "start": start,
            "use_cache": False,
            "return_dict": True,
        }
        # print(policy_trainer_inputs)
        # actor_loss = self.policy_trainer.training_step(policy_trainer_inputs)
        # actor_loss = self.policy_trainer.full_training_step(
        #     epoch, step, steps_in_epoch, policy_trainer_inputs, model,
        #     train_dataloader, resume_from_checkpoint,
        #     steps_trained_in_current_epoch, steps_trained_progress_bar,
        #     ignore_keys_for_eval)
        print("=" * 20, "before policy_trainer.full_training_step")
        kwargs = self.policy_trainer.full_training_step(policy_trainer_inputs, **kwargs)
        print("=" * 20, "after policy_trainer.full_training_step")

        value_trainer_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "old_reward_values": old_reward_values,
            "reward_returns": reward_returns,
            "sequence_mask": sequence_mask,
            "start": start,
            "use_cache": False,
            "return_dict": True,
        }
        # print(value_trainer_inputs)
        # reward_critic_loss = self.value_trainer.training_step(
        #     value_trainer_inputs)
        # reward_critic_loss = self.value_trainer.full_training_step(
        #     epoch, step, steps_in_epoch, value_trainer_inputs, model,
        #     train_dataloader, resume_from_checkpoint,
        #     steps_trained_in_current_epoch, steps_trained_progress_bar,
        #     ignore_keys_for_eval)
        print("=" * 20, "before value_trainer.full_training_step")
        kwargs = self.value_trainer.full_training_step(value_trainer_inputs, **kwargs)
        print("=" * 20, "after value_trainer.full_training_step")

        with paddle.no_grad():
            kl_divergence = ((old_log_probs - ref_log_probs) * sequence_mask)[:, start:].sum(axis=-1).mean()
            mean_generated_length = sequence_mask[:, start:].cast(paddle.float32).sum(axis=-1).mean()
            max_generated_length = sequence_mask[:, start:].cast(paddle.float32).sum(axis=-1).max()

        rewards = rewards.mean()

        # actor_loss = get_all_reduce_mean(actor_loss)
        # reward_critic_loss = get_all_reduce_mean(reward_critic_loss)
        # rewards = get_all_reduce_mean(rewards)
        # kl_divergence = get_all_reduce_mean(kl_divergence)
        # mean_generated_length = get_all_reduce_mean(mean_generated_length)
        # max_generated_length = get_all_reduce_max(max_generated_length)

        # dist.barrier()

        return {
            "train/actor_loss": kwargs["actor_loss"],
            "train/reward_critic_loss": kwargs["reward_critic_loss"],
            "train/reward": rewards,
            "train/kl_divergence": kl_divergence,
            "train/mean_generated_length": mean_generated_length,
            "train/max_generated_length": max_generated_length,
            "train/actor_lr": self.policy_trainer._get_learning_rate(),
            "train/reward_critic_lr": self.value_trainer._get_learning_rate(),
        }, kwargs

    def ptx_step(self, ptx_batch: Dict[str, paddle.Tensor], **kwargs) -> Dict[str, Any]:
        """Perform a single update step with PTX loss."""
        # ptx_loss = self.policy_trainer.training_step(ptx_batch)
        # print(ptx_batch)
        kwargs = self.policy_trainer.full_training_step(ptx_batch, **kwargs)

        # ptx_loss = get_all_reduce_mean(ptx_loss)

        return {"train/ptx_loss": kwargs["ptx_loss"]}, kwargs

    def split_ptx_micro_batches(
        self,
        ptx_batch: Dict[str, paddle.Tensor],
    ) -> List[Dict[str, paddle.Tensor]]:
        """Split a batch of PTX samples into micro-batches."""
        micro_batches = []
        total_batch_size = ptx_batch["input_ids"].shape[0]
        micro_batch_size = self.args.per_device_train_batch_size
        for i in range(0, total_batch_size, micro_batch_size):
            micro_batch = map_structure(
                # pylint: disable-next=cell-var-from-loop
                lambda tensor: tensor[i : i + micro_batch_size],  # noqa: B023
                ptx_batch,
            )
            micro_batches.append(micro_batch)
        return micro_batches

    def split_rl_micro_batches(
        self,
        prompt_only_batch: PromptOnlyBatch,
    ) -> List[PromptOnlyBatch]:
        """Split a batch of RL samples into micro-batches."""
        total_batch_size = prompt_only_batch["input_ids"].shape[0]
        micro_batch_size = self.args.per_device_train_batch_size
        micro_batches = []
        for i in range(0, total_batch_size, micro_batch_size):
            micro_batch = {}
            micro_batch = map_structure(
                lambda tensor: tensor[i : i + micro_batch_size],
                prompt_only_batch,
            )
            micro_batches.extend(self.rollout(micro_batch))
        return micro_batches

    @paddle.no_grad()
    def rollout(self, prompt_only_batch: PromptOnlyBatch) -> List[Dict[str, Any]]:
        """Rollout a batch of experiences."""
        input_ids = prompt_only_batch["input_ids"]
        # NOTE: generation output of paddlenlp do not contain prompt, we should
        # change sequences here.
        # print("=" * 20, type(self.policy_trainer.model_wrapped),
        #       type(self.policy_trainer.model))
        # sequences = self.policy_trainer.model_wrapped.generate(
        # sequences = self.actor_model.generate(
        sequences = self.policy_trainer.model.generate(
            input_ids=input_ids,
            attention_mask=prompt_only_batch["attention_mask"],
            generation_config=self.generation_config,
            synced_gpus=True,
        )[0]
        sequences = sequences.reshape([input_ids.shape[0], self.args.num_return_sequences, -1]).transpose([1, 0, 2])

        return [
            self.post_rollout(
                input_ids,
                seq,
                attention_mask=paddle.logical_and(
                    seq != self.tokenizer.pad_token_id,
                    seq != self.tokenizer.unk_token_id,
                ),
            )
            for seq in sequences
        ]

    @paddle.no_grad()
    def post_rollout(
        self,
        prompt: paddle.Tensor,
        sequence: paddle.Tensor,
        attention_mask: paddle.Tensor,
    ) -> Dict[str, Any]:
        if self.reward_tokenizer is not self.tokenizer:
            reward_tokenize_output = batch_retokenize(
                sequence,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.reward_tokenizer,
                skip_special_tokens=True,
            )
            reward_seq = reward_tokenize_output["input_ids"]
            reward_attention_mask = reward_tokenize_output["attention_mask"]
        else:
            reward_seq = sequence
            reward_attention_mask = attention_mask

        logits = self.policy_trainer.model(
            sequence,
            # logits = self.actor_model(sequence,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits
        ref_logits = self.reference_model(sequence, attention_mask=attention_mask, return_dict=True).logits

        reward_score = self.reward_model(reward_seq, attention_mask=reward_attention_mask, return_dict=True).end_scores
        reward_value = self.value_trainer.model(
            sequence,
            # reward_value = self.reward_critic_model(sequence,
            attention_mask=attention_mask,
            return_dict=True,
        ).scores

        reward_score = reward_score.squeeze(axis=-1)
        reward_value = reward_value.squeeze(axis=-1)[:, :-1]

        log_probs = gather_log_probabilities(logits[:, :-1], sequence[:, 1:])
        ref_log_probs = gather_log_probabilities(ref_logits[:, :-1], sequence[:, 1:])
        return {
            "prompt": prompt,
            "log_probs": log_probs,
            "ref_log_probs": ref_log_probs,
            "rewards": reward_score,
            "reward_values": reward_value,
            "input_ids": sequence,
            "attention_mask": attention_mask,
        }
