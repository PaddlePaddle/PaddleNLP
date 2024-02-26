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
import inspect
import itertools
import math
import os
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
from models.ppo_model_utils import RLHFPPOMixedLoss, RLHFValueLoss, create_loss
from paddle.distributed import fleet
from paddle.io import DataLoader, Dataset, DistributedBatchSampler
from paddle.utils import map_structure
from rich.console import Console
from rich.table import Table

from paddlenlp.data import DataCollator
from paddlenlp.generation import GenerationConfig
from paddlenlp.generation.utils import GenerationMixin
from paddlenlp.trainer.plugins.unified_checkpoint import flatten_list
from paddlenlp.trainer.trainer import (
    TRAINER_STATE_NAME,
    EvalLoopOutput,
    EvalPrediction,
    HybridParallelOptimizer,
    NlpDistributedBatchSampler,
    ShardingOption,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    _obtain_optimizer_parameters_list,
    distributed_file,
    distributed_isfile,
    fused_allreduce_gradients,
    logger,
    reshard_util,
    speed_metrics,
    split_inputs_sequence_dim,
)
from paddlenlp.transformers import BatchEncoding, PretrainedModel, PretrainedTokenizer
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_outputs import ModelOutput
from paddlenlp.transformers.tokenizer_utils_base import (
    PaddingStrategy,
    TruncationStrategy,
)
from paddlenlp.utils.distributed import distributed_gather

global_dev_id = 0 if paddle.get_device() == "cpu" else int(paddle.get_device().split(":")[1])


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


def init_train_model_opt(
    self: Trainer, max_steps: int, resume_from_checkpoint: bool = False, clear_master_weight: bool = False
) -> PretrainedModel:
    # Copy of model/optimizer init and resuming related code in `Trainer.train`.
    # NOTE: this `_load_from_checkpoint` is indeed to load model states in the
    # following elif-else branches, though they are apart away in `Trainer.train`.
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

    if ShardingOption.FULL_SHARD in self.args.sharding and clear_master_weight:
        # for inference model to use Trainer sharding stage3, clear master_weight
        # which is created in GroupShardedStage3.__init__
        self.optimizer._master_weights = None

    if self.args.device == "npu" and self.args.flatten_param_grads:
        from .plugins.npu_plugin import npu_accelerate_plugin

        npu_accelerate_plugin(self.optimizer)

    return model


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


def full_training_step(self: Trainer, inputs: Dict[str, paddle.Tensor], **kwargs):
    """
    Just a copy of single training step complete code in Trainer.train while loop
    which including forward+backward+step, while wraps the inputs and outputs to
    make the complicated copied code no need to change. Maybe a better way is to
    add fine-grained methods including these steps to Trainer which is similar to
    DeepSpeed engine.
    """

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

    if self.args.use_hybrid_parallel and self.args.sep_parallel_degree > 1:
        inputs = split_inputs_sequence_dim(inputs)
    self.timers and self.timers("read-data").stop()
    os.environ["TRAINER_GLOBAL_STEP"] = str(self.state.global_step)
    self.callback_handler.on_load_data_end(args, self.state, self.control, inputs=inputs)

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
        final_local_vars = locals()
        for k in kwargs.keys():
            if k in final_local_vars:
                kwargs[k] = final_local_vars[k]
        return kwargs
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

    dp_master_grad = self.args.world_size > 1 and self.args.amp_master_grad and not self.args.use_hybrid_parallel
    if dp_master_grad:
        is_no_sync = True

    if is_no_sync:
        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
        with model.no_sync():
            tr_loss_step = self.training_step(model, inputs)
    else:
        tr_loss_step = self.training_step(model, inputs)

    tr_loss += tr_loss_step

    if (step_control + 1) % args.gradient_accumulation_steps == 0 or (
        # last step in epoch but step is always smaller than gradient_accumulation_steps
        steps_in_epoch <= args.gradient_accumulation_steps
        and (step + 1) == steps_in_epoch
    ):
        if self.args.pipeline_parallel_degree <= 1 and self._enable_delay_scale_loss():
            tr_loss /= self.args.gradient_accumulation_steps

        self.timers and self.timers("forward-backward").stop()
        # Maunally collect gradients
        # Case 1: Use recompute and dp
        # Case 2: Hack dp with master_grad
        # Case 3: Pipeline or sharding overlap
        # local_rank != -1 don't means dp in networks.
        self.timers and self.timers("all-reduce").start()

        # Case 1: Use recompute and dp / sharding stage1,
        # manualy collect gradient for dp.
        if args.recompute and availiable_no_sync:
            fused_allreduce_gradients(list(model.parameters()), None)

        # Case 2: hack dp with master_grad
        if dp_master_grad and not (args.recompute and availiable_no_sync):
            fused_allreduce_gradients(list(model.parameters()), None)

        # Pipeline parallel mode,  handle gradient reduce here to overlap
        pipeline_parallel_config = (
            set(args.pipeline_parallel_config.split(" ")) if args.pipeline_parallel_degree > 1 else set()
        )
        enable_dp_comm_overlap = "enable_dp_comm_overlap" in pipeline_parallel_config
        enable_release_grads = "enable_release_grads" in pipeline_parallel_config

        # Case 3: Pipeline parallel mode, overlap with dp
        if isinstance(self.optimizer, HybridParallelOptimizer) and not self.do_grad_scaling:
            parameters_list = _obtain_optimizer_parameters_list(self.optimizer._inner_opt)

            if not enable_dp_comm_overlap:
                if self.optimizer._sharding_enable:
                    assert reshard_util.is_sharding_opt(self.optimizer)
                    self.optimizer._inner_opt.reduce_gradients(list(parameters_list), self.optimizer._hcg)

                if self.optimizer._dp_enable or getattr(self.optimizer, "_sep_enable", False):
                    fused_allreduce_gradients(list(parameters_list), self.optimizer._hcg)

        self.timers and self.timers("all-reduce").stop()
        self.timers and self.timers("optimizer-step").start()

        if self.args.gradient_accumulation_steps > 1 and self._enable_delay_scale_loss():
            for p in model._layers.parameters():
                with paddle.no_grad():
                    if hasattr(p, "main_grad") and p.main_grad is not None:
                        assert p.grad is None
                        p.main_grad.scale_(1.0 / self.args.gradient_accumulation_steps)
                    elif p.grad is not None:
                        p.grad.scale_(1.0 / self.args.gradient_accumulation_steps)

        # Optimizer step
        self.callback_handler.on_optimizer_begin(
            args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
        )
        optimizer_was_run = True
        if self.do_grad_scaling:
            scale_before = paddle.assign(self.scaler._scale)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler._scale
            optimizer_was_run = not self.scaler._cache_founf_inf
            if not optimizer_was_run:
                scale_before_value = scale_before.cpu().numpy()
                scale_after_value = scale_after.cpu().numpy()
                logger.warning(
                    f"optimizer not run, scale_before: {scale_before_value[0]}, scale_after: {scale_after_value[0]}"
                )
        elif isinstance(self.optimizer, HybridParallelOptimizer):
            self.optimizer._step(parameters_list)
        else:
            self.optimizer.step()

        self.timers and self.timers("optimizer-step").stop()

        if optimizer_was_run:
            self.lr_scheduler.step()

        if enable_release_grads and args.pipeline_parallel_degree > 1:
            self.optimizer.clear_grad(set_to_zero=False)
            for _, buffers in model._chunk_2_comm_buffers.items():
                for buffer in buffers:
                    buffer._clear_grad_storage()
        else:
            self.optimizer.clear_grad()

        self.callback_handler.on_optimizer_end(
            args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
        )

        self.state.global_step += 1
        self.state.epoch = epoch + (step + 1) / steps_in_epoch
        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
        self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
        self._print_timer()
        step_control = 0
    else:
        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
        step_control += 1

    if self.control.should_epoch_stop or self.control.should_training_stop:
        # break
        final_local_vars = locals()
        for k in kwargs.keys():
            if k in final_local_vars:
                kwargs[k] = final_local_vars[k]
        return kwargs
    self.timers and self.timers("read-data").start()

    final_local_vars = locals()
    for k in kwargs.keys():
        if k in final_local_vars:
            kwargs[k] = final_local_vars[k]
    return kwargs


def offload_tensor_to_cpu(tensor):
    if isinstance(tensor, dict):
        for _, v in tensor.items():
            offload_tensor_to_cpu(v)
    elif isinstance(tensor, paddle.Tensor):
        if not tensor.place.is_cpu_place():
            cpu_tensor = tensor._copy_to(paddle.CPUPlace(), True)
            tensor.value().get_tensor()._share_data_with(cpu_tensor.value().get_tensor())
    else:
        logger.warning(f"Can't parse for type {type(tensor)}")
        return tensor


def reload_tensor_to_gpu(tensor):
    if isinstance(tensor, dict):
        for _, v in tensor.items():
            reload_tensor_to_gpu(v)
    elif isinstance(tensor, paddle.Tensor):
        if not tensor.place.is_gpu_place():
            gpu_tensor = tensor._copy_to(paddle.CUDAPlace(global_dev_id), True)
            tensor.value().get_tensor()._share_data_with(gpu_tensor.value().get_tensor())
    else:
        logger.warning(f"Can't parse for type {type(tensor)}")
        return tensor


def cleanup_tensor_space(tensor):
    if isinstance(tensor, dict):
        for _, v in tensor.items():
            cleanup_tensor_space(v)
    elif isinstance(tensor, paddle.Tensor):
        tensor._clear_data()
    else:
        logger.warning(f"Can't parse for type {type(tensor)}")
        return tensor


def export_evaluate_model(self: Trainer, train_model, eval_model, **kwargs):
    if eval_model is None:
        return None

    with_offload = kwargs.pop("with_offload", False)
    train_tp_size = max(train_model.config.tensor_parallel_degree, 1)
    eval_tp_size = max(eval_model.config.tensor_parallel_degree, 1)

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()
    sd_group = hcg.get_sharding_parallel_group()
    dp_group = hcg.get_data_parallel_group()

    global_rank = paddle.distributed.get_rank()

    train_state_dict = train_model.state_dict()
    eval_state_dict = eval_model.state_dict()

    print(sd_group)

    if dp_group.rank <= 0 and sd_group.rank <= 0:
        train_pp_size = pp_group.nranks
        if eval_tp_size > 1 and train_tp_size != eval_tp_size:
            raise ValueError("Only support for the same tensor_parallel_degree for train and eval model for now.")

        # 单卡情况
        # tp->single
        # tp+pp -> single
        if eval_tp_size == 1:
            if train_pp_size == 1 and train_tp_size > 1:
                # tp ->single
                logger.error("using tp to single eval model.")
                # state = train_model.merge_tensor_parallel()
                tp_actions = train_model.get_tensor_parallel_convert_actions(
                    train_model.config,
                    loaded_state_dict_keys=eval_state_dict.keys(),
                    is_split=False,
                    ignore_error=False,
                )

                # print(tp_actions.keys())

                is_dst = global_rank == 0
                for key in eval_state_dict.keys():
                    # print(f"get key {key}")
                    tensor = train_state_dict[key]
                    if key in tp_actions:
                        ret = distributed_gather(tensor, dst=0, group=tp_group, offload=False)
                        action = tp_actions.pop(key)
                        tensor = action(ret) if is_dst else None
                        # print(f"merge {key}")
                    else:
                        tensor = tensor._copy_to(paddle.CPUPlace(), False) if is_dst else None

                    if tensor is not None:
                        # print(tensor.shape)
                        eval_state_dict[key].set_value(tensor)

                    if not eval_state_dict[key]._is_initialized():
                        v = eval_state_dict[key]
                        t = paddle._C_ops.full_like(v, 0, v.dtype, paddle.CUDAPlace(global_dev_id))
                        v.get_tensor()._share_data_with(t.get_tensor())

                    if with_offload:
                        offload_tensor_to_cpu(train_state_dict[key])
            else:
                # single to single
                # tp+pp -> single
                raise ValueError("Not support yet.")

        def create_send_recv_table(train_keys, eval_keys):
            recv_table = []
            send_table = []
            if pp_group.rank == 0:
                for key in eval_keys:
                    recv_table.append((key, global_rank))

            for key in train_keys:
                send_table.append((key, global_rank))

            all_recv, all_send = [], []
            paddle.distributed.all_gather_object(all_recv, [recv_table], group=pp_group)
            paddle.distributed.all_gather_object(all_send, [send_table], group=pp_group)
            all_recv = flatten_list(all_recv)
            all_send = flatten_list(all_send)

            send_dict = {}
            for k, v in all_send:
                send_dict[k] = v

            table = []
            for k, v in all_recv:
                # key, send, recv
                table.append([k, send_dict.pop(k), v])
            assert len(send_dict) == 0, f"Some key can't be recv {send_dict.keys()}"
            return table

            # pp0tp0 -> pp0tp0
            # pp0tp1 -> pp0tp1
            # pp1tp0 -> pp0tp0
            # pp1tp1 -> pp0tp1

        # tp情况
        # tp+pp->tp
        if eval_tp_size > 1 and train_pp_size > 1:
            table = create_send_recv_table(train_state_dict.keys(), eval_state_dict.keys())
            # print(table)

            for key, src_rank, dst_rank in table:
                # Init tensor for model is cleaned
                # print(key, src_rank, dst_rank, eval_state_dict[key]._is_initialized())
                # if key in train_state_dict:
                #     print(train_state_dict[key]._is_initialized())

                if not eval_state_dict[key]._is_initialized():
                    v = eval_state_dict[key]
                    t = paddle._C_ops.full_like(v, 0, v.dtype, paddle.CUDAPlace(global_dev_id))
                    v.get_tensor()._share_data_with(t.get_tensor())

                if src_rank == dst_rank and global_rank == src_rank:
                    eval_state_dict[key].copy_(train_state_dict[key], True)
                else:
                    if global_rank == src_rank:
                        dist.stream.send(train_state_dict[key], dst=dst_rank)

                    if global_rank == dst_rank:
                        dist.stream.recv(eval_state_dict[key], src=src_rank)

                # Offload train model if need
                if global_rank == src_rank and with_offload:
                    offload_tensor_to_cpu(train_state_dict[key])
    else:
        # 其他 DP rank 的state dict, 适配 offload 和初始化
        if with_offload:
            for key in list(train_state_dict.keys()):
                offload_tensor_to_cpu(train_state_dict[key])
        for k, v in eval_state_dict.items():
            if not v._is_initialized():
                # print(f"init {k}")
                t = paddle._C_ops.full_like(v, 0, v.dtype, paddle.CUDAPlace(global_dev_id))
                v.get_tensor()._share_data_with(t.get_tensor())

    paddle.distributed.barrier()
    if eval_tp_size == 1:
        for _, tensor in eval_state_dict.items():
            paddle.distributed.broadcast(tensor, src=0, group=None, sync_op=True)
    else:
        if sd_group.nranks > 1:
            if dp_group.rank <= 0:
                paddle.distributed.parallel.sync_params_buffers(
                    eval_model, comm_group=sd_group, src_rank=sd_group.ranks[0], fuse_params=False
                )
        if dp_group.nranks > 1:
            paddle.distributed.parallel.sync_params_buffers(
                eval_model, comm_group=dp_group, src_rank=dp_group.ranks[0], fuse_params=False
            )


Trainer.init_train_model_opt = init_train_model_opt
Trainer.init_train_log = init_train_log
Trainer.init_train_state = init_train_state
Trainer.full_training_step = full_training_step

Trainer.export_evaluate_model = export_evaluate_model


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
        # only used for non-PipelineParallel models
        criterion = create_loss(RLHFPPOMixedLoss, model.config, args)
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
        inputs = super()._prepare_input(inputs)
        label_names = self.criterion.__class__.label_names
        # some data fields are used both in model and loss
        shared_fields = set(["input_ids", "attention_mask"])
        labels = []
        for name in label_names:
            if name not in inputs:
                label = self.criterion.__class__.label_default_values.get(name, None)
            elif name in shared_fields:
                label = inputs[name]
            else:
                label = inputs.pop(name)
            labels.append(label)
        # "labels" is the pre-defined label name in Trainer
        inputs["labels"] = labels
        # NOTE: TensorParallel model requires non-Tensor inputs to be lists and
        # broadcast them, thus do not or optionally use these inputs. labels use
        # in criterion not send to model can workaround this.
        return inputs

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
        # only used for non-PipelineParallel models
        criterion = create_loss(RLHFValueLoss, model.config, args)
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
        inputs = super()._prepare_input(inputs)
        label_names = self.criterion.__class__.label_names
        # some data fields are used both in model and loss
        shared_fields = set(["input_ids", "attention_mask"])
        labels = []
        for name in label_names:
            if name not in inputs:
                label = self.criterion.__class__.label_default_values.get(name, None)
            elif name in shared_fields:
                label = inputs[name]
            else:
                label = inputs.pop(name)
            labels.append(label)
        # "labels" is the pre-defined label name in Trainer
        inputs["labels"] = labels
        # NOTE: TensorParallel model requires non-Tensor inputs to be lists and
        # broadcast them, thus do not or optionally use these inputs. labels use
        # in criterion not send to model can workaround this.
        return inputs

    def full_training_step(self: Trainer, inputs: Dict[str, paddle.Tensor], **kwargs):
        # TODO(guosheng): Make these training control vars mapping as class attr,
        # then PPOTrainer can extract and reuse them to avoid hard code.
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


class PipeEvalModel(GenerationMixin):
    def __init__(self, trainer: Trainer):
        self.model: fleet.model.PipelineParallel = trainer.model_wrapped
        self.config: PretrainedConfig = trainer.model.config
        self._is_gen = False
        # self.gen_fn = None
        # self.fwd_fn = None
        # use non-pipe model generetion related methods
        # self.prepare_inputs_for_generation = types.MethodType(
        #     self.model._layers._non_pipe_model_class.prepare_inputs_for_generation, self
        # )
        self.update_model_kwargs_for_generation = (
            self.model._layers._non_pipe_model_class.update_model_kwargs_for_generation
        )

    @property
    def pp_group(self):
        return self.model.pp_group

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def _broadcast_outputs(self, outputs):
        # outputs is PipelineParallel.eval_batch which is a list of batches.
        out = []
        outputs = (outputs,) if isinstance(outputs, paddle.Tensor) else outputs
        for tensors in outputs:
            if not self.model.is_pipeline_last_stage():
                tensor = tensors if isinstance(tensors, paddle.Tensor) else tensors[0]
                head_out_meta = (
                    (self.model._layers.head_out_meta,)
                    if isinstance(self.model._layers.head_out_meta, paddle.static.InputSpec)
                    else self.model._layers.head_out_meta
                )
                tensors = tuple(
                    paddle.empty(
                        shape=[
                            tensor.shape[i] if (meta.shape[i] is None or meta.shape[i] < 0) else meta.shape[i]
                            for i in range(len(meta.shape))
                        ],
                        dtype=tensor.dtype if meta.dtype is None else meta.dtype,
                    )
                    for meta in head_out_meta
                )
            else:
                # Currently use tuple instead of ModelOutput and require the
                # caller use the return result as tuple.
                tensors = (
                    (tensors,)
                    if isinstance(tensors, paddle.Tensor)
                    else tensors.to_tuple()
                    if isinstance(tensors, ModelOutput)
                    else tensors
                )

            # map_structure(
            #     lambda tensor: paddle.distributed.broadcast(
            #         tensor,
            #         src=self.model.pp_group.ranks[-1],
            #         group=self.model.pp_group), tensors)
            for tensor in tensors:
                paddle.distributed.broadcast(tensor, src=self.model.pp_group.ranks[-1], group=self.model.pp_group)
            out.append(tensors[0] if len(tensors) == 1 else tensors)
        return out[0] if len(out) == 1 else out

    def __call__(self, *args, **kwargs):
        model = self.model
        assert self.model.training is False
        if self._is_gen:
            # inputs by `prepare_inputs_for_generation` is a dict with following keys:
            # "input_ids", "position_ids", "past_key_values", "use_cache", "attention_mask"
            # NOTE: 1. cache/past_key_values should be passed across decoding steps
            # by using as model attr rather than input args to reduce comm overhead.
            # Also, pipe model defined for training not support this cache input.
            # 2. ignore use_cache since _check_data_vaild requires tensor if not None.
            # 3. attention_mask can reuse _prepare_decoder_attention_mask in LlamaEmbeddingPipe.
            # 4. position_ids pass through _prepare_pipeline_inputs_func and PipeLayer.
            inputs, labels = model._prepare_pipeline_inputs_func(*args, **kwargs)
            # currently, set accumulate_steps to 1 to avoid multi-batch eval/gen
            with guard_set_args(model, {"_compute_loss": False, "accumulate_steps": 1}):
                outputs = model.eval_batch([inputs, labels], compute_loss=False)
            # TODO(guosheng): Broadcasted logits are used to get next_scores, remove
            # it to reduce comm overhead. Also note that we still need broadcast
            # next_tokens though logits are broadcasted since pp ranks' seeds differs.
            # Currently, just slice the last token to reduce comm overhead.
            outputs = [
                micro_batch_output[:, -1, :].unsqueeze(1)
                if isinstance(micro_batch_output, paddle.Tensor)
                else micro_batch_output[0][:, -1, :].unsqueeze(1)
                for micro_batch_output in outputs
            ]
            outputs = self._broadcast_outputs(outputs)
        else:
            # use _prepare_pipeline_inputs_func to convert pipeline inputs
            inputs, labels = model._prepare_pipeline_inputs_func(*args, **kwargs)
            # NOTE(guosheng): bug seems exist. pp.eval_batch(compute_loss=False)
            # will set pp._compute_loss to False and would not set it back. Thus
            # hack here to set it back.
            with guard_set_args(model, {"_compute_loss": False, "accumulate_steps": 1}):
                outputs = model.eval_batch([inputs, labels], compute_loss=False)
            outputs = self._broadcast_outputs(outputs)
        return outputs

    def generate(self, *args, **kwargs):
        self._is_gen = True
        # patch DecoderLayerPipe to use cache, DecoderLayerPipe is subclass of
        # DecoderLayer, and would call super().forward
        ori_decoder_layer_forward = self.model._layers._non_pipe_decoder_layer_class.forward

        def decoder_layer_forward(layer_self, *args, **kwargs):
            kwargs.update({"use_cache": True, "past_key_value": getattr(layer_self, "_cache", None)})
            outputs = ori_decoder_layer_forward(layer_self, *args, **kwargs)
            output = outputs[0]
            layer_self._cache = outputs[1]
            self._has_cache = True
            return output

        with guard_set_args(self.model._layers._non_pipe_decoder_layer_class, {"forward": decoder_layer_forward}):
            outputs = super().generate(*args, **kwargs)
        self._is_gen = False
        # clear cache of decoder layers, sublayers is incursive thus suitable
        # to both 1F1B and interleave
        for layer in self.model._layers.sublayers():
            if isinstance(layer, self.model._layers._non_pipe_decoder_layer_class):
                layer._cache = None
        self._has_cache = False
        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        arg_bind = inspect.signature(self.model._layers._non_pipe_model_class.prepare_inputs_for_generation).bind(
            *((self,) + args), **kwargs
        )
        arg_bind.apply_defaults()
        arg_dict = arg_bind.arguments
        last_arg_name, last_arg_value = arg_dict.popitem()
        if arg_bind.signature.parameters[last_arg_name].kind == inspect.Parameter.VAR_KEYWORD:
            arg_dict.update(last_arg_value)
        else:
            arg_dict[last_arg_name] = last_arg_value
        arg_dict.pop("self")
        past_key_values = arg_dict.get("past_key_values", None)
        # prepare_inputs_for_generation use past_key_values to discrimate prefill
        # or decode and slice inputs accordingly.
        if getattr(self, "_has_cache", False):
            arg_dict.update({"past_key_values": True})
        model_inputs = self.model._layers._non_pipe_model_class.prepare_inputs_for_generation(self, **arg_dict)
        model_inputs.update({"past_key_values": past_key_values})
        return model_inputs


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
        with guard_set_args(
            args,
            {
                "recompute": False,
                "fp16_opt_level": "O1",
                "pipeline_parallel_degree": 1,  # workaround for pipeline parallel model check
            },
        ):
            # just used to create trival attrs might be used in the training
            # process of trainer, while changing some args to avoid model usage
            # in __init__ such as recompute and AMP-O2
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

        self.train_dataset = train_dataset
        self.ptx_dataset = ptx_dataset
        self.eval_dataset = eval_dataset

        (policy_model, reference_model, reward_model, value_model, policy_model_eval, value_model_eval) = model
        self._policy_model_eval = policy_model_eval
        self._value_model_eval = value_model_eval

        # policy_tokenizer and value_tokenizer should be same
        (policy_tokenizer, reference_tokenizer, reward_tokenizer, value_tokenizer) = tokenizer

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

        # use trainer for reference_model/reward_model to enable sharding stage-3
        # maybe we should allow models to use  different dist strategies later
        if True:  # ShardingOption.FULL_SHARD in args.sharding:
            self.reference_trainer = Trainer(
                reference_model,
                criterion,
                args,
                data_collator,
                train_dataset,
                eval_dataset,
                reference_tokenizer,
                compute_metrics,
                callbacks,
                optimizers,
                preprocess_logits_for_metrics,
            )
            self.reward_trainer = Trainer(
                reward_model,
                criterion,
                args,
                data_collator,
                train_dataset,
                eval_dataset,
                reward_tokenizer,
                compute_metrics,
                callbacks,
                optimizers,
                preprocess_logits_for_metrics,
            )
            # TODO(guosheng): sharding stage3 should create master weight optionally
            # instead of creation and clear.
            self.reference_trainer.init_train_model_opt(100, None, clear_master_weight=True)  # dummy max_steps
            self.reward_trainer.init_train_model_opt(100, None, clear_master_weight=True)  # dummy max_steps
        else:
            self._reference_model = reference_model
            self._reward_model = reward_model
        self.reference_model.eval()
        self.reward_model.eval()

        self.reward_tokenizer = reward_tokenizer
        self.tokenizer = policy_tokenizer
        if is_same_tokenizer(self.tokenizer, self.reward_tokenizer):
            self.reward_tokenizer = self.tokenizer

        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_length,
            num_return_sequences=self.args.num_return_sequences,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            # top_k=self.args.top_k,
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
        self.policy_trainer.ptx_coeff = self.ptx_coeff = self.args.ptx_coeff
        self.gamma = 1.0
        self.gae_lambda = 0.95

        # dummy class and object for model to be compaible with methods of
        # Trainer, such as evaluation_loop
        self.DummyPPOModel = type(
            "DummyPPOModel", (object,), {"eval": lambda _: self.set_eval(), "train": lambda _: self.set_train()}
        )
        self.model = self.model_wrapped = self.DummyPPOModel()
        # self.optimizer = self.policy_trainer.optimizer
        # self.scaler = self.reference_trainer.scaler = self.reward_trainer.scaler = None

    @property
    def reference_model(self):
        model = getattr(self, "_reference_model", None)
        if model is not None:
            return model
        # use model with Trainer
        if self.reference_trainer.args.pipeline_parallel_degree > 1:
            # Only accept wrapped model for pipeline_parallel mode
            # model = self.reference_trainer.model_wrapped
            model = PipeEvalModel(self.reference_trainer)
            self._reference_model = model
        else:
            model = self.reference_trainer.model
        return model

    @property
    def reward_model(self):
        model = getattr(self, "_reward_model", None)
        if model is not None:
            return model
        # use model with Trainer
        if self.reward_trainer.args.pipeline_parallel_degree > 1:
            # Only accept wrapped model for pipeline_parallel mode
            # model = self.reward_trainer.model_wrapped
            model = PipeEvalModel(self.reward_trainer)
            self._reward_model = model
        else:
            model = self.reward_trainer.model
        return model

    @property
    def actor_model(self):
        if self.training:
            return self.policy_trainer.model_wrapped
        model = getattr(self, "_actor_model", None)
        if model is not None:
            return model
        if self.policy_trainer.args.pipeline_parallel_degree > 1:
            # Only accept wrapped model for pipeline_parallel mode
            # model = self.policy_trainer.model_wrapped
            model = PipeEvalModel(self.policy_trainer)
            self._actor_model = model
        else:
            model = self.policy_trainer.model
        return model

    @property
    def reward_critic_model(self):
        if self.training:
            return self.value_trainer.model_wrapped
        model = getattr(self, "_reward_critic_model", None)
        if model is not None:
            return model
        if self.value_trainer.args.pipeline_parallel_degree > 1:
            # Only accept wrapped model for pipeline_parallel mode
            # model = self.value_trainer.model_wrapped
            model = PipeEvalModel(self.value_trainer)
            self._reward_critic_model = model
        else:
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

    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        with paddle.no_grad():
            with self.autocast_smart_context_manager():
                seq = self.actor_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=self.generation_config,
                    synced_gpus=ShardingOption.FULL_SHARD in self.policy_trainer.args.sharding,
                )[0]
                attention_mask = paddle.logical_and(
                    seq != self.tokenizer.pad_token_id,
                    seq != self.tokenizer.unk_token_id,
                )
                if self.reward_tokenizer is not self.tokenizer:
                    reward_tokenize_output = batch_retokenize(
                        input_ids=seq,
                        src_tokenizer=self.tokenizer,
                        dest_tokenizer=self.reward_tokenizer,
                        skip_special_tokens=True,
                        device=self.args.device,
                    )
                    reward_input_ids = reward_tokenize_output["input_ids"]
                    reward_attention_mask = reward_tokenize_output["attention_mask"]
                else:
                    reward_input_ids = seq
                    reward_attention_mask = attention_mask

                # unify PP with others since PP always return tuple
                reward_score = self.reward_model(
                    reward_input_ids,
                    attention_mask=reward_attention_mask,
                    # return_dict=True,
                )[
                    1
                ]  # .end_scores
                reward_score = reward_score.squeeze(axis=-1).cast(paddle.float32)

        # keep the first batch of eval output sequence to print and check
        prompt = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        generated = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
        for i, text in enumerate(generated):
            self._eval_out_file.write(text + "\n")
        if getattr(self, "_eval_seq", None) is None:
            generated = [text[len(prompt[i]) :] for i, text in enumerate(generated)]
            # prompts.extend(prompt)
            # generateds.extend(generated)
            self._eval_seq = (prompt, generated, reward_score.tolist())

        return reward_score.mean(), None, None

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_eval_iters: Optional[int] = -1,
    ) -> EvalLoopOutput:
        # to save eval generated sequence
        eval_out_file = os.path.join(
            self.args.output_dir, f"eval_out-step{self.state.global_step}-rank{self.args.local_rank}.txt"
        )
        self._eval_out_file = open(eval_out_file, "w")

        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix, max_eval_iters
        )
        output.metrics[f"{metric_key_prefix}/reward"] = output.metrics.pop(f"{metric_key_prefix}_loss")

        columns = ["Prompt", "Generated", "Reward"]
        rows = list(zip(*self._eval_seq))
        rows = [[str(item) for item in row] for row in rows]
        max_num_rows = 5
        table = Table(title="Evaluating...", show_lines=True, title_justify="left")
        for column in columns:
            table.add_column(column)
        for row in rows[:max_num_rows]:
            table.add_row(*row)
        Console(soft_wrap=True, markup=False, emoji=False).print(table)
        self._eval_seq = None

        self._eval_out_file.close()

        return output

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        with guard_set_args(self, {"data_collator": self.eval_dataset.get_collator()}):
            return super().get_eval_dataloader(eval_dataset)

    def _save_checkpoint(self, model, metrics=None):
        # maybe change args.output_dir of policy_trainer/value_trainer directly
        with guard_set_args(self.policy_trainer.args, {"output_dir": os.path.join(self.args.output_dir, "policy")}):
            self.policy_trainer._save_checkpoint(model, metrics)
        with guard_set_args(self.value_trainer.args, {"output_dir": os.path.join(self.args.output_dir, "value")}):
            self.value_trainer._save_checkpoint(model, metrics)

    # def _load_from_checkpoint(self, resume_from_checkpoint=None):
    #     with guard_set_args(self.policy_trainer.args, {"output_dir": os.path.join(self.args.output_dir, "policy")}):
    #         self.policy_trainer._load_from_checkpoint(resume_from_checkpoint)
    #     with guard_set_args(self.value_trainer.args, {"output_dir": os.path.join(self.args.output_dir, "value")}):
    #         self.value_trainer._load_from_checkpoint(resume_from_checkpoint)

    # def _load_optimizer_and_scheduler(self, checkpoint):
    #     # NOTE: `Trainer._load_optimizer_and_scheduler` would not seek the latest
    #     # state as in `_load_from_checkpoint``, and it just use `resume_from_checkpoint`
    #     # as value of `checkpoint` to load.
    #     self.policy_trainer._load_optimizer_and_scheduler(
    #         checkpoint if checkpoint is None else os.path.join(checkpoint, "policy")
    #     )
    #     self.value_trainer._load_optimizer_and_scheduler(
    #         checkpoint if checkpoint is None else os.path.join(checkpoint, "value")
    #     )

    def init_train_model_opt(
        self: Trainer, max_steps: int, resume_from_checkpoint: bool = False, clear_master_weight: bool = False
    ) -> PretrainedModel:
        # resume should be triggered here
        # maybe change args.output_dir of policy_trainer/value_trainer directly
        with guard_set_args(self.policy_trainer.args, {"output_dir": os.path.join(self.args.output_dir, "policy")}):
            policy_model = self.policy_trainer.init_train_model_opt(
                max_steps,
                os.path.join(resume_from_checkpoint, "policy")
                if isinstance(resume_from_checkpoint, str)
                else resume_from_checkpoint,
            )
        with guard_set_args(self.value_trainer.args, {"output_dir": os.path.join(self.args.output_dir, "value")}):
            value_model = self.value_trainer.init_train_model_opt(
                max_steps,
                os.path.join(resume_from_checkpoint, "value")
                if isinstance(resume_from_checkpoint, str)
                else resume_from_checkpoint,
            )
        return policy_model, value_model

    def load_sing_gen_data(self, as_batches=True, use_counter=False, data_dir="pkl_data"):
        if use_counter:
            iter_counter = getattr(self, "iter_counter", 0)
            self.iter_counter = iter_counter + 1
        else:
            iter_counter = ""
        import pickle

        from paddle.distributed import fleet

        hcg = fleet.get_hybrid_communicate_group()
        data_rank = hcg.get_sharding_parallel_rank()
        with open(os.path.join(data_dir, f"{iter_counter}rl_batch-{data_rank}.data"), "rb") as f:
            data = pickle.load(f)
        rl_batch = map_structure(lambda x: paddle.to_tensor(x), data)
        rl_batches = [rl_batch] if as_batches else rl_batch
        return rl_batches

    def save_single_gen_data(self, rl_batch, use_counter=False, data_dir="pkl_data"):
        if use_counter:
            iter_counter = getattr(self, "iter_counter", 0)
            self.iter_counter = iter_counter + 1
        else:
            iter_counter = ""
        import pickle

        import paddle.distributed as dist

        with open(os.path.join(data_dir, f"{iter_counter}rl_batch-{dist.get_rank()}.data"), "wb") as f:
            rl_batch = map_structure(lambda x: x.numpy(), rl_batch)
            pickle.dump(rl_batch, f)
        # exit(0)

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

                # self.optimizer.offload()
                if self.eval_mode is not None and "optimizer" in self.args.offload_level:
                    offload_tensor_to_cpu(self.policy_trainer.optimizer.state_dict())
                    offload_tensor_to_cpu(self.value_trainer.optimizer.state_dict())

                self.policy_trainer.export_evaluate_model(
                    self.policy_trainer.model,
                    self._policy_model_eval,
                    with_offload=self.args.offload_level is not None,
                )
                # todo: zhui
                self.value_trainer.export_evaluate_model(
                    self.value_trainer.model, self._value_model_eval, with_offload=self.args.offload_level is not None
                )
                # self.reference_model.reload()
                # self.reward_model.reload()
                # reload_tensor_to_gpu(self.reference_model.state_dict())
                # reload_tensor_to_gpu(self.reward_model.state_dict())

                # 生成数据
                rl_batches = self.split_rl_micro_batches(prompt_only_batch)
                # rl_batches = self.load_sing_gen_data(as_batches=True,
                #                                      use_counter=True)
                if self.use_ptx:
                    ptx_batches = self.split_ptx_micro_batches(ptx_batch)
                else:
                    ptx_batches = [None for _ in range(len(rl_batches))]
                paddle.device.cuda.empty_cache()

                # # 数据造好, 开始训练
                # self.reference_model.offload()
                # self.reward_model.offload()
                # policy_model_eval.cleanup()
                # value_model_eval.cleanup()
                if self.args.offload_level is not None:
                    if "eval" in self.args.offload_level:
                        cleanup_tensor_space(self._policy_model_eval.state_dict())
                        cleanup_tensor_space(self._value_model_eval.state_dict())
                    if "reward" in self.args.offload_level:
                        offload_tensor_to_cpu(self.reference_model.state_dict())
                        offload_tensor_to_cpu(self.reward_model.state_dict())

                self.set_train()
                for _ in range(self.args.update_iters):
                    for rl_batch, ptx_batch in zip(rl_batches, ptx_batches):
                        # self.save_single_gen_data(rl_batch, use_counter=True)
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
        # ##### The following code try to keep same as the Trainer.train #####
        args = self.args
        self.is_in_train = True

        # ##### trainging data and related num setting #####
        # TODO(guosheng): remove the binding method get_collator of dataset
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
                self, {"train_dataset": self.ptx_dataset, "data_collator": self.ptx_dataset.get_collator(shift=True)}
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
        # policy_model = self.policy_trainer.init_train_model_opt(max_steps, resume_from_checkpoint)
        # value_model = self.value_trainer.init_train_model_opt(max_steps, resume_from_checkpoint)
        policy_model, value_model = self.init_train_model_opt(max_steps, resume_from_checkpoint)
        paddle.device.cuda.empty_cache()
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

        # train_step_kwargs is used to provide arguments more than model inputs
        # for full_training_step which is copied from Trainer.train and needs
        # these arguments to control training process.
        train_step_kwargs = {
            "ignore_keys_for_eval": None,  # no need
            # TODO(guosheng): commented args mean to resume data, not support yet
            # "resume_from_checkpoint": resume_from_checkpoint,
            # "train_dataloader": train_dataloader,
            # "epochs_trained": epochs_trained,
            # "steps_trained_in_current_epoch": steps_trained_in_current_epoch,
            # "steps_trained_progress_bar": steps_trained_progress_bar,
            "steps_in_epoch": steps_in_epoch,  # to control training process
            # the following args are corresponding to tr_loss and model used in
            # Trainer.train, and they would be used as tr_loss and model in
            # PolicyTranier and ValueTrainer.
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
                # TODO(guosheng): make rl_step/ptx_step run with autocast_smart_context_manager

                # policy_model.reload()
                # value_model.reload()
                reload_tensor_to_gpu(self.actor_model.state_dict())
                reload_tensor_to_gpu(self.reward_critic_model.state_dict())

                rl_info, train_step_kwargs = self.rl_step(rl_batch, **train_step_kwargs)

                paddle.device.cuda.empty_cache()
                if self.use_ptx:
                    ptx_info, train_step_kwargs = self.ptx_step(ptx_batch, **train_step_kwargs)
                    rl_info.update(ptx_info)
                    paddle.device.cuda.empty_cache()

                self.state.global_step = self.value_trainer.state.global_step
                self.state.epoch = self.value_trainer.state.epoch
                if train_step_kwargs["value_step_control"] == 0:
                    # NOTE: PipelineParallel only returns a accumulated loss after
                    # accumulated steps, which is a mixed loss of ppo-loss and
                    # ptx-loss. We hack PipelineParallel._forward_step to record
                    # loss metrics and postprocess the recorded losses here.
                    # Maybe better to make the last_stage worker log to reduce
                    # comm and for simplicity.
                    if isinstance(policy_model, fleet.model.PipelineParallel):
                        with paddle.no_grad():
                            # TODO(guosheng): maybe move this to model_pp.py and
                            # using interface here is better
                            # interleave betweeen ppo-loss and ptx-loss
                            if policy_model.is_pipeline_last_stage():
                                # loss is 0D tensor, use stack rather than concat
                                mix_loss = paddle.stack(policy_model._step_losses)
                                policy_model._step_losses = None
                            else:
                                # The tessor shape is not policy_model.accumulate_steps
                                # (args.accu_steps) but policy_trainer.args.accu_steps,
                                # since policy_model is created with global pp_config
                                # using global args.accu_steps which is only half of
                                # policy_trainer.args.accu_steps, and indeed trainer hack
                                # model.accumulate_steps in training_pipeline_step to use
                                # trainer.args.accu_steps. The dtype is fp32(to be check),
                                # thus no need to broadcast.
                                mix_loss = paddle.empty(
                                    shape=[self.policy_trainer.args.gradient_accumulation_steps], dtype=paddle.float32
                                )
                            paddle.distributed.broadcast(
                                mix_loss, src=policy_model.pp_group.ranks[-1], group=policy_model.pp_group
                            )
                            real_actor_loss = mix_loss[0::2].mean()
                            real_ptx_loss = mix_loss[1::2].mean()
                        rl_info.update({"train/actor_loss": real_actor_loss, "train/ptx_loss": real_ptx_loss})
                    # on_step_end
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                else:
                    # on_sub_step_end
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(rl_info, None, epoch, ignore_keys_for_eval, inputs=inputs)

            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({self.state.max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            # argument model is not used in _maybe_log_save_evaluate, thus use None
            self._maybe_log_save_evaluate(rl_info, None, epoch, ignore_keys_for_eval, inputs=inputs)

            if self.control.should_training_stop:
                break
        # TODO(guosheng): add epilogue of training

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            for k, v in tr_loss.items():
                if isinstance(v, paddle.Tensor) and "lr" not in k and "max_generated_length" not in k:
                    v_scalar = self._nested_gather(v).mean().item()
                    # TODO(guosheng): maybe should consider self._enable_delay_scale_loss()
                    # and maybe should merge with loss postprocess in PP
                    if "train/actor_loss" == k and "train/ptx_loss" in tr_loss:
                        # use_ptx would double the gradient_accumulation_steps
                        # which causes actor_loss and ptx_loss reduced by half
                        v_scalar = v_scalar * 2
                    elif "train/ptx_loss" == k:
                        # similar to actor_loss and should double, additionally
                        # it should be divided by ptx_coeff for logging
                        v_scalar = v_scalar * 2 / self.ptx_coeff
                    logs[k] = round(v_scalar / (self.state.global_step - self._globalstep_last_logged), 8)
                    v.subtract_(v)
                    attr_name = "_total_" + k.split("/")[-1] + "_scalar"
                    attr_value = getattr(self, attr_name, 0)
                    setattr(self, attr_name, attr_value + v_scalar)
                elif "max_generated_length" in k:
                    v_scalar = self._nested_gather(v).max().item()
                    logs[k] = v_scalar
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

        # To trigger evaluation and save but avoid log again
        with guard_set_args(self.control, {"should_log": False}):
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

        # log_probs has shifted by one for predicted logits
        # TODO(guosheng): When using flash_attn with casual mask and right padding
        # inputs, responses of batch input cannot be got by sliced from start. And
        # use sequences (as labels) with full length instead of target length.
        start = prompt.shape[-1] - 1
        sequence_mask = attention_mask[:, 1:]

        with paddle.no_grad():
            # maybe these two can also be put into rollout
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
            # metric
            kl_divergence = ((old_log_probs - ref_log_probs) * sequence_mask)[:, start:].sum(axis=-1).mean()
            mean_generated_length = sequence_mask[:, start:].cast(paddle.float32).sum(axis=-1).mean()
            max_generated_length = sequence_mask[:, start:].cast(paddle.float32).sum(axis=-1).max()
            rewards = rewards.mean()
            # trainer inputs with target length
            old_log_probs = old_log_probs[:, start:]
            old_reward_values = old_reward_values[:, start:]
            sequence_mask = sequence_mask[:, start:]

        policy_trainer_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "old_log_probs": old_log_probs,
            "reward_advantages": reward_advantages,
            "sequence_mask": sequence_mask,
            # "start": start,
            # "use_cache": False,
            # "return_dict": True,
        }
        kwargs = self.policy_trainer.full_training_step(policy_trainer_inputs, **kwargs)

        value_trainer_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "old_reward_values": old_reward_values,
            "reward_returns": reward_returns,
            "sequence_mask": sequence_mask,
            # "start": start,
            # "use_cache": False,
            # "return_dict": True,
        }
        kwargs = self.value_trainer.full_training_step(value_trainer_inputs, **kwargs)

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
        kwargs = self.policy_trainer.full_training_step(ptx_batch, **kwargs)
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

        # todo, fixme zhui, self.actor_model.generate
        if self._policy_model_eval is not None:
            actor_model_in_use = self._policy_model_eval
        else:
            actor_model_in_use = self.actor_model

        sequences = actor_model_in_use.generate(
            input_ids=input_ids,
            attention_mask=prompt_only_batch["attention_mask"],
            generation_config=self.generation_config,
            synced_gpus=ShardingOption.FULL_SHARD in self.policy_trainer.args.sharding,
        )[0]
        sequences = sequences.reshape([input_ids.shape[0], self.args.num_return_sequences, -1]).transpose([1, 0, 2])
        # sequences = [self.load_sing_gen_data(as_batches=False, use_counter=False)["input_ids"]]

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
        if False:  # self.reward_tokenizer is not self.tokenizer:
            # right padding
            reward_tokenize_output = batch_retokenize(
                sequence,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.reward_tokenizer,
                skip_special_tokens=True,
            )
            reward_seq = sequence = reward_tokenize_output["input_ids"]
            reward_attention_mask = attention_mask = reward_tokenize_output["attention_mask"]
        else:
            # for text in self.tokenizer.batch_decode(
            #     sequence,
            #     skip_special_tokens=True
            # ):
            #     print(text)
            reward_seq = sequence
            reward_attention_mask = attention_mask

        # todo, fixme zhui, self.actor_model forward
        if self._policy_model_eval is not None:
            actor_model_in_use = self._policy_model_eval
        else:
            actor_model_in_use = self.actor_model

        if self._value_model_eval is not None:
            reward_critic_model_in_use = self._value_model_eval
        else:
            reward_critic_model_in_use = self.reward_critic_model

        # pipe model outputs a logits tensor with LMHead, while non-pipe model
        # outputs a tuple with logits tensor as the only one element.
        logits = actor_model_in_use(
            sequence,
            attention_mask=attention_mask,
            # return_dict=True,
        )  # .logits
        if not isinstance(logits, paddle.Tensor):
            logits = logits[0]
        ref_logits = self.reference_model(
            sequence,
            attention_mask=attention_mask,
            # return_dict=True,
        )  # .logits
        if not isinstance(ref_logits, paddle.Tensor):
            ref_logits = ref_logits[0]

        reward_score = self.reward_model(
            reward_seq,
            attention_mask=reward_attention_mask,
            # return_dict=True,
        )[
            1
        ]  # .end_scores
        reward_value = reward_critic_model_in_use(
            sequence,
            attention_mask=attention_mask,
            # return_dict=True,
        )[
            0
        ]  # .scores
        # TODO(guosheng): move these to model methods such as get_logprobs
        reward_score = reward_score.squeeze(axis=-1)
        reward_value = reward_value.squeeze(axis=-1)

        reward_value = reward_value[:, :-1]
        log_probs = gather_log_probabilities(logits[:, :-1], sequence[:, 1:])
        ref_log_probs = gather_log_probabilities(ref_logits[:, :-1], sequence[:, 1:])
        return {
            "prompt": prompt,
            "input_ids": sequence,
            "attention_mask": attention_mask,
            "rewards": reward_score,
            "reward_values": reward_value,
            "log_probs": log_probs,
            "ref_log_probs": ref_log_probs,
        }

    # @paddle.no_grad()
    # def post_rollout(
    #     self,
    #     prompt: paddle.Tensor,
    #     sequence: paddle.Tensor,
    #     attention_mask: paddle.Tensor,
    # ) -> Dict[str, Any]:
    #     if self.reward_tokenizer is not self.tokenizer:
    #         reward_tokenize_output = batch_retokenize
