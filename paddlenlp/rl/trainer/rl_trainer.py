# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed as dist
import tqdm
from paddle import nn
from paddle.distributed import fleet
from paddle.io import DataLoader, Dataset

from ...data import DataCollator
from ...trainer.trainer import (
    EvalPrediction,
    HybridParallelOptimizer,
    NlpDistributedBatchSampler,
    ShardingOption,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainingArguments,
    _obtain_optimizer_parameters_list,
    distributed_file,
    distributed_isfile,
    fused_allreduce_gradients,
    logger,
    reshard_util,
    split_inputs_sequence_dim,
)
from ...transformers import PretrainedModel, PretrainedTokenizer
from ...utils.env import TRAINER_STATE_NAME
from ..models.ppo_model_utils import create_loss
from ..utils.comm_utils import ActorStages, create_data_trans_group
from ..utils.infer_utils import InferEvalModel
from ..utils.reshard_utils import init_rollout_env
from ..utils.timer_utils import TimerScope
from .trainer_utils import PipeEvalModel

# ########## patches for Trianer ##########


def init_train_model_opt(
    self: Trainer,
    max_steps: int,
    resume_from_checkpoint: bool = False,
    clear_master_weight: bool = False,
) -> PretrainedModel:
    """
    Initialize the training model and optimizer, and return the wrapped model.

    Args:
        self (Trainer): The instance of the Trainer class.
        max_steps (int): The maximum number of training steps.
        resume_from_checkpoint (bool, optional): Whether to resume training from a checkpoint, defaults to False.
        clear_master_weight (bool, optional): When using Trainer's distributed hardware acceleration, clear the master parameter weights, defaults to False.

    Returns:
        PretrainedModel: The wrapped model ready for training.
    """
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
        # In this mode, the rank0 load all params and the _wrap_model implicitly broadcast
        # params from rank0 to the other ranks.
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
    """
    Initialize the training state.

    Args:
        self (Trainer): The instance of the Trainer class to record the training state.
        resume_from_checkpoint (bool, optional): Whether to resume training from a checkpoint, defaults to False.
        train_dataloader (DataLoader, optional): The data loader for training, defaults to None.
        max_steps (int, optional): The maximum number of training steps, defaults to -1.
        num_train_epochs (int, optional): The maximum number of training epochs, defaults to 3.
        num_update_steps_per_epoch (int, optional): The number of steps to update the model per epoch, defaults to 1.

    Returns:
        Tuple[int, int, Optional[tqdm]]:
            - epochs_trained (int): The number of epochs already trained.
            - steps_trained_in_current_epoch (int): The number of batches trained in the current epoch if not skipping data; otherwise, 0.
            - steps_trained_progress_bar (Optional[tqdm]): A tqdm progress bar to show the progress of skipping the first batch if not skipping data; otherwise, None.
    """
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

    return (
        epochs_trained,
        steps_trained_in_current_epoch,
        steps_trained_progress_bar,
    )


def init_train_log(
    self: Trainer,
    num_examples: int,
    num_train_epochs: int,
    total_train_batch_size: int,
    max_steps: int,
    num_train_samples: int,
    model: PretrainedModel,
):
    """
    Initialize the training log.

    Args:
        self (Trainer): The instance of the Trainer class containing parameters and information required for training.
        num_examples (int): The total number of samples in the training set.
        num_train_epochs (int): The number of training epochs.
        total_train_batch_size (int): The sum of the training batch sizes on a single device.
        max_steps (int): The maximum number of training steps.
        num_train_samples (int): The total number of samples in the training set.
        model (PretrainedModel): The model being trained.

    Returns:
        None, this function does not return any value.
    """
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
    logger.debug(f"  Number of trainable parameters = {per_device_trainable_numel:,} (per device)")
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
            logger.debug(f"  Number of trainable parameters = {trainable_numel:,} (all devices, roughly)")


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
    # timer_name = kwargs.get("timer_name", "")
    tr_loss = kwargs.get("tr_loss", 0.0)
    model = kwargs.get("model", self.model_wrapped)
    # needed in _maybe_log_save_evaluate
    self._globalstep_last_logged = getattr(self, "_globalstep_last_logged", 0)
    self._globalstep_last_start_time = getattr(self, "_globalstep_last_start_time", time.time())

    args = self.args

    if self.args.use_hybrid_parallel and self.args.sep_parallel_degree > 1:
        inputs = split_inputs_sequence_dim(inputs)
    # self.timers and self.timers("read-data").stop()
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
        # self.timers and self.timers(f"{timer_name}: forward-backward").start()

    dp_enabled = self.args.data_parallel_degree > 1 if self.args.use_hybrid_parallel else args.local_rank != -1
    forbidden_no_sync = False
    # stage2 and stage3 should not no_sync, because the is no DDP wrapper and no_sync API
    # hybrid_parallel (tp or pp or sharding stage 1) should not no_sync
    if self.args.use_hybrid_parallel:
        forbidden_no_sync = True

    available_no_sync = dp_enabled and not forbidden_no_sync

    is_no_sync = (
        ((step_control + 1) % args.gradient_accumulation_steps != 0)
        and available_no_sync
        and args._no_sync_in_gradient_accumulation
    ) or (args.recompute and available_no_sync)
    # sharding
    # stage1. the same as ddp
    # stage2. manually collect gradient on dp group

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

        # self.timers and self.timers(f"{timer_name}: forward-backward").stop()

        # Manually collect gradients
        # Case 1: Use recompute and dp
        # Case 2: Hack dp with master_grad
        # Case 3: Pipeline or sharding overlap
        # local_rank != -1 don't means dp in networks.
        # self.timers and self.timers(f"{timer_name}: all-reduce").start()

        # Case 1: Use recompute and dp / sharding stage1,
        # manually collect gradient for dp.
        if args.recompute and available_no_sync:
            fused_allreduce_gradients(list(model.parameters()), None)

        # Case 2: hack dp with master_grad
        if dp_master_grad and not (args.recompute and available_no_sync):
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

        # self.timers and self.timers(f"{timer_name}: all-reduce").stop()
        # self.timers and self.timers(f"{timer_name}: optimizer-step").start()

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
            args,
            self.state,
            self.control,
            scaler=self.scaler if self.do_grad_scaling else None,
        )
        optimizer_time_scope = TimerScope(self.timers, ActorStages.OPTIMIZE_STEP)
        optimizer_time_scope.start()

        optimizer_was_run = True

        if self.args.offload_optim:
            self._reload_optimizer()

        if self.do_grad_scaling:
            scale_before = paddle.assign(self.scaler._scale)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler._scale
            # Compatible with paddlepaddle 2.6.0 using typo word.
            if hasattr(self.scaler, "_cache_founf_inf"):
                optimizer_was_run = not self.scaler._cache_founf_inf
            else:
                optimizer_was_run = not self.scaler._cache_found_inf
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

        # self.timers and self.timers(f"{timer_name}: optimizer-step").stop()
        if self.args.offload_optim:
            self._offload_optimizer()

        if optimizer_was_run:
            self.lr_scheduler.step()

        if args.release_grads or enable_release_grads:
            self.optimizer.clear_grad(set_to_zero=False)
            if args.pipeline_parallel_degree > 1:
                for _, buffers in model._chunk_2_comm_buffers.items():
                    for buffer in buffers:
                        buffer._clear_grad_storage()
        else:
            self.optimizer.clear_grad()

        optimizer_time_scope.stop()

        self.callback_handler.on_optimizer_end(
            args,
            self.state,
            self.control,
            scaler=self.scaler if self.do_grad_scaling else None,
        )

        self.state.global_step += 1
        self.state.epoch = epoch + (step + 1) / steps_in_epoch
        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
        self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
        # self._print_timer()
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
    # self.timers and self.timers("read-data").start()

    final_local_vars = locals()
    for k in kwargs.keys():
        if k in final_local_vars:
            kwargs[k] = final_local_vars[k]
    return kwargs


Trainer.init_train_model_opt = init_train_model_opt
Trainer.init_train_log = init_train_log
Trainer.init_train_state = init_train_state
Trainer.full_training_step = full_training_step
# ########## patches for Trianer ##########


class RLTrainer(Trainer):
    """
    Features of RLTrainer:
    1. Trainer enhanced with step-level training combining with patches of
    Trianer. We can use this to do training whose step is composed of multi
    models via multiple instances of RLTrainer, such as PPO.
    2. Additionally, using a mixed loss and get the separated loss metrics is
    supported, which is helpful to PipelienParallel with a mixed loss.
    3. EMA is supported.
    """

    # used to create criterion for trainer, please refer to `create_criterion`
    # for details.
    loss_cls: type

    def __init__(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]] = None,
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
        # criterion is only used for non-PipelineParallel models. criterion is
        # included in model for PipelineParallel.
        self.info_buffer = {}
        if getattr(self, "loss_cls", None) and self.criterion is None:
            self.criterion = self.create_criterion()

        self.use_fusemt = getattr(args, "use_fusemt", False)
        # ablout 4s slower than infer generation without ema
        self.use_ema = getattr(args, "use_ema", False)
        self.shard_ema = getattr(args, "shard_ema", False)
        self.offload_ema = getattr(args, "offload_ema", True)
        self.ema_beta = getattr(args, "ema_beta", 0.992)
        # if self.timers:
        #     self.timers.log = types.MethodType(new_timer_log, self.timers)

    def create_criterion(self):
        """
        create loss using `loss_cls` for trainer. It would use a wrapped loss_cls
        whose label arguments are merged into one argument, this is useful to
        PipelineParallel and trainer.criterion which limit loss format.
        """
        criterion = create_loss(self.loss_cls, self.model.config, self.args, self.info_buffer, merge_labels=True)
        return criterion

    def loss_identifier(self, inputs: Dict) -> str:
        """
        Moreover, a model/RLTrainer instance may use a mixed loss which uses a
        different loss for different step and inputs, while we often want to get
        the separated loss metric. We use a callable discriminator using inputs
        (dict) as arguments and returning corresponding loss name to identify
        current loss. NOTE: please make the loss name ends with "_loss". `tr_loss`
        is the default loss name used in trainer.train.
        """
        return "tr_loss"

    def set_eval_model(self, model):
        """
        To avoid eval/generation with PipelineParallel when training with PP, we
        allow to use an extra eval model to do eval/generation, which would need
        to reshard parameters and dispatch data according to model's distributed
        topo. Currently, the eval model should cancel PP setting and keep the same
        TP setting with training.
        """
        if model is None:
            logger.warning("use None to set eval model for trainer and it would be ignored")
            return
        else:
            self._inner_eval_model = model
        # bind a new comm group for eval model data dispatch
        # param dispatch is binded in `InferEvalModel.enable`
        hcg = fleet.get_hybrid_communicate_group()
        sd_group = hcg.get_sharding_parallel_group()
        dp_group = hcg.get_data_parallel_group()
        global_rank = dist.get_rank()
        old_dp_workers = self.args.world_size // (max(sd_group.nranks, 1) * max(dp_group.nranks, 1))
        with init_rollout_env(self.args.rollout_tensor_parallel_degree):
            hcg = fleet.get_hybrid_communicate_group()
            tensor_parallel_degree = hcg.get_model_parallel_world_size()
            tensor_parallel_rank = hcg.get_model_parallel_rank()
            eval_tp_size = max(tensor_parallel_degree, 1)
            eval_tp_rank = max(tensor_parallel_rank, 0)
        group_nums = self.args.logical_process_index // old_dp_workers * eval_tp_size + eval_tp_rank
        self._data_trans_group = create_data_trans_group(global_rank, group_nums)
        # just for compatiable with old code
        self._policy_model_eval_group = self._data_trans_group

    def get_model(self, train=False):
        """
        model visitor wrapps PipelineParalle and Inference model to do evaulation
        and generation.
        """
        if train:
            return self.model_wrapped
        model = getattr(self, "_eval_model", None)
        if model is not None:
            return model
        inner_eval_model = getattr(self, "_inner_eval_model", None)
        if (self.args.pipeline_parallel_degree > 1 and inner_eval_model is None) or isinstance(
            inner_eval_model, fleet.model.PipelineParallel
        ):
            # Only accept wrapped model for pipeline_parallel mode
            model = PipeEvalModel(self)
            self._eval_model = model
        else:
            model = InferEvalModel(self)
            self._eval_model = model
        return model

    def get_train_step_vars(self, vars: Optional[Dict] = None) -> Dict:
        """
        NOTE: This is transparent to users.
        When using multiple instances of RLTrainer collaborate to do one training
        step, each should use its own vars such as loss/model/step_control which are
        local vars in Trainer.train, we define these vars by `train_step_vars`. They
        are vars needed by full_training_step for training control, as following:
        tr_loss, model, epoch, step, step_control.
        some vars such as `epoch` are meaningless, they are needed just because
        full_training_step copies code from Trainer.train which is designed for
        complete training process.

        return `train_step_vars` (dict). If not exists, create it first. If `vars`
        is not None, update `train_step_vars` with it.

        TODO(guosheng): use namedtuple or dataclass to make it more readable.
        """
        if not hasattr(self, "train_step_vars"):
            # should be called after model is wrapped since the model field should
            # use model_wrapped.

            if paddle.distributed.get_world_size() > 1:
                assert self.model is not self.model_wrapped
            self.train_step_vars = {
                # meaningless vars can pass from outter, dummy value is enough
                "epoch": 0,  # meaningless for step training
                "step": 0,  # meaningless for step training
                "steps_in_epoch": 100000,  # meaningless for step training
                "step_control": 0,  # to control training process
                "model": self.model_wrapped,
                # "tr_loss": paddle.to_tensor(0.0),  # lazy create
            }
        if vars:
            self.train_step_vars.update(vars)
        return self.train_step_vars

    @property
    def loss_names(self):
        """
        Return a list of names of all loss terms, computed only on the first call.
        If there are no loss terms, return an empty list.

        Returns:
            List[str]: A list of names of loss terms, each ending with "_loss".
        """
        if not hasattr(self, "_loss_names"):
            self._loss_names = [var_name for var_name in self.get_train_step_vars() if var_name.endswith("_loss")]
            assert len(self._loss_names) > 0
        return self._loss_names

    def full_training_step(self, **inputs) -> paddle.Tensor:
        """
        Accept any valid key word arguments of model and loss as inputs, they
        would be sent to model and then loss. Mostly it is similar to output from
        data collator.
        Return loss var. However when using PipelienParallel, the loss returned
        is 0 when not reach accumulated step and the loss returned at accumulated
        step is a mixed loss. We can use `get_step_loss` to get the actual loss.
        """
        # if model has multi losses which are combined into one mixed criterion,
        # loss statistic var may change for different training steps according
        # to inputs.
        train_step_vars = self.get_train_step_vars()
        loss_name = self.loss_identifier(inputs)
        loss_var = train_step_vars.get(loss_name, None)
        # trainer.train use `tr_loss` as loss var to accumulate loss.
        # NOTE: `tr_loss` in trainer.train not only accumulate mean loss for
        # steps in one `gradient_accumulation_steps`, but also accumulate for
        # one logging intervel which may contains more than one accumulated steps.
        # However, in RLTrainer we only want to use `tr_loss` to accumulate
        # mean loss for steps in a `gradient_accumulation_steps` range. As for
        # logging intervel loss accumulation is not take into account here and
        # should be considered in outter.
        if loss_var is None:  # the first step of current loss type
            loss_var = paddle.to_tensor(0.0)
            train_step_vars[loss_name] = loss_var
        elif self.is_accumulation_step:  # begin a new accumulation step intervel
            for name in self.loss_names:
                train_step_vars[name] = paddle.to_tensor(0.0)
            loss_var = train_step_vars[loss_name]

        train_step_vars["tr_loss"] = loss_var
        # train_step_vars["timer_name"] = self.__class__.__name__

        new_train_step_vars = super().full_training_step(inputs, **train_step_vars)

        # minimally update
        train_step_vars = self.get_train_step_vars(
            {
                "step_control": new_train_step_vars["step_control"],
                loss_name: new_train_step_vars["tr_loss"],
            }
        )
        if loss_name != "tr_loss":
            train_step_vars.pop("tr_loss")

        self.mark_step_loss(loss_name)

        return train_step_vars[loss_name]

    def _prepare_inputs(self, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> Dict[str, Union[paddle.Tensor, Any]]:
        """
        trainer.criterion only support criterion(prediction, labels), so we need
        to reorganize the inputs to extract label data into one argument. This is
        only used in non-PipelineParallel model training since loss is included
        in PipelineLayer.
        """
        inputs = super()._prepare_input(inputs)
        if self.criterion is None or getattr(self.criterion, "label_names", None) is None:
            return inputs
        # criterion created by create_loss has `label_names` and `label_default_values`
        label_names = self.criterion.__class__.label_names
        # some data fields are used both in model and loss
        shared_fields = {"input_ids", "attention_mask"}
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

    def mark_step_loss(self, loss_name):
        """
        NOTE: This is transparent to users.
        When using a mixed loss we often want to get the separated loss metrics,
        thus we mark loss type of each training step to separate them. This is
        not necessary since the loss would be returnd after each training step.
        However when using PipelienParallel, the loss returned is 0 when not reach
        accumulated step and the loss returned at accumulated step is a mixed loss.
        To separate loss metrics in PipelienParallel:
        1. We hack PipelineParallel._forward_step to record actual loss for each
           step in a list (only in training and not in evaluation currently).
        2. We mark the loss type only once for each step using `loss_step_indice`
           (dict), then wen can check out the corresponding loss metrics from the
           loss list.
        We assume a static order of multi-losses and mark the loss indice only once.
        """
        self.loss_step_indice = getattr(self, "loss_step_indice", {})
        if loss_name not in self.loss_step_indice:
            self.loss_step_indice[loss_name] = len(self.loss_step_indice)

    @paddle.no_grad()
    def get_step_loss(self, loss_prefix: str = "", loss_accumulator: Dict = {}) -> Dict[str, paddle.Tensor]:
        """
        Return a dict mapping loss name to value of current training step. This
        is mainly to get loss for metric logging, and it would not affect the
        training. This is mostly helpful to PipelienParallel with a mixed loss
        in which the loss returned is 0 when not reach accumulated step and the
        loss returned at accumulated step is a mixed loss.
        NOTE: 1. Only when reaching accumulated step the losses returned are
        accurate, and each loss is a mean loss of steps among one accumulated
        steps range.
        """
        if not self.is_accumulation_step:
            msg = "The loss returned may not be accurate when not reaching accumulated step."
            logger.error(msg)
        model = self.get_model(train=True)
        loss_dict = loss_accumulator if loss_accumulator else {}
        if isinstance(model, fleet.model.PipelineParallel) and len(self.loss_names) > 1:
            # NOTE: PipelineParallel only returns a accumulated loss after
            # accumulated steps, which is a mixed loss of ppo-loss and
            # ptx-loss. We hack PipelineParallel._forward_step to record
            # loss metrics and postprocess the recorded losses here.
            # Maybe better to make the last_stage worker log to reduce
            # comm and for simplicity.
            with paddle.no_grad():
                if model.is_pipeline_last_stage():
                    # loss is 0D tensor, use stack rather than concat
                    mix_loss = paddle.stack(model._step_losses)
                    model._step_losses = None
                else:
                    # The tessor shape is not actor_model.accumulate_steps
                    # (args.accu_steps) but actor_trainer.args.accu_steps,
                    # since actor_model is created with global pp_config
                    # using global args.accu_steps which is only half of
                    # actor_trainer.args.accu_steps, and indeed trainer hack
                    # model.accumulate_steps in training_pipeline_step to use
                    # trainer.args.accu_steps. The dtype is fp32(to be check),
                    # thus no need to broadcast.
                    mix_loss = paddle.empty(
                        shape=[self.args.gradient_accumulation_steps],
                        dtype=paddle.float32,
                    )
                paddle.distributed.broadcast(mix_loss, src=model.pp_group.ranks[-1], group=model.pp_group)
                for loss_name in self.loss_names:
                    # We assume a static order of multi-losses and mark the loss
                    # indice only once.
                    value = mix_loss[self.loss_step_indice[loss_name] :: len(self.loss_names)].mean()
                    loss_name = loss_prefix + loss_name if loss_prefix else loss_name
                    loss_dict[loss_name] = loss_dict[loss_name].add_(value) if loss_name in loss_dict else value
            return loss_dict
        elif isinstance(model, fleet.model.PipelineParallel):
            model._step_losses = None

        for loss_name in self.loss_names:
            value = self.get_train_step_vars()[loss_name]
            loss_name = loss_prefix + loss_name if loss_prefix else loss_name
            loss_dict[loss_name] = loss_dict[loss_name].add_(value) if loss_name in loss_dict else value
        return loss_dict

    @property
    def is_accumulation_step(self):
        """Indicate whether accumulation steps' training is done."""
        return self.get_train_step_vars()["step_control"] == 0

    def get_sharding_master_weight_structured_names(self, model, optimizer):
        """
        Get a list of structured names for the sharding master weights.

        Args:
            model (paddle.nn.Layer): The model object containing parameters that need to be sharded.
            optimizer (paddle.optimizer.Optimizer): The optimizer object containing parameters that need to be sharded.

        Returns:
            list[str]: A list of structured names for all parameters that are being trained on the current sharding master.
        """
        rank_param_names = [p.name for p in optimizer._rank2params[optimizer._sharding_rank]]
        structured_names = []
        # For pipeline model, using `model.state_dict()` would automatically map parameter names
        for name, p in model.state_dict().items():
            if p.name in rank_param_names:
                structured_names.append(name)
        return structured_names

    def get_master_weight_state_dict(self, model, optimizer):
        """
        Retrieve the state dictionary of model weights. If AMP is used, pipeline is supported,
        and master weights exist, return the master weights. Otherwise, return model.state_dict().

        Args:
            model (nn.Module): The model from which to retrieve the state dictionary of weights.
            optimizer (Optimizer): The optimizer associated with the model, optional, defaults to None.

        Returns:
            Union[Dict[str, Tensor], Dict[str, Any]]: A dictionary containing the state of the model weights.
            The keys in the dictionary are parameter names, and the values are corresponding Tensors or values of Any type.
            If AMP is used, pipeline is supported, and master weights exist, the returned dictionary only contains the master weights.
        """
        if self.amp_dtype in ["float16", "bfloat16"] and hasattr(optimizer, "_master_weights"):
            master_weights = dict(optimizer._master_weights)
            result = {}
            # For pipeline models, using `model.state_dict()` automatically maps parameter names
            for name, p in model.state_dict().items():
                if p.name in master_weights:
                    result[name] = master_weights[p.name]
            return result
        else:
            return model.state_dict()
