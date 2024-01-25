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

import os
import random
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddle.distributed import fleet
from paddle.io import DistributedBatchSampler
from tqdm.auto import tqdm

from paddlenlp.trainer import Trainer

from ..transformers.segment_parallel_utils import split_inputs_sequence_dim
from ..utils.batch_sampler import DistributedBatchSampler as NlpDistributedBatchSampler
from ..utils.log import logger
from .argparser import strtobool
from .trainer_callback import DefaultFlowCallback, ProgressCallback, TrainerState
from .trainer_utils import (  # set_hyrbid_parallel_seed,
    PREFIX_CHECKPOINT_DIR,
    TrainOutput,
    _exec_mode_guard,
    has_length,
    speed_metrics,
)
from .utils.helper import distributed_file, distributed_isfile  # nested_truncate,

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"

MODEL_NAME = "model"
OPTIMIZER_NAME = "optimizer"
DIST_CKPT_NAME = "dist_ckpt"
SCHEDULER_NAME = "scheduler.pdparams"
SCALER_NAME = "scaler.pdparams"


class SemiAutoTrainer(Trainer):
    def __init__(self, *args, **kwargs):

        if kwargs.get("args", None) is not None and kwargs["args"].run_static_semi_auto:
            if kwargs.get("criterion", None) is None:

                def loss_func(loss, outputs):
                    return loss

                kwargs.update({"criterion": loss_func})

        super().__init__(*args, **kwargs)
        assert self.args.use_auto_parallel

        self.global_mesh = fleet.auto.get_mesh()

        self.mesh_in_dp = self.global_mesh.get_mesh_with_dim("dp")[self.args.data_parallel_rank]
        self.comm_group_in_dp = dist.new_group(list(self.mesh_in_dp.process_ids))

    def _nested_gather(self, tensors):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        return tensors

    def _get_meshes_for_loader(self):
        def _get_mesh(pp_idx=0):
            return self.global_mesh.get_mesh_with_dim("pp")[pp_idx]

        meshes = []
        for pp_idx in range(self.args.pipeline_parallel_degree):
            meshes.append(_get_mesh(pp_idx))
        return meshes

    def _wrap_for_auto(self, model, train_dataloader):
        if self.args.run_static_semi_auto:
            return dist.to_static(model, train_dataloader, self.criterion, self.optimizer, strategy=self.args.strategy)
        else:
            self.optimizer = dist.shard_optimizer(self.optimizer)
            return model

    def _wrap_for_amp_training(self):
        pass

    def _get_item_from_loss(self, loss):
        if isinstance(loss, paddle.Tensor):
            if loss.is_dist():
                return loss._local_value().item() if loss._is_initialized() else 0.0
            else:
                return loss.item() if loss._is_initialized() else 0.0
        else:
            return loss

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

        self._sync_resume_states(resume_from_checkpoint)

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if not self.args.should_load_sharding_stage1_model:
            self._load_from_checkpoint(resume_from_checkpoint)

        train_dataloader = self.get_train_dataloader()
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.dataset_world_size
        (
            len_dataloader,
            max_steps,
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
        ) = self._get_train_steps_and_samples(args, train_dataloader, total_train_batch_size)

        delay_optimizer_creation = False

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()

        model = self._wrap_for_auto(self.model_wrapped, train_dataloader)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        self._print_trainable_numel()

        start_time = time.time()
        self._globalstep_last_start_time = time.time()
        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if (
            resume_from_checkpoint is not None
            and distributed_isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            and not self.args.ignore_load_lr_and_optim
        ):
            self.state = TrainerState.load_from_json(
                distributed_file(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            )
            if self.args.world_size > 1:
                global_step_list = []
                paddle.distributed.all_gather(
                    global_step_list, paddle.to_tensor([self.state.global_step], dtype="int64")
                )
                assert (
                    paddle.sum(paddle.stack(global_step_list) - global_step_list[0]) == 0
                ), f"Error, get different globel step, please check! step list: {[x.item() for x in global_step_list]}"

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

        self.state.max_steps = int(max_steps)
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        with _exec_mode_guard("dynamic"):
            tr_loss = paddle.to_tensor(0.0)

        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        if self.args.device == "npu" and self.args.flatten_param_grads:
            from .plugins.npu_plugin import npu_accelerate_plugin

            npu_accelerate_plugin(self.optimizer)

        self.timers and self.timers("read-data").start()

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                train_dataloader.batch_sampler, DistributedBatchSampler
            ):
                train_dataloader.batch_sampler.set_epoch(epoch)

            step_control = 0  # used in loop control, reset to 0 after every step
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
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
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step_control % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                    self.timers and self.timers("forward-backward").start()

                tr_loss_step = self.training_step(model, inputs)

                with _exec_mode_guard("dynamic"):
                    tr_loss += tr_loss_step

                disable_accumulation = self.args.pipeline_parallel_degree > 1 and self.args.run_static_semi_auto

                if (step_control + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                    or disable_accumulation
                ):
                    if self.args.pipeline_parallel_degree <= 1 and self._enable_delay_scale_loss():
                        tr_loss /= self.args.gradient_accumulation_steps

                    self.timers and self.timers("forward-backward").stop()

                    self.timers and self.timers("optimizer-step").start()

                    # Optimizer step
                    self.callback_handler.on_optimizer_begin(
                        args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
                    )

                    self.optimizer_step()

                    self.timers and self.timers("optimizer-step").stop()

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

        self._total_loss_scalar += self._get_item_from_loss(tr_loss)
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)

        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _print_trainable_numel(self):
        if not self.args.run_static_semi_auto:
            per_device_trainable_numel = sum(np.prod(p.shape) for p in self.model.parameters() if not p.stop_gradient)
        else:
            per_device_trainable_numel = sum(
                np.prod(p.shape) for p in self.model._engine._model.parameters() if not p.stop_gradient
            )
        logger.info(f"  Number of trainable parameters = {per_device_trainable_numel:,} (per device)")

        parts_num = max(self.args.tensor_parallel_degree, 1) * max(self.args.pipeline_parallel_degree, 1)
        if parts_num > 1:
            all_reduce_dtype = "int64"
            if paddle.get_device().split(":")[0] in ["npu", "xpu"]:
                # TODO(duanyanhui): fix when NPU all_reduce supports int64
                all_reduce_dtype = "float32"

            with _exec_mode_guard("dynamic"):
                trainable_numel_tensor = paddle.to_tensor(per_device_trainable_numel, dtype=all_reduce_dtype)
                paddle.distributed.all_reduce(trainable_numel_tensor)
                trainable_numel = int(trainable_numel_tensor.item()) // self.args.dataset_world_size

            if self.args.sep_parallel_degree > 0:
                trainable_numel = trainable_numel // self.args.sep_parallel_degree
            # the numel is roughly, because the tensor parallel still hold own bias or layer_norm weight without splited
            # so, the trainable numel is a little bigger than real.
            logger.info(f"  Number of trainable parameters = {trainable_numel:,} (all devices, roughly)")

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        total_batch_size_per_acc_step = self.args.per_device_train_batch_size * self.args.dataset_world_size
        total_batch_size = total_batch_size_per_acc_step * self.args.gradient_accumulation_steps
        batch_size = total_batch_size if self.args.run_static_semi_auto else total_batch_size_per_acc_step

        return paddle.io.BatchSampler(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            drop_last=self.args.dataloader_drop_last,
        )

    def get_train_dataloader(self):
        train_dataloader = super().get_train_dataloader()
        dist_loader = dist.shard_dataloader(
            dataloader=train_dataloader,
            meshes=self._get_meshes_for_loader(),
            shard_dims="dp",
        )

        return dist_loader

    def dynamic_traning(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss

    def static_traning(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        input_ids, labels = tuple(inputs.values())
        loss = model(input_ids, labels)

        if loss is not None and self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        return loss

    def training_step(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        model.train()

        inputs = self._prepare_inputs(inputs)

        if not self.args.run_static_semi_auto:
            loss = self.dynamic_traning(model, inputs)
        else:
            loss = self.static_traning(model, inputs)

        if isinstance(loss, paddle.Tensor):
            return loss.detach() if loss._is_initialized() else float(0.0)
        elif isinstance(loss, np.ndarray):
            return np.sum(loss)
        elif loss is None:
            return float(0.0)
        else:
            return float(loss)

    def optimizer_step(self):
        if not self.args.run_static_semi_auto:
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
            else:
                self.optimizer.step()

            if optimizer_was_run:
                self.lr_scheduler.step()

            self.optimizer.clear_grad()
        else:
            pass

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        with _exec_mode_guard("dynamic"):
            super()._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, **kwargs)

    def _save_checkpoint(self, model, metrics=None):

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self.args.output_dir
        output_dir = f"{run_dir}/{checkpoint_folder}"

        if self.args.should_save or self.args.should_save_model_state:
            os.makedirs(output_dir, exist_ok=True)

        if self.args.should_save:
            logger.info(f"Saving checkpoinit files into {output_dir}")

            if self.args.should_save_model_state:

                optim_state_dict = self.optimizer.state_dict()
                optim_state_dict.pop("LR_Scheduler", None)

                state_dict = {
                    MODEL_NAME: self.model.state_dict(),
                    OPTIMIZER_NAME: optim_state_dict,
                }

                self._save_ckpt_func(state_dict, os.path.join(output_dir, DIST_CKPT_NAME), self.comm_group_in_dp)
                logger.info(f"Model weights and optimizer states saved in {output_dir}/{DIST_CKPT_NAME}")

                # FIXME: maybe only save one copy
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
            "cuda": [k.current_seed() for k in paddle.get_rng_state()],
            "cpu": paddle.framework.core.default_cpu_generator().get_state().current_seed(),
        }
        # if self.args.use_hybrid_parallel:
        #     rng_states[
        #         "hybrid_parallel_rng_state_tracker"
        #     ] = fleet.meta_parallel.get_rng_state_tracker().get_states_tracker()

        if self.args.world_size > 1:
            rng_states_list = []
            paddle.distributed.all_gather_object(rng_states_list, rng_states)
            if self.args.should_save:
                os.makedirs(output_dir, exist_ok=True)
                paddle.save(rng_states_list, os.path.join(output_dir, f"rng_state_{self.args.world_size}.pth"))
        else:
            os.makedirs(output_dir, exist_ok=True)
            paddle.save(rng_states, os.path.join(output_dir, "rng_state.pth"))

        if strtobool(os.getenv("FLAG_LLM_PDC", "False")):
            # save checkpoint_done file to ensure checkpoint is complete
            if self.args.should_save_model_state and self.args.should_save:
                # For ckpt integrity
                paddle.save(self.state.global_step, os.path.join(output_dir, ".checkpoint_done"))

    def _save(self, output_dir: Optional[str] = None, state_dict=None, merge_tensor_parallel=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if self.args.should_save:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            # Good practice: save your training arguments together with the trained model
            paddle.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        if self.args.should_save_model_state:
            self._save_ckpt_func(self.model.state_dict(), os.path.join(output_dir, MODEL_NAME), self.comm_group_in_dp)
            logger.info(f"Model weights saved in {output_dir}/{MODEL_NAME}")
