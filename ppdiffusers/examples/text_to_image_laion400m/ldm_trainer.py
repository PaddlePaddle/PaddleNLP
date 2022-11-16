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

import os
import time
import numpy as np
import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.io import DistributedBatchSampler
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients
from fastcore.all import patch_to
from paddlenlp.trainer.integrations import VisualDLCallback, rewrite_logs
from typing import Dict, List, Optional, Union
from tqdm.auto import tqdm
from paddlenlp.trainer import Trainer
from paddlenlp.utils.log import logger
from paddlenlp.utils.batch_sampler import DistributedBatchSampler as NlpDistributedBatchSampler
from paddlenlp.trainer.trainer_utils import (
    TrainOutput,
    speed_metrics,
    get_last_checkpoint,
)
from paddlenlp.trainer.trainer_callback import TrainerState, PrinterCallback, ProgressCallback
from paddle.io import DataLoader, get_worker_info
import itertools

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"

OPTIMIZER_NAME = "optimizer.pdopt"
SCHEDULER_NAME = "scheduler.pdparams"
SCALER_NAME = "scaler.pdparams"

WEIGHTS_NAME = "model_state.pdparams"
CONFIG_NAME = "model_config.json"


@patch_to(VisualDLCallback)
def on_log(self, args, state, control, logs=None, **kwargs):
    if not state.is_world_process_zero:
        return

    if self.vdl_writer is None:
        self._init_summary_writer(args)

    if self.vdl_writer is not None:
        logs = rewrite_logs(logs)
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                self.vdl_writer.add_scalar(k, v, state.global_step)
            elif isinstance(v, (np.ndarray)):
                self.vdl_writer.add_image(k,
                                          v,
                                          state.global_step,
                                          dataformats="NHWC")
            else:
                logger.warning(
                    "Trainer is attempting to log a value of "
                    f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                    "This invocation of VisualDL's writer.add_scalar() "
                    "is incorrect so we dropped this attribute.")
        self.vdl_writer.flush()


# dont log image
@patch_to(PrinterCallback)
def on_log(self, args, state, control, logs=None, **kwargs):
    _ = logs.pop("total_flos", None)
    ddim1 = logs.pop("ddim-samples-1.0", None)
    ddim75 = logs.pop("ddim-samples-7.5", None)
    reconstruction = logs.pop("reconstruction", None)
    if state.is_local_process_zero:
        if type(logs) is dict:
            logger.info(", ".join(f"{k}: {v}" for k, v in logs.items()))
        else:
            logger.info(logs)


# dont log image
@patch_to(ProgressCallback)
def on_log(self, args, state, control, logs=None, **kwargs):
    if state.is_local_process_zero and self.training_bar is not None:
        _ = logs.pop("total_flos", None)
        ddim1 = logs.pop("ddim-samples-1.0", None)
        ddim75 = logs.pop("ddim-samples-7.5", None)
        reconstruction = logs.pop("reconstruction", None)
        if type(logs) is dict:
            logs_str = ", ".join(f"{k}: {v}" for k, v in logs.items())
        else:
            logs_str = str(logs)
        self.training_bar.write(logs_str)


class LatentDiffusionTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(**inputs)
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
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            if not os.path.isfile(
                    os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(
                    f"Can't find a valid checkpoint at {resume_from_checkpoint}"
                )

            logger.info(f"Loading model from {resume_from_checkpoint} .")

            # TODO: Need to load the model state dict on the CPU to avoid an OOM error.
            state_dict = paddle.load(
                os.path.join(resume_from_checkpoint, WEIGHTS_NAME))
            # If the model is on the GPU, it still works!
            self._set_state_dict_in_model(state_dict)

            # release memory
            del state_dict

        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model_wrapped)

        self.state = TrainerState()

        args = self.init_num_steps(args, len(self.train_dataset))

        if args.minimum_eval_times is not None and args.minimum_eval_times > 0:
            if args.num_training_steps // args.eval_steps < args.minimum_eval_times:
                exp_step = args.num_training_steps / args.minimum_eval_times
                exp_step = max(int(exp_step - exp_step % 10), 10)
                logger.info("Reset eval step by minimum_eval_times to %d" %
                            exp_step)
                args.eval_steps = exp_step

        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_training_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        num_examples = len(self.train_dataset)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {args.total_train_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {args.num_training_steps}")
        logger.info(f"  Total num train samples = {args.num_train_samples}")

        start_time = time.time()
        self._globalstep_last_start_time = time.time()
        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // args.num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    args.num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(
                        total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description(
                        "Skipping the first batches")
            if not args.ignore_data_skip:
                if isinstance(train_dataloader,
                              paddle.io.DataLoader) and isinstance(
                                  train_dataloader.batch_sampler,
                                  NlpDistributedBatchSampler):
                    consumed_samples = self.state.global_step * args.train_batch_size * args.gradient_accumulation_steps * args.world_size
                    train_dataloader.batch_sampler.set_epoch(
                        consumed_samples=consumed_samples)
                    logger.info(
                        f"Set DistributedBatchSampler consumed_samples to {consumed_samples}"
                    )

        epoch_iterator = train_dataloader
        # TODO junnyu
        len_dataloader = None
        steps_in_epoch = (len(epoch_iterator) if len_dataloader is not None else
                          int(args.num_training_steps) *
                          args.gradient_accumulation_steps)

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        self.state.max_steps = int(args.num_training_steps)
        self.state.num_train_epochs = args.num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control)

        tr_loss = paddle.to_tensor(0.0)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        for epoch in range(epochs_trained, args.num_train_epochs):
            if isinstance(train_dataloader,
                          paddle.io.DataLoader) and isinstance(
                              train_dataloader.batch_sampler,
                              DistributedBatchSampler):
                train_dataloader.batch_sampler.set_epoch(epoch)

            step = -1

            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                # for paddlenlp.utils.batch_sampler.DistributedBatchSampler
                # We use consumed_samples to reset the status
                if isinstance(train_dataloader,
                              paddle.io.DataLoader) and isinstance(
                                  train_dataloader.batch_sampler,
                                  NlpDistributedBatchSampler):
                    if step == 0:
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(
                                steps_trained_in_current_epoch)
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
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control)

                # TODO junnyu
                # is_no_sync = False
                is_no_sync = (((
                    (step + 1) % args.gradient_accumulation_steps != 0)
                               and args.local_rank != -1
                               and args._no_sync_in_gradient_accumulation)
                              or (args.recompute and args.local_rank != -1))

                if is_no_sync:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                tr_loss += tr_loss_step

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps and
                    (step + 1) == steps_in_epoch):

                    if (args.recompute and args.local_rank != -1):
                        fused_allreduce_gradients(list(model.parameters()),
                                                  None)

                    if self.do_grad_scaling:
                        self.scaler.minimize(self.optimizer, tr_loss)
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.clear_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch

                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control)
                    # TODO junnyu
                    with self.autocast_smart_context_manager():
                        self._maybe_log_save_evaluate(inputs, tr_loss, model,
                                                      epoch,
                                                      ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({self.state.max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control)
            # TODO junnyu
            self._maybe_log_save_evaluate(inputs, tr_loss, model, epoch,
                                          ignore_keys_for_eval)

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

            best_model_path = os.path.join(self.state.best_model_checkpoint,
                                           WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = paddle.load(best_model_path)
                # If the model is on the GPU, it still works!
                self._set_state_dict_in_model(state_dict)
            else:
                logger.warning(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train",
                                start_time,
                                num_samples=args.num_train_samples,
                                num_steps=self.state.max_steps)

        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _maybe_log_save_evaluate(self, inputs, tr_loss, model, epoch,
                                 ignore_keys_for_eval):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss.subtract_(tr_loss)

            logs["loss"] = round(
                tr_loss_scalar /
                (self.state.global_step - self._globalstep_last_logged), 8)
            logs["learning_rate"] = self._get_learning_rate()
            logs["global_step"] = int(self.state.global_step)
            # TODO junnyu
            if logs["global_step"] % (self.args.logging_steps * 20) == 0:
                logs["reconstruction"] = self.model.decode_image(
                    pixel_values=inputs["pixel_values"])
                logs["ddim-samples-1.0"] = self.model.log_image(
                    input_ids=inputs["input_ids"], guidance_scale=1.0)
                logs["ddim-samples-7.5"] = self.model.log_image(
                    input_ids=inputs["input_ids"], guidance_scale=7.5)

            logs.update(
                speed_metrics(
                    "interval",
                    self._globalstep_last_start_time,
                    num_samples=self.args.train_batch_size *
                    self.args.gradient_accumulation_steps,
                    num_steps=self.state.global_step -
                    self._globalstep_last_logged,
                ))

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self._globalstep_last_start_time = time.time()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control)

    def create_optimizer(self, lr_scheduler=None):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args)

            # TODO junnyu
            if self.args.freeze_text_encoder:
                params_to_train = self.model.unet.parameters()
            else:
                params_to_train = itertools.chain(
                    self.model.text_encoder.parameters(),
                    self.model.unet.parameters())
            self.optimizer = optimizer_cls(
                learning_rate=self.lr_scheduler
                if lr_scheduler is None else lr_scheduler,
                parameters=params_to_train,
                weight_decay=self.args.weight_decay,
                grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm)
                if self.args.max_grad_norm is not None else None,
                **optimizer_kwargs)

        return self.optimizer

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
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            worker_init_fn=worker_init_fn,
        )

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 4)
        ddim1 = logs.pop("ddim-samples-1.0", None)
        ddim75 = logs.pop("ddim-samples-7.5", None)
        reconstruction = logs.pop("reconstruction", None)
        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        need_logdata = {
            **logs,
            **{
                "ddim-samples-1.0": ddim1,
                "ddim-samples-7.5": ddim75,
                "reconstruction": reconstruction
            }
        } if ddim1 is not None else logs
        self.control = self.callback_handler.on_log(self.args, self.state,
                                                    self.control, need_logdata)


def worker_init_fn(_):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id

    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_workers = worker_info.num_workers
    worker_id = worker_info.id
    worker_global_id = local_rank * num_workers + worker_id

    dataset.rng = np.random.RandomState(worker_global_id)
    for i in range(len(dataset.file_ids)):

        file_ids = dataset.file_ids[i]
        num_chunks = world_size * num_workers
        chunk_size = len(file_ids) // num_chunks

        begin_id = worker_global_id * chunk_size
        end_id = (worker_global_id + 1) * chunk_size
        dataset.file_ids[i] = dataset.file_ids[i][begin_id:end_id]
        print(
            f'dataset {i}, local_rank: {local_rank}, worker_id: {worker_id}, worker_global_id: {worker_global_id}, file_range: ({begin_id}, {end_id})'
        )
    return np.random.seed(np.random.get_state()[1][0] + worker_id)
