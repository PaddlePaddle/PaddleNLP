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
import time
import paddle.nn as nn

import numpy as np
import paddle.distributed as dist
from typing import Dict
from paddlenlp.trainer import Trainer
from paddlenlp.trainer.trainer_utils import speed_metrics
from paddle.io import DataLoader, get_worker_info

from ppdiffusers.ppnlp_patch_utils import patch_to
from paddlenlp.trainer.integrations import VisualDLCallback, rewrite_logs, TrainerCallback
from paddlenlp.utils.log import logger


class LitEmaCallback(TrainerCallback):

    def on_step_end(self, args, state, control, model=None, **kwargs):
        model.on_train_batch_end()


@patch_to(VisualDLCallback)
def on_log(self, args, state, control, logs=None, **kwargs):
    if not state.is_world_process_zero:
        return

    if self.vdl_writer is None:
        self._init_summary_writer(args)

    if self.vdl_writer is not None:
        logs = rewrite_logs(logs)
        image_logs = kwargs.get("image_logs", None)
        if image_logs is not None:
            logs.update(image_logs)
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                self.vdl_writer.add_scalar(k, v, state.global_step)
            elif isinstance(v, np.ndarray):
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


class LatentDiffusionTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(**inputs)
        return loss

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            worker_init_fn=worker_init_fn,
        )

    def _maybe_log_save_evaluate(self,
                                 tr_loss,
                                 model,
                                 epoch,
                                 ignore_keys_for_eval,
                                 inputs=None):
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

            # log_images
            image_logs = {}
            if inputs is not None and logs[
                    "global_step"] % self.args.image_logging_steps == 0:
                with self.autocast_smart_context_manager():
                    image_logs["reconstruction"] = self.model.decode_image(
                        pixel_values=inputs["pixel_values"])
                    image_logs["ddim-samples-1.0"] = self.model.log_image(
                        input_ids=inputs["input_ids"], guidance_scale=1.0)
                    image_logs["ddim-samples-7.5"] = self.model.log_image(
                        input_ids=inputs["input_ids"], guidance_scale=7.5)

            total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
            num_steps = self.state.global_step - self._globalstep_last_logged
            logs.update(
                speed_metrics(
                    "interval",
                    self._globalstep_last_start_time,
                    num_samples=total_train_batch_size * num_steps,
                    num_steps=num_steps,
                ))

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self._globalstep_last_start_time = time.time()

            self.log(logs, image_logs=image_logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control)


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
