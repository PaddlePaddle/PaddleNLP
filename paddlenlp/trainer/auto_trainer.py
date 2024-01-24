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

from typing import Any, Dict, Optional, Union

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn

from paddlenlp.trainer import Trainer

from ..utils.log import logger
from .trainer_utils import _exec_mode_guard, has_length


class SemiAutoTrainer(Trainer):
    def __init__(self, *args, **kwargs):

        self.input_spec = kwargs.pop("input_spec", None)
        super().__init__(*args, **kwargs)
        assert self.args.use_auto_parallel

    def _nested_gather(self, tensors):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        return tensors

    def _wrap_model(self, model, training=True):
        self.optimizer = dist.shard_optimizer(self.optimizer) if not self.args.run_static_semi_auto else self.optimizer

        return model

    def _wrap_for_static(self, model, train_dataloader):
        # TODO: convert fleet.auto.Strategy to dist.Strategy
        # TODO: fix bugs in paddle/distributed/auto_parallel/api.py#L981 about sample_split of engine._prepare_data_spec
        model, dist_loader = dist.to_static(
            model, train_dataloader, self.criterion, self.optimizer, input_spec=self.input_spec, strategy=self.args.strategy
        )
        return model, dist_loader

    def _wrap_for_amp_training(self):
        pass

    def _print_trainable_numel(self):
        if not self.args.run_static_semi_auto:
            super()._print_trainable_numel()
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
        batch_size = (
            total_batch_size
            if self.args.pipeline_parallel_degree > 1 and self.args.run_static_semi_auto
            else total_batch_size_per_acc_step
        )

        return paddle.io.BatchSampler(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            drop_last=self.args.dataloader_drop_last,
        )

        # return DistributedBatchSampler(
        #     self.train_dataset,
        #     batch_size=self.args.per_device_train_batch_size,
        #     shuffle=True,
        #     num_replicas=self.args.dataset_world_size,
        #     rank=self.args.dataset_rank,
        #     drop_last=self.args.dataloader_drop_last,
        # )

    def training_step(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        model.train()

        inputs = self._prepare_inputs(inputs)

        if not self.args.run_static_semi_auto:
            with self.autocast_smart_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            input_ids, labels = tuple(inputs.values())
            loss = model(input_ids, labels)

            if self.args.pipeline_parallel_degree > 1:
                self._pp_data_buffer = {}
            elif self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

        if isinstance(loss, paddle.Tensor):
            return loss.detach()
        elif isinstance(loss, np.ndarray):
            return np.sum(loss)
        elif loss is None:
            return float(0.0)
        else:
            return float(loss)

    def synchronize_gradients(self, *args, **kwargs):
        pass

    def optimizer_step(self):
        if not self.args.run_static_semi_auto:
            super().optimizer_step()
        else:
            pass

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        with _exec_mode_guard("dynamic"):
            super()._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, **kwargs)

    # def is_local_process_zero(self) -> bool:
    #     """
    #     Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
    #     machines) main process.
    #     """
    #     return True

    def _save(self, output_dir: Optional[str] = None, state_dict=None, merge_tensor_parallel=False):
        del self.args.global_mesh
        super()._save(output_dir, state_dict, merge_tensor_parallel)
