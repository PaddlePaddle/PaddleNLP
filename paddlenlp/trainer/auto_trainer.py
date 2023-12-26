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

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.io import DataLoader, Dataset, DistributedBatchSampler

from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)

from .trainer_utils import (
    has_length,
)

class SemiAutoTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _wrap_model(self, model, training=True):
        assert self.args.use_auto_parallel

        self.optimizer = dist.shard_optimizer(self.optimizer) if not self.args.run_static_semi_auto else self.optimizer

        return model

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None 

        if self.args.world_size <= 1 or not self.args.run_static_semi_auto:
            return paddle.io.BatchSampler(
                dataset=self.train_dataset,
                shuffle=True,
                batch_size=self.args.per_device_train_batch_size * self.args.dataset_world_size,
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
            loss = model(inputs)

        return loss.detach()

    def synchronize_gradients(self, *args, **kwargs):
        pass

    def optimizer_step(self):
        if not self.args.run_static_semi_auto:
            super().optimizer_step()
        else:
            pass
