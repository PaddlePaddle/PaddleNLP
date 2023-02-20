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

from typing import Any, Dict, Union

import paddle
import paddle.nn as nn

from paddlenlp.trainer import Trainer
from paddlenlp.trainer.trainer_callback import TrainerCallback


class GLMTrainer(Trainer):
    def compute_loss(
        self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]], return_outputs: bool = False
    ):
        if self.criterion is not None:
            if "labels" in inputs:
                labels = inputs.pop("labels")
            elif self.args.label_names is not None:
                labels = []
                for label in self.label_names:
                    labels.append(inputs.pop(label))
                labels = tuple(labels)
            elif "generator_labels" in inputs:
                labels = inputs["generator_labels"]
        else:
            labels = None

        caches = [] if self.caches is None else self.caches
        logits, caches = model(**inputs, caches=caches)

        if self.criterion is not None:
            loss = self.criterion(logits, labels)
            if self.args.label_smoothing > 0:
                smooth_loss = -nn.functional.log_softmax(logits, axis=-1).mean(axis=-1)
                loss = (1 - self.args.label_smoothing) * loss + self.args.label_smoothing * smooth_loss
            if "loss_mask" in inputs and inputs["loss_mask"] is not None:
                loss_mask = inputs["loss_mask"].reshape([-1])
                loss = paddle.sum(loss.reshape([-1]) * loss_mask) / paddle.sum(loss_mask)
            outputs = (loss, logits)
        self.caches = caches

        # TODO: Clear Cache when doing optimization.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class CacheCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        state.caches = None

    def on_step_end(self, args, state, control, **kwargs):
        state.caches = None
