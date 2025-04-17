# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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

from paddle.optimizer import lr
from paddle.optimizer.lr import LRScheduler

__all__ = [
    "CosineAnnealingWithWarmupDecay",
    "LinearDecayWithWarmup",
    "CosineDecay",
]


class CosineAnnealingWithWarmupDecay(LRScheduler):
    def __init__(self, max_lr, min_lr, warmup_rate, decay_steps, last_epoch=0, verbose=False, **kwargs):

        self.decay_steps = decay_steps
        self.warmup_step = warmup_rate * decay_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.increment = 0
        super(CosineAnnealingWithWarmupDecay, self).__init__(max_lr, last_epoch, verbose)
        self.increment = int(kwargs.get("global_batch_size", 0))

    def get_lr(self):
        if self.warmup_step > 0 and self.last_epoch <= self.warmup_step:
            return float(self.max_lr) * (self.last_epoch) / self.warmup_step

        if self.last_epoch > self.decay_steps:
            return self.min_lr

        num_step_ = self.last_epoch - self.warmup_step
        decay_steps_ = self.decay_steps - self.warmup_step
        decay_ratio = float(num_step_) / float(decay_steps_)
        coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += self.increment
            self.last_lr = self.get_lr()
        else:
            self.last_epoch += epoch
            if hasattr(self, "_get_closed_form_lr"):
                self.last_lr = self._get_closed_form_lr()
            else:
                self.last_lr = self.get_lr()

        if self.verbose:
            print(
                "Epoch {}: {} set learning rate to {}.".format(self.last_epoch, self.__class__.__name__, self.last_lr)
            )


class LinearDecayWithWarmup(LRScheduler):
    def __init__(self, learning_rate, step_each_epoch, epochs, warmup=0, verbose=False, last_epoch=-1, **kwargs):
        if kwargs.get("total_steps", -1) > 0:
            self.T_max = kwargs.get("total_steps")
        else:
            self.T_max = epochs * step_each_epoch

        self.warmup_steps = warmup if isinstance(warmup, int) else int(math.floor(warmup * self.T_max))
        super(LinearDecayWithWarmup, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return self.base_lr * (float(self.last_epoch) / float(max(1, self.warmup_steps)))
        return self.base_lr * max(0.0, 1.0 - self.last_epoch / self.T_max)


class CosineDecay(lr.LRScheduler):
    def __init__(
        self,
        learning_rate,
        step_each_epoch,
        epochs,
        update_unit="epoch",
        warmups=0,
        verbose=False,
        last_epoch=-1,
        **kwargs
    ):

        self.T_max = epochs if update_unit == "epoch" else step_each_epoch * epochs
        self.warmups = warmups if update_unit == "epoch" else step_each_epoch * warmups

        assert self.warmups < self.T_max

        self.last_epoch = last_epoch
        super(CosineDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):

        progress = (self.last_epoch - self.warmups) / float(self.T_max - self.warmups)
        progress = min(1.0, max(0.0, progress))

        if self.warmups:
            lr = self.base_lr * min(1.0, self.last_epoch / self.warmups)
        else:
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))

        return lr
