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
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from paddle.optimizer.lr import LRScheduler


class InverseSquareRootSchedule(LRScheduler):
    """
    Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate  until the configured learning rate. Thereafter
    we decay proportional to the number of updates, with a decay factor set to
    align with the configured learning rate.

    Args:
        warmup_steps(int):
            The number of warmup steps. A super parameter.
        learning_rate(float, optional):
            The learning rate. It is a python float number. Defaults to 1.0.
        last_epoch(int, optional):
            The index of last epoch. Can be set to restart training. Default: -1,
            means initial learning rate.
        verbose(bool, optional):
            If ``True``, prints a message to stdout for each
            update. Defaults to ``False``.
    """

    def __init__(self, warmup_steps, learning_rate=1.0, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        warmup_end_lr = learning_rate
        self.warmup_init_lr = 0.0
        self.lr_step = (warmup_end_lr - self.warmup_init_lr) / self.warmup_steps
        self.decay_factor = warmup_end_lr * (self.warmup_steps**0.5)

        super(InverseSquareRootSchedule, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return self.warmup_init_lr + self.last_epoch * self.lr_step
        else:
            return self.decay_factor * (self.last_epoch**-0.5)
