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
import sys

import paddle
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../../")))

from ppfleetx.data import build_dataloader
from ppfleetx.distributed.apis import env
from ppfleetx.models import build_module
from ppfleetx.optims import build_lr_scheduler, build_optimizer
from ppfleetx.utils import config


class MovingAverage:
    def __init__(self):
        self.sum = 0
        self.val = [0] * self.window_size
        self.cnt = 0

    def update(self, val, n):
        self.cnt = min(self.cnt + n, self.window_size)
        offset = max(self.window_size - n, 0)
        self.sum -= sum(self.values[:-offset])
        self.sum = val * min(n, self.window_size)
        self.avg = self.sum / self.cnt


def main():
    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)
    paddle.device.set_device("gpu:0")
    env.set_seed(cfg.Global.seed)
    module = build_module(cfg)
    config.print_config(cfg)

    amp_config = cfg.Engine.mix_precision
    scale_loss = amp_config["scale_loss"]

    scaler = paddle.amp.GradScaler(init_loss_scaling=scale_loss)

    train_data_loader = build_dataloader(cfg.Data, "Train")

    enable_to_static = cfg.Global.to_static
    if str(enable_to_static).lower() == "true":
        model = paddle.jit.to_static(module.model)
    else:
        model = module.model

    cfg.Optimizer.lr.update(
        {
            "epochs": cfg.Engine.num_train_epochs,
            "step_each_epoch": len(train_data_loader),
            "total_steps": cfg.Engine.max_steps,
        }
    )
    lr_scheduler = build_lr_scheduler(cfg.Optimizer.lr)
    optimizer = build_optimizer(cfg.Optimizer, model, lr_scheduler)

    global_batch_size = cfg.Global.global_batch_size
    max_steps = cfg.Engine.max_steps
    for step, batch in enumerate(train_data_loader()):
        if step <= max_steps:
            init_time = time.time()
            tokens, position_ids, labels, loss_mask = batch

            preds = model(tokens, position_ids)
            loss = module.loss_fn(preds, labels, loss_mask)

            loss.backward()
            optimizer.step()

            optimizer.clear_grad()
            lr_scheduler.step(global_batch_size)
            after_time = time.time()
            during_time = after_time - init_time

            print(
                "step: %d/%d\t" % (step, max_steps),
                "loss:%.6f\t" % loss.numpy(),
                "lr:%.6g\t" % optimizer.get_lr(),
                "loss_scale:%.1f\t" % scaler._scale.numpy(),
                "batch time: %.4f s" % (during_time),
            )


if __name__ == "__main__":
    main()
