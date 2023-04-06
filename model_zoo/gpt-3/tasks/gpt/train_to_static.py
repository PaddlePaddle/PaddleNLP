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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../../")))

from ppfleetx.data import build_dataloader
from ppfleetx.distributed.apis import env
from ppfleetx.models import build_module
from ppfleetx.optims import build_lr_scheduler, build_optimizer
from ppfleetx.utils import config

if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)
    paddle.set_device(cfg["Global"]["device"])

    env.set_seed(cfg.Global.seed)

    module = build_module(cfg)
    config.print_config(cfg)

    amp_config = cfg.Engine.mix_precision
    amp_enable = amp_config["enable"]
    amp_dtype = amp_config.get("dtype", "float16")
    amp_level = amp_config.get("level", "O2")
    use_main_grad = amp_config.get("use_main_grad", False)
    scale_loss = amp_config["scale_loss"]
    custom_black_list = amp_config["custom_black_list"]
    custom_white_list = amp_config["custom_white_list"]

    scaler = paddle.amp.GradScaler(init_loss_scaling=scale_loss)
    if amp_level == "O2":
        module.model = paddle.amp.decorate(models=module.model, dtype=amp_dtype, level=amp_level)

    train_data_loader = build_dataloader(cfg.Data, "Train")
    eval_data_loader = build_dataloader(cfg.Data, "Eval")

    model = paddle.jit.to_static(module.model)
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
        tokens, position_ids, labels, loss_mask = batch

        with paddle.amp.auto_cast(
            amp_enable,
            custom_black_list=custom_black_list,
            custom_white_list=custom_white_list,
            dtype=amp_dtype,
            level=amp_level,
        ):
            preds = model(tokens, position_ids)
            loss = module.loss_fn(preds, labels, loss_mask)

        if amp_enable and amp_dtype == "float16":
            scaled = scaler.scale(loss)
            scaled.backward()
        else:
            loss.backward()

        if amp_enable and amp_dtype == "float16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.clear_grad()
        lr_scheduler.step(global_batch_size)

        print(
            "step: %d/%d\t" % (step, max_steps),
            "loss:%.6f\t" % loss.numpy()[0],
            "lr:%.6g\t" % optimizer.get_lr(),
            "loss_scale:%.6f" % scaler._scale.numpy()[0],
        )
