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
import numpy as np

import paddle

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../../")))

from ppfleetx.data import build_dataloader
from ppfleetx.distributed.apis import env
from ppfleetx.models import build_module
from ppfleetx.optims import build_lr_scheduler, build_optimizer
from ppfleetx.utils import config

# paddle.set_default_dtype("float64")


if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)
    paddle.set_device(cfg["Global"]["device"])

    env.set_seed(cfg.Global.seed)

    module = build_module(cfg)
    config.print_config(cfg)

    train_data_loader = build_dataloader(cfg.Data, "Train")
    eval_data_loader = build_dataloader(cfg.Data, "Eval")

    # model = paddle.jit.to_static(module.model)
    # loss_fn = paddle.jit.to_static(module.loss_fn)
    model = module.model
    loss_fn = module.loss_fn

    cfg.Optimizer.lr.update(
        {
            "epochs": cfg.Engine.num_train_epochs,
            "step_each_epoch": len(train_data_loader),
            "total_steps": cfg.Engine.max_steps,
        }
    )
    lr_scheduler = build_lr_scheduler(cfg.Optimizer.lr)
    optimizer = build_optimizer(cfg.Optimizer, model, lr_scheduler)

    print([param.name for param in module.model.parameters()])

    print("=======" * 8)
    for param in module.model.parameters():
        print(param.name, np.sum(np.abs(np.array(param))))
    print("=======" * 8)

    global_batch_size = cfg.Global.global_batch_size
    max_steps = cfg.Engine.max_steps
    for step, batch in enumerate(train_data_loader()):
        tokens, position_ids, labels, loss_mask = batch

        print("tokens:", np.sum(np.abs(np.array(tokens))))
        print("position_ids:", np.sum(np.abs(np.array(position_ids))))
        print("labels:", np.sum(np.abs(np.array(labels))))
        print("loss_mask:", np.sum(np.abs(np.array(loss_mask))))

        preds = model(tokens, position_ids)
        loss = loss_fn(preds, labels, loss_mask)

        # if step == 0:
        #     print(model.forward.concrete_program.main_program)
        #     print(loss_fn.forward.concrete_program.main_program)
        #     print(loss_fn.forward.program_cache.last()[-1][-1].backward_program)
        #     print(model.forward.program_cache.last()[-1][-1].backward_program)

        loss.backward()
        optimizer.step()

        print("step:", step)
        print("grad" + "=======" * 8)
        for param in module.model.parameters():
            print(param.grad.name, np.sum(np.abs(np.array(param.grad))))
        print("=======" * 8)

        optimizer.clear_grad()
        lr_scheduler.step(global_batch_size)

        print(
            "step: %d/%d\t" % (step, max_steps),
            "loss:%.9f\t" % loss,
            "lr:%.5e\t" % optimizer.get_lr(),
        )
        # np.save(f"./jit/loss_{step}.npy", loss)
        # np.save(f"./dy/loss_{step}.npy", loss)

        print("step:", step)
        print("param" + "=======" * 8)
        for param in module.model.parameters():
            print(param.name, np.sum(np.abs(np.array(param))))
        print("=======" * 8)

        if step >= max_steps:
            break
