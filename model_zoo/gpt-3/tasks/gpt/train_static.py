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

paddle.enable_static()

def print_param(program):
    print("======="*8)
    from paddle.fluid.framework import Parameter
    def is_parameter(var):
        return isinstance(var, Parameter)
    def is_persist(var):
        return var.persistable
    def get_tensor(var):
        t = paddle.fluid.global_scope().find_var(var.name).get_tensor()
        return np.array(t)
    def get_name(var):
        return var.name
    parameter_list = list(filter(is_persist, program.list_vars()))
    # for p in sorted(parameter_list, key=get_name):
    #     print(p)
    for p in sorted(parameter_list, key=get_name):
        if p.name in ['create_py_reader_0', 'double_buffer_0', 'create_py_reader_1', 'double_buffer_1']:
            continue
        if "comm" in p.name or "feed" in p.name or "fetch" in p.name:
            continue
        print(p.name, np.sum(get_tensor(p)))
    print("======="*8)


if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)
    paddle.set_device(cfg["Global"]["device"])

    env.set_seed(cfg.Global.seed)

    module = build_module(cfg)
    config.print_config(cfg)

    train_data_loader = build_dataloader(cfg.Data, "Train")
    eval_data_loader = build_dataloader(cfg.Data, "Eval")

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

    var_tokens = paddle.static.data("tokens", shape=[cfg.Global.local_batch_size, cfg.Data.Train.dataset.max_seq_len], dtype="int64")
    var_position_ids = paddle.static.data("position_ids", shape=[cfg.Global.local_batch_size, cfg.Data.Train.dataset.max_seq_len], dtype="int64")
    var_labels = paddle.static.data("labels", shape=[cfg.Global.local_batch_size, cfg.Data.Train.dataset.max_seq_len], dtype="int64")
    var_loss_mask = paddle.static.data("loss_mask", shape=[cfg.Global.local_batch_size, cfg.Data.Train.dataset.max_seq_len], dtype="float32")

    preds = model(var_tokens, var_position_ids)
    loss = module.loss_fn(preds, var_labels, var_loss_mask)
    _, params_grads = optimizer.minimize(loss)

    print(paddle.static.default_main_program())

    exe = paddle.static.Executor(paddle.framework.CUDAPlace(paddle.distributed.ParallelEnv().dev_id))
    outs = exe.run(paddle.static.default_startup_program(), fetch_list=[p.name for p, g in params_grads])

    print("=======" * 8)
    for idx, param in enumerate(outs):
        print(params_grads[idx][0].name, np.sum(np.abs(np.array(param))))
    print("=======" * 8)

    global_batch_size = cfg.Global.global_batch_size
    max_steps = cfg.Engine.max_steps
    for step, batch in enumerate(train_data_loader()):
        tokens, position_ids, labels, loss_mask = batch

        print("tokens:", np.sum(np.abs(np.array(tokens))))
        print("position_ids:", np.sum(np.abs(np.array(position_ids))))
        print("labels:", np.sum(np.abs(np.array(labels))))
        print("loss_mask:", np.sum(np.abs(np.array(loss_mask))))

        square_sum_list = [f"squared_l2_norm_{i}.tmp_0" for i in range(52)] + ["add_n_0.tmp_0", "fill_constant_1.tmp_0", "elementwise_div_0"]

        outs = exe.run(
            paddle.static.default_main_program(), 
            feed={
                "tokens": tokens, 
                "position_ids": position_ids, 
                "labels": labels, 
                "loss_mask": loss_mask
            }, 
            fetch_list=[loss.name] + 
                [g.name for p, g in params_grads] + 
                [p.name for p, g in params_grads] + 
                square_sum_list
        )

        zyl_fetch_list = outs[-len(square_sum_list):]
        for i in range(52):
            print(np.sum(np.abs(np.array(zyl_fetch_list[i]))))
        print("global_norm_var:", np.sum(np.abs(np.array(zyl_fetch_list[-3]))))
        print("max_global_norm:", np.sum(np.abs(np.array(zyl_fetch_list[-2]))))
        print("clip_var:", np.sum(np.abs(np.array(zyl_fetch_list[-1]))))

        outs = outs[:-len(square_sum_list)]
        print("step:", step)
        print("grad" + "=======" * 8)
        for idx, grad in enumerate(outs[1: len(params_grads) + 1]):
            print(params_grads[idx][1].name, np.sum(np.abs(grad)))
        print("=======" * 8)

        lr_scheduler.step(global_batch_size)

        print(
            "step: %d/%d\t" % (step, max_steps),
            "loss:%.9f\t" % outs[0],
            "lr:%.5e\t" % optimizer.get_lr(),
        )
        # np.save(f"./st/loss_{step}.npy", outs[0])

        print("step:", step)
        print("param" + "=======" * 8)
        for idx, param in enumerate(outs[len(params_grads) + 1:]):
            print(params_grads[idx][0].name, np.sum(np.abs(param)))
        print("=======" * 8)

        if step >= max_steps:
            break
