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

from __future__ import absolute_import, division, print_function

import os
import sys

import paddle
import paddle.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

from ppfleetx.core import AutoEngine
from ppfleetx.data import build_auto_dataset
from ppfleetx.distributed.apis import auto_env
from ppfleetx.models import build_module
from ppfleetx.utils import auto_config

if __name__ == "__main__":
    args = auto_config.parse_args()
    cfg = auto_config.get_config(args.config, overrides=args.override, show=False)

    paddle.set_device(cfg["Global"]["device"])
    if dist.get_world_size() > 1:
        auto_env.init_dist_env(cfg)

    # TODO: set seed for auto_parallel

    module = build_module(cfg)
    auto_config.print_config(cfg)

    train_data = build_auto_dataset(cfg.Data, "Train")
    eval_data = build_auto_dataset(cfg.Data, "Eval")

    cfg.Optimizer.lr.update(
        {
            "epochs": cfg.Engine.num_train_epochs,
            "step_each_epoch": len(train_data),
        }
    )

    engine = AutoEngine(configs=cfg, module=module)

    if cfg.Engine.save_load.ckpt_dir is not None:
        engine.load()

    if cfg.get("Tuning", None) and cfg.Tuning.enable:
        engine.tune(train_data)
    else:
        engine.fit(train_dataset=train_data, valid_dataset=eval_data, epoch=cfg.Engine.num_train_epochs)
