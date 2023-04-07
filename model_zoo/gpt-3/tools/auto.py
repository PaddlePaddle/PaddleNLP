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

import paddle.distributed as dist
from paddle.distributed import fleet

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

from ppfleetx.core import AutoEngine
from ppfleetx.data import build_auto_dataset
from ppfleetx.models import build_module
from ppfleetx.utils import config

# init_logger()

if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_auto_config(args.config, overrides=args.override, show=False)

    if dist.get_world_size() > 1:
        fleet.init(is_collective=True)

    module = build_module(cfg)
    config.print_config(cfg)

    train_data = build_auto_dataset(cfg.Data, "Train")
    eval_data = build_auto_dataset(cfg.Data, "Eval")

    cfg.Optimizer.lr.update({"epochs": cfg.Engine.num_train_epochs, "step_each_epoch": len(train_data)})

    engine = AutoEngine(configs=cfg, module=module)

    if cfg.Engine.save_load.ckpt_dir is not None:
        engine.load()

    if cfg.get("Tuning", None) and cfg.Tuning.enable:
        engine.tune(train_data)
    else:
        engine.fit(train_dataset=train_data, valid_dataset=eval_data, epoch=cfg.Engine.num_train_epochs)
