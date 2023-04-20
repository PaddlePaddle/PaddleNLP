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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

from ppfleetx.core import AutoEngine
from ppfleetx.models import build_module
from ppfleetx.utils import config

if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_auto_config(args.config, overrides=args.override, show=False)

    if cfg.get("Model", None) is not None:
        module = build_module(cfg)
        config.print_config(cfg)

        engine = AutoEngine(configs=cfg, module=module, mode="export")

        if cfg.Engine.save_load.ckpt_dir is not None:
            engine.load()

        engine.export()
    else:
        engine = AutoEngine(configs=cfg, mode="export")
        if cfg.Engine.save_load.ckpt_dir is None:
            raise ValueError("invalid ckpt_dir.")

        engine.export_from_prog()
