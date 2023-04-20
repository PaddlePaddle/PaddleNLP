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
import paddle.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../../")))

from ppfleetx.distributed.apis import env
from ppfleetx.models import build_module
from ppfleetx.utils import config

if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)

    if dist.get_world_size() > 1:
        env.init_dist_env(cfg)

    env.set_seed(cfg.Global.seed)

    module = build_module(cfg)
    config.print_config(cfg)

    module.model.eval()

    ckpt_dir = cfg.Engine.save_load.ckpt_dir
    if ckpt_dir is not None:
        model_path = os.path.join(ckpt_dir, "model.pdparams")
        model_dict = paddle.load(model_path)

        for key, value in model_dict.items():
            model_dict[key] = model_dict[key].astype(paddle.float32)

        module.model.set_state_dict(model_dict)

    input_text = "Hi, GPT2. Tell me who Jack Ma is."
    result = module.generate(input_text)

    print(f"Prompt: {input_text}")
    print(f"Generation: {result[0]}")
