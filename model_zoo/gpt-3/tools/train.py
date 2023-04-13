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
from distutils.util import strtobool

import paddle
import paddle.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

from ppfleetx.core import EagerEngine
from ppfleetx.data import build_dataloader
from ppfleetx.distributed.apis import env
from ppfleetx.models import build_module
from ppfleetx.ops.fused_layers import mock_layers
from ppfleetx.utils import config


def set_default_flags(flags):
    for flag_name, flag_value in flags.items():
        if os.getenv(flag_name) is None:
            paddle.set_flags({flag_name: flag_value})


if __name__ == "__main__":
    mock_layers()

    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)

    paddle.set_device(cfg["Global"]["device"])
    if dist.get_world_size() > 1:
        env.init_dist_env(cfg)

    env.set_seed(cfg.Global.seed)

    module = build_module(cfg)
    config.print_config(cfg)

    train_data_loader = build_dataloader(cfg.Data, "Train")
    eval_data_loader = build_dataloader(cfg.Data, "Eval")

    load_torch_random_175B_ckpt = os.getenv("load_torch_random_175B_ckpt", False)
    mp_rank = env.get_hcg().get_model_parallel_rank() if env.get_hcg() is not None else 0
    pp_rank = env.get_hcg().get_stage_id() if env.get_hcg() is not None else 0

    # hack code for loading part-1
    load_torch_random_175B_ckpt = False
    if load_torch_random_175B_ckpt:
        from load_t2p import trans_model_for_non_pp, trans_model_for_pp
        import torch

        dir_path = '/root/paddlejob/workspace/env_run/gpt_benchmark/dev_xys/Megatron-LM/random_init_torch_ckpt_175b_188/{}'
        file_name = 'mp_rank{}_pp_rank{}.bin'.format(mp_rank, pp_rank)

        # Load the checkpoints.
        model_checkpoint_name = dir_path.format(file_name)
        optim_checkpoint_name = model_checkpoint_name

        print("model_checkpoint_name :", model_checkpoint_name)
        torch_state = torch.load(model_checkpoint_name, map_location='cpu')

        # Load model weights.
        if cfg.Distributed.pp_degree > 1:
            paddle_state = trans_model_for_pp(torch_state, pp_rank, cfg.Distributed.pp_degree, cfg['Model']['num_layers'], cfg['Model']['max_position_embeddings'], cfg['Model']['hidden_size'])
        else:
            paddle_state = trans_model_for_non_pp(torch_state, cfg['Model']['num_layers'])

        missing_keys, unexpected_keys = module.set_state_dict(
            paddle_state)

        assert len(missing_keys) == 0 and len(unexpected_keys) == 0, "mp_rank : {}.    pp_rank : {}".format(mp_rank, pp_rank)

    cfg.Optimizer.lr.update(
        {
            "epochs": cfg.Engine.num_train_epochs,
            "step_each_epoch": len(train_data_loader),
            "total_steps": cfg.Engine.max_steps,
        }
    )

    engine = EagerEngine(configs=cfg, module=module)

    if cfg.Engine.save_load.ckpt_dir is not None:
        engine.load()

    engine.fit(
        train_data_loader=train_data_loader, valid_data_loader=eval_data_loader, epoch=cfg.Engine.num_train_epochs
    )
