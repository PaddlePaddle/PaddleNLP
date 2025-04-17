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

import paddle
import paddle.distributed as dist
from paddle.incubate.distributed.utils.io import save_for_auto_inference
from ppfleetx.distributed.apis import env
from ppfleetx.utils.log import logger


def save(output_dir, model, optimizer=None, step=0, epoch=0, sharding_stage=2):
    """
    save the state dicts of model and optimizer into an checkpoint.
    """

    nranks = dist.get_world_size()
    if nranks > 1:
        hcg = env.get_hcg()

        dp_rank = hcg.get_data_parallel_rank()
        mp_rank = hcg.get_model_parallel_rank()
        pp_rank = hcg.get_stage_id()
        sharding_rank = hcg.get_sharding_parallel_rank()
    else:
        dp_rank = 0

    if dp_rank != 0:
        logger.info("DP_Rank %d doesn't save model" % dp_rank)
        return

    if output_dir and isinstance(output_dir, str):
        output_dir = os.path.join(output_dir, "epoch_%d_step_%d" % (epoch, step))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        logger.info("Save model to %s" % output_dir)

        save_dir = (
            "{}/mp_{:0>2d}_sharding_{:0>2d}_pp_{:0>2d}".format(output_dir, mp_rank, sharding_rank, pp_rank)
            if nranks > 1
            else output_dir
        )

        if sharding_stage == 3:
            model.get_all_parameters(convert2cpu=False)

        paddle.save(model.state_dict(), os.path.join(save_dir, "model.pdparams"))

        if optimizer is not None:
            paddle.save(optimizer.state_dict(), os.path.join(save_dir, "model_state.pdopt"))

        meta_dict = {"epoch": epoch, "step": step, "cuda_rng_state": paddle.get_cuda_rng_state()}
        paddle.save(meta_dict, os.path.join(save_dir, "meta_state.pdopt"))

        save_auto_dir = os.path.join(output_dir, "auto_infer")
        save_for_auto_inference(os.path.join(save_auto_dir, "auto"), model)

    else:
        raise TypeError("`save` requires a valid value of `output_dir`.")


def load(ckpt_dir, model, optimizer=None, mode="train", load_recovery=None):
    nranks = dist.get_world_size()
    if nranks > 1:
        hcg = env.get_hcg()

        mp_rank = hcg.get_model_parallel_rank()
        pp_rank = hcg.get_stage_id()
        sharding_rank = hcg.get_sharding_parallel_rank()

    load_recovery = {} if load_recovery is None else load_recovery

    if ckpt_dir and isinstance(ckpt_dir, str):
        logger.info("Try to load checkpoint from %s " % ckpt_dir)

        if mode == "quant":
            load_dir = ckpt_dir
        else:
            load_dir = (
                "{}/mp_{:0>2d}_sharding_{:0>2d}_pp_{:0>2d}".format(ckpt_dir, mp_rank, sharding_rank, pp_rank)
                if nranks > 1
                else ckpt_dir
            )
        model_path = os.path.join(load_dir, "model.pdparams")
        opt_path = os.path.join(load_dir, "model_state.pdopt")
        meta_path = os.path.join(load_dir, "meta_state.pdopt")

        if os.path.exists(model_path):
            model_dict = paddle.load(model_path)
            for name, param in model.state_dict().items():
                assert name in model_dict.keys(), "No param named `{}` was found in checkpoint file.".format(name)

                if param.dtype != model_dict[name].dtype:
                    model_dict[name] = model_dict[name].cast(param.dtype)

            model.set_state_dict(model_dict)
        else:
            raise ValueError("No model checkpoint file found in %s." % model_path)

        if mode == "train":
            if os.path.exists(opt_path):
                opt_dict = paddle.load(opt_path)
                optimizer.set_state_dict(opt_dict)
            else:
                raise ValueError("No optimizer checkpoint file found in %s." % opt_path)

            if os.path.exists(meta_path):
                meta_dict = paddle.load(meta_path)

                load_recovery.update(
                    {"step": meta_dict["step"], "epoch": meta_dict["epoch"], "rng_state": meta_dict["cuda_rng_state"]}
                )

            else:
                raise ValueError("No meta checkpoint file found in %s." % meta_path)

        logger.info("successfully load checkpoints")
    else:
        logger.warning("`load` requires a valid value of `ckpt_dir`.")
        raise TypeError("`load` requires a valid value of `ckpt_dir`.")
