# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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


def save_checkpoint(args, global_step, model, optimizer, lr_scheduler,
                    tokenizer, loss_scale, dp_rank, mp_rank, pp_rank, pass_num,
                    file_id, epoch):
    """ save some state for each rank."""

    assert args.output_dir is not None, "output_dir is not valid."
    output_dir = os.path.join(args.output_dir, "step_{}".format(global_step))
    os.makedirs(output_dir, exist_ok=True)

    state_dict = {}
    state_dict["args"] = args
    state_dict["global_step"] = global_step
    state_dict["loss_scale"] = loss_scale
    state_dict["data_meta"] = {
        "pass_num": pass_num,
        "file_id": file_id,
        "start_epoch": epoch
    }

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()

    if lr_scheduler is not None:
        state_dict["lr_scheduler"] = lr_scheduler.state_dict()

    if args.pp_degree > 1:
        path = os.path.join(
            output_dir, "dp_{}_mp_{}_pp_{}".format(dp_rank, mp_rank, pp_rank))
        #model.save_state_dict(path)
        paddle.save(model.state_dict(),
                    os.path.join(path, "model_state.pdparams"))
        tokenizer.save_pretrained(path)
    else:
        path = os.path.join(output_dir, "dp_{}_mp_{}".format(dp_rank, mp_rank))
        tokenizer.save_pretrained(path)
        paddle.save(model.state_dict(),
                    os.path.join(path, "model_state.pdparams"))

    state_save_path = os.path.join(path, "meta_state.pdopt")
    paddle.save(state_dict, state_save_path)


def load_checkpoint(args, model, optimizer, lr_scheduler, tokenizer, dp_rank,
                    mp_rank, pp_rank):
    """ load checkpoint for all rank."""

    assert args.resume_dir is not None and len(
        args.resume_dir) > 0, "resume_dir is not valid."
    assert os.path.exists(args.resume_dir) and os.path.isdir(
        args.resume_dir), "resume_dir not exists or not a dir."

    load_path = None
    if args.pp_degree > 1:
        load_path = os.path.join(
            args.resume_dir, "dp_{}_mp_{}_pp_{}".format(dp_rank, mp_rank,
                                                        pp_rank))
        #model.set_state_dir(load_path)
        model.set_state_dict(
            paddle.load(os.path.join(load_path, "model_state.pdparams")))
    else:
        load_path = os.path.join(args.resume_dir,
                                 "dp_{}_mp_{}".format(dp_rank, mp_rank))
        model.set_state_dict(
            paddle.load(os.path.join(load_path, "model_state.pdparams")))

    tokenizer.from_pretrained(load_path)
    state_dict = paddle.load(os.path.join(load_path, "meta_state.pdopt"))

    if optimizer is not None:
        optimizer.set_state_dict(state_dict["optimizer"])

    if lr_scheduler is not None:
        lr_scheduler.set_state_dict(state_dict["lr_scheduler"])

    global_step = state_dict["global_step"]
    args.seed = state_dict["args"].seed
    loss_scale = state_dict["loss_scale"]

    resume_step = int(args.resume_dir.strip("/").split("_")[-1])
    if resume_step != global_step:
        print("Warning: resume_step is {}, but the step of checkpoint is {}.".
              format(resume_step, global_step))

    return global_step, loss_scale, state_dict["data_meta"]
