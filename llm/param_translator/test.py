# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from param_translator import ParamTranslator
from trans_dy_ckpt import MapGenerator

if __name__ == "__main__":
    dy_hand_checkpoint_path = "/root/paddlejob/workspace/env_run/output/baidu/dialogue/PaddleNLP/llm/auto_parallel/llama/checkpoints_dy/llama2_pretrain_ckpts/checkpoint-1/"
    dy_hand_model_state_name = [
        "model_state.tp00_pp00.pdparams",
        "model_state.tp00_pp01.pdparams",
        "model_state.tp00_pp02.pdparams",
        "model_state.tp00_pp03.pdparams",
        "model_state.tp01_pp00.pdparams",
        "model_state.tp01_pp01.pdparams",
        "model_state.tp01_pp02.pdparams",
        "model_state.tp01_pp03.pdparams",
    ]
    dy_hand_model_name = [
        "model_state.tp00_pp00.keymap",
        "model_state.tp00_pp01.keymap",
        "model_state.tp00_pp02.keymap",
        "model_state.tp00_pp03.keymap",
        "model_state.tp01_pp00.keymap",
        "model_state.tp01_pp01.keymap",
        "model_state.tp01_pp02.keymap",
        "model_state.tp01_pp03.keymap",
    ]
    dy_hand_optim_state_name = [
        "optimizer.tp00_pp00.pdopt",
        "optimizer.tp00_pp01.pdopt",
        "optimizer.tp00_pp02.pdopt",
        "optimizer.tp00_pp03.pdopt",
        "optimizer.tp01_pp00.pdopt",
        "optimizer.tp01_pp01.pdopt",
        "optimizer.tp01_pp02.pdopt",
        "optimizer.tp01_pp03.pdopt",
    ]

    dy2st_checkpoint_path = "/root/paddlejob/workspace/env_run/output/baidu/dialogue/PaddleNLP/llm/auto_parallel/llama/checkpoints_st/llama2_pretrain_ckpts/checkpoint-1/dist_ckpt/"
    dy2st_model_state_name = [
        "0_0.model.param",
        "2_0.model.param",
        "4_0.model.param",
        "6_0.model.param",
        "1_0.model.param",
        "3_0.model.param",
        "5_0.model.param",
        "7_0.model.param",
    ]
    dy2st_model_name = [
        "0_0.model.param.name",
        "2_0.model.param.name",
        "4_0.model.param.name",
        "6_0.model.param.name",
        "1_0.model.param.name",
        "3_0.model.param.name",
        "5_0.model.param.name",
        "7_0.model.param.name",
    ]

    need_repartition_tensor_names = [
        "llama.embed_tokens.weight",
        "embedding_0.w_0",
        "embedding_0.w_0_fp32_master_0_moment1_0",
        "embedding_0.w_0_fp32_master_0_moment2_0",
    ]
    need_repartition_files = [
        {
            "model": os.path.join(dy_hand_checkpoint_path, dy_hand_model_state_name[0]),
            "optim": os.path.join(dy_hand_checkpoint_path, dy_hand_optim_state_name[0]),
        },
        {
            "model": os.path.join(dy_hand_checkpoint_path, dy_hand_model_state_name[4]),
            "optim": os.path.join(dy_hand_checkpoint_path, dy_hand_optim_state_name[4]),
        },
    ]

    print("==========translate param=================")

    saved_checkpoint_path = "/root/paddlejob/workspace/env_run/output/baidu/dialogue/PaddleNLP/llm/auto_parallel/llama/checkpoints_trans/llama2_pretrain_ckpts/checkpoint-1/dist_ckpt/"

    rank_name_maps = MapGenerator(
        dy_hand_checkpoint_path, dy_hand_model_name, dy2st_checkpoint_path, dy2st_model_name
    ).gen()

    ParamTranslator(
        rank_name_maps,
        need_repartition_tensor_names,
        need_repartition_files,
        model_state_dict_file_names=dy_hand_model_state_name,
        optim_state_dict_file_names=dy_hand_optim_state_name,
        dy_hand_checkpoint_path=dy_hand_checkpoint_path,
        saved_checkpoint_path=saved_checkpoint_path,
        save_file_names=dy2st_model_state_name,
    ).do_translate()
