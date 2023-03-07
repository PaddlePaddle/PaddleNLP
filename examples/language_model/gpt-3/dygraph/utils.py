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

import copy
import os
import re

import numpy as np
import paddle

__all__ = ["merge_model_parallel"]

PREFIX_CHECKPOINT_DIR = "model_state"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\_mp_(\d+)" + ".pdparams$")


def get_model_parallel_paramerters(folder):
    content = os.listdir(folder)
    if "model_state.pdparams" in content:
        return [os.path.join(folder, "model_state.pdparams")]

    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isfile(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        raise ValueError("No checkpoint found within folder {}".format(folder))

    return [
        os.path.join(folder, v) for v in sorted(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0]))
    ]


def MergedKeys(num_layers):
    res = {}
    Column = [
        "gpt.decoder.layers.0.linear1.bias",
        "gpt.decoder.layers.0.linear1.weight",
        "gpt.decoder.layers.0.self_attn.qkv_proj.bias",
        "gpt.decoder.layers.0.self_attn.qkv_proj.weight",
    ]

    Row = [
        "gpt.embeddings.word_embeddings.weight",
        # 'gpt.decoder.layers.0.self_attn.out_proj.bias',
        "gpt.decoder.layers.0.self_attn.out_proj.weight",
        # 'gpt.decoder.layers.0.linear2.bias',
        "gpt.decoder.layers.0.linear2.weight",
    ]
    for v in Column:
        if "layers.0." in v:
            for i in range(num_layers):
                res[v.replace("layers.0.", f"layers.{i}.")] = "col"
        else:
            res[v] = "col"
    for v in Row:
        if "layers.0." in v:
            for i in range(num_layers):
                res[v.replace("layers.0.", f"layers.{i}.")] = "row"
        else:
            res[v] = "row"

    return res


def merge_rows(values):
    return np.concatenate(values, axis=0)


def merge_column(values):
    return np.concatenate(values, axis=-1)


def merge_model_parallel(model_path, config, as_float32=True):
    final_weight = None
    weights_path = get_model_parallel_paramerters(model_path)
    if len(weights_path) == 1:
        final_weight = paddle.load(weights_path[0], return_numpy=True)
    else:
        weights_list = []
        for path in weights_path:
            weights_list.append(paddle.load(path, return_numpy=True))

        final_weight = copy.deepcopy(weights_list[0])
        merged_keys = MergedKeys(config.num_hidden_layers)

        for k, func_name in merged_keys.items():
            func = merge_column if "col" == func_name else merge_rows
            final_weight[k] = func([weight[k] for weight in weights_list])

    if as_float32:
        for k in final_weight.keys():
            final_weight[k] = final_weight[k].astype("float32")

    return final_weight
