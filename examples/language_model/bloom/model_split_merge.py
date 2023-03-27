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
import shutil

import numpy as np
import paddle
from modeling import BloomModel

from paddlenlp.utils.env import MODEL_HOME

PREFIX_CHECKPOINT_DIR = "model_state"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\_mp_(\d+)" + ".pdparams$")


def get_model_parallel_paramerters(folder):
    content = os.listdir(folder)
    if "model_state.pdparams" in content:
        return ["model_state.pdparams"]

    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isfile(os.path.join(folder, path))
    ]
    print("checkpoints", checkpoints)
    if len(checkpoints) == 0:
        raise ValueError("No checkpoint found within folder {}".format(folder))

    return [
        os.path.join(folder, v) for v in sorted(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0]))
    ]


def MergedKeys(num_layers):
    res = {}
    Column = [
        "h.0.self_attention.query_key_value.weight",
        "h.0.self_attention.query_key_value.bias",
        "h.0.mlp.dense_h_to_4h.bias",
        "h.0.mlp.dense_h_to_4h.weight",
    ]

    Row = [
        # "h.0.self_attention.dense.bias",
        "word_embeddings.weight",
        "h.0.self_attention.dense.weight",
        "h.0.mlp.dense_4h_to_h.weight",
        # "h.0.mlp.dense_4h_to_h.bias"
    ]

    for v in Column:
        if "h.0." in v:
            for i in range(num_layers):
                res[v.replace("h.0.", f"h.{i}.")] = "col"
        else:
            res[v] = "col"
    for v in Row:
        if "h.0." in v:
            for i in range(num_layers):
                res[v.replace("h.0.", f"h.{i}.")] = "row"
        else:
            res[v] = "row"
    return res


def merge_rows(values):
    return np.concatenate(values, axis=0)


def merge_columns(values):
    return np.concatenate(values, axis=-1)


def split_rows(data, indices_or_sections):
    return np.split(data, axis=0, indices_or_sections=indices_or_sections)


def split_columns(data, indices_or_sections):
    return np.split(data, axis=-1, indices_or_sections=indices_or_sections)


def construct_sub_model_name_or_path(model_name_or_path, mp_degree, sharding_degree):
    sub_directory_name = os.path.join(
        model_name_or_path, "splits_mp_{:0>2d}_sharding_{:0>2d}".format(mp_degree, sharding_degree)
    )
    return sub_directory_name


# TODO(wawltor) just support the model parallel
def split_model_parallel(model_name_or_path, config, mp_degree, sharding_degree, as_float32=False):
    # Get the 3D rank
    state_dict = None
    is_path = True if os.path.exists(model_name_or_path) else False
    if not is_path:
        model_name = model_name_or_path
        model_name_or_path = os.path.join(MODEL_HOME, model_name_or_path)
        if not os.path.exists(os.path.join(model_name_or_path, "model_state.pdparams")):
            model = BloomModel.from_pretrained(model_name, low_cpu_mem_usage=True)
            state_dict = model.state_dict()

    # Check the model split files exists
    sub_directory_name = construct_sub_model_name_or_path(model_name_or_path, mp_degree, sharding_degree)
    is_all_splits_ready = True
    if os.path.exists(sub_directory_name):
        for i in range(0, mp_degree):
            weight_name = os.path.join(sub_directory_name, "model_state_mp_{:0>2d}.pdparams".format(i))
            if not os.path.exists(weight_name):
                is_all_splits_ready = False
                break
    else:
        is_all_splits_ready = False

    if is_all_splits_ready:
        return sub_directory_name
    if not os.path.exists(sub_directory_name):
        os.mkdir(sub_directory_name)
    # Generate the split files
    if state_dict is None:
        state_dict = paddle.load(os.path.join(model_name_or_path, "model_state.pdparams"), return_numpy=True)
    state_dict_splits = [copy.deepcopy(state_dict) for i in range(0, mp_degree)]
    merged_keys = MergedKeys(config.n_layer)
    # reversed_merged_keys = dict(zip(merged_keys.values(), merged_keys.keys()))
    for key, key_type in merged_keys.items():
        parameter = state_dict[key]
        if key_type == "row":
            parameter_splits = split_rows(parameter, mp_degree)
        else:
            parameter_splits = split_columns(parameter, mp_degree)
        for idx, state_dict_split in enumerate(state_dict_splits):
            state_dict_split[key] = parameter_splits[idx]
    if as_float32:
        for state_dict_split in state_dict_splits:
            for k in state_dict_split.keys():
                state_dict_split[k] = state_dict_split[k].astype("float32")
    # Copy the config to the subset directory
    for file_name in os.listdir(model_name_or_path):
        if file_name.count(".json"):
            source_file = os.path.join(model_name_or_path, file_name)
            target_file = os.path.join(sub_directory_name, file_name)
            shutil.copyfile(source_file, target_file)

    # Save the split files
    for idx, state_dict_split in enumerate(state_dict_splits):
        weight_name = os.path.join(sub_directory_name, "model_state_mp_{:0>2d}.pdparams".format(idx))
        paddle.save(state_dict_split, weight_name)
    return sub_directory_name


def merge_model_parallel(model_name_or_path, config, as_float32=False):
    # Get the 3D rank
    is_path = True if os.path.exists(model_name_or_path) else False
    if not is_path:
        raise "Please input the path for the model"
    weight_file_name = os.path.join(model_name_or_path, "model_state.pdparams")
    if os.path.exists(os.path.join(model_name_or_path, "model_state.pdparams")):
        return weight_file_name

    # Collect the split files
    file_list = []
    for file_name in os.listdir(model_name_or_path):
        if file_name.count("model_state_mp") and file_name.count("pdparams"):
            file_list.append(file_name)
    file_list.sort()
    state_dict_list = []
    for file_name in file_list:
        state_dict = paddle.load(os.path.join(model_name_or_path, file_name), return_numpy=True)
        state_dict_list.append(state_dict)

    # Merge the state_dict
    final_weight = copy.deepcopy(state_dict_list[0])
    merged_keys = MergedKeys(config.n_layer)
    for k, func_name in merged_keys.items():
        func = merge_columns if "col" == func_name else merge_rows
        k = "{}.{}".format(config.model_type, k)
        final_weight[k] = func([weight[k] for weight in state_dict_list])

    if as_float32:
        for k in final_weight.keys():
            final_weight[k] = final_weight[k].astype("float32")

    # Save the merge state dict
    paddle.save(final_weight, weight_file_name)

    return weight_file_name
