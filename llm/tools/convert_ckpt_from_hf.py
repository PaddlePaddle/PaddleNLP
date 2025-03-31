# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import collections
import copy
import json
import os
import shutil

import paddle
import torch
from safetensors.numpy import load_file as numpy_load_file
from safetensors.numpy import save_file as numpy_save_file
from safetensors.torch import load_file as torch_load_file
from safetensors.torch import save_file as torch_save_file

from paddlenlp.transformers import AutoConfig
from paddlenlp.utils import import_module
from paddlenlp.utils.log import logger
from paddlenlp.utils.serialization import load_torch

dtype_mapping = {
    torch.float8_e4m3fn: (torch.int8, "float8_e4m3fn"),
}


def dlpack_torch_to_paddle(t):
    if t.dtype not in dtype_mapping:
        capsule = torch.utils.dlpack.to_dlpack(t)
        t = paddle.utils.dlpack.from_dlpack(capsule)
        return t.numpy()
    else:
        dst, dst_pd = dtype_mapping[t.dtype]
        t = dlpack_torch_to_paddle(t.view(dst))
        return t.view(dtype=dst_pd)


def execute_cmd(cmd, file_path):
    cmd = cmd + " " + file_path
    os.system(cmd)


def convert_from_torch_to_paddle(
    torch_path=None,
    paddle_path=None,
    torch_prefix_key="model.",
    delete_after_convert=False,
    extra_name_mappings=None,
):
    assert torch_path is not None, "torch_path is None"
    if paddle_path is None:
        paddle_path = torch_path + "-paddle"
    if not os.path.exists(paddle_path):
        # os.mkdir(paddle_path)
        os.makedirs(paddle_path, exist_ok=True)

    config = AutoConfig.from_pretrained(torch_path)
    paddle_class = import_module(f"paddlenlp.transformers.{config.architectures[0]}")
    name_mappings = paddle_class._get_name_mappings(config=config)
    if extra_name_mappings is not None:
        # 拼接name_mappings list
        name_mappings.extend(extra_name_mappings)

    torch_prefix_key = torch_prefix_key
    paddle_prefix_key = paddle_class.base_model_prefix + "."

    if os.path.exists(os.path.join(torch_path, "original", "tokenizer.model")):
        # copy config.json and other files
        shutil.copy(
            os.path.join(torch_path, "original", "tokenizer.model"),
            os.path.join(paddle_path, "tokenizer.model"),
        )

    if os.path.exists(os.path.join(torch_path, "model.safetensors.index.json")):
        index = json.load(open(os.path.join(torch_path, "model.safetensors.index.json")))
        dst_index = copy.deepcopy(index)

        for key in list(dst_index["weight_map"].keys()):
            paddle_key = key.replace(torch_prefix_key, paddle_prefix_key)
            dst_index["weight_map"][paddle_key] = dst_index["weight_map"].pop(key)

        files = set(index["weight_map"].values())
        logger.info(files)

        for file_name in sorted(os.listdir(torch_path)):
            # skip hidden files
            if file_name.startswith("."):
                continue

            logger.info(file_name)
            if file_name in files:
                # convert safetensors to safetensors(paddle)
                convert_safetensors_from_torch_to_paddle(
                    file_name,
                    torch_path,
                    paddle_path,
                    torch_prefix_key,
                    paddle_prefix_key,
                    name_mappings,
                    delete_after_convert=False,
                )
            elif file_name.endswith(".bin") or file_name == "pytorch_model.bin.index.json":
                pass
            else:
                if os.path.isdir(os.path.join(torch_path, file_name)):
                    shutil.copytree(
                        os.path.join(torch_path, file_name),
                        os.path.join(paddle_path, file_name),
                        dirs_exist_ok=True,
                    )
                else:
                    # copy config.json and other files
                    shutil.copy(
                        os.path.join(torch_path, file_name),
                        os.path.join(paddle_path, file_name),
                    )

        json.dump(
            dst_index,
            open(os.path.join(paddle_path, "model.safetensors.index.json"), "w"),
            indent=2,
        )
    else:
        for file_name in sorted(os.listdir(torch_path)):
            # skip hidden files
            if file_name.startswith("."):
                continue

            logger.info(file_name)
            if file_name == "model.safetensors":
                convert_safetensors_from_torch_to_paddle(
                    file_name,
                    torch_path,
                    paddle_path,
                    torch_prefix_key,
                    paddle_prefix_key,
                    name_mappings,
                    delete_after_convert=False,
                )
            else:
                if os.path.isdir(os.path.join(torch_path, file_name)):
                    shutil.copytree(
                        os.path.join(torch_path, file_name),
                        os.path.join(paddle_path, file_name),
                    )
                else:
                    # copy config.json and other files
                    shutil.copy(
                        os.path.join(torch_path, file_name),
                        os.path.join(paddle_path, file_name),
                    )

    execute_cmd(
        cmd="sed -i -e  's/torch_dtype/dtype/g' ",
        file_path=os.path.join(paddle_path, "config.json"),
    )

    return paddle_path


def convert_from_paddle_to_torch(
    torch_path=None,
    paddle_path=None,
    torch_prefix_key="model.",
    delete_after_convert=False,
):
    assert paddle_path is not None
    if torch_path is None:
        torch_path = paddle_path + "-torch"
    if not os.path.exists(torch_path):
        os.mkdir(torch_path)

    config = AutoConfig.from_pretrained(paddle_path)
    paddle_class = import_module(f"paddlenlp.transformers.{config.architectures[0]}")
    name_mappings = paddle_class._get_name_mappings(config=config)

    torch_prefix_key = torch_prefix_key
    paddle_prefix_key = paddle_class.base_model_prefix + "."

    if os.path.exists(os.path.join(paddle_path, "model.safetensors.index.json")):
        index = json.load(open(os.path.join(paddle_path, "model.safetensors.index.json")))
        dst_index = copy.deepcopy(index)

        for key in list(dst_index["weight_map"].keys()):
            torch_key = key.replace(paddle_prefix_key, torch_prefix_key)
            dst_index["weight_map"][torch_key] = dst_index["weight_map"].pop(key)

        files = set(index["weight_map"].values())
        logger.info(files)

        for file_name in sorted(os.listdir(paddle_path)):
            # skip hidden files
            if file_name.startswith("."):
                continue

            logger.info(file_name)
            if file_name in files:
                # convert safetensors to safetensors(paddle)
                convert_safetensors_from_paddle_to_torch(
                    file_name,
                    torch_path,
                    paddle_path,
                    torch_prefix_key,
                    paddle_prefix_key,
                    name_mappings,
                    delete_after_convert=False,
                )
            else:
                if os.path.isdir(os.path.join(paddle_path, file_name)):
                    # copy config.json and other files
                    shutil.copy(
                        os.path.join(paddle_path, file_name),
                        os.path.join(torch_path, file_name),
                    )
                else:
                    # copy config.json and other files
                    shutil.copy(
                        os.path.join(paddle_path, file_name),
                        os.path.join(torch_path, file_name),
                    )

        json.dump(
            dst_index,
            open(os.path.join(torch_path, "model.safetensors.index.json"), "w"),
            indent=2,
        )
    else:
        for file_name in sorted(os.listdir(paddle_path)):
            # skip hidden files
            if file_name.startswith("."):
                continue

            logger.info(file_name)
            if file_name == "model.safetensors":
                convert_safetensors_from_paddle_to_torch(
                    file_name,
                    torch_path,
                    paddle_path,
                    torch_prefix_key,
                    paddle_prefix_key,
                    name_mappings,
                    delete_after_convert=False,
                )
            else:
                if os.path.isdir(os.path.join(paddle_path, file_name)):
                    # copy config.json and other files
                    shutil.copy(
                        os.path.join(paddle_path, file_name),
                        os.path.join(torch_path, file_name),
                    )
                else:
                    # copy config.json and other files
                    shutil.copy(
                        os.path.join(paddle_path, file_name),
                        os.path.join(torch_path, file_name),
                    )

    execute_cmd(
        cmd="sed -i -e  's/dtype/torch_dtype/g' ",
        file_path=os.path.join(torch_path, "config.json"),
    )
    return torch_path


def convert_safetensors_from_torch_to_paddle(
    file_name,
    torch_path,
    paddle_path,
    torch_prefix_key,
    paddle_prefix_key,
    name_mappings,
    delete_after_convert=False,
):
    tensors = torch_load_file(os.path.join(torch_path, file_name))

    transpose_state_dict = {}
    torch_to_paddle_key_mappings = {}
    for name_mapping in name_mappings:
        torch_to_paddle_key_mappings[name_mapping.source_name] = name_mapping.target_name

        if name_mapping.action == "transpose":
            transpose_state_dict[name_mapping.target_name] = True
        else:
            transpose_state_dict[name_mapping.target_name] = False

        # for key without prefix
        if name_mapping.source_name.replace(torch_prefix_key, "") not in torch_to_paddle_key_mappings:
            torch_to_paddle_key_mappings[name_mapping.source_name.replace(torch_prefix_key, "")] = (
                name_mapping.target_name.replace(paddle_prefix_key, "")
            )
            transpose_state_dict[name_mapping.target_name.replace(paddle_prefix_key, "")] = transpose_state_dict[
                name_mapping.target_name
            ]

        # for weight_scale_inv
        if name_mapping.source_name.replace(".weight", ".weight_scale_inv") not in torch_to_paddle_key_mappings:
            torch_to_paddle_key_mappings[name_mapping.source_name.replace(".weight", ".weight_scale_inv")] = (
                name_mapping.target_name.replace(".weight", ".weight_scale_inv")
            )
            transpose_state_dict[name_mapping.target_name.replace(".weight", ".weight_scale_inv")] = (
                transpose_state_dict[name_mapping.target_name]
            )

    for key in list(tensors.keys()):
        if key not in torch_to_paddle_key_mappings:
            tensors.pop(key)
            Warning("{} is not in the name_mappings, it will be ignored.".format(key))
            continue
        paddle_key = torch_to_paddle_key_mappings[key]
        logger.info("{} {}".format(key, tensors[key].shape))
        if transpose_state_dict[paddle_key]:
            t = tensors.pop(key).cuda().t().contiguous()
        else:
            t = tensors.pop(key).cuda()
        tensors[paddle_key] = dlpack_torch_to_paddle(t)
        # tensors[dst_key] = paddle.to_tensor(tensors.pop(key).cuda().float().cpu().numpy(), dtype="bfloat16").numpy()
        logger.info("{} {}".format(paddle_key, tensors[paddle_key].shape))

    numpy_save_file(tensors, os.path.join(paddle_path, file_name), metadata={"format": "np"})
    if delete_after_convert:
        os.remove(os.path.join(torch_path, file_name))


def convert_safetensors_from_paddle_to_torch(
    file_name,
    torch_path,
    paddle_path,
    torch_prefix_key,
    paddle_prefix_key,
    name_mappings,
    delete_after_convert=False,
):
    tensors = numpy_load_file(os.path.join(paddle_path, file_name))

    transpose_state_dict = {}
    for name_mapping in name_mappings:
        if name_mapping.action == "transpose":
            transpose_state_dict[name_mapping.source_name] = True
        else:
            transpose_state_dict[name_mapping.source_name] = False

    for key in list(tensors.keys()):
        torch_key = key.replace(paddle_prefix_key, torch_prefix_key)
        logger.info("{} {}".format(key, tensors[key].shape))
        if transpose_state_dict[torch_key]:
            t = paddle.Tensor(tensors.pop(key), zero_copy=True).t().contiguous()
        else:
            t = paddle.Tensor(tensors.pop(key), zero_copy=True).contiguous()

        capsule = paddle.utils.dlpack.to_dlpack(t)
        t = torch.utils.dlpack.from_dlpack(capsule)

        tensors[torch_key] = t
        logger.info("{} {}".format(torch_key, tensors[torch_key].shape))

    torch_save_file(tensors, os.path.join(torch_path, file_name), metadata={"format": "pt"})
    if delete_after_convert:
        os.remove(os.path.join(paddle_path, file_name))


def EAGLE_convert_from_torch_to_paddle(torch_path, torch_lm_head_weight_path=None, extra_name_mappings=None):
    if torch_lm_head_weight_path is not None:
        lm_head_weight = torch_load_file(torch_lm_head_weight_path)["lm_head.weight"].numpy()
    else:
        lm_head_weight = None
        Warning("No lm_head_weight is provided, the lm_head.weight will not be initialized.")

    if torch_path.endswith(".bin"):
        state_dict = load_torch(torch_path)
    else:
        state_dict = torch_load_file(torch_path)

    if lm_head_weight is not None:
        state_dict["lm_head.weight"] = lm_head_weight

    torch_safetensors_path = os.path.join(os.path.abspath(os.path.dirname(torch_path)), "model.safetensors")
    numpy_save_file(state_dict, torch_safetensors_path, metadata={"format": "np"})

    return convert_from_torch_to_paddle(
        torch_path=os.path.abspath(os.path.dirname(torch_path)),
        extra_name_mappings=extra_name_mappings,
    )


def patch_shared_weights_in_state_dict(state_dict):
    ptrs = collections.defaultdict(list)
    for name, tensor in state_dict.items():
        ptrs[tensor.data_ptr()].append(name)
    # These are all the pointers of shared tensors.
    shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
    print(shared_ptrs)
    for ptr, names in shared_ptrs.items():
        assert len(names) == 2
        state_dict[names[0]] = state_dict[names[0]]
        state_dict[names[1]] = copy.deepcopy(state_dict[names[0]])
    return state_dict


def convert_torch_bin_to_safetensors(torch_path):
    for file_name in os.listdir(torch_path):
        if file_name.endswith(".bin"):
            state_dict = torch.load(os.path.join(torch_path, file_name))
            state_dict = patch_shared_weights_in_state_dict(state_dict)
            safe_tensor_filename = file_name.replace(".bin", ".safetensors").replace("pytorch_model-", "model-")
            torch_safetensors_path = os.path.join(os.path.join(torch_path, safe_tensor_filename))
            torch_save_file(state_dict, torch_safetensors_path)

    if os.path.exists(os.path.join(torch_path, "pytorch_model.bin.index.json")):
        index = json.load(open(os.path.join(torch_path, "pytorch_model.bin.index.json")))
        dst_index = copy.deepcopy(index)
        for key in list(dst_index["weight_map"].keys()):
            dst_value = (
                dst_index["weight_map"][key].replace(".bin", ".safetensors").replace("pytorch_model-", "model-")
            )
            dst_index["weight_map"][key] = dst_value
        json.dump(
            dst_index,
            open(os.path.join(torch_path, "model.safetensors.index.json"), "w"),
            indent=2,
        )


parser = argparse.ArgumentParser()
parser.add_argument("--torch_path", type=str, default=None)
parser.add_argument("--paddle_path", type=str, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    convert_from_torch_to_paddle(
        torch_path=args.torch_path,
        paddle_path=args.paddle_path,
    )
