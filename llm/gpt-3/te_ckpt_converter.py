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

import argparse
import json
import os
from collections import OrderedDict

import paddle

PREFIX = "gpt.decoder.layers."

pd_to_te_param_name_map = {
    "self_attn.qkv_proj.weight": "transformer.self_attention.layernorm_qkv.weight",  # need transpose
    "self_attn.qkv_proj.bias": "transformer.self_attention.layernorm_qkv.bias",
    "self_attn.out_proj.weight": "transformer.self_attention.proj.weight",  # need transpose
    "self_attn.out_proj.bias": "transformer.self_attention.proj.bias",
    "norm1.weight": "transformer.self_attention.layernorm_qkv.ln_weight",
    "norm1.bias": "transformer.self_attention.layernorm_qkv.ln_bias",
    "norm2.weight": "transformer.layernorm_mlp.ln_weight",
    "norm2.bias": "transformer.layernorm_mlp.ln_bias",
    "linear1.weight": "transformer.layernorm_mlp.fc1_weight",
    "linear1.bias": "transformer.layernorm_mlp.fc1_bias",
    "linear2.weight": "transformer.layernorm_mlp.fc2_weight",
    "linear2.bias": "transformer.layernorm_mlp.fc2_bias",
}

# reverse the map
te_to_pd_param_name_map = {v: k for k, v in pd_to_te_param_name_map.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_ckpt_path",
        type=str,
        default=None,
        help="The path of the input checkpoint to be converted.",
    )
    parser.add_argument(
        "--output_ckpt_path",
        type=str,
        default=None,
        help="The path of the output checkpoint to be saved.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pd2te",
        choices=["pd2te", "te2pd"],
        help="The mode of the conversion.",
    )

    return parser.parse_args()


def get_ckpt_filename(ckpt_path):
    # filename format: model_state.xxx.pdparams
    ckpt_files = os.listdir(ckpt_path)
    ckpt_filename_list = [file for file in ckpt_files if file.startswith("model_state") and file.endswith(".pdparams")]
    return ckpt_filename_list


def convert_sharded_ckpt_index(ckpt_path, output_path, mode):
    # has model_state.pdparams.index.json
    ckpt_files = os.listdir(ckpt_path)
    index_filename_list = [
        file for file in ckpt_files if file.endswith(".json") and file.startswith("model_state.pdparams.index")
    ]
    if len(index_filename_list) == 0:
        return

    for index_file in index_filename_list:
        index_file_full_path = os.path.join(ckpt_path, index_file)
        # read json file, replace the param name in the map, and save it to output_path
        with open(index_file_full_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        new_index = OrderedDict()
        new_index["weight_map"] = OrderedDict()
        for name, param in index.items():
            if name != "weight_map":
                new_index[name] = param
        for name, param in index["weight_map"].items():
            new_name = convert_param_name(mode, name)
            new_index["weight_map"][new_name] = param

        output_file_full_path = os.path.join(output_path, index_file)
        with open(output_file_full_path, "w", encoding="utf-8") as f:
            json.dump(new_index, f, indent=4)


def convert_param_name(mode, param_name):
    """
    Convert the paddle parameter name from PD to TE, or from TE to PD.
    """
    if not param_name.startswith(PREFIX):
        return param_name
    param_name = param_name[len(PREFIX) :]
    layer_id = None
    if param_name[0] >= "0" and param_name[0] <= "9":
        layer_id = int(param_name[0])
        if param_name[1] >= "0" and param_name[1] <= "9":
            layer_id = layer_id * 10 + int(param_name[1])
            param_name = param_name[3:]
        else:
            param_name = param_name[2:]
    else:
        param_name = param_name[1:]

    name_map = pd_to_te_param_name_map if mode == "pd2te" else te_to_pd_param_name_map
    new_param_name = PREFIX + str(layer_id) + "." + name_map[param_name]
    return new_param_name


def check_params(all_params_name, mode):
    """
    Check the parameters in the state dict.
    """
    te_name_exist = False
    pd_name_exist = False
    for name in all_params_name:
        if "transformer" in name:
            te_name_exist = True
        if "self_attn" in name:
            pd_name_exist = True

    if (te_name_exist and pd_name_exist) or (not te_name_exist and not pd_name_exist):
        raise ValueError("The input checkpoint is not a valid checkpoint.")
    if mode == "pd2te":
        assert pd_name_exist and not te_name_exist, "The input checkpoint is not a paddle checkpoint."
    if mode == "te2pd":
        assert te_name_exist and not pd_name_exist, "The input checkpoint is not a TE checkpoint."


def convert_ckpt(args):
    # check_args
    assert args.input_ckpt_path is not None, "Please specify the input checkpoint path."
    assert args.output_ckpt_path is not None, "Please specify the output checkpoint path."
    assert args.mode in ["pd2te", "te2pd"], "Unsupported conversion mode."

    # create dir if not exist
    if not os.path.exists(args.output_ckpt_path):
        os.makedirs(args.output_ckpt_path)

    # output dir must be empty
    assert (
        len(os.listdir(args.output_ckpt_path)) == 0
    ), "Output directory must be empty. Please remove all files in the output directory."

    # get all ckpt filenames
    ckpt_filename_list = get_ckpt_filename(args.input_ckpt_path)
    assert len(ckpt_filename_list) > 0, "No checkpoint file found in the input path."

    # check if params name match the mode
    all_params_name = []
    for ckpt_filename in ckpt_filename_list:
        ckpt_file_full_path = os.path.join(args.input_ckpt_path, ckpt_filename)
        state_dict = paddle.load(ckpt_file_full_path)
        for name, param in state_dict.items():
            all_params_name.append(name)
    check_params(all_params_name, args.mode)

    # convert ckpt
    for ckpt_filename in ckpt_filename_list:
        # convert pd model to TE model
        ckpt_file_full_path = os.path.join(args.input_ckpt_path, ckpt_filename)
        state_dict = paddle.load(ckpt_file_full_path)
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            # print(f"PD Layer: {name} | Size: {param.shape}")
            new_param_name = convert_param_name(args.mode, name)
            # if param is tensor and has shape [n, m], transpose it to [m, n]
            if isinstance(param, paddle.Tensor) and len(param.shape) == 2 and "decoder" in name:
                new_param = param.transpose([1, 0])
                new_state_dict[new_param_name] = new_param
            else:
                new_state_dict[new_param_name] = param
            # print(f"TE Layer: {te_param_name} | Size: {te_state_dict[te_param_name].shape}")
        output_ckpt_file_full_path = os.path.join(args.output_ckpt_path, ckpt_filename)
        paddle.save(new_state_dict, output_ckpt_file_full_path)

    # convert sharded ckpt index if exists
    convert_sharded_ckpt_index(args.input_ckpt_path, args.output_ckpt_path, args.mode)

    print(
        f"Convert checkpoint from {args.input_ckpt_path} to {args.output_ckpt_path} with mode {args.mode} successfully."
    )


if __name__ == "__main__":
    args = parse_args()
    convert_ckpt(args)
