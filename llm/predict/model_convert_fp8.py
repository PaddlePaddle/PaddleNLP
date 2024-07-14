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
"""
PTQ_SCALES_MAP, FP8_PTQ_SCALES_MAP,GLOBAL dict for load scales and weight from slim ckpt
"""
# cipher_token=WjI1fQOvhN  # do not edit this line
PTQ_SCALES_MAP = {
    "act_scale": {
        "qkv_in_scale": "llama.layers.#.self_attn.q_proj.activation_quanter",
        "out_linear_in_scale": "llama.layers.#.self_attn.o_proj.activation_quanter",
        "ffn1_in_scale": "llama.layers.#.mlp.gate_proj.activation_quanter",
        "ffn2_in_scale": "llama.layers.#.mlp.down_proj.activation_quanter",
    },
    "weight_scale": {
        "q_weight_scale": "llama.layers.#.self_attn.q_proj.weight_quanter",
        "k_weight_scale": "llama.layers.#.self_attn.k_proj.weight_quanter",
        "v_weight_scale": "llama.layers.#.self_attn.v_proj.weight_quanter",
        "out_linear_weight_scale": "llama.layers.#.self_attn.o_proj.weight_quanter",
        "ffn1_1_weight_scale": "llama.layers.#.mlp.gate_proj.weight_quanter",
        "ffn1_2_weight_scale": "llama.layers.#.mlp.up_proj.weight_quanter",
        "ffn2_weight_scale": "llama.layers.#.mlp.down_proj.weight_quanter",
    },
    "cachekv_scale": {
        "cache_k_scale": "llama.layers.#.self_attn.cachek_matmul.activation_quanter",
        "cache_v_scale": "llama.layers.#.self_attn.cachev_matmul.activation_quanter",
    },
}

FP8_PTQ_SCALES_MAP = {
    "act_scale": {
        "qkv_in_scale": "llama.layers.#.self_attn.q_proj.activation_quanter",
        "out_linear_in_scale": "llama.layers.#.self_attn.o_proj.activation_quanter",
        "ffn1_in_scale": "llama.layers.#.mlp.gate_proj.activation_quanter",
        "ffn2_in_scale": "llama.layers.#.mlp.down_proj.activation_quanter",
    },
    "weight_scale": {
        "qkv_weight_scale": "llama.layers.#.self_attn.qkv_proj.weight_quanter",
        "out_linear_weight_scale": "llama.layers.#.self_attn.o_proj.weight_quanter",
        "ffn1_0_weight_scale": "llama.layers.#.mlp.gate_proj.weight_quanter",
        "ffn1_1_weight_scale": "llama.layers.#.mlp.up_proj.weight_quanter",
        "ffn2_weight_scale": "llama.layers.#.mlp.down_proj.weight_quanter",
    },
    "cachekv_scale": {
        "cache_k_scale": "llama.layers.#.self_attn.cachek_matmul.activation_quanter",
        "cache_v_scale": "llama.layers.#.self_attn.cachev_matmul.activation_quanter",
    },
}


import argparse
import json
import os

import numpy as np


def setup_args():
    """Setup export arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="parameters file.")
    args = parser.parse_args()
    return args


class LoadActScaleJson:
    """
    Load Act Scale from json file
    """

    def __init__(
        self,
        scale_json_file_path="act_scales.json",
        key_map_dict=None,
        num_of_layers=None,
    ):
        """
            Args:
            scale_json_file_path (str, optional): JSON file path containing scales. Defaults to "act_scales.json".
            key_map_dict (dict, optional): Dictionary mapping scale types to templates. Defaults to None.
            num_of_layers (int, optional): Number of layers. Defaults to None.

        Raises:
            FileNotFoundError: If the specified JSON file does not exist.
        """
        with open(scale_json_file_path) as json_file:
            self.scale_dict = json.load(json_file)
        self.key_map = key_map_dict
        self.scale = {}
        for scale_type, key_template in self.key_map.items():
            self.scale[scale_type] = np.full([num_of_layers], fill_value=-1.0)
            for i in range(num_of_layers):
                if key_template.replace("#", str(i)) in self.scale_dict.keys():
                    self.scale[scale_type][i] = 1 / self.scale_dict[key_template.replace("#", str(i))]


if __name__ == "__main__":
    args = setup_args()
    path_name = args.model_name_or_path

    config_path = os.path.join(path_name, "config.json")
    with open(config_path, "r") as model_config_file:
        model_config = json.load(model_config_file)
        nums_layers = model_config["num_hidden_layers"]

    # linear_weights_name = '.self_attn.o_proj.weight'
    # linear_bias_name = '.self_attn.o_proj.bias'

    # ffn1_0_weights_name = '.mlp.gate_proj.weight'
    # ffn1_1_weights_name = '.mlp.up_proj.weight'

    # ffn2_weights_name = '.mlp.down_proj.weight'
    # ffn2_bias_name = '.mlp.down_proj.bias'

    # # qkv_weights_name = '.self_attn.qkv_proj.weight'
    # q_weights_name = '.self_attn.q_proj.weight'
    # k_weights_name = '.self_attn.k_proj.weight'
    # v_weights_name = '.self_attn.v_proj.weight'

    # params_states = paddle.load(os.path.join(path_name, "model_state.pdparams"))
    # new_path = os.path.join(path_name, "model_state.pdparams")

    # scale_map_dict = FP8_PTQ_SCALES_MAP
    # act_scale_map_dict = scale_map_dict["act_scale"]

    # act_scale_json_path = os.path.join(path_name+"/act_scales.json")

    # act_scales = LoadActScaleJson(
    #     act_scale_json_path, act_scale_map_dict, num_of_layers=nums_layers
    # )

    # new_weight_scale ={}

    # for i in range(0, nums_layers):
    #     linear_weights = params_states['llama.layers.'+str(i)+linear_weights_name]
    #     # linear_bias = params_states['llama.layers.'+str(i)+linear_bias_name]
    #     ffn1_weights_0 = params_states['llama.layers.'+str(i)+ffn1_0_weights_name]
    #     ffn1_weights_1 = params_states['llama.layers.'+str(i)+ffn1_1_weights_name]
    #     ffn2_weights = params_states['llama.layers.'+str(i)+ffn2_weights_name]
    #     # ffn2_bias = params_states['llama.layers.'+str(i)+ffn2_bias_name]

    #     q_weights = params_states['llama.layers.'+str(i)+q_weights_name]
    #     k_weights = params_states['llama.layers.'+str(i)+k_weights_name]
    #     v_weights = params_states['llama.layers.'+str(i)+v_weights_name]

    #     new_weight_scale["llama.layers."+str(i)+".self_attn.o_proj.weight_quanter"] = paddle.cast(linear_weights,'float').numpy().max()
    #     new_linear_weights = paddle.cast(linear_weights *448/linear_weights.max(), 'float8_e4m3fn')
    #     new_weight_scale["llama.layers."+str(i)+".mlp.gate_proj.weight_quanter"] = paddle.cast(ffn1_weights_0,'float').numpy().max()
    #     new_weight_scale["llama.layers."+str(i)+".mlp.up_proj.weight_quanter"] = paddle.cast(ffn1_weights_1,'float').numpy().max()

    #     params_states['llama.layers.'+str(i)+ffn1_0_weights_name] = paddle.cast(ffn1_weights_0*448/ ffn1_weights_0.max(), 'float8_e4m3fn')
    #     params_states['llama.layers.'+str(i)+ffn1_1_weights_name] = paddle.cast(ffn1_weights_1*448/ ffn1_weights_1.max(), 'float8_e4m3fn')

    #     new_weight_scale["llama.layers."+str(i)+".mlp.down_proj.weight_quanter"] = paddle.cast(ffn2_weights,'float').numpy().max()
    #     new_ffn2_weights = paddle.cast(ffn2_weights*448/ ffn2_weights.max(), 'float8_e4m3fn')

    #     qkv_weight_scale = max(paddle.cast(q_weights, 'float').numpy().max(),
    #                            paddle.cast(k_weights, 'float').numpy().max(),
    #                            paddle.cast(v_weights, 'float').numpy().max())

    #     new_weight_scale["llama.layers."+str(i)+".self_attn.qkv_proj.weight_quanter"] = qkv_weight_scale

    #     qkv_weights_max = max(q_weights.max(), k_weights.max(), v_weights.max())

    #     new_q_weights = paddle.cast(q_weights*448/ qkv_weights_max, 'float8_e4m3fn')
    #     new_k_weights = paddle.cast(k_weights*448/ qkv_weights_max, 'float8_e4m3fn')
    #     new_v_weights = paddle.cast(v_weights*448/ qkv_weights_max, 'float8_e4m3fn')

    #     params_states['llama.layers.'+str(i)+linear_weights_name] = new_linear_weights
    #     params_states['llama.layers.'+str(i)+ffn2_weights_name] = new_ffn2_weights
    #     params_states['llama.layers.'+str(i)+q_weights_name] = new_q_weights
    #     params_states['llama.layers.'+str(i)+k_weights_name] = new_k_weights
    #     params_states['llama.layers.'+str(i)+v_weights_name] = new_v_weights
    #     # params_states['llama.layers.'+str(i)+linear_bias_name] = linear_bias
    #     # params_states['llama.layers.'+str(i)+ffn2_bias_name] = ffn2_bias

    # with open(path_name+'/weight_scales.json', 'w') as weight_scales_file:
    #     json.dump(new_weight_scale, weight_scales_file)

    with open(config_path, "w") as model_config_file:
        model_config["quantization_config"] = {"quant_type": "a8w8_fp8"}
        json.dump(model_config, model_config_file)

    # paddle.save(params_states,new_path)
