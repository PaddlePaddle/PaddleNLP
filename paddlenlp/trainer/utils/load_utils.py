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

import json
import re
from collections import defaultdict
from typing import List, Optional

import paddle
from paddle.distributed import fleet
from safetensors import safe_open

# develop: "_layers.<idx>.<rest>"
_LAYER_RE = re.compile(r"^_layers\.(\d+)(?:\.(.*))?$")
_EXPERT_W1_RE = re.compile(r"^mlp\.experts\.(\d+)\.w1(?:\.weight)?$")
_EXPERT_W2_RE = re.compile(r"^mlp\.experts\.(\d+)\.w2(?:\.weight)?$")

custom_name_map = {
    "mlp.router.weight": "mlp.gate.weight",
    "mlp.router.e_score_correction_bias": "mlp.gate.e_score_correction_bias",
}


def _layers_match(name: str):
    return _LAYER_RE.match(name)


def simple_safe_call(model, method_name, *args, **kwargs):
    if hasattr(model, method_name):
        return getattr(model, method_name)(*args, **kwargs)
    if hasattr(model, "_layers") and hasattr(model._layers, method_name):
        return getattr(model._layers, method_name)(*args, **kwargs)
    raise AttributeError(f"{type(model).__name__} (or its wrapper) has no method {method_name}")


def add_prefix_to_keys(d, prefix):
    print("Input dict:", d)

    mappings = {}
    for key, value in d.items():
        if key == "embed_tokens.weight":
            new_key = "_layers.0.embed_tokens.weight"
        elif key == "lm_head.weight":
            new_key = "_layers.64.weight"
        else:
            new_key = f"{prefix}{key}"
        mappings[new_key] = value
    return mappings


def _get_hf_prefix_develop(idx: int) -> str:
    if idx == 0:
        return "model"  # embedding
    if idx == 63:
        return "model"  # final norm
    if idx == 64:
        return "lm_head"  # lm_head
    return f"model.layers.{idx - 1}"  # decoder layer


def _handle_expert_weights(hf_prefix: str, rest: str) -> Optional[List[str]]:
    if m := _EXPERT_W1_RE.match(rest):
        expert_id = int(m.group(1))
        return [
            f"{hf_prefix}.mlp.experts.{expert_id}.gate_proj.weight",
            f"{hf_prefix}.mlp.experts.{expert_id}.up_proj.weight",
        ]
    if m := _EXPERT_W2_RE.match(rest):
        expert_id = int(m.group(1))
        return [
            f"{hf_prefix}.mlp.experts.{expert_id}.down_proj.weight",
        ]
    return None


def _handle_mlp_weights(hf_prefix: str, rest: str) -> Optional[List[str]]:
    if rest == "mlp.w1":
        return [
            f"{hf_prefix}.mlp.gate_proj.weight",
            f"{hf_prefix}.mlp.up_proj.weight",
        ]
    if rest == "mlp.w2":
        return [
            f"{hf_prefix}.mlp.down_proj.weight",
        ]
    return None


def paddle_name_to_hf_names(paddle_name: str) -> List[str]:
    """
    Mapping Function for Paddle Parameter Names to Hugging Face Names
    """
    m = _layers_match(paddle_name)
    if not m:
        return []
    idx = int(m.group(1))
    rest = m.group(2) or ""

    hf_prefix = _get_hf_prefix_develop(idx)

    # 专项重命名
    if rest in custom_name_map:
        return [f"{hf_prefix}.{custom_name_map[rest]}"]

    # 历史专家
    if expert_names := _handle_expert_weights(hf_prefix, rest):
        return expert_names

    # 历史mlp
    if mlp_names := _handle_mlp_weights(hf_prefix, rest):
        return mlp_names

    return [f"{hf_prefix}.{rest}"] if rest else [hf_prefix]


def prepare_tensor(tensor, pd_param, tensor_parallel_mappings, mp_degree, dst_shape):
    """
    Converting weight tensors to match the target model’s shape involves
    automatically adjusting for transposing, concatenating, and slicing by columns or lengths.
    """

    if isinstance(tensor, list):
        tensor = paddle.concat(
            [
                paddle.transpose(tensor[0], perm=[1, 0]).contiguous(),
                paddle.transpose(tensor[1], perm=[1, 0]).contiguous(),
            ],
            axis=-1,
        )
    # match for transpose
    if len(tensor.shape) == 2:
        if (tensor.shape[0] == dst_shape[1] or tensor.shape[1] == dst_shape[0]) and tensor.shape != dst_shape:
            tensor = paddle.transpose(tensor, perm=[1, 0]).contiguous()
        print(f"after transpose get hf tensor shape {tensor.shape}, paddle shape {dst_shape}")

    if mp_degree > 1 and pd_param in tensor_parallel_mappings:
        tensor = tensor_parallel_mappings[pd_param](tensor)
    if tensor.shape == dst_shape:
        return tensor
    raise ValueError(f"Unexpected tensor shape: got {tensor.shape}, want {dst_shape}")


def load_paddle_model_from_safetensors(
    model,
    weight_map_path: str,
    ckpt_pre: str,
    verbose: bool = True,
):
    """
    Load safetensors into a Paddle  model using the weight mappings outlined in index.json.
    """

    tensor_parallel_mappings = {}
    mp_degree = fleet.get_hybrid_communicate_group().get_model_parallel_world_size()
    print("fuck mp degree!!!!!!!!!", mp_degree)

    if mp_degree > 1:
        print("load with mp_degree:", mp_degree)
        tensor_parallel_mappings = simple_safe_call(model, "get_tensor_parallel_mappings", is_split=True)
        tensor_parallel_mappings = add_prefix_to_keys(tensor_parallel_mappings, "_")

    for k, v in tensor_parallel_mappings.items():
        print("tensor_parallel_mappings:", k, v)

    with open(weight_map_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    required_files = set()
    file_to_pd_param_name = defaultdict(list)
    pd_param_name_to_file = defaultdict(list)

    for pd_name, _ in model.named_parameters():
        hf_names = paddle_name_to_hf_names(pd_name)
        if verbose:
            print(f"paddle_name_to_hf_names: {pd_name} -> {hf_names}")
        if not hf_names:
            if verbose:
                print(f"Warning: {pd_name} can not be mapped")
            continue
        for i, hf_name in enumerate(hf_names):
            if hf_name in weight_map:
                filename = weight_map[hf_name]
                required_files.add(filename)
                file_to_pd_param_name[filename].append(pd_name)
                if filename not in pd_param_name_to_file[pd_name]:
                    pd_param_name_to_file[pd_name].append(filename)
            else:
                if verbose:
                    print(f"Warning: {pd_name} -> {hf_name} not found in weight map")

    check_list = []
    if verbose:
        print("---- start load param ----")
        for key, value in tensor_parallel_mappings.items():
            print(key, value)
    for filename in required_files:
        try:
            with safe_open(ckpt_pre + filename, framework="paddle", device="cpu") as f:
                pd_params = file_to_pd_param_name[filename]
                for pd_param in pd_params:
                    if pd_param in check_list:
                        continue
                    if verbose:
                        print("load for pd_param:", pd_param)
                    hf_names = paddle_name_to_hf_names(pd_param)
                    if not hf_names:
                        continue
                    if len(hf_names) == 1:
                        tensor = f.get_tensor(hf_names[0])
                        value = prepare_tensor(
                            tensor, pd_param, tensor_parallel_mappings, mp_degree, model.state_dict()[pd_param].shape
                        )

                        model.state_dict()[pd_param].set_value(paddle.cast(value, model.state_dict()[pd_param].dtype))
                    else:
                        files = pd_param_name_to_file[pd_param]
                        if len(files) == 1:
                            tensor0 = f.get_tensor(hf_names[0])
                            tensor1 = f.get_tensor(hf_names[1])
                        else:
                            if weight_map[hf_names[0]] == filename:
                                tensor0 = f.get_tensor(hf_names[0])
                                with safe_open(
                                    ckpt_pre + weight_map[hf_names[1]], framework="paddle", device="cpu"
                                ) as f2:
                                    tensor1 = f2.get_tensor(hf_names[1])
                            else:
                                with safe_open(
                                    ckpt_pre + weight_map[hf_names[0]], framework="paddle", device="cpu"
                                ) as f2:
                                    tensor0 = f2.get_tensor(hf_names[0])
                                tensor1 = f.get_tensor(hf_names[1])
                        value = prepare_tensor(
                            [tensor0, tensor1],
                            pd_param,
                            tensor_parallel_mappings,
                            mp_degree,
                            model.state_dict()[pd_param].shape,
                        )
                        model.state_dict()[pd_param].set_value(value)
                    check_list.append(pd_param)
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            raise

    if verbose:
        print("All parameters loaded.")
