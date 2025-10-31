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

import argparse
import os

import paddle

from paddlenlp.peft import DisLoRAConfig, DisLoRAModel
from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from paddlenlp.utils.env import CONFIG_NAME


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, help="The directory of pretrained model.")
    parser.add_argument("--dislora_path", default="", help="The directory of dislora parameters. Default to None")
    parser.add_argument(
        "--merge_dislora_model_path",
        default="",
        help="The directory of merged parameters. Default to None",
    )
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    parser.add_argument(
        "--low_gpu_mem", type=bool, default=True, help="Whether to use low gpu memory. Default to False"
    )
    return parser.parse_args()


def weight_process(name, dislora_config, state_dict):
    """
    Based on the DisLoRA algorithm for processing weight merging:
    The final weight = W_prin + W_res + W_TSD
    However, here we do not directly add the adapter to the base model; instead, we reconstruct the entire weight matrix.
    Args:
        name: Layer name (e.g. "model.layers.0.self_attn.q_proj")
        dislora_config: DisLoRA configuration
        state_dict: Model state dictionary
    # Define the weight_process function to handle the DisLoRA weight merging. The parameters include the layer name, DisLoRA configuration, and the model state dictionary.
    """

    weight_key = name + ".weight"

    if weight_key not in state_dict:
        print(f"Warning: {weight_key} not found in state_dict")
        return

    w_prin = state_dict[weight_key]
    print(f"Processing layer: {name}")
    print(f"  W_prin shape: {w_prin.shape}")

    scaling = dislora_config.dislora_alpha / dislora_config.r

    final_weight = w_prin.clone()

    ur_key = name + ".Direc_Ur.weight"
    sr_key = name + ".Direc_Sr"
    vhr_key = name + ".Direc_Vhr.weight"

    w_res_added = False

    if all(key in state_dict for key in [ur_key, sr_key, vhr_key]):

        direc_ur = state_dict[ur_key]  # [r, out_features]
        direc_sr = state_dict[sr_key]  # [r]
        direc_vhr = state_dict[vhr_key]  # [in_features, r]

        s_diag = paddle.diag(direc_sr)  # [r, r]

        w_res = direc_vhr @ s_diag @ direc_ur * scaling  # [in_features, out_features]

        if w_res.shape != w_prin.shape:
            print(f"  Error: W_res shape {w_res.shape} doesn't match W_prin shape {w_prin.shape}")
            return

        final_weight += w_res
        w_res_added = True
        print(f"  ✓ Added W_res with scaling factor: {scaling}")
    else:
        print(f"  ⚠ W_res components not found for {name}")

    utsd_key = name + ".Direc_Utsd.weight"
    stsd_key = name + ".Direc_Stsd"
    vhtsd_key = name + ".Direc_Vhtsd.weight"

    w_tsd_added = False
    if all(key in state_dict for key in [utsd_key, stsd_key, vhtsd_key]):

        direc_utsd = state_dict[utsd_key]  # [s_tsd, out_features]
        direc_stsd = state_dict[stsd_key]  # [s_tsd]
        direc_vhtsd = state_dict[vhtsd_key]  # [in_features, s_tsd]

        if not paddle.all(direc_stsd == 0.0):

            s_diag_tsd = paddle.diag(direc_stsd)  # [s_tsd, s_tsd]

            w_tsd = direc_vhtsd @ s_diag_tsd @ direc_utsd * scaling  # [in_features, out_features]

            if w_tsd.shape != w_prin.shape:
                print(f"  Error: W_TSD shape {w_tsd.shape} doesn't match W_prin shape {w_prin.shape}")
                return

            final_weight += w_tsd
            w_tsd_added = True
            print(f"  ✓ Added W_TSD with scaling factor: {scaling}")
        else:
            print(f"  ⚠ W_TSD parameters are uninitialized (all zeros) for {name}")
    else:
        print(f"  ⚠ W_TSD components not found for {name}")

    state_dict[weight_key] = final_weight

    keys_to_remove = []
    for key in state_dict.keys():
        if key.startswith(name + ".Direc_") or key == name + ".step":
            keys_to_remove.append(key)

    for key in keys_to_remove:
        removed_param = state_dict.pop(key)
        print(f"  ✓ Removed DisLoRA parameter: {key} (shape: {removed_param.shape})")

    components = []
    if w_res_added:
        components.append("W_res")
    if w_tsd_added:
        components.append("W_TSD")

    if components:
        print(f"  ✓ Successfully merged: W_prin + {' + '.join(components)}")
    else:
        print("  ✓ Kept original W_prin (no adaptations found)")
    print()


def merge():

    args = parse_arguments()
    paddle.set_device(args.device)

    print("Loading DisLoRA configuration...")
    dislora_config = DisLoRAConfig.from_pretrained(args.dislora_path)
    if dislora_config.base_model_name_or_path is None:
        if args.model_name_or_path is None:
            raise ValueError("We can not find a valid model_name_or_path.")
        else:
            dislora_config.base_model_name_or_path = args.model_name_or_path

    print("Loading model configuration...")
    if os.path.isfile(os.path.join(args.dislora_path, CONFIG_NAME)):
        config = AutoConfig.from_pretrained(args.dislora_path)
    elif args.model_name_or_path is not None:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            f"We can not find config.json in dislora_path: {args.dislora_path} or find a valid model_name_or_path."
        )

    config.dtype = dislora_config.dtype

    if (
        dislora_config.dtype == "bfloat16"
        or (
            hasattr(config, "quantization_config")
            and hasattr(config.quantization_config, "weight_quantize_algo")
            and config.quantization_config.weight_quantize_algo in ["nf4", "fp4"]
        )
    ) and args.device == "cpu":
        raise ValueError("We can not apply bfloat16 or nf4/fp4 dislora merge on cpu.")

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        dislora_config.base_model_name_or_path,
        config=config,
        low_cpu_mem_usage=args.low_gpu_mem,
    )

    print("Loading DisLoRA model...")
    model = DisLoRAModel.from_pretrained(model=model, dislora_path=args.dislora_path, dislora_config=dislora_config)

    model.eval()
    model_state_dict = model.model.state_dict()

    print(f"Total parameters in state_dict: {len(model_state_dict)}")

    step_keys = [key for key in model_state_dict.keys() if key.endswith(".step")]
    if step_keys:
        print(f"Found {len(step_keys)} step parameters in loaded model:")
        for key in step_keys[:5]:
            print(f"  {key}")
        if len(step_keys) > 5:
            print(f"  ... and {len(step_keys) - 5} more")
    else:
        print("No step parameters found in loaded model")
    print()

    print("Identifying DisLoRA layers...")
    dislora_name_set = set()
    for key in model_state_dict.keys():
        if any(
            dislora_param in key
            for dislora_param in ["Direc_Ur", "Direc_Sr", "Direc_Vhr", "Direc_Utsd", "Direc_Stsd", "Direc_Vhtsd"]
        ):

            for param_type in ["Direc_Ur", "Direc_Sr", "Direc_Vhr", "Direc_Utsd", "Direc_Stsd", "Direc_Vhtsd"]:
                if f".{param_type}" in key:
                    layer_name = key.split(f".{param_type}")[0]
                    dislora_name_set.add(layer_name)
                    break

    dislora_name_list = sorted(list(dislora_name_set))

    print(f"Found {len(dislora_name_list)} DisLoRA layers:")
    for i, name in enumerate(dislora_name_list, 1):
        print(f"  {i:2d}. {name}")
    print()

    print("Merging DisLoRA parameters...")

    for i, name in enumerate(dislora_name_list, 1):
        print(f"[{i}/{len(dislora_name_list)}] Processing: {name}")
        weight_process(name, dislora_config, model_state_dict)

    print("Cleaning up remaining step parameters...")
    step_keys_to_remove = [key for key in model_state_dict.keys() if key.endswith(".step")]
    for key in step_keys_to_remove:
        removed_param = model_state_dict.pop(key)
        print(f"  ✓ Removed step parameter: {key} (shape: {removed_param.shape})")

    if step_keys_to_remove:
        print(f"✓ Removed {len(step_keys_to_remove)} step parameters")
    else:
        print("✓ No step parameters found")
    print()

    print("Verifying parameter cleanup...")
    remaining_dislora_params = []
    remaining_step_params = []
    for key in model_state_dict.keys():
        if any(
            dislora_param in key
            for dislora_param in ["Direc_Ur", "Direc_Sr", "Direc_Vhr", "Direc_Utsd", "Direc_Stsd", "Direc_Vhtsd"]
        ):
            remaining_dislora_params.append(key)
        if key.endswith(".step"):
            remaining_step_params.append(key)

    if remaining_dislora_params:
        print(f"Warning: {len(remaining_dislora_params)} DisLoRA parameters still remain:")
        for param in remaining_dislora_params:
            print(f"  - {param}")
    else:
        print("✓ All DisLoRA parameters successfully removed")

    if remaining_step_params:
        print(f"Warning: {len(remaining_step_params)} step parameters still remain:")
        for param in remaining_step_params:
            print(f"  - {param}")
    else:
        print("✓ All step parameters successfully removed")
    print()

    print("Saving merged model...")
    os.makedirs(args.merge_dislora_model_path, exist_ok=True)
    model.model.save_pretrained(args.merge_dislora_model_path, state_dict=model_state_dict)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(dislora_config.base_model_name_or_path)
    tokenizer.save_pretrained(args.merge_dislora_model_path)

    print("=" * 80)
    print("✓ DisLoRA merge completed successfully!")
    print(f"✓ Merged model saved to: {args.merge_dislora_model_path}")
    print(f"✓ Processed {len(dislora_name_list)} DisLoRA layers")
    print("=" * 80)


if __name__ == "__main__":
    merge()
