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
import importlib
import os
import re

import paddle

from paddlenlp.transformers import AutoConfig
from paddlenlp.transformers.auto.modeling import MAPPING_NAMES
from paddlenlp.utils.log import logger


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, required=True, help="The directory of model.")
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    parser.add_argument("--pipeline_parallel_degree", "--pp", type=int, required=True, help="pp degree")
    parser.add_argument("--tensor_parallel_degree", "--tp", type=int, required=True, help="tp degree")
    return parser.parse_args()


def validate_model_file(path: str, tp_degree: int, pp_degree: int) -> None:
    files = os.listdir(path)
    pattern = r"model_state\.tp0\d*_pp0\d*\.pdparams|model_state\.tp0\d*\.pdparams|model_state\.pp0\d*\.pdparams"
    if pp_degree == 0:
        target_files = [f"model_state.tp{tp:0>2d}.pdparams" for tp in range(tp_degree)]
    elif tp_degree == 0:
        target_files = [f"model_state.pp{pp:0>2d}.pdparams" for pp in range(pp_degree)]
    else:
        target_files = [
            f"model_state.tp{tp:0>2d}_pp{pp:0>2d}.pdparams" for tp in range(tp_degree) for pp in range(pp_degree)
        ]

    exist_required_files = []
    for file in files:
        if re.match(pattern, file):
            exist_required_files.append(file)

    missing_files = set(target_files) - set(exist_required_files)
    if len(missing_files) > 0:
        raise FileNotFoundError(f"Please check your pp/tp degree, missing files {list(missing_files)}")


def load_tp_params(tp_degree, path):
    tp_state_dict_list = []
    for tp in range(tp_degree):
        tp_state_dict = {}
        tmp = paddle.load(os.path.join(path, f"model_state.tp{tp:0>2d}.pdparams"), return_numpy=True)
        for k, v in tmp.items():
            tp_state_dict[k] = v
        tp_state_dict_list.append(tp_state_dict)

    return tp_state_dict_list


def load_tp_and_pp_params(tp_degree, pp_degree, path):
    tp_state_dict_list = []
    for tp in range(tp_degree):
        tp_state_dict = {}
        for pp in range(pp_degree):
            tmp = paddle.load(os.path.join(path, f"model_state.tp{tp:0>2d}_pp{pp:0>2d}.pdparams"), return_numpy=True)
            for k, v in tmp.items():
                tp_state_dict[k] = v
        tp_state_dict_list.append(tp_state_dict)
    return tp_state_dict_list


def load_pp_params(pp_degree, path):
    pp_state_dict = {}
    for pp in range(pp_degree):
        tmp = paddle.load(os.path.join(path, f"model_state.pp{pp:0>2d}.pdparams"), return_numpy=True)
        for k, v in tmp.items():
            pp_state_dict[k] = v
    return pp_state_dict


def merge_tensor_parallel(model_class, state_dict_list, config) -> None:
    """the entry of converting config and converting model file

    Args:
        input_dir (str | None): the input dir which contains `pytorch_model.bin` and `config.json` file
        config (PretrainedConfig): the PretrainedConfig instance of model
    """
    name_action_mappings = model_class._get_tensor_parallel_mappings(config, is_split=False)
    state_keys_map = model_class._resolve_prefix_keys(name_action_mappings.keys(), state_dict_list[0].keys())

    for k, v in state_keys_map.items():
        name_action_mappings[v] = name_action_mappings.pop(k)

    state_dict_to_save = {}
    for key in state_dict_list[0].keys():
        tensor = state_dict_list[0][key]
        if key in name_action_mappings:
            ret = [x[key] for x in state_dict_list]
            action = name_action_mappings.pop(key)
            tensor = action(ret)

        state_dict_to_save[key] = tensor

    if len(name_action_mappings) > 0:
        for x in name_action_mappings.keys():
            logger.warning(f"key <{x}> need to merge tensor parallel but we can't find in model state.")

    logger.info("Finally, we merging state dict to fellowing tensors.")
    for k, v in state_dict_to_save.items():
        logger.info(f"{k}, {v.shape}, {v.dtype}")

    return state_dict_to_save


def main():
    args = parse_arguments()
    paddle.set_device(args.device)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    init_class = config["architectures"][0]
    if args.pipeline_parallel_degree > 1:
        # using pp
        import_class = importlib.import_module(f"paddlenlp.transformers.{MAPPING_NAMES[init_class[:-15]]}.modeling_pp")
    else:
        # tp only
        import_class = importlib.import_module(f"paddlenlp.transformers.{MAPPING_NAMES[init_class[:-11]]}.modeling")
    model_class = getattr(import_class, init_class)

    validate_model_file(args.model_name_or_path, args.tensor_parallel_degree, args.pipeline_parallel_degree)

    if args.tensor_parallel_degree > 1:
        if args.pipeline_parallel_degree > 1:
            tp_state_dict_list = load_tp_and_pp_params(
                args.tensor_parallel_degree, args.pipeline_parallel_degree, args.model_name_or_path
            )
        else:
            tp_state_dict_list = load_tp_params(args.tensor_parallel_degree, args.model_name_or_path)
        state_dict_to_save = merge_tensor_parallel(
            model_class=model_class, state_dict_list=tp_state_dict_list, config=config
        )
        logger.info("Saving")
        paddle.save(state_dict_to_save, os.path.join(args.model_name_or_path, "model_state.pdparams"))
    elif args.pipeline_parallel_degree > 1:
        state_dict_to_save = load_pp_params(args.pipeline_parallel_degree, args.model_name_or_path)
        logger.info("Saving")
        paddle.save(state_dict_to_save, os.path.join(args.model_name_or_path, "model_state.pdparams"))
    else:
        logger.info("No need to merge since config.tensor_parallel_degree <= 1.")


if __name__ == "__main__":
    main()
