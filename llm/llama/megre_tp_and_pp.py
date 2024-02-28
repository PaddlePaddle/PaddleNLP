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

import os

import paddle

from paddlenlp.transformers import LlamaConfig, LlamaForCausalLM
from paddlenlp.utils.log import logger


def merge_pipeline_parallel(tp_degree, pp_degree, path):
    tp_state_dict_list = []
    for tp in range(tp_degree):
        tp_state_dict = {}
        for pp in range(pp_degree):
            tmp = paddle.load(os.path.join(path, f"model_state.tp{tp:0>2d}_pp{pp:0>2d}.pdparams"), return_numpy=True)
            for k, v in tmp.items():
                tp_state_dict[k] = v

        tp_state_dict_list.append(tp_state_dict)

    return tp_state_dict_list


def merge_tensor_parallel(cls, state_dict_list, config) -> None:
    """the entry of converting config and converting model file

    Args:
        input_dir (str | None): the input dir which contains `pytorch_model.bin` and `config.json` file
        config (PretrainedConfig): the PretrainedConfig instance of model
    """
    name_action_mappings = cls._get_tensor_parallel_mappings(config, is_split=False)
    state_keys_map = cls._resolve_prefix_keys(name_action_mappings.keys(), state_dict_list[0].keys())

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

    print("Finally, we merging state dict to fellowing tensors.")
    for k, v in state_dict_to_save.items():
        print(k, v.shape, v.dtype)

    return state_dict_to_save


def main():
    tp_degree = 2
    pp_degree = 2
    model_name_or_path = "temp_dir_to_your_ckpt"

    assert tp_degree > 1
    assert pp_degree > 1
    config = LlamaConfig.from_pretrained(model_name_or_path)
    cls = LlamaForCausalLM

    tp_state_dict_list = merge_pipeline_parallel(tp_degree, pp_degree, model_name_or_path)
    state_dict_to_save = merge_tensor_parallel(cls=cls, state_dict_list=tp_state_dict_list, config=config)
    print("saving")
    paddle.save(state_dict_to_save, os.path.join(model_name_or_path, "model_state.pdparams"))


if __name__ == "__main__":
    main()
