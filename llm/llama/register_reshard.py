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
import re


# register pp_reshard information to aid pp reshard
def register_pp_reshard_information(num_hidden_layers):

    from paddlenlp.trainer.utils.reshard.pp_reshard import (
        register_index_layer_func,
        register_layername_prefix,
        regitser_extract_layer_name_func,
    )

    # register layer names
    register_layername_prefix("column_sequence_parallel_linear")
    register_layername_prefix("row_sequence_parallel_linear")
    register_layername_prefix("linear")
    register_layername_prefix("embedding")
    register_layername_prefix("create_parameter")
    register_layername_prefix("llama_lm_head")

    # register func to extract layer from stuctural param name
    # register func to extract layer index  from stuctural param name

    def extract_layer_name(param_name):
        patterns = [r"^llama\.embed_tokens", "^llama\.norm", r"^lm_head", r"^llama\.layers((\.\d+))"]
        # match 1
        for p in patterns:
            match = re.search(p, param_name)
            if match:
                return match.group()

    def index_layer(layer_name):
        if layer_name == "llama.embed_tokens":
            return 0
        elif layer_name == "llama.norm":
            return num_hidden_layers + 1
        elif layer_name == "lm_head":
            return num_hidden_layers + 2
        else:
            pattern = r"llama\.layers((\.(\d+)))"
            match = re.search(pattern, layer_name)
            assert match
            index = int(match.group(3)) + 1
            assert index <= num_hidden_layers, f"{index} {num_hidden_layers}"
            return index

    regitser_extract_layer_name_func(extract_layer_name)
    register_index_layer_func(index_layer)
