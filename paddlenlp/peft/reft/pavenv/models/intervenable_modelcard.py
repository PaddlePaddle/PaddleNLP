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

# from .constants import *
from .llama.modelings_intervenable_llama import (
    llama_lm_type_to_dimension_mapping,
    llama_lm_type_to_module_mapping,
    llama_type_to_module_mapping,
)

#########################################################################
"""
Below are functions that you need to modify if you add
a new model arch type in this library.

We put them in front so it is easier to keep track of
things that need to be changed.
"""
import paddlenlp

global type_to_module_mapping
global type_to_dimension_mapping
global output_to_subcomponent_fn_mapping
global scatter_intervention_output_fn_mapping

type_to_module_mapping = {
    paddlenlp.transformers.llama.modeling.LlamaModel: llama_type_to_module_mapping,
    paddlenlp.transformers.llama.modeling.LlamaForCausalLM: llama_lm_type_to_module_mapping,
}

type_to_dimension_mapping = {
    paddlenlp.transformers.llama.modeling.LlamaForCausalLM: llama_lm_type_to_dimension_mapping,
}
