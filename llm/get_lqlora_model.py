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

import paddle
from utils import get_lora_target_modules

from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.peft.lora.lqlora_utils import transform_lora_layers
from paddlenlp.transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/llama-7b")
target_modules = get_lora_target_modules(model)
lora_config = LoRAConfig(
    target_modules=target_modules,
    r=8,
    lora_alpha=16,
    merge_weights=False,
    tensor_parallel_degree=1,
    dtype=paddle.float16,
    base_model_name_or_path="facebook/llama-7b",
)
model = LoRAModel(model, lora_config)
quantization_config_dict = paddle.load("./ilp_data/merge/qconfig_dict")
transform_lora_layers(model, quantization_config_dict)

state_dict = model.state_dict()
paddle.save(state_dict, "./ilp_data/merge/lqlora_state_dict.pth")
