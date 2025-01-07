#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

# import os
# os.environ['ENABLE_EXPERIMENTAL_FLAGS'] = '1'
# os.environ['VISUALIZATION_MODE'] = '0'
# os.environ['GRAPH_VISUALIZATION'] = '1'
# os.environ["HABANA_LOGS"] = "logs"
# os.environ["LOG_LEVEL_ALL"] = "0"
# os.environ['GLOG_v'] = '10'


paddle.set_device("intel_hpu")
paddle.set_default_dtype("bfloat16")

model = "meta-llama/Llama-2-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, dtype="bfloat16")

input_features = tokenizer("please introduce llm", return_tensors="pd")

with paddle.amp.auto_cast(dtype="bfloat16", custom_white_list={"elementwise_add", "rms_norm"}):
    outputs = model.generate(**input_features, max_length=128)

print(tokenizer.batch_decode(outputs[0]))
