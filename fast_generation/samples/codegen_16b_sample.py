# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers import CodeGenForCausalLM, CodeGenTokenizer

# Can be load on A100-40G
paddle.set_default_dtype("float16")
model_name = "Salesforce/codegen-16B-mono"

tokenizer = CodeGenTokenizer.from_pretrained(model_name)
model = CodeGenForCausalLM.from_pretrained(model_name, load_state_as_np=True)
model.eval()

inputs = "def hello"
input_ids = tokenizer([inputs], return_tensors="pd")["input_ids"]

# Enable FastGeneration
outputs, _ = model.generate(
    input_ids=input_ids, max_length=128, decode_strategy="greedy_search", use_fp16_decoding=True, use_fast=True
)

result = tokenizer.decode(outputs[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])

print("Model input:", inputs)
print("Result:", result)
