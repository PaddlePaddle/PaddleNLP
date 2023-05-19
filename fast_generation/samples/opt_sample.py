# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers import GPTTokenizer, OPTForCausalLM

model_name = "facebook/opt-350m"

tokenizer = GPTTokenizer.from_pretrained(model_name)
model = OPTForCausalLM.from_pretrained(model_name)
model.eval()

inputs = """a chat between a curious human and Statue of Liberty.
Human: What is your name?
Statue: I am statue of liberty.
Human: where do you live?
Statue: New york city.
Human: how long have you lived there?。"""

inputs_ids = tokenizer([inputs])["input_ids"]
inputs_ids = paddle.to_tensor(inputs_ids, dtype="int64")

outputs, _ = model.generate(
    input_ids=inputs_ids,
    max_length=20,
    decode_strategy="greedy_search",
    use_fast=True,
)

result = tokenizer.convert_ids_to_string(outputs[0].numpy().tolist())

print("Model input:", inputs)
print("Result:", result)
