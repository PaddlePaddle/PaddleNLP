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

from paddlenlp.transformers import GPTJForCausalLM, GPTJTokenizer

paddle.set_default_dtype("float16")
model_name = "EleutherAI/gpt-j-6B"

tokenizer = GPTJTokenizer.from_pretrained(model_name)
model = GPTJForCausalLM.from_pretrained(model_name, load_state_as_np=True)
model.eval()

inputs = "What is PaddleNLP?"
input_ids = tokenizer([inputs], return_tensors="pd")["input_ids"]

outputs, _ = model.generate(
    input_ids=input_ids,
    max_length=100,
    decode_strategy="sampling",
    temperature=0.8,
    top_p=0.9,
    use_fp16_decoding=True,
    use_fast=True,
)

result = tokenizer.decode(outputs[0])

print("Model input:", inputs)
print("Result:", result)
