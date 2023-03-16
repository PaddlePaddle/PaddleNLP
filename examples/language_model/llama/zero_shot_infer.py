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

import paddle
from modeling import LLaMAForCausalLM
from tokenizer import LLaMATokenizer

paddle.set_default_dtype("float16")
# paddle.set_default_dtype("float32")
# paddle.set_device("cpu")


model = LLaMAForCausalLM.from_pretrained("./llama-7b", load_state_as_np=True)

tokenizer = LLaMATokenizer.from_pretrained("./llama-7b")

inputs = tokenizer("My name is", return_tensors="pd")

output = model.generate(
    inputs["input_ids"][:, 1:],
    max_length=100,
    use_cache=True,
)[0]

print(tokenizer.decode(output.tolist()[0], skip_special_tokens=True))
