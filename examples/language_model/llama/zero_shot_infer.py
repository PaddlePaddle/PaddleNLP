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

from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

paddle.seed(100)

paddle.set_default_dtype("float16")
# paddle.set_default_dtype("float32")
# paddle.set_device("cpu")

model = AutoModelForCausalLM.from_pretrained("./llama-7b", load_state_as_np=True)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "./llama-7b",
    add_bos_token=False,
)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.unk_token

inputs = tokenizer(
    ["My name is", "I am"],
    padding=True,
    return_tensors="pd",
    return_attention_mask=True,
    return_position_ids=True,
)

output = model.generate(
    inputs["input_ids"],
    inputs["attention_mask"],
    inputs["position_ids"],
    max_length=100,
    min_length=0,
    use_cache=True,
    temperature=1.0,
    top_k=1,
    top_p=1.0,
    repetition_penalty=1.0,
    decode_strategy="sampling",
)[0]

for out in output.tolist():
    print(tokenizer.decode(out, skip_special_tokens=True))
    print("-" * 20)
