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

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args(prog=None):
    """
    parse_args
    """
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--model_name_or_path", type=str, help="model name or local path", required=True)
    parser.add_argument("--do_forward", action="store_true", help="fowrward test")
    parser.add_argument("--do_generate", action="store_true", help="generate test")
    return parser.parse_args()


@torch.no_grad()
def predict_generate(model, inputs):
    for i in range(10):
        start = time.perf_counter()
        generate_ids = model.generate(
            inputs.input_ids,
            max_length=100,
            do_sample=False,
        )
        hf_cost = (time.perf_counter() - start) * 1000
        print("Speed test:", hf_cost)
        result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(result)


@torch.no_grad()
def predict_forward(model, inputs):
    for i in range(10):
        start = time.perf_counter()
        _ = model(**inputs)
        hf_cost = (time.perf_counter() - start) * 1000
        print("Speed test:", hf_cost)


if __name__ == "__main__":
    args = parse_args()
    all_texts = [
        "你好",
        "去年9月，拼多多海外版“Temu”正式在美国上线。数据显示，截至2023年2月23日，Temu App新增下载量4000多万，新增用户增速第一，AppStore购物榜霸榜69天、Google Play购物榜霸榜114天。",
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if "llama" in args.model_name_or_path:
        tokenizer.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).cuda()
    model = model.eval()
    if args.do_forward:
        for input_text in all_texts:
            print(f"text: {input_text}")
            inputs = tokenizer([input_text], return_tensors="pt", max_length=50, padding=True)
            inputs = inputs.to("cuda")
            predict_forward(model, inputs)

    if args.do_generate:
        for input_text in all_texts:
            print(f"text: {input_text}")
            inputs = tokenizer([input_text], return_tensors="pt", max_length=50, padding=True)
            inputs = inputs.to("cuda")
            predict_generate(model, inputs)
