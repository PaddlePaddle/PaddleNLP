# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
from pprint import pprint

import numpy as np
import paddle
import pynvml

from paddlenlp.transformers import CodeGenForCausalLM, CodeGenTokenizer

pynvml.nvmlInit()


def query_by_id(gpu_id=2):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.used // 1024 // 1024


def perf_pd(args):
    start_mem = query_by_id(args.gpu_id)
    place = "gpu"
    place = paddle.set_device(place)
    tokenizer = CodeGenTokenizer.from_pretrained(args.model_name_or_path)
    model = CodeGenForCausalLM.from_pretrained(args.model_name_or_path, load_state_as_np=True)
    model.eval()
    load_mem = query_by_id(args.gpu_id)

    input_ids_np = [
        np.random.choice(list(tokenizer.decoder.keys())[:-1], args.input_len) for _ in range(args.batch_size)
    ]
    input_ids = paddle.to_tensor(input_ids_np)

    num_loop = 100
    with paddle.no_grad():
        for i in range(num_loop):
            # For warmup.
            if num_loop // 2 == i:
                # PaddlePaddle >= 2.2
                paddle.device.cuda.synchronize(place)
                start = time.perf_counter()
            model.generate(
                input_ids=input_ids,
                max_length=args.generate_len,
                min_length=args.generate_len,
                decode_strategy="sampling",
                top_k=args.top_k,
                top_p=args.top_p,
                use_fast=args.use_faster,
                use_fp16_decoding=args.use_fp16_decoding,
            )
            generate_mem = query_by_id(args.gpu_id)
        paddle.device.cuda.synchronize(place)
        pd_cost = (time.perf_counter() - start) / (num_loop - num_loop // 2) * 1000
    return pd_cost, load_mem - start_mem, generate_mem - start_mem


def perf_hf(args):
    import torch
    from transformers import CodeGenForCausalLM as hf_codegen
    from transformers import CodeGenTokenizer as hf_tokenizer

    start_mem = query_by_id(args.gpu_id)
    device = torch.device("cuda")
    tokenizer = hf_tokenizer.from_pretrained(args.model_name_or_path)
    model = hf_codegen.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()
    load_mem = query_by_id(args.gpu_id)

    input_ids_np = [np.random.choice(list(tokenizer.decoder.keys()), args.input_len) for _ in range(args.batch_size)]
    input_ids = torch.tensor(input_ids_np)
    input_ids = input_ids.to(device)
    num_loop = 100
    with torch.no_grad():
        for i in range(num_loop):
            # For warmup.
            if num_loop // 2 == i:
                torch.cuda.synchronize()
                start = time.perf_counter()
            model.generate(
                input_ids,
                do_sample=True,
                max_length=args.generate_len + input_ids.shape[-1],
                min_length=args.generate_len + input_ids.shape[-1],
                top_k=args.top_k,
                top_p=args.top_p,
            )
            generate_mem = query_by_id(args.gpu_id)
        torch.cuda.synchronize()
        hf_cost = (time.perf_counter() - start) / (num_loop - num_loop // 2) * 1000
    return hf_cost, load_mem - start_mem, generate_mem - start_mem


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--perf_type",
        default="pd",
        type=str,
        choices=["pd", "pd_faster_fp32", "pd_faster_fp16", "hf"],
        help="The type of perf.  ",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="Salesforce/codegen-350M-mono",
        type=str,
        choices=[
            "Salesforce/codegen-350M-mono",
            "Salesforce/codegen-2B-mono",
            "Salesforce/codegen-6B-mono",
            "Salesforce/codegen-16B-mono",
        ],
        help="The model name to specify the bart to use.  ",
    )
    parser.add_argument("--top_k", default=4, type=int, help="The number of candidate to procedure topk sampling. ")
    parser.add_argument(
        "--top_p", default=1.0, type=float, help="The probability threshold to procedure topp sampling. "
    )
    parser.add_argument("--batch_size", default=1, type=int, help="The size of input batch. ")
    parser.add_argument("--input_len", default=60, type=int, help="The size of model input. ")
    parser.add_argument("--generate_len", default=20, type=int, help="Length of output . ")
    parser.add_argument("--gpu_id", default=2, type=int, help="The id of GPU . ")
    parser.add_argument(
        "--use_faster", action="store_true", help="Whether to process inference using faster codegen. "
    )

    parser.add_argument("--use_fp16_decoding", action="store_true", help="Whether to use fp16 decoding to predict. ")
    args = parser.parse_args()
    return args


def do_predict(args):
    try:
        if args.perf_type == "pd":
            args.use_faster = False
            cost, load_mem, generate_mem = perf_pd(args)
        elif args.perf_type == "pd_faster_fp32":
            args.use_faster = True
            args.use_fp16_decoding = False
            cost, load_mem, generate_mem = perf_pd(args)
        elif args.perf_type == "pd_faster_fp16":
            args.use_faster = True
            args.use_fp16_decoding = True
            paddle.set_default_dtype("float16")
            cost, load_mem, generate_mem = perf_pd(args)
        else:
            cost, load_mem, generate_mem = perf_hf(args)
        pprint(args)
        print(
            f"CodeGenPerfResult: cost_time: {cost} ms, load_mem: {load_mem} MB, generate_mem:{generate_mem} MB, args:{args}\n"
        )
    except Exception as e:
        pprint(args)
        print(f"CodeGenPerfResult: ERROR: {e}, args:{args}\n")


if __name__ == "__main__":
    args = parse_args()
    do_predict(args)
