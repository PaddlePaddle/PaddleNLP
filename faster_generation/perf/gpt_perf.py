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

import paddle
import torch
from paddlenlp.transformers import GPTLMHeadModel, GPTTokenizer
from transformers import GPT2LMHeadModel as hf_gpt_model
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2-en",
        type=str,
        choices=['gpt2-en', 'gpt2-medium-en', 'gpt2-large-en'],
        help=
        "The model name to specify the bart to use. Can be one of ['gpt2-en', 'gpt2-medium-en', 'gpt2-large-en']. "
    )
    parser.add_argument(
        "--decode_strategy",
        default='sampling',
        type=str,
        choices=['greedy_search', 'sampling'],
        help="The decoding strategy. Can be one of ['greedy_search', 'sampling']"
    )
    parser.add_argument(
        "--top_k",
        default=4,
        type=int,
        help="The number of candidate to procedure beam search. ")
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="The size of input batch. ")
    parser.add_argument(
        "--top_p",
        default=1.0,
        type=float,
        help="The probability threshold to procedure topp sampling. ")
    parser.add_argument("--max_length",
                        default=32,
                        type=int,
                        help="Maximum output length. ")
    parser.add_argument("--use_fp16_decoding",
                        action="store_true",
                        help="Whether to use fp16 decoding to predict. ")
    args = parser.parse_args()
    return args


def do_predict(args):
    place = "gpu"
    place = paddle.set_device(place)

    tokenizer = GPTTokenizer.from_pretrained(args.model_name_or_path)
    model = GPTLMHeadModel.from_pretrained(args.model_name_or_path)
    # Set evaluate mode
    model.eval()
    bos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    input_ids_np = np.array([[bos_id] for i in range(args.batch_size)
                             ]).astype("int64").reshape([args.batch_size, 1])
    input_ids = paddle.to_tensor(input_ids_np)
    # Define model
    num_loop = 100
    with paddle.no_grad():
        for i in range(num_loop):
            # For warmup.
            if 50 == i:
                # PaddlePaddle >= 2.2
                paddle.device.cuda.synchronize(place)
                start = time.perf_counter()
            output, _ = model.generate(input_ids=input_ids,
                                       max_length=args.max_length,
                                       decode_strategy=args.decode_strategy,
                                       top_k=args.top_k,
                                       top_p=args.top_p,
                                       bos_token_id=bos_id,
                                       eos_token_id=eos_id,
                                       use_faster=True,
                                       use_fp16_decoding=args.use_fp16_decoding)
        paddle.device.cuda.synchronize(place)
        faster_cost = (time.perf_counter() - start) / 50 * 1000

    if args.use_fp16_decoding:
        pprint(args)
        print("Faster FP16 cost:", faster_cost)
        return
    with paddle.no_grad():
        for i in range(num_loop):
            # For warmup.
            if 50 == i:
                # PaddlePaddle >= 2.2
                paddle.device.cuda.synchronize(place)
                start = time.perf_counter()
            output, _ = model.generate(input_ids=input_ids,
                                       max_length=args.max_length,
                                       decode_strategy=args.decode_strategy,
                                       top_k=args.top_k,
                                       top_p=args.top_p,
                                       bos_token_id=bos_id,
                                       eos_token_id=eos_id)
        paddle.device.cuda.synchronize(place)
        pd_cost = (time.perf_counter() - start) / 50 * 1000

    device = torch.device("cuda:0")
    hf_model = hf_gpt_model.from_pretrained(args.model_name_or_path[:-3])
    hf_model.to(device)
    hf_model.eval()

    hf_input_ids = torch.tensor(input_ids_np)
    hf_input_ids = hf_input_ids.to(device)

    if args.decode_strategy == 'sampling':
        do_sample = True
    else:
        do_sample = False
    with torch.no_grad():
        for i in range(num_loop):
            # For warmup.
            if 50 == i:
                torch.cuda.synchronize()
                start = time.perf_counter()
            output = hf_model.generate(hf_input_ids,
                                       do_sample=do_sample,
                                       max_length=args.max_length + 1,
                                       bos_token_id=bos_id,
                                       eos_token_id=eos_id,
                                       pad_token_id=0,
                                       top_k=args.top_k,
                                       top_p=args.top_p)
        torch.cuda.synchronize()
        hf_cost = (time.perf_counter() - start) / 50 * 1000

    pprint(args)
    print("Faster FP32 cost:", faster_cost)
    print("PD cost:", pd_cost)
    print("HF cost:", hf_cost)
    print("Speed up Faster FP32/PD:", pd_cost / faster_cost)
    print("Speed up Faster FP32/HF:", hf_cost / faster_cost)


if __name__ == "__main__":
    args = parse_args()
    do_predict(args)
