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
from paddlenlp.transformers import BartForConditionalGeneration, BartTokenizer
from transformers import BartForConditionalGeneration as hf_bart_model
from paddlenlp.data import Pad


def prepare_input(tokenizer, sentences):
    word_pad = Pad(tokenizer.pad_token_id, dtype="int64")
    tokenized = tokenizer(sentences)
    inputs = word_pad([i["input_ids"] for i in tokenized])
    input_ids = paddle.to_tensor(inputs)
    return input_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="bart-base",
        type=str,
        choices=['bart-base', 'bart-large'],
        help=
        "The model name to specify the bart to use. Can be one of ['bart-base', 'bart-large']. "
    )
    parser.add_argument(
        "--decode_strategy",
        default='sampling',
        type=str,
        choices=['greedy_search', 'beam_search', 'sampling'],
        help=
        "The decoding strategy. Can be one of ['greedy_search', 'beam_search', 'sampling']"
    )
    parser.add_argument("--num_beams",
                        default=4,
                        type=int,
                        help="The parameters for beam search. ")
    parser.add_argument(
        "--top_k",
        default=4,
        type=int,
        help="The number of candidate to procedure beam search. ")
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

    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    model = BartForConditionalGeneration.from_pretrained(
        args.model_name_or_path)
    # Set evaluate mode
    model.eval()
    sentences = [
        "I love that girl, but <mask> does not <mask> me.",
        "She is so <mask> that I can not help glance at <mask>.",
        "Nothing's gonna <mask> my love for you.",
        "Drop everything now. Meet me in the pouring <mask>. Kiss me on the sidewalk.",
    ]

    input_ids = prepare_input(tokenizer, sentences)

    # Define model
    model.eval()

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
                                       num_beams=args.num_beams,
                                       early_stopping=True,
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
                                       num_beams=args.num_beams,
                                       early_stopping=True)
        paddle.device.cuda.synchronize(place)
        pd_cost = (time.perf_counter() - start) / 50 * 1000

    device = torch.device("cuda:0")
    hf_model = hf_bart_model.from_pretrained("facebook/" +
                                             args.model_name_or_path)
    hf_model.to(device)
    hf_model.eval()
    hf_input_ids = prepare_input(tokenizer, sentences)
    hf_input_ids = torch.tensor(hf_input_ids.numpy())
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
                                       top_k=args.top_k,
                                       top_p=args.top_p,
                                       num_beams=args.num_beams,
                                       no_repeat_ngram_size=0,
                                       length_penalty=0.0)
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
