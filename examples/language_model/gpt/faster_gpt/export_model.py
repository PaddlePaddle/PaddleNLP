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
import sys
import os
import numpy as np
from attrdict import AttrDict
import argparse
import time

import paddle

import yaml
from pprint import pprint

from paddlenlp.ops import FasterGPT
from paddlenlp.transformers import GPTModel, GPTLMHeadModel
from paddlenlp.transformers import GPTChineseTokenizer, GPTTokenizer

from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "gpt-cpm-large-cn": (GPTLMHeadModel, GPTChineseTokenizer),
    "gpt2-medium-en": (GPTLMHeadModel, GPTTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2-medium-en",
        type=str,
        help="The model name to specify the gpt to use. Can be one of ['gpt2-en', 'gpt2-medium-en', 'gpt-cpm-large-cn']. "
    )
    parser.add_argument(
        "--decoding_lib",
        default="../../build/lib/libdecoding_op.so",
        type=str,
        help="Path of libdecoding_op.so. ")
    parser.add_argument(
        "--inference_model_dir",
        default="./infer_model/",
        type=str,
        help="Path to save inference model of gpt. ")
    parser.add_argument(
        "--top_k",
        default=4,
        type=int,
        help="The number of candidate to procedure beam search. ")
    parser.add_argument(
        "--top_p",
        default=0.0,
        type=float,
        help="The probability threshold to procedure top_p sampling. ")
    parser.add_argument(
        "--max_out_len", default=32, type=int, help="Maximum output length. ")
    parser.add_argument(
        "--start_token",
        default="<|endoftext|>",
        type=str,
        help="The start token. Defaults to <|endoftext|>. ")
    parser.add_argument(
        "--end_token",
        default="<|endoftext|>",
        type=str,
        help="The end token. Defaults to <|endoftext|>. ")
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        help="The temperature to set. ")
    parser.add_argument(
        "--use_fp16_decoding",
        action="store_true",
        help="Whether to use fp16 decoding to predict. ")
    args = parser.parse_args()
    return args


def do_predict(args):
    place = "gpu"
    place = paddle.set_device(place)

    model_class, tokenizer_class = MODEL_CLASSES[args.model_name_or_path]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    logger.info('Loading the model parameters, please wait...')
    model = model_class.from_pretrained(
        args.model_name_or_path, max_predict_len=args.max_out_len)

    bos_id = tokenizer.convert_tokens_to_ids(args.start_token)
    eos_id = tokenizer.convert_tokens_to_ids(args.end_token)

    gpt = FasterGPT(
        model=model,
        top_k=args.top_k,
        top_p=args.top_p,
        max_out_len=args.max_out_len,
        bos_id=bos_id,
        eos_id=eos_id,
        temperature=args.temperature,
        decoding_lib=args.decoding_lib,
        use_fp16_decoding=args.use_fp16_decoding)

    # Set evaluate mode
    gpt.eval()

    # Convert dygraph model to static graph model 
    gpt = paddle.jit.to_static(
        gpt,
        input_spec=[
            # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int32")
        ])

    # Save converted static graph model
    paddle.jit.save(gpt, os.path.join(args.inference_model_dir, "gpt"))
    logger.info("GPT has been saved to {}".format(args.inference_model_dir))

    gpt.save_resources(tokenizer, args.inference_model_dir)


if __name__ == "__main__":
    args = parse_args()
    pprint(args)
    do_predict(args)
