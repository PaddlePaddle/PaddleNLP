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

import sys
import os
import numpy as np
from attrdict import AttrDict
import argparse
import time
from pprint import pprint

import paddle
from paddlenlp.ops import FasterErnie3
from paddlenlp.transformers import Ernie3Tokenizer, Ernie3ForGeneration
from paddlenlp.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="ernie3-10b",
        type=str,
        help="The model name to specify the ernie3 to use. ")
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
        "--topk",
        default=4,
        type=int,
        help="The number of candidate to procedure beam search. ")
    parser.add_argument(
        "--topp",
        default=0.0,
        type=float,
        help="The probability threshold to procedure topp sampling. ")
    parser.add_argument(
        "--min_out_len", default=10, type=int, help="Minimum output length. ")
    parser.add_argument(
        "--max_out_len", default=20, type=int, help="Maximum output length. ")
    parser.add_argument(
        "--num_return_sequence",
        default=1,
        type=int,
        help="The number of returned sequence. ")
    parser.add_argument(
        "--temperature",
        default=0.9,
        type=float,
        help="The temperature to set. ")
    parser.add_argument(
        "--repetition_penalty",
        default=1.1,
        type=float,
        help="The temperature to set. ")
    parser.add_argument(
        "--decoding_strategy",
        default="greedy_search",
        choices=["sampling", "greedy_search"],
        type=str,
        help="The main strategy to decode. ")
    parser.add_argument(
        "--use_fp32_decoding",
        action="store_true",
        help="Whether to use fp32 decoding to predict. ")
    args = parser.parse_args()
    return args


def do_predict(args):
    place = "gpu"
    place = paddle.set_device(place)
    paddle.set_default_dtype('float16')

    tokenizer = Ernie3Tokenizer.from_pretrained(args.model_name_or_path)
    logger.info('Loading the model parameters, please wait...')
    model = Ernie3ForGeneration.from_pretrained(
        args.model_name_or_path, load_state_as_np=True)

    ernie3 = FasterErnie3(
        model=model,
        decoding_lib=args.decoding_lib,
        use_fp16_decoding=not args.use_fp32_decoding)

    # Set evaluate mode
    ernie3.eval()

    # Convert dygraph model to static graph model 
    ernie3 = paddle.jit.to_static(
        ernie3,
        input_spec=[
            # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int32"),
            #
            # If to provide mem_seq_len and attention_mask,
            # the parameters should be:
            # mem_seq_len
            # paddle.static.InputSpec(shape=[None, None], dtype="int32"),
            # attention_mask
            # paddle.static.InputSpec(shape=[None, None, None], dtype="float16" if args.use_fp16_decoding else "float32"),
            #
            None,  # mem_seq_len
            None,  # attention_mask
            args.topk,
            args.topp,
            args.min_out_len,
            args.max_out_len,
            model.ernie3.bos_token_id,
            model.ernie3.end_token_id,
            model.ernie3.pad_token_id,
            None,  # forced_eos_token_id
            args.temperature,
            args.repetition_penalty,
            args.decoding_strategy,
            args.num_return_sequence
        ])

    # Save converted static graph model
    paddle.jit.save(ernie3, os.path.join(args.inference_model_dir, "ernie3"))
    logger.info("Ernie3 has been saved to {}".format(args.inference_model_dir))

    ernie3.save_resources(tokenizer, args.inference_model_dir)


if __name__ == "__main__":
    args = parse_args()
    pprint(args)
    do_predict(args)
