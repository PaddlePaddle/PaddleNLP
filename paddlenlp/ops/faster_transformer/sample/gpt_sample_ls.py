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
from paddlenlp.transformers import GPTModel, GPTForGreedyGeneration
from paddlenlp.transformers import GPTChineseTokenizer, GPTTokenizer
import lightseq.inference as lsi
from pd_gpt2_export import extract_paddle_gpt_weights
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "gpt-cpm-large-cn": (GPTForGreedyGeneration, GPTChineseTokenizer),
    "gpt2-medium-en": (GPTForGreedyGeneration, GPTTokenizer),
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
        default="../build/lib/libdecoding_op.so",
        type=str,
        help="Path of libdecoding_op.so. ")
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size. ")
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
    output_lightseq_model_name_paddle = "lightseq" + args.model_name_or_path + str(
        args.topk)

    input_paddlenlp_gpt_model = args.model_name_or_path

    tokenizer = GPTTokenizer.from_pretrained(args.model_name_or_path)
    logger.info('Loading the model parameters, please wait...')

    bos_id = 1  # tokenizer.convert_tokens_to_ids(args.start_token)
    eos_id = tokenizer.convert_tokens_to_ids(args.end_token)
    if not os.path.exists(output_lightseq_model_name_paddle + '.hdf5'):
        extract_paddle_gpt_weights(
            output_lightseq_model_name_paddle,
            input_paddlenlp_gpt_model,
            head_num=16,  # layer number
            generation_method="topk",
            topk=args.topk,
            topp=args.topp,
            eos_id=eos_id,
            pad_id=50257,
            max_step=args.max_out_len, )
    ls_model = lsi.Gpt(output_lightseq_model_name_paddle + '.hdf5',
                       max_batch_size=16)

    input_ids = np.array(
        [[bos_id] for i in range(args.batch_size * 1)]).astype("int32").reshape(
            [args.batch_size, 1])
    input_ids = paddle.to_tensor(input_ids)

    with paddle.no_grad():
        for i in range(100):
            # For warmup. 
            if 50 == i:
                paddle.fluid.core._cuda_synchronize(place)
                start = time.time()
            out_seq = ls_model.sample(input_ids)
        paddle.fluid.core._cuda_synchronize(place)
        logger.info("Average test time for decoding is %f ms" % (
            (time.time() - start) / 50 * 1000))
        output_sequence = out_seq
    for i in range(args.batch_size):
        print("========== Sample-%d ==========" % i)
        print(tokenizer.convert_ids_to_string(output_sequence[i][1:]))


if __name__ == "__main__":
    args = parse_args()
    pprint(args)
    do_predict(args)
