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

from paddlenlp.ops import FasterTransformer

from paddlenlp.utils.log import logger
from paddlenlp.data import Pad


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./sample/config/decoding.sample.yaml",
        type=str,
        help="Path of the config file. ")
    parser.add_argument(
        "--decoding_lib",
        default="./build/lib/libdecoding_op.so",
        type=str,
        help="Path of libdecoding_op.so. ")
    parser.add_argument(
        "--use_fp16_decoding",
        action="store_true",
        help="Whether to use fp16 decoding to predict. ")
    args = parser.parse_args()
    return args


def generate_src_word(batch_size, vocab_size, max_length, eos_idx, pad_idx):
    memory_sequence_length = np.random.randint(
        low=1, high=max_length, size=batch_size).astype(np.int32)
    data = []
    for i in range(batch_size):
        data.append(
            np.random.randint(
                low=3,
                high=vocab_size,
                size=memory_sequence_length[i],
                dtype=np.int64))

    word_pad = Pad(pad_idx)
    src_word = word_pad([list(word) + [eos_idx] for word in data])

    return paddle.to_tensor(src_word, dtype="int64")


def do_predict(args):
    place = "gpu"
    paddle.set_device(place)

    # Define model
    transformer = FasterTransformer(
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
        max_length=args.max_length + 1,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        weight_sharing=args.weight_sharing,
        bos_id=args.bos_idx,
        eos_id=args.eos_idx,
        decoding_strategy=args.decoding_strategy,
        beam_size=args.beam_size,
        topk=args.topk,
        topp=args.topp,
        max_out_len=args.max_out_len,
        decoding_lib=args.decoding_lib,
        use_fp16_decoding=args.use_fp16_decoding)

    # Set evaluate mode
    transformer.eval()

    src_word = generate_src_word(
        batch_size=args.infer_batch_size,
        vocab_size=args.src_vocab_size,
        max_length=args.max_length,
        eos_idx=args.eos_idx,
        pad_idx=args.bos_idx)

    with paddle.no_grad():
        for i in range(100):
            # For warmup. 
            if 50 == i:
                start = time.time()
            transformer(src_word=src_word)
        logger.info("Average test time for encoder-decoding is %f ms" % (
            (time.time() - start) / 50 * 1000))


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)
    args.decoding_lib = ARGS.decoding_lib
    args.use_fp16_decoding = ARGS.use_fp16_decoding

    do_predict(args)
