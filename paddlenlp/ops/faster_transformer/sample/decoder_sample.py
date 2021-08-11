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
import os
from attrdict import AttrDict
import argparse
import time

import yaml
from pprint import pprint
import paddle

from paddlenlp.ops import FasterDecoder
from paddlenlp.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./config/decoder.sample.yaml",
        type=str,
        help="Path of the config file. ")
    parser.add_argument(
        "--decoder_lib",
        default="../../build/lib/libdecoder_op.so",
        type=str,
        help="Path of libdecoder_op.so. ")
    parser.add_argument(
        "--use_fp16_decoder",
        action="store_true",
        help="Whether to use fp16 decoder to predict. ")
    args = parser.parse_args()
    return args


def do_predict(args):
    place = "gpu"
    paddle.set_device(place)

    # Define model
    transformer = FasterDecoder(
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
        max_out_len=args.max_out_len,
        decoder_lib=args.decoder_lib,
        use_fp16_decoder=args.use_fp16_decoder)

    # Load checkpoint.
    transformer.load(
        os.path.join(args.init_from_params, "transformer.pdparams"))
    # Set evaluate mode
    transformer.eval()

    # Generate data randomly
    dec_input = paddle.randn(
        shape=[args.infer_batch_size, 1, args.d_model], dtype='float32')
    enc_output = paddle.randn(
        shape=[args.infer_batch_size, args.max_length, args.d_model],
        dtype='float32')
    mem_seq_lens = paddle.full(
        shape=[args.infer_batch_size, 1],
        fill_value=args.max_length,
        dtype='int32')
    dtype = 'float32'
    if args.use_fp16_decoder:
        dtype = 'float16'
        dec_input = paddle.cast(dec_input, dtype=dtype)
        enc_output = paddle.cast(enc_output, dtype=dtype)
    self_cache = paddle.zeros(
        shape=[args.n_layer, 2, 0, args.infer_batch_size, args.d_model],
        dtype=dtype)
    mem_cache = paddle.zeros(
        shape=[
            args.n_layer, 2, args.infer_batch_size, args.max_length,
            args.d_model
        ],
        dtype=dtype)

    with paddle.no_grad():
        for i in range(100):
            # For warmup. 
            if 50 == i:
                start = time.time()
            dec_output, self_cache, mem_cache = transformer.decoder(
                from_tensor=dec_input,
                memory_tensor=enc_output,
                mem_seq_len=mem_seq_lens,
                self_cache=self_cache,
                mem_cache=mem_cache)
        logger.info("Average test time for decoder is %f ms" % (
            (time.time() - start) / 50 * 1000))


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)
    args.decoder_lib = ARGS.decoder_lib
    args.use_fp16_decoder = ARGS.use_fp16_decoder

    do_predict(args)
