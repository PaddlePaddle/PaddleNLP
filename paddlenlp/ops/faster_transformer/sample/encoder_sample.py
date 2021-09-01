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

import time
import argparse
from attrdict import AttrDict
import yaml
from pprint import pprint
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.ops import FasterEncoder
from paddlenlp.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./config/encoder.sample.yaml",
        type=str,
        help="Path of the config file. ")
    parser.add_argument(
        "--encoder_lib",
        default="../../build/lib/libencoder_op.so",
        type=str,
        help="Path of libencoder_op.so. ")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument(
        "--use_fp16_encoder",
        action="store_true",
        help="Whether to use fp16 encoder to predict. ")
    args = parser.parse_args()
    return args


def do_predict(args):
    place = "gpu"
    place = paddle.set_device(place)
    paddle.seed(args.seed)
    # Define model
    encoder = FasterEncoder(
        src_vocab_size=args.src_vocab_size,
        max_seq_length=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        bos_id=args.bos_idx,
        int8_mode=args.int8_mode,
        allow_gemm_test=args.allow_gemm_test,
        use_trt_kernel=args.use_trt_kernel,
        remove_padding=args.remove_padding,
        encoder_lib=args.encoder_lib,
        use_fp16_encoder=args.use_fp16_encoder)
    encoder.load('base_trained_models/step_final/transformer.pdparams')
    seq_len = args.seq_len
    batch_size = args.batch_size

    src_word = paddle.randint(
        0, 30000, shape=[args.batch_size, args.seq_len], dtype='int64')
    mem_seq_lens = paddle.randint(1, seq_len + 1, (batch_size, ), dtype="int32")

    # if args.remove_padding:
    #     if args.avg_seq_len > 0:
    #         mem_seq_lens = paddle.ones(
    #             (batch_size, ), dtype="int32") * args.avg_seq_len
    #     elif args.avg_seq_len == -1:
    #         mem_seq_lens = paddle.ones(
    #             (batch_size, ), dtype="int32") * seq_len / 2
    #     else:
    #         raise ValueError("wrong avg_seq_len")

    src_max_len = paddle.shape(src_word)[-1]

    src_mask = F.sequence_mask(
        mem_seq_lens, src_max_len, dtype=paddle.get_default_dtype())
    src_word *= src_mask

    src_pos = paddle.cast(
        src_word != args.bos_idx, dtype="int64") * paddle.arange(
            start=0, end=src_max_len)

    src_emb = encoder.src_word_embedding(src_word)
    src_pos_emb = encoder.src_pos_embedding(src_pos)
    src_emb = src_emb + src_pos_emb
    enc_input = F.dropout(
        src_emb, p=encoder.dropout,
        training=encoder.training) if args.dropout else src_emb

    src_mask = paddle.unsqueeze(src_mask, axis=[1, 2])
    src_mask_1 = paddle.transpose(src_mask, [0, 1, 3, 2])
    src_mask = src_mask * src_mask_1
    src_mask.stop_gradient = True

    src_slf_attn_bias = (1 - src_mask) * -1e4
    src_slf_attn_bias.stop_gradient = True

    custom_enc_output = encoder(enc_input, src_mask, mem_seq_lens)
    # '''
    encoder.transformer.encoder.eval()
    enc_output = encoder.transformer.encoder(enc_input, src_slf_attn_bias)

    # Max abs diff
    out1, out2 = enc_output.numpy(), custom_enc_output.numpy()
    diff = abs(out1 - out2)
    max_index = np.unravel_index(np.argmax(diff), diff.shape)

    print("Max abs diff: ", out1[max_index], out2[max_index],
          out1[max_index] - out2[max_index])

    # Max relative diff
    max_index = np.unravel_index(np.argmax(abs(out1 - out2) / out2), diff.shape)
    print("Max relative diff: ", out1[max_index], out2[max_index],
          (out1[max_index] - out2[max_index]) / out2[max_index])

    np.testing.assert_allclose(out1, out2)

    with paddle.no_grad():
        for i in range(1):
            # For warmup. 
            if 0 == i:
                start = time.time()
            encoder_out = encoder(src_word=src_word, seq_len=seq_len)
        logger.info("Average test time for decoder is %f ms" % (
            (time.time() - start) / 50 * 1000))
    # '''


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)
    args.encoder_lib = ARGS.encoder_lib
    print("lib path", args.encoder_lib)
    args.use_fp16_encoder = ARGS.use_fp16_encoder

    do_predict(args)
