import sys
import os
import numpy as np
from attrdict import AttrDict
import argparse
import time

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import yaml
from pprint import pprint

from paddlenlp.transformers import TransformerModel
from paddlenlp.transformers import position_encoding_init
from paddlenlp.ops import FasterTransformer

sys.path.append("../")
import reader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../configs/transformer.base.yaml",
        type=str,
        help="Path of the config file. ")
    parser.add_argument(
        "--decoding-lib",
        default="../../../../paddlenlp/ops/src/build/lib/libdecoding_op.so",
        type=str,
        help="Path of libdecoding_op.so. ")
    parser.add_argument(
        "--use-fp16-decoding",
        action="store_true",
        help="Whether to use fp16 decoding to predict. ")
    args = parser.parse_args()
    return args


def do_predict(args):
    paddle.enable_static()
    place = "gpu"
    place = paddle.set_device(place)

    test_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(test_program, startup_program):
        src_word = paddle.static.data(
            name="src_word", shape=[None, None], dtype="int64")

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
            decoding_strategy="beam_search",
            beam_size=args.beam_size,
            max_out_len=args.max_out_len,
            decoding_lib=args.decoding_lib,
            use_fp16_decoding=args.use_fp16_decoding)

        finished_seq = transformer(src_word=src_word)

    test_program = test_program.clone(for_test=True)

    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    # Load checkpoint.
    transformer.export_params(
        init_from_params=os.path.join(args.init_from_params,
                                      "transformer.pdparams"),
        place=place)

    paddle.static.save_inference_model(
        os.path.join(args.inference_model_dir, "transformer"),
        feed_vars=src_word,
        fetch_vars=finished_seq,
        executor=exe,
        program=test_program)


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)
    args.decoding_lib = ARGS.decoding_lib
    args.use_fp16_decoding = ARGS.use_fp16_decoding

    do_predict(args)
