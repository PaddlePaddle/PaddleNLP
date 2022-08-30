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
from paddlenlp.utils.log import logger

sys.path.append("../")
import reader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="../configs/transformer.base.yaml",
                        type=str,
                        help="Path of the config file. ")
    parser.add_argument(
        "--decoding_lib",
        default="../../../../paddlenlp/ops/build/lib/libdecoding_op.so",
        type=str,
        help="Path of libdecoding_op.so. ")
    parser.add_argument("--use_fp16_decoding",
                        action="store_true",
                        help="Whether to use fp16 decoding to predict. ")
    parser.add_argument(
        "--enable_faster_encoder",
        action="store_true",
        help=
        "Whether to use faster version encoder to predict. This is experimental option for now. "
    )
    parser.add_argument("--use_fp16_encoder",
                        action="store_true",
                        help="Whether to use fp16 encoder to predict. ")
    parser.add_argument(
        "--decoding_strategy",
        default="beam_search",
        type=str,
        choices=[
            "beam_search", "topk_sampling", "topp_sampling", "beam_search_v2"
        ],
        help=
        "Decoding strategy. Can be one of ['beam_search', 'topk_sampling', 'topp_sampling', 'beam_search_v2']. "
    )
    parser.add_argument("--beam_size", default=4, type=int, help="Beam size. ")
    parser.add_argument("--topk",
                        default=4,
                        type=int,
                        help="The k value for topk_sampling. Default is 4. ")
    parser.add_argument(
        "--topp",
        default=0.0,
        type=float,
        help=
        "The probability threshold for topp_sampling. Default is 0.0 which means it won't go through topp_sampling. "
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=
        "Whether to print logs on each cards and use benchmark vocab. Normally, not necessary to set --benchmark. "
    )
    parser.add_argument(
        "--vocab_file",
        default=None,
        type=str,
        help=
        "The vocab file. Normally, it shouldn't be set and in this case, the default WMT14 dataset will be used."
    )
    parser.add_argument(
        "--unk_token",
        default=None,
        type=str,
        help=
        "The unknown token. It should be provided when use custom vocab_file. ")
    parser.add_argument(
        "--bos_token",
        default=None,
        type=str,
        help="The bos token. It should be provided when use custom vocab_file. "
    )
    parser.add_argument(
        "--eos_token",
        default=None,
        type=str,
        help="The eos token. It should be provided when use custom vocab_file. "
    )
    args = parser.parse_args()
    return args


def do_predict(args):
    place = "gpu"
    place = paddle.set_device(place)
    reader.adapt_vocab_size(args)

    # Define model
    transformer = FasterTransformer(
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
        max_length=args.max_length + 1,
        num_encoder_layers=args.n_layer,
        num_decoder_layers=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        weight_sharing=args.weight_sharing,
        bos_id=args.bos_idx,
        eos_id=args.eos_idx,
        decoding_strategy=args.decoding_strategy,
        beam_size=args.beam_size,
        max_out_len=args.max_out_len,
        decoding_lib=args.decoding_lib,
        use_fp16_decoding=args.use_fp16_decoding,
        enable_faster_encoder=args.enable_faster_encoder,
        use_fp16_encoder=args.use_fp16_encoder,
        rel_len=args.use_rel_len,
        alpha=args.alpha)

    # Set evaluate mode
    transformer.eval()

    # Load checkpoint.
    transformer.load(init_from_params=os.path.join(args.init_from_params,
                                                   "transformer.pdparams"))

    # Convert dygraph model to static graph model
    transformer = paddle.jit.to_static(
        transformer,
        input_spec=[
            # src_word
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            # trg_word
            # Support exporting model which support force decoding
            # NOTE: Data type MUST be int32 !
            # paddle.static.InputSpec(
            #     shape=[None, None], dtype="int32")
        ])

    # Save converted static graph model
    paddle.jit.save(transformer,
                    os.path.join(args.inference_model_dir, "transformer"))
    logger.info("Transformer has been saved to {}".format(
        args.inference_model_dir))


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
    args.decoding_lib = ARGS.decoding_lib
    args.use_fp16_decoding = ARGS.use_fp16_decoding
    args.enable_faster_encoder = ARGS.enable_faster_encoder
    args.use_fp16_encoder = ARGS.use_fp16_encoder
    args.decoding_strategy = ARGS.decoding_strategy
    args.beam_size = ARGS.beam_size
    args.topk = ARGS.topk
    args.topp = ARGS.topp
    args.benchmark = ARGS.benchmark
    args.vocab_file = ARGS.vocab_file
    args.unk_token = ARGS.unk_token
    args.bos_token = ARGS.bos_token
    args.eos_token = ARGS.eos_token
    pprint(args)

    do_predict(args)
