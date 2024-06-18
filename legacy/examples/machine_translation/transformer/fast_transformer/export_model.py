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

import argparse
import os
import sys
from pprint import pprint

import paddle
import yaml
from easydict import EasyDict as AttrDict

from paddlenlp.ops import FasterTransformer
from paddlenlp.utils.log import logger

sys.path.append("../")
import reader  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="../configs/transformer.base.yaml", type=str, help="Path of the config file. "
    )
    parser.add_argument(
        "--decoding_lib",
        default="../../../../paddlenlp/ops/build/lib/libdecoding_op.so",
        type=str,
        help="Path of libdecoding_op.so. ",
    )
    parser.add_argument("--use_fp16_decoding", action="store_true", help="Whether to use fp16 decoding to predict. ")
    parser.add_argument(
        "--enable_fast_encoder",
        action="store_true",
        help="Whether to use fast version encoder to predict. This is experimental option for now. ",
    )
    parser.add_argument("--use_fp16_encoder", action="store_true", help="Whether to use fp16 encoder to predict. ")
    parser.add_argument(
        "--decoding_strategy",
        default="beam_search",
        type=str,
        choices=["beam_search", "topk_sampling", "topp_sampling", "beam_search_v2"],
        help="Decoding strategy. Can be one of ['beam_search', 'topk_sampling', 'topp_sampling', 'beam_search_v2']. ",
    )
    parser.add_argument("--beam_size", default=4, type=int, help="Beam size. ")
    parser.add_argument("--topk", default=4, type=int, help="The k value for topk_sampling. Default is 4. ")
    parser.add_argument(
        "--topp",
        default=0.0,
        type=float,
        help="The probability threshold for topp_sampling. Default is 0.0 which means it won't go through topp_sampling. ",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Whether to print logs on each cards and use benchmark vocab. Normally, not necessary to set --benchmark. ",
    )
    parser.add_argument(
        "--vocab_file",
        default=None,
        type=str,
        help="The vocab file. Normally, it shouldn't be set and in this case, the default WMT14 dataset will be used.",
    )
    parser.add_argument(
        "--src_vocab",
        default=None,
        type=str,
        help="The vocab file for source language. If --vocab_file is given, the --vocab_file will be used. ",
    )
    parser.add_argument(
        "--trg_vocab",
        default=None,
        type=str,
        help="The vocab file for target language. If --vocab_file is given, the --vocab_file will be used. ",
    )
    parser.add_argument(
        "--bos_token", default=None, type=str, help="The bos token. It should be provided when use custom vocab_file. "
    )
    parser.add_argument(
        "--eos_token", default=None, type=str, help="The eos token. It should be provided when use custom vocab_file. "
    )
    parser.add_argument(
        "--pad_token",
        default=None,
        type=str,
        help="The pad token. It should be provided when use custom vocab_file. And if it's None, bos_token will be used. ",
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
        pad_id=args.pad_idx,
        decoding_strategy=args.decoding_strategy,
        beam_size=args.beam_size,
        max_out_len=args.max_out_len,
        decoding_lib=args.decoding_lib,
        use_fp16_decoding=args.use_fp16_decoding,
        enable_fast_encoder=args.enable_fast_encoder,
        use_fp16_encoder=args.use_fp16_encoder,
        rel_len=args.use_rel_len,
        alpha=args.alpha,
    )

    # Set evaluate mode
    transformer.eval()

    # Load checkpoint.
    transformer.load(init_from_params=os.path.join(args.init_from_params, "transformer.pdparams"))

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
        ],
    )

    # Save converted static graph model
    paddle.jit.save(transformer, os.path.join(args.inference_model_dir, "transformer"))
    logger.info("Transformer has been saved to {}".format(args.inference_model_dir))


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, "rt") as f:
        args = AttrDict(yaml.safe_load(f))
    args.decoding_lib = ARGS.decoding_lib
    args.use_fp16_decoding = ARGS.use_fp16_decoding
    args.enable_fast_encoder = ARGS.enable_fast_encoder
    args.use_fp16_encoder = ARGS.use_fp16_encoder
    args.decoding_strategy = ARGS.decoding_strategy
    args.beam_size = ARGS.beam_size
    args.topk = ARGS.topk
    args.topp = ARGS.topp
    args.benchmark = ARGS.benchmark

    if ARGS.vocab_file is not None:
        args.src_vocab = ARGS.vocab_file
        args.trg_vocab = ARGS.vocab_file
        args.joined_dictionary = True
    elif ARGS.src_vocab is not None and ARGS.trg_vocab is None:
        args.vocab_file = args.trg_vocab = args.src_vocab = ARGS.src_vocab
        args.joined_dictionary = True
    elif ARGS.src_vocab is None and ARGS.trg_vocab is not None:
        args.vocab_file = args.trg_vocab = args.src_vocab = ARGS.trg_vocab
        args.joined_dictionary = True
    else:
        args.src_vocab = ARGS.src_vocab
        args.trg_vocab = ARGS.trg_vocab
        args.joined_dictionary = not (
            args.src_vocab is not None and args.trg_vocab is not None and args.src_vocab != args.trg_vocab
        )
    if args.weight_sharing != args.joined_dictionary:
        if args.weight_sharing:
            raise ValueError("The src_vocab and trg_vocab must be consistency when weight_sharing is True. ")
        else:
            raise ValueError(
                "The src_vocab and trg_vocab must be specified respectively when weight sharing is False. "
            )

    args.bos_token = ARGS.bos_token
    args.eos_token = ARGS.eos_token
    args.pad_token = ARGS.pad_token
    pprint(args)

    do_predict(args)
