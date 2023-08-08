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

import numpy as np
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
        choices=["beam_search", "beam_search_v2", "topk_sampling", "topp_sampling"],
        help="Decoding strategy. Can be one of ['beam_search', 'topk_sampling', 'topp_sampling']. ",
    )
    parser.add_argument("--beam_size", default=4, type=int, help="Beam size. ")
    parser.add_argument("--diversity_rate", default=0.0, type=float, help="The diversity rate for beam search. ")
    parser.add_argument("--topk", default=4, type=int, help="The k value for topk_sampling. Default is 4. ")
    parser.add_argument(
        "--topp",
        default=0.0,
        type=float,
        help="The probability threshold for topp_sampling. Default is 0.0 which means it won't go through topp_sampling. ",
    )
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size. ")
    parser.add_argument(
        "--profile", action="store_true", help="Whether to profile the performance using newstest2014 dataset. "
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The dir of train, dev and test datasets. If data_dir is given, train_file and dev_file and test_file will be replaced by data_dir/[train|dev|test].\{src_lang\}-\{trg_lang\}.[\{src_lang\}|\{trg_lang\}]. ",
    )
    parser.add_argument(
        "--test_file",
        nargs="+",
        default=None,
        type=str,
        help="The files for test. Can be set by using --test_file source_language_file. If it's None, the default WMT14 en-de dataset will be used. ",
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
    parser.add_argument("-s", "--src_lang", default=None, type=str, help="Source language. ")
    parser.add_argument("-t", "--trg_lang", default=None, type=str, help="Target language. ")
    parser.add_argument(
        "--unk_token",
        default=None,
        type=str,
        help="The unknown token. It should be provided when use custom vocab_file. ",
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


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [idx for idx in seq[: eos_pos + 1] if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)]
    return seq


def do_predict(args):
    place = "gpu"
    place = paddle.set_device(place)

    # Define data loader
    # NOTE: Data yielded by DataLoader may be on CUDAPinnedPlace,
    # but custom op doesn't support CUDAPinnedPlace. Hence,
    # disable using CUDAPinnedPlace in DataLoader.
    paddle.io.reader.use_pinned_memory(False)
    test_loader, to_tokens = reader.create_infer_loader(args)

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
        diversity_rate=args.diversity_rate,
        decoding_lib=args.decoding_lib,
        use_fp16_decoding=args.use_fp16_decoding,
        enable_fast_encoder=args.enable_fast_encoder,
        use_fp16_encoder=args.use_fp16_encoder,
    )

    # Set evaluate mode
    transformer.eval()

    # Load checkpoint.
    transformer.load(init_from_params=os.path.join(args.init_from_params, "transformer.pdparams"))

    # Providing model_dict still works.
    # state_dict = paddle.load(os.path.join(args.init_from_params,
    #                          "transformer.pdparams"))
    # transformer.load(state_dict=state_dict)

    f = open(args.output_file, "w")
    with paddle.no_grad():
        if args.profile:
            import time

            start = time.time()
        for (src_word,) in test_loader:
            finished_seq = transformer(src_word=src_word)
            if not args.profile:
                if args.decoding_strategy == "beam_search" or args.decoding_strategy == "beam_search_v2":
                    finished_seq = finished_seq.numpy().transpose([1, 2, 0])
                elif args.decoding_strategy == "topk_sampling" or args.decoding_strategy == "topp_sampling":
                    finished_seq = np.expand_dims(finished_seq.numpy().transpose([1, 0]), axis=1)
                for ins in finished_seq:
                    for beam_idx, beam in enumerate(ins):
                        if beam_idx >= args.n_best:
                            break
                        id_list = post_process_seq(beam, args.bos_idx, args.eos_idx)
                        word_list = to_tokens(id_list)
                        sequence = " ".join(word_list) + "\n"
                        f.write(sequence)
        if args.profile:
            if args.decoding_strategy == "beam_search" or args.decoding_strategy == "beam_search_v2":
                logger.info(
                    "Setting info: batch size: {}, beam size: {}, use fp16: {}. ".format(
                        args.infer_batch_size, args.beam_size, args.use_fp16_decoding
                    )
                )
            elif args.decoding_strategy == "topk_sampling":
                logger.info(
                    "Setting info: batch size: {}, topk: {}, use fp16: {}. ".format(
                        args.infer_batch_size, args.topk, args.use_fp16_decoding
                    )
                )
            elif args.decoding_strategy == "topp_sampling":
                logger.info(
                    "Setting info: batch size: {}, topp: {}, use fp16: {}. ".format(
                        args.infer_batch_size, args.topp, args.use_fp16_decoding
                    )
                )
            paddle.device.cuda.synchronize(place)
            logger.info(
                "Average time latency is {} ms/batch. ".format((time.time() - start) / len(test_loader) * 1000)
            )


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
    args.diversity_rate = ARGS.diversity_rate
    args.topk = ARGS.topk
    args.topp = ARGS.topp
    args.profile = ARGS.profile
    args.benchmark = ARGS.benchmark
    if ARGS.batch_size:
        args.infer_batch_size = ARGS.batch_size
    args.data_dir = ARGS.data_dir
    args.test_file = ARGS.test_file

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

    if ARGS.src_lang is not None:
        args.src_lang = ARGS.src_lang
    if ARGS.trg_lang is not None:
        args.trg_lang = ARGS.trg_lang

    args.unk_token = ARGS.unk_token
    args.bos_token = ARGS.bos_token
    args.eos_token = ARGS.eos_token
    args.pad_token = ARGS.pad_token
    pprint(args)

    do_predict(args)
