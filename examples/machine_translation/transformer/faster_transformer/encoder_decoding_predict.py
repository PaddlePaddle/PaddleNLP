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
        "--decoding_lib",
        default="../../../../paddlenlp/ops/build/lib/libdecoding_op.so",
        type=str,
        help="Path of libdecoding_op.so. ")
    parser.add_argument(
        "--use_fp16_decoding",
        action="store_true",
        help="Whether to use fp16 decoding to predict. ")
    parser.add_argument(
        "--decoding_strategy",
        default="beam_search",
        type=str,
        choices=["beam_search", "topk_sampling", "topp_sampling"],
        help="Decoding strategy. Can be one of ['beam_search', 'topk_sampling', 'topp_sampling']. "
    )
    parser.add_argument("--beam_size", default=5, type=int, help="Beam size. ")
    parser.add_argument(
        "--topk",
        default=4,
        type=int,
        help="The k value for topk_sampling. Default is 4. ")
    parser.add_argument(
        "--topp",
        default=0.0,
        type=float,
        help="The probability threshold for topp_sampling. Default is 0.0 which means it won't go through topp_sampling. "
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
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def do_predict(args):
    place = "gpu"
    paddle.set_device(place)

    # Define data loader
    test_loader, to_tokens = reader.create_infer_loader(args)

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
        max_out_len=args.max_out_len,
        decoding_lib=args.decoding_lib,
        use_fp16_decoding=args.use_fp16_decoding)

    # Set evaluate mode
    transformer.eval()

    # Load checkpoint.
    transformer.load(init_from_params=os.path.join(args.init_from_params,
                                                   "transformer.pdparams"))

    f = open(args.output_file, "w")
    with paddle.no_grad():
        for (src_word, ) in test_loader:
            finished_seq = transformer(src_word=src_word)
            if args.decoding_strategy == "beam_search":
                finished_seq = finished_seq.numpy().transpose([1, 2, 0])
            elif args.decoding_strategy == "topk_sampling" or args.decoding_strategy == "topp_sampling":
                finished_seq = np.expand_dims(
                    finished_seq.numpy().transpose([1, 0]), axis=1)
            for ins in finished_seq:
                for beam_idx, beam in enumerate(ins):
                    if beam_idx >= args.n_best:
                        break
                    id_list = post_process_seq(beam, args.bos_idx, args.eos_idx)
                    word_list = to_tokens(id_list)
                    sequence = " ".join(word_list) + "\n"
                    f.write(sequence)


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)
    args.decoding_lib = ARGS.decoding_lib
    args.use_fp16_decoding = ARGS.use_fp16_decoding
    args.decoding_strategy = ARGS.decoding_strategy
    args.beam_size = ARGS.beam_size
    args.topk = ARGS.topk
    args.topp = ARGS.topp
    args.benchmark = False

    do_predict(args)
