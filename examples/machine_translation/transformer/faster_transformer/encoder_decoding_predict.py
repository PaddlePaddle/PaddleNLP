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
            "beam_search", "beam_search_v2", "topk_sampling", "topp_sampling"
        ],
        help=
        "Decoding strategy. Can be one of ['beam_search', 'topk_sampling', 'topp_sampling']. "
    )
    parser.add_argument("--beam_size", default=4, type=int, help="Beam size. ")
    parser.add_argument("--diversity_rate",
                        default=0.0,
                        type=float,
                        help="The diversity rate for beam search. ")
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
    parser.add_argument("--batch_size",
                        default=None,
                        type=int,
                        help="Batch size. ")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Whether to profile the performance using newstest2014 dataset. ")
    parser.add_argument(
        "--test_file",
        nargs='+',
        default=None,
        type=str,
        help=
        "The file for testing. Normally, it shouldn't be set and in this case, the default WMT14 dataset will be used to process testing."
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
    place = paddle.set_device(place)

    # Define data loader
    # NOTE: Data yielded by DataLoader may be on CUDAPinnedPlace,
    # but custom op doesn't support CUDAPinnedPlace. Hence,
    # disable using CUDAPinnedPlace in DataLoader.
    paddle.fluid.reader.use_pinned_memory(False)
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
        decoding_strategy=args.decoding_strategy,
        beam_size=args.beam_size,
        max_out_len=args.max_out_len,
        diversity_rate=args.diversity_rate,
        decoding_lib=args.decoding_lib,
        use_fp16_decoding=args.use_fp16_decoding,
        enable_faster_encoder=args.enable_faster_encoder,
        use_fp16_encoder=args.use_fp16_encoder)

    # Set evaluate mode
    transformer.eval()

    # Load checkpoint.
    transformer.load(init_from_params=os.path.join(args.init_from_params,
                                                   "transformer.pdparams"))

    f = open(args.output_file, "w")
    with paddle.no_grad():
        if args.profile:
            import time
            start = time.time()
        for (src_word, ) in test_loader:
            finished_seq = transformer(src_word=src_word)
            if not args.profile:
                if args.decoding_strategy == "beam_search" or args.decoding_strategy == "beam_search_v2":
                    finished_seq = finished_seq.numpy().transpose([1, 2, 0])
                elif args.decoding_strategy == "topk_sampling" or args.decoding_strategy == "topp_sampling":
                    finished_seq = np.expand_dims(
                        finished_seq.numpy().transpose([1, 0]), axis=1)
                for ins in finished_seq:
                    for beam_idx, beam in enumerate(ins):
                        if beam_idx >= args.n_best:
                            break
                        id_list = post_process_seq(beam, args.bos_idx,
                                                   args.eos_idx)
                        word_list = to_tokens(id_list)
                        sequence = " ".join(word_list) + "\n"
                        f.write(sequence)
        if args.profile:
            if args.decoding_strategy == "beam_search" or args.decoding_strategy == "beam_search_v2":
                logger.info(
                    "Setting info: batch size: {}, beam size: {}, use fp16: {}. "
                    .format(args.infer_batch_size, args.beam_size,
                            args.use_fp16_decoding))
            elif args.decoding_strategy == "topk_sampling":
                logger.info(
                    "Setting info: batch size: {}, topk: {}, use fp16: {}. ".
                    format(args.infer_batch_size, args.topk,
                           args.use_fp16_decoding))
            elif args.decoding_strategy == "topp_sampling":
                logger.info(
                    "Setting info: batch size: {}, topp: {}, use fp16: {}. ".
                    format(args.infer_batch_size, args.topp,
                           args.use_fp16_decoding))
            paddle.device.cuda.synchronize(place)
            logger.info("Average time latency is {} ms/batch. ".format(
                (time.time() - start) / len(test_loader) * 1000))


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
    args.diversity_rate = ARGS.diversity_rate
    args.topk = ARGS.topk
    args.topp = ARGS.topp
    args.profile = ARGS.profile
    args.benchmark = ARGS.benchmark
    if ARGS.batch_size:
        args.infer_batch_size = ARGS.batch_size
    args.test_file = ARGS.test_file
    args.vocab_file = ARGS.vocab_file
    args.unk_token = ARGS.unk_token
    args.bos_token = ARGS.bos_token
    args.eos_token = ARGS.eos_token
    pprint(args)

    do_predict(args)
