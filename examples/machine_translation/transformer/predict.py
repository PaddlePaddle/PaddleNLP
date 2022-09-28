import os
import yaml
import logging
import argparse
import numpy as np
from pprint import pprint
from attrdict import AttrDict

import paddle
from paddlenlp.ops import TransformerGenerator

import reader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="./configs/transformer.big.yaml",
                        type=str,
                        help="Path of the config file. ")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=
        "Whether to print logs on each cards and use benchmark vocab. Normally, not necessary to set --benchmark. "
    )
    parser.add_argument(
        "--test_file",
        nargs='+',
        default=None,
        type=str,
        help=
        "The file for testing. Normally, it shouldn't be set and in this case, the default WMT14 dataset will be used to process testing."
    )
    parser.add_argument("--without_ft",
                        action="store_true",
                        help="Whether to use FasterTransformer to do predict. ")
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
    if args.device == "gpu":
        place = "gpu"
    else:
        place = "cpu"

    paddle.set_device(place)

    # Define data loader
    test_loader, to_tokens = reader.create_infer_loader(args)

    # Define model
    # `TransformerGenerator` automatically chioces using `FasterTransformer`
    # (with jit building) or the slower verison `InferTransformerModel`.
    transformer = TransformerGenerator(
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
        beam_size=args.beam_size,
        max_out_len=args.max_out_len,
        use_ft=not args.without_ft,
        beam_search_version=args.beam_search_version,
        rel_len=args.use_rel_len,  # only works when using FT or beam search v2
        alpha=args.alpha,  # only works when using beam search v2
        diversity_rate=args.diversity_rate,  # only works when using FT
        use_fp16_decoding=False)  # only works when using FT

    # Load the trained model
    assert args.init_from_params, (
        "Please set init_from_params to load the infer model.")

    transformer.load(os.path.join(args.init_from_params,
                                  "transformer.pdparams"))

    # Set evaluate mode
    transformer.eval()

    f = open(args.output_file, "w", encoding="utf-8")
    with paddle.no_grad():
        for (src_word, ) in test_loader:
            # When `output_time_major` argument is `True` for TransformerGenerator,
            # the shape of finished_seq is `[seq_len, batch_size, beam_size]`
            # for beam search v1 or `[seq_len, batch_size, beam_size * 2]` for
            # beam search v2.
            finished_seq = transformer(src_word=src_word)
            finished_seq = finished_seq.numpy().transpose([1, 2, 0])
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
    args.benchmark = ARGS.benchmark
    args.test_file = ARGS.test_file
    args.without_ft = ARGS.without_ft
    args.vocab_file = ARGS.vocab_file
    args.unk_token = ARGS.unk_token
    args.bos_token = ARGS.bos_token
    args.eos_token = ARGS.eos_token
    pprint(args)

    do_predict(args)
