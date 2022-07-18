import os
import time
import sys

import argparse
import logging
import numpy as np
import yaml
from attrdict import AttrDict
from pprint import pprint

import paddle

from paddlenlp.transformers import InferTransformerModel

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import reader


def cast_parameters_to_fp32(place, program, scope=None):
    all_parameters = []
    for block in program.blocks:
        all_parameters.extend(block.all_parameters())

    var_scope = scope if scope else paddle.static.global_scope()
    for param in all_parameters:
        tensor = var_scope.find_var(param.name).get_tensor()
        if 'fp16' in str(tensor._dtype()).lower() and \
            'fp32' in str(param.dtype).lower():
            data = np.array(tensor)
            tensor.set(np.float32(data), place)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="../configs/transformer.big.yaml",
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
    paddle.enable_static()
    if args.device == "gpu":
        place = paddle.set_device("gpu")
    else:
        place = paddle.set_device("cpu")

    # Define data loader
    test_loader, to_tokens = reader.create_infer_loader(args)

    test_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(test_program, startup_program):
        src_word = paddle.static.data(name="src_word",
                                      shape=[None, None],
                                      dtype=args.input_dtype)

        # Define model
        transformer = InferTransformerModel(src_vocab_size=args.src_vocab_size,
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
                                            max_out_len=args.max_out_len)

        finished_seq = transformer(src_word=src_word)

    test_program = test_program.clone(for_test=True)

    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    assert (
        args.init_from_params), "must set init_from_params to load parameters"
    paddle.static.load(test_program,
                       os.path.join(args.init_from_params, "transformer"), exe)
    print("finish initing model from params from %s" % (args.init_from_params))

    # cast weights from fp16 to fp32 after loading
    if args.use_pure_fp16:
        cast_parameters_to_fp32(place, test_program)

    f = open(args.output_file, "w")
    for data in test_loader:
        finished_sequence, = exe.run(test_program,
                                     feed={'src_word': data[0]},
                                     fetch_list=finished_seq.name)
        finished_sequence = finished_sequence.transpose([0, 2, 1])
        for ins in finished_sequence:
            for beam_idx, beam in enumerate(ins):
                if beam_idx >= args.n_best:
                    break
                id_list = post_process_seq(beam, args.bos_idx, args.eos_idx)
                word_list = to_tokens(id_list)
                sequence = " ".join(word_list) + "\n"
                f.write(sequence)

    paddle.disable_static()


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
    args.benchmark = ARGS.benchmark
    args.test_file = ARGS.test_file
    args.vocab_file = ARGS.vocab_file
    args.unk_token = ARGS.unk_token
    args.bos_token = ARGS.bos_token
    args.eos_token = ARGS.eos_token
    pprint(args)

    do_predict(args)
