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
from pprint import pprint

import paddle
import reader
import yaml
from easydict import EasyDict as AttrDict

from paddlenlp.ops import TransformerGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="./configs/transformer.big.yaml", type=str, help="Path of the config file. "
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Whether to print logs on each cards and use benchmark vocab. Normally, not necessary to set --benchmark. ",
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
    parser.add_argument("--without_ft", action="store_true", help="Whether to use FastGeneration to do predict. ")
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
    parser.add_argument(
        "--device", default="gpu", choices=["gpu", "cpu", "xpu", "npu", "mlu"], help="Device selected for inference."
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
    if args.device == "gpu":
        place = "gpu"
    elif args.device == "xpu":
        place = "xpu"
    elif args.device == "npu":
        place = "npu"
    elif args.device == "mlu":
        place = "mlu"
    else:
        place = "cpu"

    paddle.set_device(place)

    # Define data loader
    test_loader, to_tokens = reader.create_infer_loader(args)

    # Define model
    # `TransformerGenerator` automatically chioces using `FastGeneration`
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
        pad_id=args.pad_idx,
        beam_size=args.beam_size,
        max_out_len=args.max_out_len,
        use_ft=not args.without_ft,
        beam_search_version=args.beam_search_version,
        normalize_before=args.get("normalize_before", True),
        rel_len=args.use_rel_len,  # only works when using FT or beam search v2
        alpha=args.alpha,  # only works when using beam search v2
        diversity_rate=args.diversity_rate,  # only works when using FT
        use_fp16_decoding=False,
    )  # only works when using FT

    # Load the trained model
    assert args.init_from_params, "Please set init_from_params to load the infer model."

    transformer.load(os.path.join(args.init_from_params, "transformer.pdparams"))

    # Providing model_dict still works.
    # state_dict = paddle.load(os.path.join(args.init_from_params,
    #                          "transformer.pdparams"))
    # transformer.load(state_dict=state_dict)

    # Set evaluate mode
    transformer.eval()

    f = open(args.output_file, "w", encoding="utf-8")
    with paddle.no_grad():
        for (src_word,) in test_loader:
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
    with open(yaml_file, "rt") as f:
        args = AttrDict(yaml.safe_load(f))
    args.benchmark = ARGS.benchmark
    args.without_ft = ARGS.without_ft
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

    args.device = ARGS.device
    pprint(args)

    do_predict(args)
