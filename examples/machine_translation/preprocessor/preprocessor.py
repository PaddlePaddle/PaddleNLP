# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import argparse
from itertools import zip_longest
from pprint import pprint

from paddlenlp.data import Vocab
from paddlenlp.utils.log import logger

import fastBPE


def get_preprocessing_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s",
                        "--source_lang",
                        default=None,
                        type=str,
                        help="Source language. ")
    parser.add_argument("-t",
                        "--target_lang",
                        default=None,
                        type=str,
                        help="Target language. ")
    parser.add_argument(
        "--train_pref",
        default=None,
        type=str,
        help="The prefix for train file and also used to save dict. ")
    parser.add_argument(
        "--dev_pref",
        default=None,
        type=str,
        help="The prefixes for dev file and use comma to separate. "
        "(words missing from train set are replaced with <unk>)")
    parser.add_argument(
        "--test_pref",
        default=None,
        type=str,
        help="The prefixes for test file and use comma to separate. "
        "(words missing from train set are replaced with <unk>)")
    parser.add_argument(
        "--dest_dir",
        default="./data/",
        type=str,
        help="The destination dir to save processed train, dev and test file. ")
    parser.add_argument(
        "--threshold_tgt",
        default=0,
        type=int,
        help="Map words appearing less than threshold times to unknown. ")
    parser.add_argument(
        "--threshold_src",
        default=0,
        type=int,
        help="Map words appearing less than threshold times to unknown. ")
    parser.add_argument("--src_vocab",
                        default=None,
                        type=str,
                        help="Reuse given source dictionary. ")
    parser.add_argument("--tgt_vocab",
                        default=None,
                        type=str,
                        help="Reuse given target dictionary. ")
    parser.add_argument("--nwords_tgt",
                        default=None,
                        type=int,
                        help="The number of target words to retain. ")
    parser.add_argument("--nwords_src",
                        default=None,
                        type=int,
                        help="The number of source words to retain. ")
    parser.add_argument("--align_file",
                        default=None,
                        help="An alignment file (optional). ")
    parser.add_argument("--joined_dictionary",
                        action="store_true",
                        help="Generate joined dictionary. ")
    parser.add_argument("--only_source",
                        action="store_true",
                        help="Only process the source language. ")
    parser.add_argument(
        "--dict_only",
        action='store_true',
        help="Only builds a dictionary and then exits if it's set.")
    parser.add_argument("--bos_token",
                        default="<s>",
                        type=str,
                        help="bos_token. ")
    parser.add_argument("--eos_token",
                        default="</s>",
                        type=str,
                        help="eos_token. ")
    parser.add_argument(
        "--pad_token",
        default=None,
        type=str,
        help=
        "The token used for padding. If it's None, the bos_token will be used. Defaults to None. "
    )
    parser.add_argument("--unk_token",
                        default="<unk>",
                        type=str,
                        help="Unk token. ")
    parser.add_argument("--apply_bpe",
                        action="store_true",
                        help="Whether to apply bpe to the files. ")
    parser.add_argument(
        "--bpe_code",
        default=None,
        type=str,
        help="The code used for bpe. Must be provided when --apply_bpe is set. "
    )

    args = parser.parse_args()
    return args


def _train_path(lang, train_pref):
    return "{}{}".format(train_pref, ("." + lang) if lang else "")


def _dev_path(lang, dev_pref):
    return "{}{}".format(dev_pref, ("." + lang) if lang else "")


def _test_path(lang, test_pref):
    return "{}{}".format(test_pref, ("." + lang) if lang else "")


def _file_name(prefix, lang):
    fname = prefix
    if lang is not None:
        fname += ".{lang}".format(lang=lang)
    return fname


def _dest_path(prefix, lang, dest_dir):
    return os.path.join(dest_dir, _file_name(prefix, lang))


def _dict_path(lang, dest_dir):
    return _dest_path("dict", lang, dest_dir) + ".txt"


def _build_dictionary(filenames, args, src=False, tgt=False):
    assert src ^ tgt, "src and tgt cannot be both True or both False. "

    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]

    tokens = []
    for file in filenames:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                tokens.append(line.strip().split())

    return Vocab.build_vocab(
        tokens,
        max_size=args.nwords_src if src else args.nwords_tgt,
        min_freq=args.threshold_src if src else args.threshold_tgt,
        unk_token=args.unk_token,
        pad_token=args.pad_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token)


def _make_dataset(vocab, input_prefix, output_prefix, lang, args):
    # Copy original text file to destination folder
    output_text_file = _dest_path(
        output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
        lang,
        args.dest_dir,
    )

    shutil.copyfile(_file_name(input_prefix, lang), output_text_file)


def _make_all(lang, vocab, args):
    if args.train_pref:
        _make_dataset(vocab, args.train_pref, "train", lang, args=args)

    if args.dev_pref:
        for k, dev_pref in enumerate(args.dev_pref.split(",")):
            out_prefix = "dev{}".format(k) if k > 0 else "dev"
            _make_dataset(vocab, dev_pref, out_prefix, lang, args=args)

    if args.test_pref:
        for k, test_pref in enumerate(args.test_pref.split(",")):
            out_prefix = "test{}".format(k) if k > 0 else "test"
            _make_dataset(vocab, test_pref, out_prefix, lang, args=args)


def _align_files(args, src_vocab, tgt_vocab):
    assert args.train_pref, "--train_pref must be set if --align_file is specified"
    src_file_name = _train_path(args.source_lang, args.train_pref)
    tgt_file_name = _train_path(args.target_lang, args.train_pref)
    freq_map = {}

    with open(args.align_file, "r", encoding="utf-8") as align_file:
        with open(src_file_name, "r", encoding="utf-8") as src_file:
            with open(tgt_file_name, "r", encoding="utf-8") as tgt_file:
                for a, s, t in zip_longest(align_file, src_file, tgt_file):
                    si = src_vocab.to_indices(s)
                    ti = tgt_vocab.to_indices(t)
                    ai = list(map(lambda x: tuple(x.split("\t")), a.split()))
                    for sai, tai in ai:
                        src_idx = si[int(sai)]
                        tgt_idx = ti[int(tai)]
                        if src_idx != src_vocab.get_unk_token_id(
                        ) and tgt_idx != tgt_vocab.get_unk_token_id():
                            assert src_idx != src_vocab.get_pad_token_id()
                            assert src_idx != src_vocab.get_eos_token_id()
                            assert tgt_idx != tgt_vocab.get_pad_token_id()
                            assert tgt_idx != tgt_vocab.get_eos_token_id()
                            if src_idx not in freq_map:
                                freq_map[src_idx] = {}
                            if tgt_idx not in freq_map[src_idx]:
                                freq_map[src_idx][tgt_idx] = 1
                            else:
                                freq_map[src_idx][tgt_idx] += 1

    align_dict = {}
    for src_idx in freq_map.keys():
        align_dict[src_idx] = max(freq_map[src_idx], key=freq_map[src_idx].get)

    with open(
            os.path.join(
                args.dest_dir,
                "alignment.{}-{}.txt".format(args.source_lang,
                                             args.target_lang),
            ),
            "w",
            encoding="utf-8",
    ) as f:
        for k, v in align_dict.items():
            print("{} {}".format(src_vocab[k], tgt_vocab[v]), file=f)


def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)
    pprint(args)

    if args.apply_bpe:
        bpe = fastBPE.fastBPE(args.bpe_code)
        filenames = [
            _train_path(lang, args.train_pref)
            for lang in [args.source_lang, args.target_lang]
        ]
        for k, dev_pref in enumerate(args.dev_pref.split(",")):
            filenames.extend([
                _dev_path(lang, args.dev_pref)
                for lang in [args.source_lang, args.target_lang]
            ])
        for k, test_pref in enumerate(args.test_pref.split(",")):
            filenames.extend([
                _test_path(lang, args.test_pref)
                for lang in [args.source_lang, args.target_lang]
            ])

        for file in filenames:
            sequences = []
            with open(file, "r") as f:
                lines = f.readlines()
                for seq in lines:
                    sequences.append(seq.strip())

            bpe_sequences = bpe.apply(sequences)
            os.makedirs(os.path.join(args.train_pref, "tmp_bpe"), exist_ok=True)
            shutil.copyfile(
                file,
                os.path.join(args.train_pref, "tmp_bpe",
                             os.path.split(file)[-1]))

            with open(file, "w") as f:
                for bpe_seq in bpe_sequences:
                    f.write(bpe_seq + "\n")

    # build dictionaries
    target = not args.only_source

    if not args.src_vocab and os.path.exists(
            _dict_path(args.source_lang, args.dest_dir)):
        raise FileExistsError(_dict_path(args.source_lang, args.dest_dir))

    if (target and not args.tgt_vocab
            and os.path.exists(_dict_path(args.target_lang, args.dest_dir))):
        raise FileExistsError(_dict_path(args.target_lang, args.dest_dir))

    if args.joined_dictionary:
        assert (
            not args.src_vocab or not args.tgt_vocab
        ), "Cannot use both --src_vocab and --tgt_vocab with --joined_dictionary"

        if args.src_vocab:
            src_vocab = Vocab.load_vocabulary(filepath=args.src_vocab,
                                              unk_token=args.unk_token,
                                              bos_token=args.bos_token,
                                              eos_token=args.eos_token,
                                              pad_token=args.pad_token)
        elif args.tgt_vocab:
            src_vocab = Vocab.load_vocabulary(filepath=args.tgt_vocab,
                                              unk_token=args.unk_token,
                                              bos_token=args.bos_token,
                                              eos_token=args.eos_token,
                                              pad_token=args.pad_token)
        else:
            assert (
                args.train_pref
            ), "--train_pref must be set if --src_vocab is not specified. "
            src_vocab = _build_dictionary([
                _train_path(lang, args.train_pref)
                for lang in [args.source_lang, args.target_lang]
            ],
                                          args=args,
                                          src=True)

        tgt_vocab = src_vocab
    else:
        if args.src_vocab:
            src_vocab = Vocab.load_vocabulary(filepath=args.src_vocab,
                                              unk_token=args.unk_token,
                                              bos_token=args.bos_token,
                                              eos_token=args.eos_token,
                                              pad_token=args.pad_token)
        else:
            assert (
                args.train_pref
            ), "--train_pref must be set if --src_vocab is not specified"
            src_vocab = _build_dictionary(
                [_train_path(args.source_lang, args.train_pref)],
                args=args,
                src=True)

        if target:
            if args.tgt_vocab:
                tgt_vocab = Vocab.load_vocabulary(filepath=args.tgt_vocab,
                                                  unk_token=args.unk_token,
                                                  bos_token=args.bos_token,
                                                  eos_token=args.eos_token,
                                                  pad_token=args.pad_token)
            else:
                assert (
                    args.train_pref
                ), "--train_pref must be set if --tgt_vocab is not specified"
                tgt_vocab = _build_dictionary(
                    [_train_path(args.target_lang, args.train_pref)],
                    args=args,
                    tgt=True)
        else:
            tgt_vocab = None

    # save dictionaries
    src_vocab.save_vocabulary(_dict_path(args.source_lang, args.dest_dir))
    if target and tgt_vocab is not None:
        tgt_vocab.save_vocabulary(_dict_path(args.target_lang, args.dest_dir))

    if args.dict_only:
        return

    _make_all(args.source_lang, src_vocab, args)
    if target:
        _make_all(args.target_lang, tgt_vocab, args)

    logger.info("Wrote preprocessed data to {}".format(args.dest_dir))

    if args.align_file:
        _align_files(args, src_vocab=src_vocab, tgt_vocab=tgt_vocab)


if __name__ == "__main__":
    args = get_preprocessing_parser()
    main(args)
