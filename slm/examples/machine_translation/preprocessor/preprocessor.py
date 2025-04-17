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
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil
from itertools import zip_longest
from pprint import pprint

from paddlenlp.data import Vocab
from paddlenlp.utils.log import logger


def get_preprocessing_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--src_lang", default=None, type=str, help="Source language. ")
    parser.add_argument("-t", "--trg_lang", default=None, type=str, help="Target language. ")
    parser.add_argument(
        "--train_pref", default=None, type=str, help="The prefix for train file and also used to save dict. "
    )
    parser.add_argument(
        "--dev_pref",
        default=None,
        type=str,
        help="The prefixes for dev file and use comma to separate. "
        "(words missing from train set are replaced with <unk>)",
    )
    parser.add_argument(
        "--test_pref",
        default=None,
        type=str,
        help="The prefixes for test file and use comma to separate. "
        "(words missing from train set are replaced with <unk>)",
    )
    parser.add_argument(
        "--dest_dir",
        default="./data/",
        type=str,
        help="The destination dir to save processed train, dev and test file. ",
    )
    parser.add_argument(
        "--threshold_trg", default=0, type=int, help="Map words appearing less than threshold times to unknown. "
    )
    parser.add_argument(
        "--threshold_src", default=0, type=int, help="Map words appearing less than threshold times to unknown. "
    )
    parser.add_argument("--src_vocab", default=None, type=str, help="Reuse given source dictionary. ")
    parser.add_argument("--trg_vocab", default=None, type=str, help="Reuse given target dictionary. ")
    parser.add_argument("--nwords_trg", default=None, type=int, help="The number of target words to retain. ")
    parser.add_argument("--nwords_src", default=None, type=int, help="The number of source words to retain. ")
    parser.add_argument("--align_file", default=None, help="An alignment file (optional). ")
    parser.add_argument("--joined_dictionary", action="store_true", help="Generate joined dictionary. ")
    parser.add_argument("--only_source", action="store_true", help="Only process the source language. ")
    parser.add_argument(
        "--dict_only", action="store_true", help="Only builds a dictionary and then exits if it's set."
    )
    parser.add_argument("--bos_token", default="<s>", type=str, help="bos_token. ")
    parser.add_argument("--eos_token", default="</s>", type=str, help="eos_token. ")
    parser.add_argument(
        "--pad_token",
        default=None,
        type=str,
        help="The token used for padding. If it's None, the bos_token will be used. Defaults to None. ",
    )
    parser.add_argument("--unk_token", default="<unk>", type=str, help="Unk token. ")
    parser.add_argument("--apply_bpe", action="store_true", help="Whether to apply bpe to the files. ")
    parser.add_argument(
        "--bpe_code", default=None, type=str, help="The code used for bpe. Must be provided when --apply_bpe is set. "
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


def _build_dictionary(filenames, args, src=False, trg=False):
    assert src ^ trg, "src and trg cannot be both True or both False. "

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
        max_size=args.nwords_src if src else args.nwords_trg,
        min_freq=args.threshold_src if src else args.threshold_trg,
        unk_token=args.unk_token,
        pad_token=args.pad_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
    )


def _make_dataset(vocab, input_prefix, output_prefix, lang, args):
    # Copy original text file to destination folder
    output_text_file = _dest_path(
        output_prefix + ".{}-{}".format(args.src_lang, args.trg_lang),
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


def _align_files(args, src_vocab, trg_vocab):
    assert args.train_pref, "--train_pref must be set if --align_file is specified"
    src_file_name = _train_path(args.src_lang, args.train_pref)
    trg_file_name = _train_path(args.trg_lang, args.train_pref)
    freq_map = {}

    with open(args.align_file, "r", encoding="utf-8") as align_file:
        with open(src_file_name, "r", encoding="utf-8") as src_file:
            with open(trg_file_name, "r", encoding="utf-8") as trg_file:
                for a, s, t in zip_longest(align_file, src_file, trg_file):
                    si = src_vocab.to_indices(s)
                    ti = trg_vocab.to_indices(t)
                    ai = list(map(lambda x: tuple(x.split("\t")), a.split()))
                    for sai, tai in ai:
                        src_idx = si[int(sai)]
                        trg_idx = ti[int(tai)]
                        if src_idx != src_vocab.get_unk_token_id() and trg_idx != trg_vocab.get_unk_token_id():
                            assert src_idx != src_vocab.get_pad_token_id()
                            assert src_idx != src_vocab.get_eos_token_id()
                            assert trg_idx != trg_vocab.get_pad_token_id()
                            assert trg_idx != trg_vocab.get_eos_token_id()
                            if src_idx not in freq_map:
                                freq_map[src_idx] = {}
                            if trg_idx not in freq_map[src_idx]:
                                freq_map[src_idx][trg_idx] = 1
                            else:
                                freq_map[src_idx][trg_idx] += 1

    align_dict = {}
    for src_idx in freq_map.keys():
        align_dict[src_idx] = max(freq_map[src_idx], key=freq_map[src_idx].get)

    with open(
        os.path.join(
            args.dest_dir,
            "alignment.{}-{}.txt".format(args.src_lang, args.trg_lang),
        ),
        "w",
        encoding="utf-8",
    ) as f:
        for k, v in align_dict.items():
            print("{} {}".format(src_vocab[k], trg_vocab[v]), file=f)


def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)
    pprint(args)

    if args.apply_bpe:
        import fastBPE

        bpe = fastBPE.fastBPE(args.bpe_code)
        filenames = [_train_path(lang, args.train_pref) for lang in [args.src_lang, args.trg_lang]]
        for k, dev_pref in enumerate(args.dev_pref.split(",")):
            filenames.extend([_dev_path(lang, args.dev_pref) for lang in [args.src_lang, args.trg_lang]])
        for k, test_pref in enumerate(args.test_pref.split(",")):
            filenames.extend([_test_path(lang, args.test_pref) for lang in [args.src_lang, args.trg_lang]])

        for file in filenames:
            sequences = []
            with open(file, "r") as f:
                lines = f.readlines()
                for seq in lines:
                    sequences.append(seq.strip())

            bpe_sequences = bpe.apply(sequences)
            os.makedirs(os.path.join(args.train_pref, "tmp_bpe"), exist_ok=True)
            shutil.copyfile(file, os.path.join(args.train_pref, "tmp_bpe", os.path.split(file)[-1]))

            with open(file, "w") as f:
                for bpe_seq in bpe_sequences:
                    f.write(bpe_seq + "\n")

    # build dictionaries
    target = not args.only_source

    if not args.src_vocab and os.path.exists(_dict_path(args.src_lang, args.dest_dir)):
        raise FileExistsError(_dict_path(args.src_lang, args.dest_dir))

    if target and not args.trg_vocab and os.path.exists(_dict_path(args.trg_lang, args.dest_dir)):
        raise FileExistsError(_dict_path(args.trg_lang, args.dest_dir))

    if args.joined_dictionary:
        assert (
            not args.src_vocab or not args.trg_vocab
        ), "Cannot use both --src_vocab and --trg_vocab with --joined_dictionary"

        if args.src_vocab:
            src_vocab = Vocab.load_vocabulary(
                filepath=args.src_vocab,
                unk_token=args.unk_token,
                bos_token=args.bos_token,
                eos_token=args.eos_token,
                pad_token=args.pad_token,
            )
        elif args.trg_vocab:
            src_vocab = Vocab.load_vocabulary(
                filepath=args.trg_vocab,
                unk_token=args.unk_token,
                bos_token=args.bos_token,
                eos_token=args.eos_token,
                pad_token=args.pad_token,
            )
        else:
            assert args.train_pref, "--train_pref must be set if --src_vocab is not specified. "
            src_vocab = _build_dictionary(
                [_train_path(lang, args.train_pref) for lang in [args.src_lang, args.trg_lang]], args=args, src=True
            )

        trg_vocab = src_vocab
    else:
        if args.src_vocab:
            src_vocab = Vocab.load_vocabulary(
                filepath=args.src_vocab,
                unk_token=args.unk_token,
                bos_token=args.bos_token,
                eos_token=args.eos_token,
                pad_token=args.pad_token,
            )
        else:
            assert args.train_pref, "--train_pref must be set if --src_vocab is not specified"
            src_vocab = _build_dictionary([_train_path(args.src_lang, args.train_pref)], args=args, src=True)

        if target:
            if args.trg_vocab:
                trg_vocab = Vocab.load_vocabulary(
                    filepath=args.trg_vocab,
                    unk_token=args.unk_token,
                    bos_token=args.bos_token,
                    eos_token=args.eos_token,
                    pad_token=args.pad_token,
                )
            else:
                assert args.train_pref, "--train_pref must be set if --trg_vocab is not specified"
                trg_vocab = _build_dictionary([_train_path(args.trg_lang, args.train_pref)], args=args, trg=True)
        else:
            trg_vocab = None

    # save dictionaries
    src_vocab.save_vocabulary(_dict_path(args.src_lang, args.dest_dir))
    if target and trg_vocab is not None:
        trg_vocab.save_vocabulary(_dict_path(args.trg_lang, args.dest_dir))

    if args.dict_only:
        return

    _make_all(args.src_lang, src_vocab, args)
    if target:
        _make_all(args.trg_lang, trg_vocab, args)

    logger.info("Wrote preprocessed data to {}".format(args.dest_dir))

    if args.align_file:
        _align_files(args, src_vocab=src_vocab, trg_vocab=trg_vocab)


if __name__ == "__main__":
    args = get_preprocessing_parser()
    main(args)
