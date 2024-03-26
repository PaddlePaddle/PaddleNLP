# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import sentencepiece as spm


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain_files_dir", default=None, required=True, help="The first-level directory of pre-training data."
    )
    parser.add_argument("--model_prefix", default=None, required=True, help="The model_prefix of the tokenizer model.")
    parser.add_argument(
        "--model_type", default="unigram", required=False, help="The model_type used to train the tokenizer model ."
    )
    parser.add_argument("--vocab_size", default=30000, required=False, help="The vocab_size of the tokenizer model.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    dirnames = os.listdir(args.pretrain_files_dir)
    print("starting to merge all pretraining files into one file named all_pretrain_data.txt")
    with open(os.path.join(args.pretrain_files_dir, "all_pretrain_data.txt"), "w") as f:
        for dirname in dirnames:
            if dirname == "all_pretrain_data.txt":
                continue
            dirpath = os.path.join(args.pretrain_files_dir, dirname)
            filenames = os.listdir(dirpath)
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                with open(filepath, "r") as f_1:
                    lines = f_1.readlines()
                    for line in lines:
                        f.write(line)
    print("merging files precedure Done.\n")
    print("straing to train tokenizer model.\n")
    spm.SentencePieceTrainer.train(
        input=os.path.join(args.pretrain_files_dir, "all_pretrain_data.txt"),
        input_format="text",
        model_prefix=args.model_prefix,
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        character_coverage=0.9995,
        num_threads=32,
        train_extremely_large_corpus="true",
    )


if __name__ == "__main__":
    main()
