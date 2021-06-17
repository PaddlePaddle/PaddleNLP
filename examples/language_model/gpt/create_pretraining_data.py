# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

import os
import re
import argparse
import json
import multiprocessing

import numpy as np
from paddlenlp.transformers import GPTTokenizer
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path', type=str, required=True, help='Path to input JSON')
    parser.add_argument(
        '--model_name', type=str, required=True, help='What model to use.')
    parser.add_argument(
        '--append_eos',
        action='store_true',
        help='Append an <eod> token to the end of a document.')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes to launch')
    args = parser.parse_args()
    return args


class Converter(object):
    def __init__(self, model_name, append_eos):
        self.append_eos = append_eos
        self.tokenizer = GPTTokenizer.from_pretrained(model_name)
        self.eos_id = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)

    def encode(self, text):
        tokens = self.tokenizer(text)["input_ids"]
        if self.append_eos:
            tokens.append(self.eos_id)
        return tokens, len(tokens)


def main():
    args = get_args()
    file_paths = []
    if os.path.isfile(args.input_path):
        file_paths.append(args.input_path)
    else:
        for root, _, fs in os.walk(args.input_path):
            for f in fs:
                file_paths.append(os.path.join(root, f))
    all_doc_ids = []
    lens = []
    convert = Converter(args.model_name, args.append_eos)
    pool = multiprocessing.Pool(args.workers)
    if convert.vocab_size < 65500:
        save_dtype = np.uint16
    else:
        save_dtype = np.int32

    for file_path in tqdm(file_paths):
        text = open(file_path, 'r', encoding='utf-8').read()
        text = re.sub('[\n]+', '\n', text)
        text = re.sub('[ ]+', ' ', text)
        encoded_docs = pool.imap(convert.encode, [text], 25)
        for tokens, sizes in encoded_docs:
            all_doc_ids.extend(tokens)
            lens.append(sizes)
    all_doc_ids = np.array(all_doc_ids, dtype=save_dtype)
    lens = np.array(lens, dtype=save_dtype)
    np.savez(args.input_path + "_ids.npz", ids=all_doc_ids, lens=lens)


if __name__ == "__main__":
    main()
