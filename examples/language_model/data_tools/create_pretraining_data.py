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
import sys
import time
import profile

import numpy as np
import paddlenlp.transformers as tfs
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', type=str, required=True, help='What model to use.')
    parser.add_argument(
        '--tokenizer_name',
        type=str,
        required=True,
        choices=[
            'ErnieTokenizer', 'BertTokenizer', 'GPTTokenizer',
            'GPTChineseTokenizer'
        ],
        help='What type of tokenizer to use.')
    group = parser.add_argument_group(title='input data')
    group.add_argument(
        '--input_path', type=str, required=True, help='Path to input JSON')
    group.add_argument(
        '--data_format',
        type=str,
        required=True,
        choices=['JSON', 'RAW', 'RAW_SPLIT'],
        help='What type of the raw string data..')
    group.add_argument(
        '--json_keys',
        nargs='+',
        default=['text'],
        help='For JSON format. Space separate listed of keys to extract from json'
    )
    group.add_argument(
        '--split_dimer',
        type=str,
        default=' ',
        help="For RWA_SPLIT format. Split dimer for tokens.")
    group.add_argument(
        '--append_eos',
        action='store_true',
        help='Append an <eos> token to the end of a document.')
    group.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='Interval between progress updates')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes to launch')
    args = parser.parse_args()
    return args


class Converter(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Converter.tokenizer = getattr(
            tfs, self.args.tokenizer_name).from_pretrained(self.args.model_name)

    def encode(self, json_line):
        text = json.loads(json_line)['text']
        #tokens = self.tokenizer(text)["input_ids"]
        text = re.sub('[\n]+', '\n', text)
        text = re.sub('[ ]+', ' ', text)
        tokens = Converter.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(text))
        if self.args.append_eos:
            tokens.append(Converter.tokenizer.eos_token_id)
        return tokens, len(json_line)


def main():
    args = get_args()
    startup_start = time.time()

    file_paths = []
    if os.path.isfile(args.input_path):
        file_paths.append(args.input_path)
    else:
        for root, _, fs in os.walk(args.input_path):
            for f in fs:
                file_paths.append(os.path.join(root, f))
    all_doc_ids = []
    lens = []
    convert = Converter(args)

    # try tokenizer is availiable
    sample_tokenizer = getattr(
        tfs, args.tokenizer_name).from_pretrained(args.model_name)
    if sample_tokenizer.vocab_size < 2**16 - 1:
        save_dtype = np.uint16
    else:
        save_dtype = np.int32

    pool = multiprocessing.Pool(args.workers, initializer=convert.initializer)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for file_path in tqdm(file_paths):
        text = open(file_path, 'r', encoding='utf-8')
        encoded_docs = pool.imap(convert.encode, text, 25)
        for i, (tokens, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed

            master_start = time.time()
            all_doc_ids.extend(tokens)
            lens.append(len(tokens))

            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(
                    f"Processed {i} documents",
                    f"({i/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)
    pool.close()

    all_doc_ids = np.array(all_doc_ids, dtype=save_dtype)
    lens = np.array(lens, dtype=np.uint32)
    np.savez(args.input_path + "_ids.npz", ids=all_doc_ids, lens=lens)
    print("Total documents num: %d" % len(lens))
    print("Total tokens num: %d" % len(all_doc_ids))
    print("Average tokens per doc: %.2f" % (len(all_doc_ids) / len(lens)))


if __name__ == "__main__":
    main()
    #profile.run("main()", "testprof")
