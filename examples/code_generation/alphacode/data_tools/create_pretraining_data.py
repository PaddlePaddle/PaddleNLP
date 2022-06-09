# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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
import io
import argparse
import json
import multiprocessing
import sys
import time

import numpy as np
from tqdm import tqdm
from paddlenlp.transformers import BartTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='data input/output')
    parser.add_argument('--model_name',
                        type=str,
                        required=True,
                        help='What model to use.')
    group.add_argument('--input_path',
                       type=str,
                       required=True,
                       help='Path to input JSON files.')
    group.add_argument('--output_prefix',
                       type=str,
                       required=True,
                       help='Output prefix to store output file.')
    group.add_argument(
        '--data_format',
        type=str,
        default='text',
        choices=['JSON'],
        help='Only support json format for now. One document per line.')
    group.add_argument(
        '--json_key',
        type=str,
        default='text',
        help=
        'For JSON format. Space separate listed of keys to extract from json')
    group = parser.add_argument_group(title='common config')
    group.add_argument('--append_eos',
                       action='store_true',
                       help='Append an <eos> token to the end of a document.')
    group.add_argument('--log_interval',
                       type=int,
                       default=100,
                       help='Interval between progress updates')
    group.add_argument('--workers',
                       type=int,
                       default=1,
                       help='Number of worker processes to launch')
    args = parser.parse_args()
    return args


class IdentitySplitter(object):

    def tokenize(self, *text):
        return text


class Converter(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        Converter.tokenizer = BartTokenizer.from_pretrained(
            self.args.model_name)

        def process(text):
            tokens = Converter.tokenizer.tokenize(text)
            tokens = Converter.tokenizer.convert_tokens_to_ids(tokens)
            return tokens

        Converter.process = process
        Converter.splitter = IdentitySplitter()

    def encode(self, json_line):
        try:
            text = json.loads(json_line)[self.args.json_key]
        except:
            print(f'Failed json parse: {json_line}')
            return [], 0
        doc_ids = []
        for sentence in Converter.splitter.tokenize(text):
            sentence_ids = Converter.process(sentence)
            if len(sentence_ids) > 0:
                doc_ids.append(sentence_ids)

        if len(doc_ids) > 0 and self.args.append_eos:
            doc_ids[-1].append(Converter.tokenizer.eos_token_id)

        return doc_ids, len(text.encode("utf-8"))


def main():
    args = get_args()

    file_paths = []
    if os.path.isfile(args.input_path):
        file_paths.append(args.input_path)
    else:
        for root, _, fs in os.walk(args.input_path):
            for f in fs:
                file_paths.append(os.path.join(root, f))
    convert = Converter(args)

    # Try tokenizer is availiable
    sample_tokenizer = BartTokenizer.from_pretrained(args.model_name)
    print(f"Vocab size: {sample_tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    if sample_tokenizer.vocab_size < 2**16 - 1:
        save_dtype = np.uint16
    else:
        save_dtype = np.int32

    pool = multiprocessing.Pool(args.workers, initializer=convert.initializer)

    # We use BytesIO to store the ids.
    token_ids_stream = io.BytesIO()
    sentlens_stream = io.BytesIO()
    doc_cumsum_stream = io.BytesIO()
    doc_cumsum_stream.write((0).to_bytes(8, byteorder='little', signed=True))

    sent_count = 0
    file_paths.sort()

    step = 0
    total_bytes_processed = 0
    startup_start = time.time()
    for file_path in tqdm(file_paths):
        if file_path.endswith(".zst"):
            import zstandard
            cctx = zstandard.ZstdDecompressor()
            fh = open(file_path, 'rb')
            text = io.BufferedReader(cctx.stream_reader(fh))
        elif file_path.endswith(".jsonl"):
            text = open(file_path, 'r', encoding='utf-8')
        else:
            print("Unexpected data format, skiped %s" % file_path)
            continue
        encoded_docs = pool.imap(convert.encode, text, 256)
        print("Processing %s" % file_path)
        for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
            step += 1
            total_bytes_processed += bytes_processed
            if len(doc) == 0:
                continue
            for sentence in doc:
                sentence_len = len(sentence)
                if sentence_len == 0:
                    continue
                sentlens_stream.write(
                    sentence_len.to_bytes(4, byteorder='little', signed=True))
                sent_count += 1
                token_ids_stream.write(
                    np.array(sentence, dtype=save_dtype).tobytes(order='C'))
            doc_cumsum_stream.write(
                sent_count.to_bytes(8, byteorder='little', signed=True))

            if step % args.log_interval == 0:
                current = time.time()
                elapsed = current - startup_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {step} documents",
                      f"({step/elapsed:.2f} docs/s, {mbs:.4f} MB/s).",
                      file=sys.stderr)

    pool.close()
    print("Saving tokens to files...")
    all_doc_ids = np.frombuffer(token_ids_stream.getbuffer(), dtype=save_dtype)
    lens = np.frombuffer(sentlens_stream.getbuffer(), dtype=np.int32)
    docs = np.frombuffer(doc_cumsum_stream.getbuffer(), dtype=np.int64)
    np.save(args.output_prefix + "_ids.npy", all_doc_ids)
    np.savez(args.output_prefix + "_idx.npz", lens=lens, docs=docs)

    print("Total sentences num: %d" % len(lens))
    print("Total documents num: %d" % (len(docs) - 1))
    print("Total tokens num: %d" % len(all_doc_ids))
    print("Average tokens per sentence: %.2f" % (len(all_doc_ids) / len(lens)))
    print("Average tokens per document: %.2f" % (len(all_doc_ids) /
                                                 (len(docs) - 1)))


if __name__ == "__main__":
    main()
