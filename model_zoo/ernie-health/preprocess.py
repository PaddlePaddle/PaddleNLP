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
import re
import argparse
import json
import multiprocessing
import sys
import time

import numpy as np
from tqdm import tqdm

from paddlenlp.transformers import ElectraTokenizer


def parse_args():
    parser = argparse.ArgumentParser('Preprocessor for ERNIE-Health')
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='The path to input text files where a sentence per line.')
    parser.add_argument('--output_file',
                        type=str,
                        required=True,
                        help='The output file path of preprocessed ids.')
    parser.add_argument('--tokenize_tool',
                        type=str,
                        default='lac',
                        choices=['lac', 'seg', 'jieba'],
                        help='The tokenization tool for chinese words.')
    parser.add_argument('--logging_steps',
                        type=int,
                        default=100,
                        help='The interval between progress updates.')
    parser.add_argument('--num_worker',
                        type=int,
                        default=1,
                        help='Number of worker processes to launch.')

    args = parser.parse_args()
    return args


def lac_segmentation():
    from LAC import LAC
    tool = LAC(mode='lac')

    def process(text):
        words, _ = tool.run(text)
        return words

    return process


def seg_segmentation():
    from LAC import LAC
    tool = LAC(mode='seg')

    def process(text):
        words = tool.run(text)
        return words

    return process


def jieba_segmentation():
    import jieba

    def process(text):
        words = jieba.cut(text)
        return list(words)

    return process


SEGMENTATION_FN = {
    'lac': lac_segmentation(),
    'seg': seg_segmentation(),
    'jieba': jieba_segmentation()
}


class ProcessFn(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        ProcessFn.tokenizer = ElectraTokenizer.from_pretrained(
            'ernie-health-chinese')
        ProcessFn.segmenter = SEGMENTATION_FN[self.args.tokenize_tool]
        # Update vocabulary with '##'-prefixed chinese characters.
        # The ids should coincide with those in run_pretrain.py.
        suffix_vocab = {}
        for idx, token in enumerate(range(0x4E00, 0x9FA6)):
            suffix_vocab['##' + chr(token)] = len(ProcessFn.tokenizer) + idx
        ProcessFn.tokenizer.added_tokens_encoder.update(suffix_vocab)

        def mark_word_in_tokens(tokens, words, max_word_length=4):
            word_set = set(words)
            index = 0
            while index < len(tokens):
                # Skip non-chinese characters.
                if len(re.findall('[\u4E00-\u9FA5]', tokens[index])) == 0:
                    index += 1
                    continue
                # Find the word with maximum length and mark it.
                find_word = False
                for length in range(max_word_length, 0, -1):
                    if index + length > len(tokens):
                        continue
                    if ''.join(tokens[index:index + length]) in word_set:
                        for i in range(1, length):
                            tokens[index + i] = '##' + tokens[index + i]
                        index += length
                        find_word = True
                        break

                if not find_word:
                    index += 1
            return tokens

        def process(text):
            words = ProcessFn.segmenter(text.strip())
            tokens = ProcessFn.tokenizer.tokenize(''.join(words))
            tokens = mark_word_in_tokens(tokens, words)
            tokens = ProcessFn.tokenizer.convert_tokens_to_ids(tokens)
            return tokens

        ProcessFn.process = process

    def encode(self, text):
        token_ids = ProcessFn.process(text)
        return token_ids, len(text.encode('utf-8'))


def main():
    args = parse_args()

    file_paths = []
    if os.path.isfile(args.input_path):
        file_paths.append(args.input_path)
    else:
        for root, dirs, files in os.walk(args.input_path):
            for file_name in files:
                file_paths.append(os.path.join(root, file_name))
    file_paths.sort()

    tokenizer = ElectraTokenizer.from_pretrained('ernie-health-chinese')
    save_dtype = np.uint16 if tokenizer.vocab_size < 2**16 - 1 else np.int32
    processer = ProcessFn(args)

    pool = multiprocessing.Pool(args.num_worker,
                                initializer=processer.initializer)

    token_id_stream = io.BytesIO()
    sent_len_stream = io.BytesIO()

    step = 0
    sent_count = 0
    total_bytes_processed = 0
    start_tic = time.time()

    for path in tqdm(file_paths):
        text_fp = open(path, 'r')
        processed_text = pool.imap(processer.encode, text_fp, 256)
        print('Processing %s' % path)
        for i, (tokens, bytes_processed) in enumerate(processed_text, start=1):
            step += 1
            total_bytes_processed += bytes_processed

            sentence_len = len(tokens)
            if sentence_len == 0:
                continue
            sent_len_stream.write(
                sentence_len.to_bytes(4, byteorder='little', signed=True))
            sent_count += 1
            token_id_stream.write(
                np.array(tokens, dtype=save_dtype).tobytes(order='C'))

            if step % args.logging_steps == 0:
                time_cost = time.time() - start_tic
                mbs = total_bytes_processed / time_cost / 1024 / 1024
                print(f'Processed {step} sentences',
                      f'({step/time_cost:.2f} sentences/s, {mbs:.4f} MB/s).',
                      file=sys.stderr)

    pool.close()
    print('Saving tokens to files...')
    all_token_ids = np.frombuffer(token_id_stream.getbuffer(), dtype=save_dtype)
    all_sent_lens = np.frombuffer(sent_len_stream.getbuffer(), dtype=np.int32)
    np.save(args.output_file + '_ids.npy', all_token_ids)
    np.savez(args.output_file + '_idx.npz', lens=all_sent_lens)

    print('Total sentences num: %d' % len(all_sent_lens))
    print('Total tokens num: %d' % len(all_token_ids))
    print('Average tokens per sentence: %.2f' %
          (len(all_token_ids) / len(all_sent_lens)))


if __name__ == '__main__':
    main()
