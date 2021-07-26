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
import io
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
    group = parser.add_argument_group(title='data input/output')
    group.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to input JSON files.')
    group.add_argument(
        '--output_prefix',
        type=str,
        required=True,
        help='Output prefix to store output file.')
    group.add_argument(
        '--data_format',
        type=str,
        required=True,
        choices=['JSON'],
        help='Only support json format for now. One document per line.')
    group.add_argument(
        '--json_key',
        type=str,
        default='text',
        help='For JSON format. Space separate listed of keys to extract from json'
    )

    group = parser.add_argument_group(title='chinese words')
    group.add_argument(
        '--chinese_words_segment',
        action='store_true',
        help="Is corpus need words segmentation step for chinese words.")
    group.add_argument(
        '--chinese_splited',
        action='store_true',
        help="Is chinese corpus is splited in to words.")
    group.add_argument(
        '--chinese_split_dimer',
        type=str,
        default=' ',
        help="Split dimer between chinese words.")
    group.add_argument(
        '--chinese_seg_func',
        type=str,
        default='lac',
        choices=['lac', 'seg'],
        help='Words segment function for chinese words.')

    group = parser.add_argument_group(title='common config')
    group.add_argument(
        '--append_eos',
        action='store_true',
        help='Append an <eos> token to the end of a document.')
    group.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='Interval between progress updates')
    group.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes to launch')

    args = parser.parse_args()
    return args


def lexical_analysis_fn():
    from LAC import LAC
    lac = LAC(mode="lac")

    def process(line):
        words, _ = lac.run(line)
        return words

    return process


def chinese_segmentation_fn():
    from LAC import LAC
    lac_cws = LAC(mode='seg')

    def process(line):
        words = lac.run(line)
        return words

    return process


CHINESE_SEG_FUNC = {
    'lac': lexical_analysis_fn(),
    'seg': chinese_segmentation_fn(),
}


# Complexity is k*n**2, too high for long documents
def get_whole_word_mask_tokens(tokens, words, max_word_length=4):
    """
    Do whole word mask on Chinese word.
    First, we do Chinese word segmentation on the sequence of tokens, which are from the WordPiece tokenization.
    Then, we add the '##' mark on chinese characters which are in the middle of Chinese words.
    And if the tokens are not chinese characters, we just exploit the results of WordPiece tokenization as words.
    Such as, 
         - text line : 通过利用mercer核，将样本从输入空间映射到高维特征空间，使原来没有显现的特征突现出来，取得了很好的图像分割效果。
         - the input tokens (after WordPiece): 
            ['通', '过', '利', '用', 'me', '##rc', '##er', '核', '，', '将', '样', '本', '从', '输', '入', '空', '间', '映', 
            '射', '到', '高', '维', '特', '征', '空', '间', '，', '使', '原', '来', '没', '有', '显', '现', '的', '特', '征', 
            '突', '现', '出', '来', '，', '取', '得', '了', '很', '好', '的', '图', '像', '分', '割', '效', '果', '。']
        - the Chinese words (after Chinese word segmentation like jieba)
            ['通过', '利用', 'mercer', '核', '，', '将', '样本', '从', '输入', '空间', '映射', '到', '高维', '特征', 
            '空间', '，', '使', '原来', '没有', '显现', '的', '特征', '突现', '出来', '，', '取得', '了', '很', '好', 
            '的', '图像', '分割', '效果', '。']
        - the output whole word mask tokens:
            ['通', '##过', '利', '##用', 'me', '##rc', '##er', '核', '，', '将', '样', '##本', '从', '输', '##入', 
            '空', '##间', '映', '##射', '到', '高', '##维', '特', '##征', '空', '##间', '，', '使', '原', '##来', 
            '没', '##有', '显', '##现', '的', '特', '##征', '突', '##现', '出', '##来', '，', '取', '##得', '了', 
            '很', '好', '的', '图', '##像', '分', '##割', '效', '##果', '。']

    Args:
        tokens(list(str)): The sequence of tokens, which are from the WordPiece tokenization.
        words(list(str)): The sequence of Chinese words.
        max_word_length(int, optional): 
            The maximum chinese character in Chinese words. It avoids too long Chinese word to be masked.
            Defaults as 4.

    Returns:
         new_tokens(list(str)): The new token will be done with whole word masking strategy.

    """

    new_tokens = []
    # opt for long document
    words_set = set(words)
    i = 0
    while i < len(tokens):
        # non-chinese character, then do word piece 
        if len(re.findall('[\u4E00-\u9FA5]', tokens[i])) == 0:
            new_tokens.append(tokens[i])
            i += 1
            continue

        # add "##" mark on the middel tokens of Chinese words
        # such as ["通过", "利用"] -> ["通", "##过"， "利", "##用"] 
        has_add = False
        for length in range(max_word_length, 0, -1):
            if i + length > len(tokens):
                continue
            if ''.join(tokens[i:i + length]) in words_set:
                new_tokens.append(tokens[i])
                for l in range(1, length):
                    new_tokens.append('##' + tokens[i + l])
                i += length
                has_add = True
                break

        if not has_add:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


class Converter(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Converter.tokenizer = getattr(
            tfs, self.args.tokenizer_name).from_pretrained(self.args.model_name)

        if self.args.chinese_words_segment:
            if self.args.chinese_splited:
                Converter.segment_func = lambda text: text.split(self.args.chinese_split_dimer)
            else:
                Converter.segment_func = CHINESE_SEG_FUNC[
                    self.args.chinese_seg_func]

            def process(text):
                words = Converter.segment_func(text)
                tokens = Converter.tokenizer.tokenize("".join(words))
                tokens = get_whole_word_mask_tokens(tokens, words)
                tokens = Converter.tokenizer.convert_tokens_to_ids(tokens)
                return tokens

            Converter.process = process

        else:
            Converter.process = lambda text : Converter.tokenizer.convert_tokens_to_ids(
                Converter.tokenizer.tokenize(text))

    def encode(self, json_line):
        text = json.loads(json_line)[self.args.json_key]
        text = re.sub('[\n]+', '\n', text)
        tokens = Converter.process(text)
        #text = re.sub('[ ]+', ' ', text)
        # if self.args.chinese_words_segment:
        #     if self.args.splited:
        #         words = text.split(self.args.split_dimer) 
        #     else:
        #         words = Converter.segment_func(text)
        #     tokens = Converter.tokenizer.tokenize(text)
        #     tokens = get_whole_word_mask_tokens(tokens, words) 
        #     tokens = Converter.tokenizer.convert_tokens_to_ids(tokens)
        # else:
        #     tokens = Converter.tokenizer.convert_tokens_to_ids(
        #         Converter.tokenizer.tokenize(text))

        # if self.args.append_eos:
        #     tokens.append(Converter.tokenizer.eos_token_id)

        return tokens, len(json_line.encode("utf-8"))


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
    convert = Converter(args)

    # try tokenizer is availiable
    sample_tokenizer = getattr(
        tfs, args.tokenizer_name).from_pretrained(args.model_name)
    if sample_tokenizer.vocab_size < 2**16 - 1:
        save_dtype = np.uint16
    else:
        save_dtype = np.int32

    pool = multiprocessing.Pool(args.workers, initializer=convert.initializer)

    # We use BytesIO to store the ids.
    memory_stream = io.BytesIO()
    lens = []

    for file_path in tqdm(file_paths):
        total_bytes_processed = 0
        text = open(file_path, 'r', encoding='utf-8')
        encoded_docs = pool.imap(convert.encode, text, 256)

        startup_end = time.time()
        proc_start = time.time()
        print("Time to startup:", startup_end - startup_start)
        print("Processing %s" % file_path)
        for i, (tokens, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed

            lens.append(len(tokens))
            memory_stream.write(
                np.array(
                    tokens, dtype=save_dtype).tobytes(order='C'))

            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(
                    f"Processed {i} documents",
                    f"({i/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)

    pool.close()
    print("Saving tokens to npz file...")
    all_doc_ids = np.frombuffer(memory_stream.getbuffer(), dtype=save_dtype)
    lens = np.array(lens, dtype=np.uint32)
    np.savez(args.output_prefix + "_ids_lens.npz", ids=all_doc_ids, lens=lens)
    print("Total documents num: %d" % len(lens))
    print("Total tokens num: %d" % len(all_doc_ids))
    print("Average tokens per doc: %.2f" % (len(all_doc_ids) / len(lens)))


if __name__ == "__main__":
    main()
    #profile.run("main()", "testprof")
