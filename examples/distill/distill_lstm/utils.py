# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import jieba

import numpy as np


def convert_small_example(example,
                          task_name,
                          vocab,
                          is_tokenized=False,
                          max_seq_length=128,
                          is_test=False):
    input_ids = []
    if task_name == 'senta':
        if is_tokenized:
            input_ids = [vocab[token] for token in example[0]]
        else:
            for i, token in enumerate(jieba.cut(example[0])):
                if i == max_seq_length:
                    break
                token_id = vocab[token]
                input_ids.append(token_id)
    else:
        if is_tokenized:
            tokens = example[0][:max_seq_length]
        else:
            tokens = vocab(example[0])[:max_seq_length]
        input_ids = vocab.convert_tokens_to_ids(tokens)

    valid_length = np.array(len(input_ids), dtype='int64')

    if not is_test:
        label = np.array(example[-1], dtype="int64")
        return input_ids, valid_length, label
    return input_ids, valid_length


def convert_pair_example(example,
                         task_name,
                         vocab,
                         is_tokenized=True,
                         max_seq_length=128,
                         is_test=False):
    seq1 = convert_small_example([example[0], example[2]], task_name, vocab,
                                 is_tokenized, max_seq_length, is_test)[:2]

    seq2 = convert_small_example([example[1], example[2]], task_name, vocab,
                                 is_tokenized, max_seq_length, is_test)
    pair_features = seq1 + seq2

    return pair_features


def convert_two_example(example,
                        task_name,
                        tokenizer,
                        label_list,
                        max_seq_length,
                        vocab,
                        is_tokenized=True,
                        is_test=False):
    bert_features = convert_example(
        example[1:] if task_name == 'senta' and is_tokenized else example,
        tokenizer=tokenizer,
        label_list=label_list,
        is_tokenized=is_tokenized,
        max_seq_length=max_seq_length,
        is_test=is_test)

    if task_name == 'qqp':
        small_features = convert_pair_example(
            example, task_name, vocab, is_tokenized, max_seq_length, is_test)
    else:
        small_features = convert_small_example(
            example, task_name, vocab, is_tokenized, max_seq_length, is_test)

    return bert_features[:2] + small_features


def convert_example(example,
                    tokenizer,
                    label_list,
                    is_tokenized=False,
                    max_seq_length=512,
                    is_test=False):
    """convert a glue example into necessary features"""

    def _truncate_seqs(seqs, max_seq_length):
        if len(seqs) == 1:  # single sentence
            # Account for [CLS] and [SEP] with "- 2"
            seqs[0] = seqs[0][0:(max_seq_length - 2)]
        else:  # Sentence pair
            # Account for [CLS], [SEP], [SEP] with "- 3"
            tokens_a, tokens_b = seqs
            max_seq_length -= 3
            while True:  # Truncate with longest_first strategy
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_seq_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
        return seqs

    def _concat_seqs(seqs, separators, seq_mask=0, separator_mask=1):
        concat = sum((seq + sep for sep, seq in zip(separators, seqs)), [])
        segment_ids = sum(
            ([i] * (len(seq) + len(sep))
             for i, (sep, seq) in enumerate(zip(separators, seqs))), [])
        if isinstance(seq_mask, int):
            seq_mask = [[seq_mask] * len(seq) for seq in seqs]
        if isinstance(separator_mask, int):
            separator_mask = [[separator_mask] * len(sep) for sep in separators]
        p_mask = sum((s_mask + mask
                      for sep, seq, s_mask, mask in zip(
                          separators, seqs, seq_mask, separator_mask)), [])
        return concat, segment_ids, p_mask

    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example[-1]
        example = example[:-1]
        # Create label maps if classification task
        if label_list:
            label_map = {}
            for (i, l) in enumerate(label_list):
                label_map[l] = i
            label = label_map[label]
        label = np.array([label], dtype=label_dtype)

    if is_tokenized:
        tokens_raw = example
    else:
        # Tokenize raw text
        tokens_raw = [tokenizer(l) for l in example]
    # Truncate to the truncate_length,
    tokens_trun = _truncate_seqs(tokens_raw, max_seq_length)

    # Concate the sequences with special tokens
    tokens_trun[0] = [tokenizer.cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = _concat_seqs(tokens_trun, [[tokenizer.sep_token]] *
                                          len(tokens_trun))
    # Convert the token to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    valid_length = len(input_ids)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    # input_mask = [1] * len(input_ids)
    if not is_test:
        return input_ids, segment_ids, valid_length, label
    return input_ids, segment_ids, valid_length
