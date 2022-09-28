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


def convert_example_for_lstm(example,
                             task_name,
                             vocab,
                             is_tokenized=False,
                             max_seq_length=128,
                             is_test=False):
    """convert a example for lstm's input"""
    input_ids = []
    if task_name == 'chnsenticorp':
        if is_tokenized:
            lstm_tokens = example['lstm_tokens'][:max_seq_length]
            input_ids = [vocab[token] for token in lstm_tokens]
        else:
            tokenized_text = list(jieba.cut(example['text']))[:max_seq_length]
            input_ids = vocab[tokenized_text]
    else:
        if is_tokenized:
            tokens = example['sentence'][:max_seq_length]
        else:
            tokens = vocab.tokenize(example['sentence'])[:max_seq_length]
        input_ids = vocab.convert_tokens_to_ids(tokens)

    valid_length = np.array(len(input_ids), dtype='int64')
    if not is_test:
        label = np.array(
            example['label'],
            dtype="int64") if task_name == 'chnsenticorp' else np.array(
                example['labels'], dtype="int64")
        return input_ids, valid_length, label
    return input_ids, valid_length


def convert_pair_example(example,
                         task_name,
                         vocab,
                         is_tokenized=True,
                         max_seq_length=128,
                         is_test=False):
    seq1 = convert_example_for_lstm(
        {
            "sentence": example['sentence1'],
            "labels": example['labels']
        }, task_name, vocab, is_tokenized, max_seq_length, is_test)[:2]

    seq2 = convert_example_for_lstm(
        {
            "sentence": example['sentence2'],
            "labels": example['labels']
        }, task_name, vocab, is_tokenized, max_seq_length, is_test)
    pair_features = seq1 + seq2

    return pair_features


def convert_example_for_distill(example,
                                task_name,
                                tokenizer,
                                label_list,
                                max_seq_length,
                                vocab,
                                is_tokenized=True,
                                is_test=False):
    bert_features = convert_example_for_bert(example,
                                             tokenizer=tokenizer,
                                             label_list=label_list,
                                             is_tokenized=is_tokenized,
                                             max_seq_length=max_seq_length,
                                             is_test=is_test)
    if task_name == 'qqp':
        small_features = convert_pair_example(example, task_name, vocab,
                                              is_tokenized, max_seq_length,
                                              is_test)
    else:
        small_features = convert_example_for_lstm(example, task_name, vocab,
                                                  is_tokenized, max_seq_length,
                                                  is_test)
    return bert_features[:2] + small_features


def convert_example_for_bert(example,
                             tokenizer,
                             label_list,
                             is_tokenized=False,
                             max_seq_length=512,
                             is_test=False):
    """convert a example for bert's input"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example['labels'] if 'labels' in example else example['label']
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    if 'sentence1' in example:
        example = tokenizer(example['sentence1'],
                            text_pair=example['sentence2'],
                            max_seq_len=max_seq_length,
                            is_split_into_words=is_tokenized)
    else:
        if 'sentence' in example:
            text = example['sentence']
        elif 'text' in example:
            text = example['text']
        else:
            text = example['bert_tokens']
        example = tokenizer(text,
                            max_seq_len=max_seq_length,
                            is_split_into_words=is_tokenized)

    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']
