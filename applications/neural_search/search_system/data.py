# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import random

import numpy as np
import paddle


def gen_id2corpus(corpus_file):
    id2corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            id2corpus[idx] = line.rstrip()
    return id2corpus


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=batchify_fn,
                                return_list=True)


def convert_example(example, tokenizer, max_seq_length=512, do_evalute=False):
    """
    Builds model inputs from a sequence.
        
    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``

    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
    """

    result = []

    for key, text in example.items():
        if 'label' in key:
            # do_evaluate
            result += [example['label']]
        else:
            # do_train
            encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length)
            input_ids = encoded_inputs["input_ids"]
            token_type_ids = encoded_inputs["token_type_ids"]
            result += [input_ids, token_type_ids]

    return result


def convert_example_test(example,
                         tokenizer,
                         max_seq_length=512,
                         pad_to_max_seq_len=False):
    """
    Builds model inputs from a sequence.
        
    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``

    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
    """

    result = []
    for key, text in example.items():
        encoded_inputs = tokenizer(text=text,
                                   max_seq_len=max_seq_length,
                                   pad_to_max_seq_len=pad_to_max_seq_len)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        result += [input_ids, token_type_ids]
    return result


def read_simcse_text(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip()
            yield {'text_a': data, 'text_b': data}


def read_text_pair(data_path, is_test=False):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if is_test == True:
                if len(data) != 3:
                    continue
                yield {'text_a': data[0], 'text_b': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    continue

                yield {'text_a': data[0], 'text_b': data[1]}


def gen_text_file(similar_text_pair_file):
    text2similar_text = {}
    texts = []
    with open(similar_text_pair_file, 'r', encoding='utf-8') as f:
        for line in f:
            splited_line = line.rstrip().split("\t")
            if len(splited_line) != 2:
                continue

            text, similar_text = line.rstrip().split("\t")

            if not text or not similar_text:
                continue

            text2similar_text[text] = similar_text
            texts.append({"text": text})
    return texts, text2similar_text


def word_repetition(input_ids, token_type_ids, dup_rate=0.32):
    """Word Reptition strategy."""
    input_ids = input_ids.numpy().tolist()
    token_type_ids = token_type_ids.numpy().tolist()

    batch_size, seq_len = len(input_ids), len(input_ids[0])
    repetitied_input_ids = []
    repetitied_token_type_ids = []
    rep_seq_len = seq_len
    for batch_id in range(batch_size):
        cur_input_id = input_ids[batch_id]
        actual_len = np.count_nonzero(cur_input_id)
        dup_word_index = []
        # If sequence length is less than 5, skip it
        if (actual_len > 5):
            dup_len = random.randint(a=0, b=max(2, int(dup_rate * actual_len)))
            # Skip cls and sep position
            dup_word_index = random.sample(list(range(1, actual_len - 1)),
                                           k=dup_len)

        r_input_id = []
        r_token_type_id = []
        for idx, word_id in enumerate(cur_input_id):
            # Insert duplicate word
            if idx in dup_word_index:
                r_input_id.append(word_id)
                r_token_type_id.append(token_type_ids[batch_id][idx])
            r_input_id.append(word_id)
            r_token_type_id.append(token_type_ids[batch_id][idx])
        after_dup_len = len(r_input_id)
        repetitied_input_ids.append(r_input_id)
        repetitied_token_type_ids.append(r_token_type_id)

        if after_dup_len > rep_seq_len:
            rep_seq_len = after_dup_len
    # Padding the data to the same length
    for batch_id in range(batch_size):
        after_dup_len = len(repetitied_input_ids[batch_id])
        pad_len = rep_seq_len - after_dup_len
        repetitied_input_ids[batch_id] += [0] * pad_len
        repetitied_token_type_ids[batch_id] += [0] * pad_len

    return paddle.to_tensor(repetitied_input_ids,
                            dtype='int64'), paddle.to_tensor(
                                repetitied_token_type_ids, dtype='int64')
