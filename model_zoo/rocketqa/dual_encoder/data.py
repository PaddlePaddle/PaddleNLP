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

import paddle
from paddlenlp.utils.log import logger


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


def convert_train_example(example,
                          tokenizer,
                          query_max_seq_length=32,
                          title_max_seq_length=128):
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
    encoded_inputs = tokenizer(text=example["query"],
                               max_seq_len=query_max_seq_length)
    query_input_ids = encoded_inputs["input_ids"]
    query_token_type_ids = encoded_inputs["token_type_ids"]

    # Place title to sentence 1 and paragraph to sentence 2
    encoded_inputs = tokenizer(text=example["pos_title"],
                               text_pair=example["pos_para"],
                               max_seq_len=title_max_seq_length,
                               truncation_strategy="longest_first")
    pos_title_input_ids = encoded_inputs["input_ids"]
    pos_title_token_type_ids = encoded_inputs["token_type_ids"]

    encoded_inputs = tokenizer(text=example["neg_title"],
                               text_pair=example["neg_para"],
                               max_seq_len=title_max_seq_length,
                               truncation_strategy="longest_first")
    neg_title_input_ids = encoded_inputs["input_ids"]
    neg_title_token_type_ids = encoded_inputs["token_type_ids"]

    result = [
        query_input_ids, query_token_type_ids, pos_title_input_ids,
        pos_title_token_type_ids, neg_title_input_ids, neg_title_token_type_ids
    ]

    return result


def convert_inference_example(example, tokenizer, max_seq_length=128):
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

    encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_length)
    text_input_ids = encoded_inputs["input_ids"]
    text_token_type_ids = encoded_inputs["token_type_ids"]
    result = [text_input_ids, text_token_type_ids]
    return result


def convert_inference_example_para(example, tokenizer, max_seq_length=128):
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

    encoded_inputs = tokenizer(text=example['title'],
                               text_pair=example["para"],
                               max_seq_len=max_seq_length,
                               truncation_strategy="longest_first")
    text_input_ids = encoded_inputs["input_ids"]
    text_token_type_ids = encoded_inputs["token_type_ids"]
    result = [text_input_ids, text_token_type_ids]
    return result


def read_train_data(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) != 6:
                continue
            yield {
                'query': data[0],
                'pos_title': data[1],
                'pos_para': data[2],
                'neg_title': data[3],
                'neg_para': data[4]
            }


def read_text(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) != 1:
                continue
            yield {'text': data[0]}


def read_dev_text(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            yield {'text': data[0]}


def read_passage_text(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            yield {'title': data[1], 'para': data[2]}
