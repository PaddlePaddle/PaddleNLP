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

import json

import numpy as np
import paddle


def convert_example(example,
                    tokenzier,
                    max_seq_len=512,
                    max_cls_len=5,
                    summary_num=2,
                    is_test=False):
    """
    Builds model inputs from a sequence for noun phrase classification task.
    A prompt template is added to the end of the sequence.

    Prompt template:

    - ``[是] + [MASK] * max_cls_len``

    Model input example:

    - ``[CLS0][CLS1] X [是][MASK]...[MASK][SEP]``
    
        where X is the input text.

    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        max_cls_len(obj:`int`): The maximum length of labels.
        summary_num(obj:`int`): The number of summary tokens, e.g. `[CLS0]` and `[CLS1]`. 
        is_test(obj:`bool`): If True, it will not return the label.

    """

    if len(example["text"]) + max_cls_len + 1 + summary_num + 1 > max_seq_len:
        example["text"] = example["text"][:(
            max_seq_len - (max_cls_len + 1 + summary_num + 1))]

    tokens = list(example["text"]) + ["是"] + ["[MASK]"] * max_cls_len
    inputs = tokenzier(tokens,
                       return_length=True,
                       is_split_into_words=True,
                       max_length=max_seq_len)

    label_indices = list(
        range(inputs["seq_len"] - 1 - max_cls_len, inputs["seq_len"] - 1))

    if is_test:
        return inputs["input_ids"], inputs["token_type_ids"], label_indices

    label_tokens = list(
        example["label"]) + ["[PAD]"] * (max_cls_len - len(example["label"]))
    labels = np.full([inputs["seq_len"]], fill_value=-100, dtype=np.int64)
    labels[label_indices] = tokenzier.convert_tokens_to_ids(label_tokens)
    return inputs["input_ids"], inputs["token_type_ids"], labels


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


def read_custom_data(filename):
    """Reads data"""
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.strip().split('\t')
            yield {'text': text, 'label': label}
