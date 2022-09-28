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

import re
import numpy as np
import pandas as pd
import paddle


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed 
    to be used in a sequence-pair classification task.
        
    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``
    - pair of sequences: ``[CLS] A [SEP] B [SEP]``

    A BERT sequence pair mask has the following format:
    ::
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |

    If only one sequence, only returns the first portion of the mask (0's).


    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array(example["label"], dtype="float32")
        return input_ids, token_type_ids, label
    return input_ids, token_type_ids


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


def read_custom_data(filename, is_test=False):
    """Reads data."""
    data = pd.read_csv(filename)
    for line in data.values:
        if is_test:
            text = line[1]
            yield {"text": clean_text(text), "label": ""}
        else:
            text, label = line[1], line[2:]
            yield {"text": clean_text(text), "label": label}


def clean_text(text):
    text = text.replace("\r", "").replace("\n", "")
    text = re.sub(r"\\n\n", ".", text)
    return text


def write_test_results(filename, results, label_info):
    """write test results"""
    data = pd.read_csv(filename)
    qids = [line[0] for line in data.values]
    results_dict = {k: [] for k in label_info}
    results_dict["id"] = qids
    results = list(map(list, zip(*results)))
    for key in results_dict:
        if key != "id":
            for result in results:
                results_dict[key] = result
    df = pd.DataFrame(results_dict)
    df.to_csv("sample_test.csv", index=False)
    print("Test results saved")
