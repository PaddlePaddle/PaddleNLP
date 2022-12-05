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
import numpy as np
import paddle


def convert_example(example, tokenizer, max_seq_length=512, is_test=False, is_pair=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed
    to be used in a sequence-pair classification task.

    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``

    It returns the first portion of the mask (0's).


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

    if is_pair:
        text = example["text_a"]
        text_pair = example["text_b"]
    else:
        text = example["text"]
        text_pair = None
    encoded_inputs = tokenizer(text=text, text_pair=text_pair, max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if is_test:
        return input_ids, token_type_ids
    label = np.array([example["label"]], dtype="int64")
    return input_ids, token_type_ids, label


def read_text_pair(data_path):
    """Reads data."""
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) != 3:
                continue
            yield {"text_a": data[0], "text_b": data[1]}


def read_data(data_path):
    """Reads data."""
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            # Skip column name
            if i == 0:
                continue
            data = line.rstrip("\n").split("\t")
            if len(data) != 3:
                print(data)
                continue
            query = data[0]
            title = data[1]
            label = data[-1]
            # breakpoint()
            yield {"text_a": query, "text_b": title, "label": int(label)}


def create_dataloader(dataset, mode="train", batch_size=1, batchify_fn=None, trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train" else False
    if mode == "train":
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)
