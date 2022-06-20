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
from paddlenlp.datasets import load_dataset


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    """
    Create dataloader.

    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        batchify_fn(obj:`callable`, optional, defaults to `None`): function to generate mini-batch data by merging
            the sample list, None for only stack each fields of sample in axis
            0(same as :attr::`np.stack(..., axis=0)`).
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a data sample to input ids, etc.

    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle)
    dataloader = paddle.io.DataLoader(dataset,
                                      batch_sampler=sampler,
                                      collate_fn=batchify_fn)
    return dataloader


def preprocess_prediction_data(data,
                               tokenizer,
                               pad_token_id=0,
                               max_ngram_filter_size=3):
    """
    It process the prediction data as the format used as training.

    Args:
        data (obj:`list[str]`): The prediction data whose each element is a tokenized text.
        tokenizer(obj: paddlenlp.data.JiebaTokenizer): It use jieba to cut the chinese string.
        pad_token_id(obj:`int`, optional, defaults to 0): The pad token index.
        max_ngram_filter_size (obj:`int`, optional, defaults to 3) Max n-gram size in TextCNN model.
            Users should refer to the ngram_filter_sizes setting in TextCNN, if ngram_filter_sizes=(1, 2, 3)
            then max_ngram_filter_size=3

    Returns:
        examples (obj:`list`): The processed data whose each element 
            is a `list` object, which contains 
            
            - word_ids(obj:`list[int]`): The list of word ids.
    """
    examples = []
    for text in data:
        ids = tokenizer.encode(text)
        seq_len = len(ids)
        # Sequence length should larger or equal than the maximum ngram_filter_size in TextCNN model
        if seq_len < max_ngram_filter_size:
            ids.extend([pad_token_id] * (max_ngram_filter_size - seq_len))
        examples.append(ids)
    return examples


def convert_example(example, tokenizer):
    """convert_example"""
    input_ids = tokenizer.encode(example["text"])
    input_ids = np.array(input_ids, dtype='int64')

    label = np.array(example["label"], dtype="int64")
    return input_ids, label


def read_custom_data(filename):
    """Reads data."""
    with open(filename, 'r', encoding='utf-8') as f:
        # Skip head
        next(f)
        for line in f:
            data = line.strip().split("\t")
            label, text = data
            yield {"text": text, "label": label}
