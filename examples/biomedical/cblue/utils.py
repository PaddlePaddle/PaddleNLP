# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers import normalize_chars, tokenize_special_chars


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequences for sequence
    classification tasks by concatenating and adding special tokens. And
    creates a mask from the two sequences for sequence-pair classification
    tasks.

    The convention in Electra/EHealth is:

    - single sequence:
        input_ids:      ``[CLS] X [SEP]``
        token_type_ids: ``  0   0   0``
        position_ids:   ``  0   1   2``

    - a senquence pair:
        input_ids:      ``[CLS] X [SEP] Y [SEP]``
        token_type_ids: ``  0   0   0   1   1``
        position_ids:   ``  0   1   2   3   4``

    Args:
        example (obj:`dict`):
            A dictionary of input data, containing text and label if it has.
        tokenizer (obj:`PretrainedTokenizer`):
            A tokenizer inherits from :class:`paddlenlp.transformers.PretrainedTokenizer`.
            Users can refer to the superclass for more information.
        max_seq_length (obj:`int`):
            The maximum total input sequence length after tokenization.
            Sequences longer will be truncated, and the shorter will be padded.
        is_test (obj:`bool`, default to `False`):
            Whether the example contains label or not.

    Returns:
        input_ids (obj:`list[int]`):
            The list of token ids.
        token_type_ids (obj:`list[int]`):
            List of sequence pair mask.
        position_ids (obj:`list[int]`):
            List of position ids.
        label(obj:`numpy.array`, data type of int64, optional):
            The input label if not is_test.
    """
    text_a = example['text_a']
    text_b = example.get('text_b', None)

    text_a = tokenize_special_chars(normalize_chars(text_a))
    if text_b is not None:
        text_b = tokenize_special_chars(normalize_chars(text_b))

    encoded_inputs = tokenizer(
        text=text_a,
        text_pair=text_b,
        max_seq_len=max_seq_length,
        return_position_ids=True)
    input_ids = encoded_inputs['input_ids']
    token_type_ids = encoded_inputs['token_type_ids']
    position_ids = encoded_inputs['position_ids']

    if is_test:
        return input_ids, token_type_ids, position_ids
    label = np.array([example['label']], dtype='int64')
    return input_ids, token_type_ids, position_ids, label


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
