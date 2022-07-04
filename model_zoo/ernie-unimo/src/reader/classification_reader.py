#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""data reader for text classification tasks"""

import os
import csv
import numpy as np
import copy
from collections import namedtuple
from model import tokenization
from reader.batching import pad_batch_data

from paddle.io import Dataset


class ClassifyReader(Dataset):
    """ClassifyReader"""
    def __init__(self, input_file, tokenizer, args):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id

        self.max_seq_len = args.max_seq_len
        self.in_tokens = args.in_tokens

        self.examples = self._read_tsv(input_file)


    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            headers = next(reader)
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"
            ]
            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                example = Example(*line)
                examples.append(example)
            return examples


    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`."""
        text_a = tokenization.convert_to_unicode(example.text_a)
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None
        if "text_b" in example._fields:
            text_b = tokenization.convert_to_unicode(example.text_b)
            tokens_b = tokenizer.tokenize(text_b)
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        text_type_ids = []
        tokens.append("[CLS]")
        text_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            text_type_ids.append(0)
        tokens.append("[SEP]")
        text_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                text_type_ids.append(1)
            tokens.append("[SEP]")
            text_type_ids.append(1)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(2, len(token_ids) + 2))
        label_id = example.label

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'label_id'])

        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            label_id=label_id)
        return record


    def __len__(self):
        """get_num_examples"""
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        record = self._convert_example_to_record(example, self.max_seq_len,
                                                     self.tokenizer)
        return record
                

def pad_batch_records(batch_records, tokenizer):
    batch_token_ids = [record.token_ids for record in batch_records]
    batch_text_type_ids = [record.text_type_ids for record in batch_records]
    batch_position_ids = [record.position_ids for record in batch_records]
    batch_labels = [record.label_id for record in batch_records]
    batch_labels = np.array(batch_labels).astype('int64').reshape([-1, 1])

    # padding
    padded_token_ids, input_mask = pad_batch_data(
        batch_token_ids, pretraining_task='nlu', pad_idx=tokenizer.pad_token_id, return_input_mask=True)
    padded_text_type_ids = pad_batch_data(
        batch_text_type_ids, pretraining_task='nlu', pad_idx=tokenizer.pad_token_id)
    padded_position_ids = pad_batch_data(
        batch_position_ids, pretraining_task='nlu', pad_idx=tokenizer.pad_token_id)
    input_mask = np.matmul(input_mask, np.transpose(input_mask, (0, 2, 1)))

    return_list = [
        padded_token_ids, padded_text_type_ids, padded_position_ids,
        input_mask, batch_labels
    ]

    return return_list

if __name__ == '__main__':
    pass
