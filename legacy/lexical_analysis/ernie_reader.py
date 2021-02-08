#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
This module provides reader for ernie model
"""

import sys

from collections import namedtuple
import numpy as np

sys.path.append("../shared_modules/")
from preprocess.ernie.task_reader import BaseReader, tokenization


def pad_batch_data(insts,
                   pad_idx=0,
                   max_len=128,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and input mask.
    """
    return_list = []
    # max_len = max(len(inst) for inst in insts)
    max_len = max_len
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] *
                                    (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape([-1])]

    return return_list if len(return_list) > 1 else return_list[0]


class SequenceLabelReader(BaseReader):
    """SequenceLabelReader"""

    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_label_ids = [record.label_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, batch_seq_lens = pad_batch_data(
            batch_token_ids,
            max_len=self.max_seq_len,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, max_len=self.max_seq_len, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, max_len=self.max_seq_len, pad_idx=self.pad_id)
        padded_label_ids = pad_batch_data(
            batch_label_ids,
            max_len=self.max_seq_len,
            pad_idx=len(self.label_map) - 1)

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            input_mask, padded_label_ids, batch_seq_lens
        ]
        return return_list

    def _reseg_token_label(self, tokens, labels, tokenizer):
        assert len(tokens) == len(labels)
        ret_tokens = []
        ret_labels = []
        for token, label in zip(tokens, labels):
            sub_token = tokenizer.tokenize(token)
            if len(sub_token) == 0:
                continue
            ret_tokens.extend(sub_token)
            ret_labels.append(label)
            if len(sub_token) < 2:
                continue
            sub_label = label
            if label.startswith("B-"):
                sub_label = "I-" + label[2:]
            ret_labels.extend([sub_label] * (len(sub_token) - 1))

        assert len(ret_tokens) == len(ret_labels)
        return ret_tokens, ret_labels

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        tokens = tokenization.convert_to_unicode(example.text_a).split(u"")
        labels = tokenization.convert_to_unicode(example.label).split(u"")
        tokens, labels = self._reseg_token_label(tokens, labels, tokenizer)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))
        text_type_ids = [0] * len(token_ids)
        no_entity_id = len(self.label_map) - 1
        labels = [
            label if label in self.label_map else u"O" for label in labels
        ]
        label_ids = [no_entity_id] + [
            self.label_map[label] for label in labels
        ] + [no_entity_id]

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'label_ids'])
        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            label_ids=label_ids)
        return record
