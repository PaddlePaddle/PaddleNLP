#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import traceback
import logging
import json

import numpy as np
import paddle

from text2sql.utils import nn_utils


def collate_batch_data_v2(origin_batch, config):
    """format origin batch data for model forward
    """
    TOKEN_IDS = []
    SENT_IDS = []
    INPUT_MASK = []
    POSITION_IDS = []
    TASK_IDS = []

    QUESTION_TOKENS_INDEX = []
    TABLE_INDEX = []
    COLUMN_INDEX = []
    VALUE_INDEX = []
    RELATION_MATRIXES = []

    lst_orig_inputs = []
    lst_orig_labels = []
    for orig_input, orig_label in origin_batch:
        if orig_input.value_indexes[-1] > 510:
            logging.warning('sequence is too long: %d. question is %s',
                            orig_input.value_indexes[-1] + 2,
                            orig_input.question)
            continue
        lst_orig_inputs.append(orig_input)
        lst_orig_labels.append(orig_label)

        TOKEN_IDS.append(orig_input.token_ids)
        SENT_IDS.append(orig_input.sent_ids)

        # orig_input.span_lens[0] 即 question 包含 [cls], [sep] 的长度
        QUESTION_TOKENS_INDEX.append(
            list(range(1, orig_input.column_indexes[0] - 1)))
        TABLE_INDEX.append(orig_input.table_indexes)
        COLUMN_INDEX.append(orig_input.column_indexes)
        VALUE_INDEX.append(orig_input.value_indexes)

        relations = orig_input.relations
        RELATION_MATRIXES.append(
            np.pad(relations, (0, config.max_seq_len - relations.shape[0])))

    TOKEN_IDS = nn_utils.pad_sequences(TOKEN_IDS, max_len=config.max_seq_len)
    SENT_IDS = nn_utils.pad_sequences(SENT_IDS, max_len=config.max_seq_len)

    QUESTION_TOKENS_INDEX = nn_utils.pad_sequences(
        QUESTION_TOKENS_INDEX, max_len=config.max_question_len)
    TABLE_INDEX = nn_utils.pad_sequences(TABLE_INDEX,
                                         max_len=config.max_table_num)
    COLUMN_INDEX = nn_utils.pad_sequences(COLUMN_INDEX,
                                          max_len=config.max_column_num)
    VALUE_INDEX = nn_utils.pad_sequences(VALUE_INDEX,
                                         max_len=config.max_column_num * 2)

    inputs = {
        'src_ids': TOKEN_IDS,
        'sent_ids': SENT_IDS,
        'question_tokens_index': QUESTION_TOKENS_INDEX,
        'table_indexes': TABLE_INDEX,
        'column_indexes': COLUMN_INDEX,
        'value_indexes': VALUE_INDEX,
        'orig_inputs': lst_orig_inputs,
    }
    RELATION_MATRIXES = np.array(RELATION_MATRIXES).astype(np.int64)
    inputs["relations"] = RELATION_MATRIXES

    for key, value in inputs.items():
        if key in ('orig_inputs', ):
            continue
        inputs[key] = paddle.to_tensor(value)
    return (inputs, lst_orig_labels)


class DataLoader(object):
    """Data Loader for train, test and inference"""

    def __init__(self,
                 config,
                 dataset,
                 batch_size=1,
                 collate_fn=collate_batch_data_v2,
                 shuffle=False,
                 drop_last=False,
                 use_data_parallel=False,
                 use_multiprocess=False):
        super(DataLoader, self).__init__()
        assert batch_size > 0, "batch_size must be an interger that > 0"

        self.config = config
        self._dataset = dataset
        self._batch_size = batch_size
        self._collate_fn = collate_fn
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._use_data_parallel = use_data_parallel
        self._use_multiprocess = use_multiprocess

        self.dataloader = paddle.io.DataLoader.from_generator(
            capacity=1000, return_list=True, use_multiprocess=use_multiprocess)
        self.dataloader.set_batch_generator(self.create_generator())
        if use_data_parallel:
            self.dataloader = paddle.distributed_batch_reader(self.dataloader)

    def __call__(self):
        """call"""
        return self.create_generator()()

    def create_generator(self):
        """Returns a generator, each iteration returns a batch of data"""

        def _reader():
            range_fn = np.random.permutation if self._shuffle else np.arange
            batch = []
            for iid in range_fn(len(self._dataset)):
                batch.append(self._dataset[iid])
                if len(batch) == self._batch_size:
                    outputs = self._collate_fn(batch, self.config.model)
                    batch = []
                    if len(outputs[1]) == 0:
                        continue
                    yield outputs

            if len(batch) > 0 and not self._drop_last:
                yield self._collate_fn(batch, self.config.model)

        return _reader

    @property
    def name(self):
        """read property of name"""
        return self._dataset.name


if __name__ == "__main__":
    """run simple tests"""
    if len(sys.argv) != 5:
        print("usage: %s schema content data grammar_file" % (sys.argv[0]))
        sys.exit(1)
