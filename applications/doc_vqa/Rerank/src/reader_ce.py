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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import sys
import os
import json
import random
import logging
import numpy as np
import six
from io import open
from collections import namedtuple

import tokenization
from batching import pad_batch_data

log = logging.getLogger(__name__)

if six.PY3:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def csv_reader(fd, delimiter='\t', trainer_id=0, trainer_num=1):

    def gen():
        for i, line in enumerate(fd):
            if i % trainer_num == trainer_id:
                slots = line.rstrip('\n').split(delimiter)
                if len(slots) == 1:
                    yield slots,
                else:
                    yield slots

    return gen()


class BaseReader(object):

    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 total_num=0,
                 do_lower_case=True,
                 in_tokens=False,
                 is_inference=False,
                 random_seed=None,
                 tokenizer="FullTokenizer",
                 for_cn=True,
                 task_id=0):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path,
                                                    do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens
        self.is_inference = is_inference
        self.for_cn = for_cn
        self.task_id = task_id

        np.random.seed(random_seed)

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0
        self.total_num = total_num

        if label_map_config:
            with open(label_map_config, encoding='utf8') as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, 'r', encoding='utf8') as f:
            reader = csv_reader(f)
            headers = next(reader)
            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                example = Example(*line)
                examples.append(example)
            return examples

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
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

        query = tokenization.convert_to_unicode(example.query)
        tokens_a = tokenizer.tokenize(query)
        tokens_b = None

        title = tokenization.convert_to_unicode(example.title)
        tokens_b = tokenizer.tokenize(title)

        para = tokenization.convert_to_unicode(example.para)
        tokens_para = tokenizer.tokenize(para)

        tokens_b.extend(tokens_para)

        self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        # The convention in BERT/ERNIE is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
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
        position_ids = list(range(len(token_ids)))

        if self.is_inference:
            Record = namedtuple('Record',
                                ['token_ids', 'text_type_ids', 'position_ids'])
            record = Record(token_ids=token_ids,
                            text_type_ids=text_type_ids,
                            position_ids=position_ids)
        else:
            if self.label_map:
                label_id = self.label_map[example.label]
            else:
                label_id = example.label

            Record = namedtuple('Record', [
                'token_ids', 'text_type_ids', 'position_ids', 'label_id', 'qid'
            ])

            qid = None
            if "qid" in example._fields:
                qid = example.qid

            record = Record(token_ids=token_ids,
                            text_type_ids=text_type_ids,
                            position_ids=position_ids,
                            label_id=label_id,
                            qid=qid)
        return record

    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            record = self._convert_example_to_record(example, self.max_seq_len,
                                                     self.tokenizer)
            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records)
                batch_records, max_len = [record], len(record.token_ids)

        if batch_records:
            yield self._pad_batch_records(batch_records)

    def get_num_examples(self, input_file):
        #        examples = self._read_tsv(input_file)
        #        return len(examples)
        return self.num_examples

    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       dev_count=1,
                       trainer_id=0,
                       trainer_num=1,
                       shuffle=True,
                       phase=None):

        if phase == 'train':
            #            examples = examples[trainer_id: (len(examples) //trainer_num) * trainer_num : trainer_num]
            self.num_examples_per_node = self.total_num // trainer_num
            self.num_examples = self.num_examples_per_node * trainer_num
            examples = self._read_tsv(input_file,
                                      trainer_id=trainer_id,
                                      trainer_num=trainer_num,
                                      num_examples=self.num_examples_per_node)
            log.info('apply sharding %d/%d' % (trainer_id, trainer_num))
        else:
            examples = self._read_tsv(input_file)

        def wrapper():
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                if shuffle:
                    np.random.shuffle(examples)

                for batch_data in self._prepare_batch_data(examples,
                                                           batch_size,
                                                           phase=phase):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        for batch in all_dev_batches:
                            yield batch
                        all_dev_batches = []

        def f():
            try:
                for i in wrapper():
                    yield i
            except Exception as e:
                import traceback
                traceback.print_exc()

        return f


class ClassifyReader(BaseReader):

    def _read_tsv(self,
                  input_file,
                  quotechar=None,
                  trainer_id=0,
                  trainer_num=1,
                  num_examples=0):
        """Reads a tab separated value file."""
        with open(input_file, 'r', encoding='utf8') as f:
            reader = csv_reader(f,
                                trainer_id=trainer_id,
                                trainer_num=trainer_num)
            #            headers = next(reader)
            headers = 'query\ttitle\tpara\tlabel'.split('\t')
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"
            ]
            Example = namedtuple('Example', headers)

            examples = []
            for cnt, line in enumerate(reader):
                if num_examples != 0 and cnt == num_examples:
                    break
                for index, text in enumerate(line):
                    if index in text_indices:
                        if self.for_cn:
                            line[index] = text.replace(' ', '')
                        else:
                            line[index] = text
                example = Example(*line)
                examples.append(example)
            return examples

    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]

        if not self.is_inference:
            batch_labels = [record.label_id for record in batch_records]
            batch_labels = np.array(batch_labels).astype("int64").reshape(
                [-1, 1])

            if batch_records[0].qid:
                batch_qids = [record.qid for record in batch_records]
                batch_qids = np.array(batch_qids).astype("int64").reshape(
                    [-1, 1])
            else:
                batch_qids = np.array([]).astype("int64").reshape([-1, 1])

        # padding
        padded_token_ids, input_mask = pad_batch_data(batch_token_ids,
                                                      pad_idx=self.pad_id,
                                                      return_input_mask=True)
        padded_text_type_ids = pad_batch_data(batch_text_type_ids,
                                              pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(batch_position_ids,
                                             pad_idx=self.pad_id)
        padded_task_ids = np.ones_like(padded_token_ids,
                                       dtype="int64") * self.task_id

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            padded_task_ids, input_mask
        ]
        if not self.is_inference:
            return_list += [batch_labels, batch_qids]

        return return_list
