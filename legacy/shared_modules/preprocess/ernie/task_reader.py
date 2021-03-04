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
This module provides reader for classification and sequence labing
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import csv
import json

import numpy as np

from preprocess.ernie import tokenization
from preprocess.padding import pad_batch_data
import io


def csv_reader(fd, delimiter='\t'):
    def gen():
        for i in fd:
            slots = i.rstrip('\n').split(delimiter)
            if len(slots) == 1:
                yield slots,
            else:
                yield slots

    return gen()


class BaseReader(object):
    """BaseReader for classify and sequence labeling task"""

    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 random_seed=None):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens

        np.random.seed(random_seed)

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        if label_map_config:
            with open(label_map_config) as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="utf8") as f:
            reader = csv_reader(f, delimiter="\t")
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

        if self.label_map:
            label_id = self.label_map[example.label]
        else:
            label_id = example.label

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'label_id', 'qid'])

        qid = None
        if "qid" in example._fields:
            qid = example.qid

        record = Record(
            token_ids=token_ids,
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
        """return total number of examples"""
        examples = self._read_tsv(input_file)
        return len(examples)

    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       shuffle=True,
                       phase=None):
        """return generator which yields batch data for pyreader"""
        examples = self._read_tsv(input_file)

        def _wrapper():
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                if shuffle:
                    np.random.shuffle(examples)

                for batch_data in self._prepare_batch_data(
                        examples, batch_size, phase=phase):
                    yield batch_data

        return _wrapper


class ClassifyReader(BaseReader):
    """ClassifyReader"""

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="utf8") as f:
            reader = csv_reader(f, delimiter="\t")
            headers = next(reader)
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"
            ]
            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                for index, text in enumerate(line):
                    if index in text_indices:
                        line[index] = text.replace(' ', '')
                example = Example(*line)
                examples.append(example)
            return examples

    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_labels = [record.label_id for record in batch_records]
        batch_labels = np.array(batch_labels).astype("int64").reshape([-1, 1])

        # padding
        padded_token_ids, input_mask, seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            input_mask, batch_labels, seq_lens
        ]

        return return_list


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
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        padded_label_ids = pad_batch_data(
            batch_label_ids, pad_idx=len(self.label_map) - 1)

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


class ExtractEmbeddingReader(BaseReader):
    """ExtractEmbeddingReader"""

    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            input_mask, seq_lens
        ]

        return return_list


if __name__ == '__main__':
    pass
