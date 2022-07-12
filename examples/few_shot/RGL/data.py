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

from abc import abstractmethod
import os
import csv
import json
import pandas as pd
from typing import Optional, Union, List
from dataclasses import dataclass, field
from collections import defaultdict

import paddle
from paddle.metric import Accuracy

import paddlenlp
from paddle.fluid.reader import default_collate_fn
from paddlenlp.utils.log import logger
from paddlenlp.datasets import MapDataset
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman


@dataclass
class InputExample(object):
    """Data structure of every example in datasets."""
    uid: str = field(default=None,
                     metadata={'help': 'A unique identifier of the example.'})
    text_a: str = field(
        default=None,
        metadata={'help': 'The first text sequence in each example.'})
    text_b: str = field(
        default=None,
        metadata={'help': 'The other text sequences in each example.'})
    cls_label: int = field(
        default=None, metadata={'help': 'The label of classification tasks.'})
    seq_label: list = field(default=None,
                            metadata={'help': 'The label of generation tasks.'})
    meta: dict = field(
        default=None,
        metadata={
            'help': 'An optional dictionary of other data for each example.'
        })

    def __repr__(self):
        content = {k: v for k, v in self.__dict__.items() if v is not None}
        content = json.dumps(content, indent=2, sort_keys=True) + '\n'
        return str(content)

    def keys(self, keep_none=False):
        return [
            key for key in self.__dict__.keys()
            if getattr(self, key) is not None
        ]


class InputFeatures(dict):
    """
    Data structure of every wrapped example or a batch of examples as the input of model.
    
    Args:
        input_ids (paddle.Tensor):
            The token ids.
        attention_mask (paddle.Tensor):
            The mask ids.
        token_type_ids (paddle.Tensor, optional):
            The token type ids.
        input_embeds (paddle.Tensor, optional):
            The embeddings of soft tokens.
        mask_ids (paddle.Tensor, optional):
            The mask ids where 1 denotes that a token is a mask, 0 denotes it is not a mask.
        cls_label (list, optional):
            The label of classification task.
        seq_label (list, optional):
            The label of generation task.
        uid (list, optional):
            The unique id(s) for example(s).
    """
    input_keys = [
        'input_ids', 'attention_mask', 'token_type_ids', 'input_embeds',
        'cls_label', 'seq_label', 'label', 'uid', 'mask_ids', 'soft_token_ids'
    ]

    def __init__(self,
                 input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 input_embeds=None,
                 mask_ids=None,
                 label=None,
                 cls_label=None,
                 seq_label=None,
                 uid=None,
                 soft_token_ids=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_embeds = input_embeds
        self.label = label
        self.cls_label = cls_label
        self.seq_label = seq_label
        self.mask_ids = mask_ids
        self.uid = uid
        self.soft_token_ids = soft_token_ids

    @classmethod
    def add_keys(cls, *args):
        cls.input_keys.extend(args)

    def keys(self, keep_none=False):
        if keep_none:
            return self.input_keys
        else:
            return [
                key for key in self.input_keys if getattr(self, key) is not None
            ]

    def values(self, keep_none=False):
        return [getattr(self, key) for key in self.keys(keep_none=keep_none)]

    def items(self):
        return [(key, getattr(self, key)) for key in self.keys()]

    def __len__(self):
        return len(self.keys())

    def __repr__(self):
        return str(json.dumps(self.items()) + "\n")

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, key, keep_none):
        return key in self.keys(keep_none)

    def __setitem__(self, key, value):
        if key not in self.input_keys:
            raise KeyError(
                "{} not in predefined keys, use add_keys to add it.".format(
                    key))
        setattr(self, key, value)

    @staticmethod
    def collate_fn(batch):
        """Collate batch data in form of InputFeatures."""
        new_batch = {}
        for key in batch[0]:
            values = [b[key] for b in batch]
            try:
                new_batch[key] = paddle.to_tensor(values)
            except ValueError:
                new_batch[key] = values

        return InputFeatures(**new_batch)


class DataProcessor(object):
    """Base class for reading datasets from files."""

    def __init__(self, labels=None):
        self._labels = labels
        if labels is not None:
            self._labels = sorted(labels)

    @property
    def labels(self):
        if not getattr(self, '_labels'):
            raise ValueError('labels and label_mappings are not setted yet.')
        return self._labels

    @labels.setter
    def labels(self, labels):
        if labels is not None:
            self._labels = sorted(labels)

    @property
    def label_mapping(self):
        if not getattr(self, '_labels'):
            raise ValueError('labels and label_mappings are not setted yet.')
        if not getattr(self, '_label_mapping'):
            self._label_mapping = {k: i for i, k in enumerate(self._labels)}
        return self._label_mapping

    @label_mapping.setter
    def label_mapping(self, label_mapping):
        if getattr(self, '_labels'):
            assert self._labels == sorted(list(label_mapping.keys()))
        self._label_mapping = label_mapping

    @abstractmethod
    def get_examples(self, data_dir, split):
        raise NotImplementedError

    def get_train_examples(self, data_dir):
        return self.get_examples(data_dir, 'train')

    def get_dev_examples(self, data_dir):
        return self.get_examples(data_dir, 'dev')

    def get_test_exaples(self, data_dir):
        return self.get_examples(data_dir, 'test')

    @classmethod
    def read_tsv(cls, input_file, quotechar=None):
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            data = csv.reader(f, delimiter='\t', quotechar=quotechar)
            return [x for x in data]

    @classmethod
    def read_csv(cls, input_file, header=None):
        data = pd.read_csv(input_file, header=header)
        return data.values.tolist()

    @classmethod
    def read_json(cls, input_file):
        with open(input_file, 'r') as f:
            data = [json.loads(x) for x in f.readlines()]
            return data


class BoolQProcessor(DataProcessor):

    def __init__(self):
        super().__init__(['False', 'True'])
        self.split_map = {'train': 'train', 'dev': 'dev32', 'test': 'val'}

    def get_examples(self, data_dir, split):
        split = self.split_map[split]
        raw_data = self.read_json(os.path.join(data_dir, split + '.jsonl'))
        examples = []
        for i, line in enumerate(raw_data):
            examples.append(
                InputExample(uid='%s-%d' % (split, i),
                             text_a=line['passage'],
                             text_b=line['question'],
                             cls_label=str(line['label'])))

        return examples


class MrpcProcesser(DataProcessor):

    def __init__(self):
        super().__init__(['0', '1'])

    def get_examples(self, data_dir, split):
        raw_data = self.read_tsv(os.path.join(data_dir, split + '.tsv'))
        examples = []
        for i, line in enumerate(raw_data):
            if i == 0:
                continue
            examples.append(
                InputExample(uid='%s-%d' % (split, i),
                             text_a=line[3],
                             text_b=line[4],
                             cls_label=line[0]))

        return examples


class MnliProcessor(DataProcessor):

    def __init__(self):
        super().__init__(["contradiction", "entailment", "neutral"])

    def _process_file(self, split):
        if split in ['dev', 'test']:
            return split + '_matched'
        return split

    def get_examples(self, data_dir, split):
        split = self._process_file(split)
        raw_data = self.read_tsv(os.path.join(data_dir, split + '.tsv'))
        examples = []
        for i, line in enumerate(raw_data):
            if i == 0:
                continue
            examples.append(
                InputExample(uid='%s-%s' % (split, line[0]),
                             text_a=line[8],
                             text_b=line[9],
                             cls_label=line[-1]))
        return examples


class MnliMismatchedProcessor(MnliProcessor):

    def _process_file(self, split):
        if split == 'dev':
            return split + '_matched'
        if split == 'test':
            return split + '_mismatched'
        return split


class SnliProcessor(DataProcessor):

    def __init__(self):
        super().__init__(["contradiction", "entailment", "neutral"])

    def get_examples(self, data_dir, split):
        raw_data = self.read_tsv(os.path.join(data_dir, split + '.tsv'))
        examples = []
        for i, line in enumerate(raw_data):
            if i == 0:
                continue
            examples.append(
                InputExample(uid='%s-%s' % (split, line[0]),
                             text_a=line[7],
                             text_b=line[8],
                             cls_label=line[-1]))
        return examples


class ColaProcessor(DataProcessor):

    def __init__(self):
        super().__init__(['0', '1'])

    def get_examples(self, data_dir, split):
        raw_data = self.read_tsv(os.path.join(data_dir, split + '.tsv'))
        examples = []
        for i, line in enumerate(raw_data):
            examples.append(
                InputExample(uid='%s-%d' % (split, i),
                             text_a=line[3],
                             text_b=None,
                             cls_label=line[1]))
        return examples


class Sst2Processor(DataProcessor):

    def __init__(self):
        super().__init__(['0', '1'])

    def get_examples(self, data_dir, split):
        raw_data = self.read_tsv(os.path.join(data_dir, split + '.tsv'))
        examples = []
        for i, line in enumerate(raw_data):
            if i == 0:
                continue
            examples.append(
                InputExample(uid='%s-%d' % (split, i),
                             text_a=line[0],
                             text_b=None,
                             cls_label=line[1]))
        return examples


class StsbProcessor(DataProcessor):

    def __init__(self):
        super().__init__(['0', '1'])

    def get_examples(self, data_dir, split):
        raw_data = self.read_tsv(os.path.join(data_dir, split + '.tsv'))
        examples = []
        for i, line in enumerate(raw_data):
            if i == 0:
                continue
            examples.append(
                InputExample(uid='%s-%s' % (split, line[0]),
                             text_a=line[7],
                             text_b=line[8],
                             cls_label=line[-1]))
        return examples


class QqpProcessor(DataProcessor):

    def __init__(self):
        super().__init__(['0', '1'])

    def get_examples(self, data_dir, split):
        raw_data = self.read_tsv(os.path.join(data_dir, split + '.tsv'))
        examples = []
        for i, line in enumerate(raw_data):
            if i == 0:
                continue
            try:
                examples.append(
                    InputExample(uid='%s-%s' % (split, line[0]),
                                 text_a=line[3],
                                 text_b=line[4],
                                 cls_label=line[5]))
            except IndexError:
                continue
        return examples


class QnliProcessor(DataProcessor):

    def __init__(self):
        super().__init__(['entailment', 'not_entailment'])

    def get_examples(self, data_dir, split):
        raw_data = self.read_tsv(os.path.join(data_dir, split + '.tsv'))
        examples = []
        for i, line in enumerate(raw_data):
            if i == 0:
                continue
            examples.append(
                InputExample(uid='%s-%s' % (split, line[0]),
                             text_a=line[1],
                             text_b=line[2],
                             cls_label=line[-1]))
        return examples


class RteProcessor(DataProcessor):

    def __init__(self):
        super().__init__(['entailment', 'not_entailment'])

    def get_examples(self, data_dir, split):
        raw_data = self.read_tsv(os.path.join(data_dir, split + '.tsv'))
        examples = []
        for i, line in enumerate(raw_data):
            if i == 0:
                continue
            examples.append(
                InputExample(uid='%s-%s' % (split, line[0]),
                             text_a=line[1],
                             text_b=line[2],
                             cls_label=line[-1]))
        return examples


class WnliProcessor(DataProcessor):

    def __init__(self):
        super().__init__(['0', '1'])

    def get_examples(self, data_dir, split):
        raw_data = self.read_tsv(os.path.join(data_dir, split + '.tsv'))
        examples = []
        for i, line in enumerate(raw_data):
            if i == 0:
                continue
            examples.append(
                InputExample(uid='%s-%s' % (split, line[0]),
                             text_a=line[1],
                             text_b=line[2],
                             cls_label=line[-1]))
        return examples


class TextClassificationProcessor(DataProcessor):

    def __init__(self, task_name):
        NUM_LABELS = {
            'mr': 2,
            'sst-5': 5,
            'subj': 2,
            'trec': 6,
            'cr': 2,
            'mpqa': 2
        }
        assert task_name in NUM_LABELS, 'task_name not supported.'
        self.task_name = task_name
        self._labels = list(range(NUM_LABELS[self.task_name]))

    def get_examples(self, data_dir, split):
        raw_data = self.read_csv(os.path.join(data_dir, split + '.csv'))
        examples = []
        for i, line in enumerate(raw_data):
            examples.append(
                InputExample(uid='%s-%d' % (split, i),
                             text_a=line[1],
                             cls_label=line[0]))
        return examples


# The processor mapping for datasets in RGL paper.
PROCESSOR_MAPPING = {
    'mrpc': MrpcProcesser(),
    'mnli': MnliProcessor(),
    'mnli-mm': MnliMismatchedProcessor(),
    'snli': SnliProcessor(),
    'cola': ColaProcessor(),
    'sst-2': Sst2Processor(),
    'sts-b': StsbProcessor(),
    'qqp': QqpProcessor(),
    'qnli': QnliProcessor(),
    'rte': RteProcessor(),
    'wnli': WnliProcessor(),
    'cr': TextClassificationProcessor('cr'),
    'mr': TextClassificationProcessor('mr'),
    'sst-5': TextClassificationProcessor('sst-5'),
    'subj': TextClassificationProcessor('subj'),
    'mpqa': TextClassificationProcessor('mpqa'),
    'trec': TextClassificationProcessor('trec'),
    'boolq': BoolQProcessor()
}

# The task mapping for datasets.
TASK_MAPPING = defaultdict(lambda: 'classification')
TASK_MAPPING['sts-b'] = 'regression'

# The metric mapping for datasets.
METRIC_MAPPING = defaultdict(Accuracy)
METRIC_MAPPING.update({
    'mrpc':
    AccuracyAndF1(name=['acc', 'precision', 'recall', 'f1', 'acc_and_f1']),
    'qqp':
    AccuracyAndF1(name=['acc', 'precision', 'recall', 'f1', 'acc_and_f1']),
    'cola':
    Mcc(),
    'sts-b':
    PearsonAndSpearman(name=['pearson', 'spearman', 'corr'])
})


def load_dataset(dataset, data_path=None, splits=[]):
    """
    Read datasets from files.
    
    Args:
        dataset (str):
            The dataset name in lowercase.
        data_path (str):
            The path to the dataset directory, including train, dev or test file.
        splits (list):
            Which file(s) of dataset to read, such as ['train', 'dev', 'test'].

    """
    assert len(splits) > 0, 'No splits, can not load dataset {}'.format(dataset)
    processor = PROCESSOR_MAPPING[dataset]
    data = []
    if 'train' in splits:
        train_examples = processor.get_train_examples(data_path)
        data.append(MapDataset(train_examples))
    if 'dev' in splits:
        dev_examples = processor.get_dev_examples(data_path)
        data.append(MapDataset(dev_examples))
    if 'test' in splits:
        test_examples = processor.get_test_exaples(data_path)
        data.append(MapDataset(test_examples))
    data.append(processor.labels)
    return data
