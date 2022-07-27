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
from collections import defaultdict
import os
import csv
import json
import inspect
from typing import Optional, Union, List
from dataclasses import dataclass, field

import numpy as np
import paddle
from ..utils.log import logger
from ..datasets import MapDataset

__all__ = ["InputExample", "InputFeatures", "FewShotSampler"]


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
    labels: int = field(default=None,
                        metadata={'help': 'The label in each example.'})
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
        inputs_embeds (paddle.Tensor, optional):
            The embeddings of soft tokens.
        mask_ids (paddle.Tensor, optional):
            The mask ids where 1 denotes that a token is a mask, 0 denotes it is not a mask.
        labels (list, optional):
            The labels of classification task.
        uid (list, optional):
            The unique id(s) for example(s).
    """
    input_keys = [
        'input_ids', 'attention_mask', 'token_type_ids', 'inputs_embeds', 'uid',
        'labels', 'mask_ids', 'soft_token_ids'
    ]
    tensorable = [
        'input_ids', 'attention_mask', 'token_type_ids', 'inputs_embeds',
        'labels', 'mask_ids', 'soft_token_ids'
    ]

    def __init__(self,
                 input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 inputs_embeds=None,
                 mask_ids=None,
                 labels=None,
                 uid=None,
                 soft_token_ids=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.inputs_embeds = inputs_embeds
        self.labels = labels
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

    @property
    def tensorable_keys(self, keep_none=False):
        if keep_none:
            return self.tensorable
        else:
            return [
                key for key in self.tensorable if getattr(self, key) is not None
            ]

    @tensorable_keys.setter
    def tensorable_keys(self, keys):
        diff_keys = set(keys) - set(self.input_keys)
        if len(diff_keys) > 0:
            raise ValueError("{} not in predefined keys.".format(
                ["`%s`" % k for k in diff_keys].join(", ")))
        self.tensorable = keys

    def values(self, keep_none=False):
        return [getattr(self, key) for key in self.keys(keep_none=keep_none)]

    def items(self):
        return [(key, getattr(self, key)) for key in self.keys()]

    def __len__(self):
        return len(self.keys())

    def __repr__(self):
        content = {}
        for key, value in self.items():
            if isinstance(value, paddle.Tensor):
                value = value.numpy()
            elif isinstance(value, paddle.static.Variable):
                value = value.to_string(True)
            content[key] = value
        return str(json.dumps(content) + "\n")

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, key, keep_none):
        return key in self.keys(keep_none)

    def __setitem__(self, key, value):
        if key not in self.input_keys:
            logger.warning(
                "`{}` is not a predefined key in InputFeatures. Perhaps it "\
                "brings unexpected results.".format(key))
        self.add_keys(key)
        setattr(self, key, value)

    @classmethod
    def collate_fn(cls, batch):
        """Collate batch data in form of InputFeatures."""
        new_batch = {}
        for key in batch[0]:
            values = [b[key] for b in batch]
            if key in cls.tensorable:
                new_batch[key] = paddle.to_tensor(values)
            else:
                new_batch[key] = values

        return InputFeatures(**new_batch)


def signature(fn):
    """
    Obtain the input arguments of the given function.
    """
    sig = inspect.signature(fn)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    return args


class FewShotSampler(object):
    """
    Sampling from datasets for few-shot learning.
    Args:
        dataset
    """

    def __init__(self,
                 num_sample_per_label=None,
                 num_sample_total=None,
                 eval_num_sample_per_label=None,
                 eval_num_sample_total=None):
        if num_sample_per_label is None and num_sample_total is None:
            raise ValueError("Either `num_sample_per_label` or `num_sample_total`"\
                             " should be set.")
        if num_sample_per_label is not None and num_sample_total is not None:
            logger.info(
                "`num_sample_per_label` will overwrite `num_sample_total`.")
            self.num_sample_per_label = num_sample_per_label
            self.num_sample_total = None
        else:
            self.num_sample_per_label = num_sample_per_label
            self.num_sample_total = num_sample_total

        if eval_num_sample_per_label is None and eval_num_sample_total is None:
            self.eval_num_sample_per_label = self.num_sample_per_label
            self.eval_num_sample_total = self.num_sample_total
        elif eval_num_sample_per_label is not None and eval_num_sample_total is not None:
            logger.info(
                "`eval_num_sample_per_label` will overwrite `eval_num_sample_total`."
            )
            self.eval_num_sample_per_label = eval_num_sample_per_label
            self.eval_num_sample_total = None
        else:
            self.eval_num_sample_per_label = eval_num_sample_per_label
            self.eval_num_sample_total = eval_num_sample_total

    def sample_datasets(self, train_dataset, dev_dataset=None, seed=None):
        """
        Sample from every given dataset seperately.
        """
        self.rng = np.random.RandomState(seed)
        indices = np.arange(len(train_dataset))
        labels = [x.labels for x in train_dataset]
        train_indices = self._sample(indices, labels, self.num_sample_per_label,
                                     self.num_sample_total)
        logger.info(f"{len(train_indices)} examples sampled for train dataset.")
        train_ds = MapDataset([train_dataset[i] for i in train_indices],
                              label_list=train_dataset.label_list)

        if dev_dataset is None:
            return train_ds

        indices = np.arange(len(dev_dataset))
        labels = [x.labels for x in dev_dataset]
        eval_indices = self._sample(indices, labels,
                                    self.eval_num_sample_per_label,
                                    self.eval_num_sample_total)
        logger.info(f"{len(train_indices)} examples sampled for train dataset.")
        dev_ds = MapDataset([dev_dataset[i] for i in eval_indices],
                            label_list=dev_dataset.label_list)
        return train_ds, dev_ds

    def sample_and_partition(self, dataset, seed=None):
        """
        Sample from a single dataset and divide it into train, dev and test.
        """
        self.rng = np.random.RandomState(seed)
        total_indices = np.arange(len(dataset))
        total_labels = [x.labels for x in dataset]
        train_indices = self._sample(total_indices, total_labels,
                                     self.num_sample_per_label,
                                     self.num_sample_total)
        logger.info(f"{len(train_indices)} examples sampled for train dataset.")

        non_train_indices = [
            i for i in total_indices if i not in set(train_indices)
        ]
        non_train_labels = [total_labels[i] for i in non_train_indices]
        eval_indices = self._sample(non_train_indices, non_train_labels,
                                    self.eval_num_sample_per_label,
                                    self.eval_num_sample_total)
        logger.info(f"{len(eval_indices)} examples sampled for dev dataset.")

        test_indices = [
            i for i in non_train_indices if i not in set(eval_indices)
        ]
        logger.info(f"{len(test_indices)} examples left as test dataset.")

        train_ds = MapDataset([dataset.data[i] for i in train_indices],
                              label_list=dataset.label_list)
        dev_ds = MapDataset([dataset.data[i] for i in eval_indices],
                            label_list=dataset.label_list)
        test_ds = MapDataset([dataset.data[i] for i in test_indices],
                             label_list=dataset.label_list)
        return train_ds, dev_ds, test_ds

    def _sample(self, indices, labels, num_per_label, num_total):
        if num_per_label is not None:
            sampled_ids = self._sample_per_label(indices, labels, num_per_label)
        else:
            sampled_ids = self._sample_random(indices, num_total)
        return sampled_ids

    def _sample_random(self, indices, num_sample):
        if num_sample > len(indices):
            logger.info("Number to sample exceeds the number of examples " +
                        f"remaining. Only {len(indices)} sampled.")
        self.rng.shuffle(indices)
        return indices[:num_sample]

    def _sample_per_label(self, indices, labels, num_per_label):
        label_dict = defaultdict(list)
        for idx, label in zip(indices, labels):
            label_dict[label].append(idx)

        sampled = []
        for label, index in label_dict.items():
            if num_per_label > len(index):
                logger.info("Number to sample exceeds the number of examples" +
                            f" with label {label}, {len(index)} sampled.")
            self.rng.shuffle(index)
            sampled.extend(index[:num_per_label])

        return sampled
