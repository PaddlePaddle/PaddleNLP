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

import os
import json
import numpy as np
from collections import defaultdict
import random

import paddle
from paddlenlp.utils.log import logger
from paddlenlp.datasets import MapDataset


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=batchify_fn,
                                return_list=True)


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    sentence1 = example["sentence1"]
    sentence2 = example["sentence2"]
    encoded_inputs = tokenizer(text=sentence1,
                               text_pair=sentence2,
                               max_seq_len=max_seq_length,
                               truncation_strategy="only_first")

    src_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = example["label"]

    if is_test:
        return src_ids, token_type_ids
    else:
        return src_ids, token_type_ids, label


class DataProcessor(object):
    """Base class for data converters for sequence classification datasets."""

    def __init__(self, negative_num=1):
        # Random negative sample number for efl strategy
        self.neg_num = negative_num

    def get_train_datasets(self, datasets, task_label_description):
        """See base class."""
        return self._create_examples(datasets, "train", task_label_description)

    def get_dev_datasets(self, datasets, task_label_description):
        """See base class."""
        return self._create_examples(datasets, "dev", task_label_description)

    def get_test_datasets(self, datasets, task_label_description):
        """See base class."""
        return self._create_examples(datasets, "test", task_label_description)


class IflytekProcessor(DataProcessor):
    """Processor for the IFLYTEK dataset (CLUE version)."""

    def _create_examples(self, datasets, phase, task_label_description):
        """Creates examples for the training and dev sets."""

        examples = []

        if phase == "train":
            for example in datasets:
                true_label = str(example["label"])
                neg_examples = []
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description

                    # Todo: handle imbanlanced example, maybe hurt model performance
                    if true_label == label:
                        new_example["label"] = 1
                        examples.append(new_example)
                    else:
                        new_example["label"] = 0
                        neg_examples.append(new_example)
                neg_examples = random.sample(neg_examples, self.neg_num)
                examples.extend(neg_examples)

        elif phase == "dev":
            for example in datasets:
                true_label = str(example["label"])
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description
                    # Get true_label's index at task_label_description for evaluate
                    true_label_index = list(
                        task_label_description.keys()).index(true_label)
                    new_example["label"] = true_label_index
                    examples.append(new_example)

        elif phase == "test":
            for example in datasets:
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description
                    examples.append(new_example)

        return MapDataset(examples)


class OcnliProcessor(DataProcessor):
    """Processor for the IFLYTEK dataset (CLUE version)."""

    def _create_examples(self, datasets, phase, task_label_description):
        """Creates examples for the training and dev sets."""

        examples = []

        if phase == "train":
            for example in datasets:
                true_label = str(example["label"])
                neg_examples = []
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence1']
                    new_example[
                        "sentence2"] = label_description + example['sentence2']

                    # Todo: handle imbanlanced example, maybe hurt model performance
                    if true_label == label:
                        new_example["label"] = 1
                        examples.append(new_example)
                    else:
                        new_example["label"] = 0
                        neg_examples.append(new_example)
                neg_examples = random.sample(neg_examples, self.neg_num)
                examples.extend(neg_examples)

        elif phase == "dev":
            for example in datasets:
                true_label = example["label"]
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence1']
                    new_example[
                        "sentence2"] = label_description + example['sentence2']
                    # Get true_label's index at task_label_description for evaluate
                    true_label_index = list(
                        task_label_description.keys()).index(true_label)
                    new_example["label"] = true_label_index
                    examples.append(new_example)

        elif phase == "test":
            for example in datasets:
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence1']
                    new_example[
                        "sentence2"] = label_description + example['sentence2']
                    examples.append(new_example)

        return MapDataset(examples)


class TnewsProcessor(DataProcessor):
    """Processor for the Tnews dataset (CLUE version)."""

    def _create_examples(self, datasets, phase, task_label_description):
        """Creates examples for the training and dev sets."""

        examples = []
        if phase == "train":
            for example in datasets:
                true_label = str(example["label"])
                neg_examples = []
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description

                    # Todo: handle imbanlanced example, maybe hurt model performance
                    if true_label == label:
                        new_example["label"] = 1
                        examples.append(new_example)
                    else:
                        new_example["label"] = 0
                        neg_examples.append(new_example)
                neg_examples = random.sample(neg_examples, self.neg_num)
                examples.extend(neg_examples)

        elif phase == "dev":
            for example in datasets:
                true_label = str(example["label"])
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description

                    # Get true_label's index at task_label_description for evaluate
                    true_label_index = list(
                        task_label_description.keys()).index(true_label)
                    new_example["label"] = true_label_index
                    examples.append(new_example)

        elif phase == "test":
            for example in datasets:
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description
                    examples.append(new_example)

        return MapDataset(examples)


class BustmProcessor(DataProcessor):
    """Processor for the Bustum dataset (CLUE version)."""

    def _create_examples(self, datasets, phase, task_label_description):
        """Creates examples for the training and dev sets."""

        examples = []

        if phase == "train":
            for example in datasets:
                true_label = example["label"]
                neg_examples = []
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence1']
                    new_example[
                        "sentence2"] = label_description + example['sentence2']

                    # Todo: handle imbanlanced example, maybe hurt model performance
                    if true_label == label:
                        new_example["label"] = 1
                        examples.append(new_example)
                    else:
                        new_example["label"] = 0
                        neg_examples.append(new_example)
                neg_examples = random.sample(neg_examples, self.neg_num)
                examples.extend(neg_examples)

        elif phase == "dev":
            for example in datasets:
                true_label = example["label"]
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence1']
                    new_example[
                        "sentence2"] = label_description + example['sentence2']
                    # Get true_label's index at task_label_description for evaluate
                    true_label_index = list(
                        task_label_description.keys()).index(true_label)
                    new_example["label"] = true_label_index
                    examples.append(new_example)

        elif phase == "test":
            for example in datasets:
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence1']
                    new_example[
                        "sentence2"] = label_description + example['sentence2']
                    examples.append(new_example)

        return MapDataset(examples)


class EprstmtProcessor(DataProcessor):
    """Processor for the Eprstmt dataset (CLUE version)."""

    def _create_examples(self, datasets, phase, task_label_description):
        """Creates examples for the training and dev sets."""

        examples = []

        if phase == "train":
            for example in datasets:
                true_label = example["label"]
                neg_examples = []
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description

                    # Todo: handle imbanlanced example, maybe hurt model performance
                    if true_label == label:
                        new_example["label"] = 1
                        examples.append(new_example)
                    else:
                        new_example["label"] = 0
                        neg_examples.append(new_example)
                neg_examples = random.sample(neg_examples, self.neg_num)
                examples.extend(neg_examples)

        elif phase == "dev":
            for example in datasets:
                true_label = str(example["label"])
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description

                    # Get true_label's index at task_label_description for evaluate
                    true_label_index = list(
                        task_label_description.keys()).index(true_label)
                    new_example["label"] = true_label_index
                    examples.append(new_example)

        elif phase == "test":
            for example in datasets:
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description
                    examples.append(new_example)

        return MapDataset(examples)


class CsldcpProcessor(DataProcessor):
    """Processor for the Csldcp dataset (CLUE version)."""

    def _create_examples(self, datasets, phase, task_label_description):
        """Creates examples for the training and dev sets."""

        examples = []

        if phase == "train":
            for example in datasets:
                true_label = example["label"]
                neg_examples = []
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['content']
                    new_example["sentence2"] = label_description

                    # Todo: handle imbanlanced example, maybe hurt model performance
                    if true_label == label:
                        new_example["label"] = 1
                        examples.append(new_example)
                    else:
                        new_example["label"] = 0
                        neg_examples.append(new_example)
                neg_examples = random.sample(neg_examples, self.neg_num)
                examples.extend(neg_examples)

        elif phase == "dev":
            for example in datasets:
                true_label = str(example["label"])
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['content']
                    new_example["sentence2"] = label_description

                    # Get true_label's index at task_label_description for evaluate
                    true_label_index = list(
                        task_label_description.keys()).index(true_label)
                    new_example["label"] = true_label_index
                    examples.append(new_example)

        elif phase == "test":
            for example in datasets:
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['content']
                    new_example["sentence2"] = label_description
                    examples.append(new_example)

        return MapDataset(examples)


class CslProcessor(DataProcessor):
    """Processor for the Csl dataset (CLUE version)."""

    def _create_examples(self, datasets, phase, task_label_description):
        """Creates examples for the training and dev sets."""

        examples = []

        if phase == "train":
            for example in datasets:
                true_label = example["label"]
                neg_examples = []
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['abst']
                    new_example["sentence2"] = label_description + " ".join(
                        example['keyword'])

                    # Todo: handle imbanlanced example, maybe hurt model performance
                    if true_label == label:
                        new_example["label"] = 1
                        examples.append(new_example)
                    else:
                        new_example["label"] = 0
                        neg_examples.append(new_example)
                neg_examples = random.sample(neg_examples, self.neg_num)
                examples.extend(neg_examples)

        elif phase == "dev":
            for example in datasets:
                true_label = str(example["label"])
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['abst']
                    new_example["sentence2"] = label_description + " ".join(
                        example['keyword'])

                    # Get true_label's index at task_label_description for evaluate
                    true_label_index = list(
                        task_label_description.keys()).index(true_label)
                    new_example["label"] = true_label_index
                    examples.append(new_example)

        elif phase == "test":
            for example in datasets:
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['abst']
                    new_example["sentence2"] = label_description + " ".join(
                        example['keyword'])
                    examples.append(new_example)

        return MapDataset(examples)


class CluewscProcessor(DataProcessor):
    """Processor for the ClueWSC dataset (CLUE version)."""

    def _create_examples(self, datasets, phase, task_label_description):
        """Creates examples for the training and dev sets."""

        examples = []

        if phase == "train":
            for example in datasets:
                true_label = example["label"]
                neg_examples = []
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example["text"]
                    new_example["sentence2"] = example["target"][
                        "span1_text"] + label_description + example["target"][
                            "span2_text"]

                    # Todo: handle imbanlanced example, maybe hurt model performance
                    if true_label == label:
                        new_example["label"] = 1
                        examples.append(new_example)
                    else:
                        new_example["label"] = 0
                        neg_examples.append(new_example)
                neg_examples = random.sample(neg_examples, self.neg_num)
                examples.extend(neg_examples)

        elif phase == "dev":
            for example in datasets:
                true_label = str(example["label"])
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example["text"]
                    new_example["sentence2"] = example["target"][
                        "span1_text"] + label_description + example["target"][
                            "span2_text"]

                    # Get true_label's index at task_label_description for evaluate
                    true_label_index = list(
                        task_label_description.keys()).index(true_label)
                    new_example["label"] = true_label_index
                    examples.append(new_example)

        elif phase == "test":
            for example in datasets:
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example["text"]
                    new_example["sentence2"] = example["target"][
                        "span1_text"] + label_description + example["target"][
                            "span2_text"]
                    examples.append(new_example)

        return MapDataset(examples)


class ChidProcessor(DataProcessor):
    """Processor for the CHID dataset (CLUE version)."""

    def _create_examples(self, datasets, phase, task_label_description):
        """Creates examples for the training and dev sets."""

        examples = []

        if phase == "train":
            for example in datasets:
                neg_examples = []
                true_label_index = int(example["answer"])
                candidates = example["candidates"]
                for idx, cantidate in enumerate(candidates):
                    new_example = dict()
                    new_example["sentence1"] = example["content"]
                    new_example["sentence2"] = "位置#idiom#处的成语应该填写" + cantidate

                    if idx == true_label_index:
                        new_example["label"] = 1
                        examples.append(new_example)
                    else:
                        new_example["label"] = 0
                        neg_examples.append(new_example)
                examples.extend(neg_examples)

        elif phase == "dev":
            for example in datasets:
                true_label = str(example["answer"])
                candidates = example["candidates"]
                for idx, cantidate in enumerate(candidates):
                    new_example = dict()
                    new_example["sentence1"] = example["content"]
                    new_example["sentence2"] = "位置#idiom#处的成语应该填写" + cantidate

                    # Get true_label's index at task_label_description for evaluate
                    true_label_index = int(true_label)
                    new_example["label"] = true_label_index
                    examples.append(new_example)

        elif phase == "test":
            for example in datasets:
                candidates = example["candidates"]
                for idx, cantidate in enumerate(candidates):
                    new_example = dict()
                    new_example["sentence1"] = example["content"]
                    new_example["sentence2"] = "位置#idiom#处的成语应该填写" + cantidate
                    examples.append(new_example)

        return MapDataset(examples)


processor_dict = {
    "iflytek": IflytekProcessor,
    "tnews": TnewsProcessor,
    "eprstmt": EprstmtProcessor,
    "bustm": BustmProcessor,
    "ocnli": OcnliProcessor,
    "csl": CslProcessor,
    "csldcp": CsldcpProcessor,
    "cluewsc": CluewscProcessor,
    "chid": ChidProcessor
}
