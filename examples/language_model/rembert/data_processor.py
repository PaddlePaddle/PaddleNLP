# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from paddlenlp.transformers import RemBertTokenizer
import csv
from paddle.io import Dataset

tokenization = RemBertTokenizer.from_pretrained('rembert')


class InputExample(object):
    """
  Use classes to store each example
  """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class MrpcProcessor(object):
    """Load the dataset and convert each example text to ids"""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_2k.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_2k.tsv")), "test")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization(line[1])['input_ids']
            text_b = tokenization(line[2])['input_ids']
            label = int(line[3])
            examples.append(
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=text_b,
                             label=label))
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class XNLIProcessor(object):
    """Load the dataset and convert each example text to ids"""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "multinli.train.en.tsv")),
            "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "xnli.test.tsv")), "test")

    def get_labels(self):
        return ["neutral", "entailment", "contradictory"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == 'train':
                text_a = ' '.join(line[0].strip().split(' '))
                text_b = ' '.join(line[1].strip().split(' '))
                text_a = tokenization(text_a)['input_ids']
                text_b = tokenization(text_b)['input_ids']
                label = self.get_labels().index(line[2].strip())
                examples.append(
                    InputExample(guid=guid,
                                 text_a=text_a,
                                 text_b=text_b,
                                 label=label))
            else:
                text_a = ' '.join(line[6].strip().split(' '))
                text_b = ' '.join(line[7].strip().split(' '))
                if line[1] == 'contradiction':
                    line[1] = 'contradictory'
                label = self.get_labels().index(line[1].strip())
                text_a = tokenization(text_a)['input_ids']
                text_b = tokenization(text_b)['input_ids']
                examples.append(
                    InputExample(guid=guid,
                                 text_a=text_a,
                                 text_b=text_b,
                                 label=label))
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class DataGenerator(Dataset):
    """Data generator is used to feed features into dataloader."""

    def __init__(self, features):
        super(DataGenerator, self).__init__()
        self.features = features

    def __getitem__(self, item):
        text_a = self.features[item].text_a
        text_b = self.features[item].text_b
        text_a_token_type_ids = [0] * len(text_a)
        text_b_token_type_ids = [1] * len(text_b)
        label = [self.features[item].label]

        return dict(text_a=text_a,
                    text_b=text_b,
                    text_a_token_type_ids=text_a_token_type_ids,
                    text_b_token_type_ids=text_b_token_type_ids,
                    label=label)

    def __len__(self):
        return len(self.features)
