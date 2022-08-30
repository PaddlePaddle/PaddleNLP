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

import json
import os

from tqdm import tqdm

ENTITY_TOKEN = "[ENTITY]"


class InputExample(object):

    def __init__(self, id_, text, span, labels):
        self.id = id_
        self.text = text
        self.span = span
        self.labels = labels


class InputFeatures(object):

    def __init__(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        labels,
    ):
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids
        self.entity_segment_ids = entity_segment_ids
        self.entity_attention_mask = entity_attention_mask
        self.labels = labels


class DatasetProcessor(object):

    def get_train_examples(self, data_dir):
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(data_dir, "test")

    def get_label_list(self, data_dir):
        labels = set()
        for example in self.get_train_examples(data_dir):
            labels.update(example.labels)
        return sorted(labels)

    def _create_examples(self, data_dir, set_type):
        with open(os.path.join(data_dir, set_type + ".json"), "r") as f:
            data = json.load(f)
        return [
            InputExample(i, item["sent"], (item["start"], item["end"]),
                         item["labels"]) for i, item in enumerate(data)
        ]


def convert_examples_to_features(examples, label_list, tokenizer,
                                 max_mention_length):
    label_map = {label: i for i, label in enumerate(label_list)}

    conv_tables = (
        ("-LRB-", "("),
        ("-LCB-", "("),
        ("-LSB-", "("),
        ("-RRB-", ")"),
        ("-RCB-", ")"),
        ("-RSB-", ")"),
    )
    features = []
    for example in tqdm(examples):

        def preprocess_and_tokenize(text, start, end=None):
            target_text = text[start:end].rstrip()
            for a, b in conv_tables:
                target_text = target_text.replace(a, b)

            return tokenizer.tokenize(target_text, add_prefix_space=True)

        tokens = [tokenizer.cls_token]
        tokens += preprocess_and_tokenize(example.text, 0, example.span[0])
        mention_start = len(tokens)
        tokens.append(ENTITY_TOKEN)
        tokens += preprocess_and_tokenize(example.text, example.span[0],
                                          example.span[1])
        tokens.append(ENTITY_TOKEN)
        mention_end = len(tokens)

        tokens += preprocess_and_tokenize(example.text, example.span[1])
        tokens.append(tokenizer.sep_token)

        word_ids = tokenizer.convert_tokens_to_ids(tokens)
        word_attention_mask = [1] * len(tokens)
        word_segment_ids = [0] * len(tokens)

        entity_ids = [2, 0]
        entity_attention_mask = [1, 0]
        entity_segment_ids = [0, 0]
        entity_position_ids = list(range(mention_start,
                                         mention_end))[:max_mention_length]
        entity_position_ids += [-1] * (max_mention_length - mention_end +
                                       mention_start)
        entity_position_ids = [entity_position_ids, [-1] * max_mention_length]

        labels = [0] * len(label_map)

        for label in example.labels:
            labels[label_map[label]] = 1

        features.append(
            InputFeatures(
                word_ids=word_ids,
                word_segment_ids=word_segment_ids,
                word_attention_mask=word_attention_mask,
                entity_ids=entity_ids,
                entity_position_ids=entity_position_ids,
                entity_segment_ids=entity_segment_ids,
                entity_attention_mask=entity_attention_mask,
                labels=labels,
            ))

    return features
