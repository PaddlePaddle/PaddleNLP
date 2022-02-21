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
"""squad_data process file"""
import os
import joblib
import logging
from .wiki_link_db import WikiLinkDB
from .dataset import SquadV2Processor, SquadV1Processor
from .feature import convert_examples_to_features
from paddle.io import Dataset, DataLoader
import numpy as np
from paddlenlp.transformers import LukeTokenizer


def collate_fn(batch):
    def create_padded_sequence(k, padding_value):
        """Pad sequence to maximum length"""
        new_data = []
        max_len = 0
        for each_batch in batch:
            # find the max length
            if len(each_batch[k]) > max_len:
                max_len = len(each_batch[k])
        for each_batch in batch:
            # pad to max length
            new_data.append(each_batch[k] + [padding_value] * (
                    max_len - len(each_batch[k])))
        return np.array(new_data, dtype='int64')

    return (
        create_padded_sequence(0, 1),  # pad word_ids
        create_padded_sequence(1, 0),  # pad word_segment_ids
        create_padded_sequence(2, 0),  # pad word_attention_mask
        create_padded_sequence(3, 0),  # pad entity_ids
        create_padded_sequence(4, 0),  # pad entity_position_ids
        create_padded_sequence(5, 0),  # pad entity_segment_ids
        create_padded_sequence(6, 0),  # pad entity_attention_mask
        create_padded_sequence(7, 0),  # convert to numpy array
        create_padded_sequence(8, 0),  # convert to numpy array
        create_padded_sequence(9, 0),  # convert to numpy array
    )


def load_examples(args, data_file='train-v1.json'):
    """load examples"""

    tokenizer = LukeTokenizer.from_pretrained(args.model_type)

    wiki_link_db = WikiLinkDB(os.path.join(args.data_dir, args.wiki_link_db_file))
    model_redirect_mappings = joblib.load(os.path.join(args.data_dir, args.model_redirects_file))
    link_redirect_mappings = joblib.load(os.path.join(args.data_dir, args.link_redirects_file))

    if args.with_negative:
        processor = SquadV2Processor()
    else:
        processor = SquadV1Processor()

    if 'train' not in data_file:
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_train_examples(args.data_dir)

    logging.info("Creating features from the dataset...")
    features = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        entity_vocab=tokenizer.entity_vocab,
        wiki_link_db=wiki_link_db,
        model_redirect_mappings=model_redirect_mappings,
        link_redirect_mappings=link_redirect_mappings,
        max_seq_length=args.max_seq_length,
        max_mention_length=args.max_mention_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        min_mention_link_prob=args.min_mention_link_prob,
        segment_b_id=0,
        add_extra_sep_token=True,
        is_training='train' in data_file)

    dataset = DataGenerator(features, 'train' in data_file)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size,
                            shuffle='train' in data_file, collate_fn=collate_fn)

    return dataloader, examples, features, processor


class DataGenerator(Dataset):
    def __init__(self, features, is_train=False):
        super(DataGenerator, self).__init__()
        self.features = features
        self.is_train = is_train

    def __getitem__(self, item):
        word_ids = self.features[item].word_ids
        word_segment_ids = self.features[item].word_segment_ids
        word_attention_mask = self.features[item].word_attention_mask
        entity_ids = self.features[item].entity_ids
        entity_position_ids = self.features[item].entity_position_ids
        entity_segment_ids = self.features[item].entity_segment_ids
        entity_attention_mask = self.features[item].entity_attention_mask
        start_positions = [self.features[item].start_positions[0] if self.is_train else self.features[item].start_positions]
        end_positions = [self.features[item].end_positions[0] if self.is_train else self.features[item].end_positions]
        example_index = [self.features[item].example_index]
        return (word_ids, word_segment_ids, word_attention_mask,
                entity_ids, entity_position_ids, entity_segment_ids,
                entity_attention_mask, start_positions, end_positions,
                example_index)

    def __len__(self):
        return len(self.features)
