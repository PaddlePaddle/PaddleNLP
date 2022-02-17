#encoding=utf8
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

import logging
import argparse
from paddle.io import Dataset, DataLoader
import numpy as np
from paddlenlp.transformers import LukeTokenizer
from paddlenlp.transformers import LukeForEntityClassification
from utils.processor import *
import paddle
import json
from tqdm import tqdm
from utils.trainer import Trainer
import os

parser = argparse.ArgumentParser(description="LUKE FOR OPEN ENTITY")

parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--do_eval", type=bool, default=True)
parser.add_argument("--do_train", type=bool, default=True)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--num_train_epochs", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train_batch_size", type=int, default=2)
parser.add_argument("--device", type=str, default='gpu')
parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_proportion", type=float, default=0.06)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--adam_b1", type=float, default=0.9)
parser.add_argument("--adam_b2", type=float, default=0.98)
parser.add_argument("--model_type", type=str, default='luke-base')
parser.add_argument("--max_mention_length", type=str, default=30)

args = parser.parse_args()
args.tokenizer = LukeTokenizer.from_pretrained(args.model_type)
args.entity_vocab = args.tokenizer.get_entity_vocab()
args.tokenizer.add_special_tokens(
    dict(additional_special_tokens=[ENTITY_TOKEN]))


class DataGenerator(Dataset):
    def __init__(self, features, args):
        super(DataGenerator, self).__init__()
        self.args = args
        self.all_word_ids = [f.word_ids for f in features]
        self.all_word_segment_ids = [f.word_segment_ids for f in features]
        self.all_word_attention_mask = [f.word_attention_mask for f in features]
        self.all_entity_ids = [f.entity_ids for f in features]
        self.all_entity_position_ids = [f.entity_position_ids for f in features]
        self.all_entity_segment_ids = [f.entity_segment_ids for f in features]
        self.all_entity_attention_mask = [
            f.entity_attention_mask for f in features
        ]
        self.all_labels = [f.labels for f in features]

    def __getitem__(self, item):
        word_ids = self.all_word_ids[item]
        word_segment_ids = self.all_word_segment_ids[item]
        word_attention_mask = self.all_word_attention_mask[item]
        entity_ids = self.all_entity_ids[item]
        entity_position_ids = self.all_entity_position_ids[item]
        entity_segment_ids = self.all_entity_segment_ids[item]
        entity_attention_mask = self.all_entity_attention_mask[item]
        label = self.all_labels[item]

        return word_ids, \
               word_segment_ids, \
               word_attention_mask, \
               entity_ids, \
               entity_position_ids, \
               entity_segment_ids, \
               entity_attention_mask, \
               label

    def __len__(self):
        return len(self.all_word_ids)


@paddle.no_grad()
def evaluate(args, model, fold="dev", output_file=None):
    dataloader, _, _, label_list = load_examples(args, fold=fold)
    model.eval()

    all_logits = []
    all_labels = []

    for batch in tqdm(dataloader, desc=fold):
        logits = model(
            word_ids=batch[0],
            word_segment_ids=batch[1],
            word_attention_mask=batch[2],
            entity_ids=batch[3],
            entity_position_ids=batch[4],
            entity_segment_ids=batch[5],
            entity_attention_mask=batch[6],
            labels=None)

        logits = logits.tolist()
        labels = batch[7].tolist()

        all_logits.extend(logits)
        all_labels.extend(labels)

    all_predicted_indexes = []
    all_label_indexes = []
    for logits, labels in zip(all_logits, all_labels):
        all_predicted_indexes.append([i for i, v in enumerate(logits) if v > 0])
        all_label_indexes.append([i for i, v in enumerate(labels) if v > 0])

    if output_file:
        with open(output_file, "w") as f:
            for predicted_indexes, label_indexes in zip(all_predicted_indexes,
                                                        all_label_indexes):
                data = dict(
                    predictions=[label_list[ind] for ind in predicted_indexes],
                    labels=[label_list[ind] for ind in label_indexes], )
                f.write(json.dumps(data) + "\n")

    num_predicted_labels = 0
    num_gold_labels = 0
    num_correct_labels = 0

    for predicted_indexes, label_indexes in zip(all_predicted_indexes,
                                                all_label_indexes):
        num_predicted_labels += len(predicted_indexes)
        num_gold_labels += len(label_indexes)
        num_correct_labels += len(
            frozenset(predicted_indexes).intersection(
                frozenset(label_indexes)))

    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.0

    recall = num_correct_labels / num_gold_labels
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return dict(precision=precision, recall=recall, f1=f1)


def load_examples(args, fold="train"):
    processor = DatasetProcessor()
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    label_list = processor.get_label_list(args.data_dir)

    logging.info("Creating features from the dataset...")
    features = convert_examples_to_features(
        examples, label_list, args.tokenizer, args.max_mention_length)

    data_generator = DataGenerator(features, args)

    def collate_fn(batch):
        def create_padded_sequence(k, padding_value):
            new_data = []
            max_len = 0
            for each_batch in batch:
                if len(each_batch[k]) > max_len:
                    max_len = len(each_batch[k])
            for each_batch in batch:
                new_data.append(each_batch[k] + [padding_value] * (
                    max_len - len(each_batch[k])))
            return np.array(new_data, dtype='int64')

        return (
            create_padded_sequence(0, 1),
            create_padded_sequence(1, 0),
            create_padded_sequence(2, 0),
            create_padded_sequence(3, 0),
            create_padded_sequence(4, 0),
            create_padded_sequence(5, 0),
            create_padded_sequence(6, 0),
            create_padded_sequence(7, 0), )

    if fold in ("dev", "test"):
        dataloader = DataLoader(
            data_generator,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn)
    else:
        dataloader = DataLoader(
            data_generator,
            shuffle=True,
            batch_size=args.train_batch_size,
            collate_fn=collate_fn)

    return dataloader, examples, features, label_list


if __name__ == '__main__':
    results = {}
    train_dataloader, _, features, _ = load_examples(args, fold="train")
    num_labels = len(features[0].labels)
    num_train_steps_per_epoch = len(
        train_dataloader) // args.gradient_accumulation_steps
    num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)
    model = LukeForEntityClassification.from_pretrained(
        args.model_type, num_labels=num_labels)
    trainer = Trainer(
        args,
        model=model,
        dataloader=train_dataloader,
        num_train_steps=num_train_steps)
    trainer.train(is_op=True)
    output_file = os.path.join(args.output_dir, f"test_predictions.jsonl")
    results.update({
        f"test_{k}": v
        for k, v in evaluate(args, model, 'test', output_file).items()
    })

    print("Results: %s", json.dumps(results, indent=2, sort_keys=True))
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f)
