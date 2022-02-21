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
from utils.open_entity_processor import convert_examples_to_features, DatasetProcessor
import paddle
import json
from tqdm import tqdm
from utils.trainer import Trainer
import os

ENTITY_TOKEN = "[ENTITY]"

parser = argparse.ArgumentParser(description="LUKE FOR OPEN ENTITY")

parser.add_argument("--output_dir",
                    type=str,
                    required=True,
                    help="Use to store all outputs during training and evaluation.")
parser.add_argument("--data_dir",
                    type=str,
                    required=True,
                    help="Dataset folder")
parser.add_argument("--eval_batch_size",
                    type=int,
                    default=32,
                    help="Batch size per GPU/CPU for evaluating.")
parser.add_argument("--num_train_epochs",
                    type=int,
                    default=2,
                    help="Number of training cycles")
parser.add_argument("--seed",
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument("--train_batch_size",
                    type=int,
                    default=8,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--device",
                    type=str,
                    default='gpu',
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--gradient_accumulation_steps",
                    type=int,
                    default=3,
                    help="Gradient accumulated before each parameter update.")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.01,
                    help="Weight decay if we apply some")
parser.add_argument("--warmup_proportion",
                    type=float,
                    default=0.06,
                    help="Proportion of training steps to perform linear learning rate warmup for.")
parser.add_argument("--learning_rate",
                    type=float,
                    default=1e-5,
                    help="The initial learning rate for Adam.")
parser.add_argument("--model_type",
                    type=str,
                    default='luke-base',
                    help="Type of pre-trained model.")
parser.add_argument("--max_mention_length",
                    type=int,
                    default=30,
                    help="Max entity position's length")

args = parser.parse_args()


class DataGenerator(Dataset):
    def __init__(self, features):
        super(DataGenerator, self).__init__()
        self.features = features

    def __getitem__(self, item):
        word_ids = self.features[item].word_segment_ids
        word_segment_ids = self.features[item].word_segment_ids
        word_attention_mask = self.features[item].word_attention_mask
        entity_ids = self.features[item].entity_ids
        entity_position_ids = self.features[item].entity_position_ids
        entity_segment_ids = self.features[item].entity_segment_ids
        entity_attention_mask = self.features[item].entity_attention_mask
        labels = self.features[item].labels

        return (word_ids,
                word_segment_ids,
                word_attention_mask,
                entity_ids,
                entity_position_ids,
                entity_segment_ids,
                entity_attention_mask,
                labels)

    def __len__(self):
        return len(self.features)


@paddle.no_grad()
def evaluate(args, model, fold="dev", output_file=None):
    dataloader, _, _, label_list = load_examples(args, fold=fold)
    model.eval()

    all_logits = []
    all_labels = []

    for batch in tqdm(dataloader, desc=fold):
        logits = model(
            input_ids=batch[0],
            token_type_ids=batch[1],
            attention_mask=batch[2],
            entity_ids=batch[3],
            entity_position_ids=batch[4],
            entity_segment_ids=batch[5],
            entity_attention_mask=batch[6])

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
    tokenizer = LukeTokenizer.from_pretrained(args.model_type)
    tokenizer.add_special_tokens(
        dict(additional_special_tokens=[ENTITY_TOKEN]))
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
        examples, label_list, tokenizer, args.max_mention_length)

    dataset = DataGenerator(features)

    def collate_fn(batch):
        def create_padded_sequence(k, padding_value):
            """Pad sequence to maximum length"""
            new_data = []
            max_len = 0
            for each_batch in batch:
                if len(each_batch[k]) > max_len:
                    max_len = len(each_batch[k])
            for each_batch in batch:
                new_data.append(each_batch[k] + [padding_value] * (
                        max_len - len(each_batch[k])))
            return np.array(new_data, dtype='int64')

        return (create_padded_sequence(0, 1),  # pad word_ids
                create_padded_sequence(1, 0),  # pad word_segment_ids
                create_padded_sequence(2, 0),  # pad word_attention_mask
                create_padded_sequence(3, 0),  # pad entity_ids
                create_padded_sequence(4, 0),  # pad entity_position_ids
                create_padded_sequence(5, 0),  # pad entity_segment_ids
                create_padded_sequence(6, 0),  # pad entity_attention_mask
                create_padded_sequence(7, 0),) # convert to numpy array


    if fold in ("dev", "test"):
        dataloader = DataLoader(
            dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn)
    else:
        dataloader = DataLoader(
            dataset,
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
        args.model_type, num_classes=num_labels)
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
