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

import argparse
import functools
import os
import random

import numpy as np
import paddle
from paddle.io import BatchSampler, DataLoader
from trustai.interpretation import FeatureSimilarityModel

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.utils.log import logger

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--dataset_dir", required=True, type=str, help="The dataset directory should include train.txt,dev.txt and test.txt files.")
parser.add_argument("--params_path", default="../checkpoint/", type=str, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--seed", type=int, default=3, help="random seed for initialization")
parser.add_argument("--top_k", type=int, default=3, help="Top K important training data.")
parser.add_argument("--train_file", type=str, default="train.txt", help="Train dataset file name")
parser.add_argument("--interpret_input_file", type=str, default="bad_case.txt", help="interpretation file name")
parser.add_argument("--interpret_result_file", type=str, default="sent_interpret.txt", help="interpreted file name")
args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """
    Set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_local_dataset(path):
    """
    Read dataset file
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split("\t")
            if items[0] == "Text":
                continue
            if len(items) == 3:
                yield {"text": items[0], "label": items[1], "predict": items[2]}
            elif len(items) == 2:
                yield {"text": items[0], "label": items[1], "predict": ""}
            elif len(items) == 1:
                yield {"text": items[0], "label": "", "predict": ""}
            else:
                logger.info(line.strip())
                raise ValueError("{} should be in fixed format.".format(path))


def preprocess_function(examples, tokenizer, max_seq_length):
    """
    Preprocess dataset
    """
    result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
    return result


class LocalDataCollatorWithPadding(DataCollatorWithPadding):
    """
    Convert the  result of DataCollatorWithPadding from dict dictionary to a list
    """

    def __call__(self, features):
        batch = super().__call__(features)
        batch = list(batch.values())
        return batch


def find_positive_influence_data():

    set_seed(args.seed)
    paddle.set_device(args.device)

    # Define model & tokenizer
    if os.path.exists(args.params_path):
        model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
        tokenizer = AutoTokenizer.from_pretrained(args.params_path)
    else:
        raise ValueError("The {} should exist.".format(args.params_path))

    # Prepare & preprocess dataset
    train_path = os.path.join(args.dataset_dir, args.train_file)
    interpret_path = os.path.join(args.dataset_dir, args.interpret_input_file)

    train_ds = load_dataset(read_local_dataset, path=train_path, lazy=False)
    interpret_ds = load_dataset(read_local_dataset, path=interpret_path, lazy=False)
    trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    train_ds = train_ds.map(trans_func)
    interpret_ds = interpret_ds.map(trans_func)

    # Batchify dataset
    collate_fn = LocalDataCollatorWithPadding(tokenizer)
    train_batch_sampler = BatchSampler(train_ds, batch_size=args.batch_size, shuffle=False)
    interpret_batch_sampler = BatchSampler(interpret_ds, batch_size=args.batch_size, shuffle=False)
    train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    interpret_data_loader = DataLoader(
        dataset=interpret_ds, batch_sampler=interpret_batch_sampler, collate_fn=collate_fn
    )

    # Classifier_layer_name is the layer name of the last output layer
    feature_sim = FeatureSimilarityModel(model, train_data_loader, classifier_layer_name="classifier")
    # Feature similarity analysis & select sparse data
    analysis_result = []
    for batch in interpret_data_loader:
        analysis_result += feature_sim(batch, sample_num=args.top_k)
    with open(os.path.join(args.dataset_dir, args.interpret_result_file), "w") as f:
        for i in range(len(analysis_result)):
            f.write("text: " + interpret_ds.data[i]["text"] + "\n")
            if "predict" in interpret_ds.data[i]:
                f.write("predict label: " + interpret_ds.data[i]["predict"] + "\n")
            if "label" in interpret_ds.data[i]:
                f.write("label: " + interpret_ds.data[i]["label"] + "\n")
            f.write("examples with positive influence\n")
            for i, (idx, score) in enumerate(zip(analysis_result[i].pos_indexes, analysis_result[i].pos_scores)):
                f.write(
                    "support{} text: ".format(i + 1)
                    + train_ds.data[idx]["text"]
                    + "\t"
                    + "label: "
                    + train_ds.data[idx]["label"]
                    + "\t"
                    + "score: "
                    + "{:.5f}".format(score)
                    + "\n"
                )
    f.close()


if __name__ == "__main__":
    find_positive_influence_data()
