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

import paddle
import paddle.nn.functional as F
from paddle.io import BatchSampler, DataLoader
from utils import preprocess_function, read_local_dataset

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.utils.log import logger

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--dataset_dir", required=True, default=None, type=str, help="Local dataset directory should include data.txt and label.txt")
parser.add_argument("--params_path", default="./checkpoint/", type=str, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--data_file", type=str, default="data.txt", help="Unlabeled data file name")
parser.add_argument("--label_file", type=str, default="label.txt", help="Label file name")
args = parser.parse_args()
# yapf: enable


@paddle.no_grad()
def predict():
    """
    Predicts the data labels.
    """
    paddle.set_device(args.device)
    model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
    tokenizer = AutoTokenizer.from_pretrained(args.params_path)

    label_list = []
    label_path = os.path.join(args.dataset_dir, args.label_file)
    with open(label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            label_list.append(line.strip())

    data_ds = load_dataset(
        read_local_dataset, path=os.path.join(args.dataset_dir, args.data_file), is_test=True, lazy=False
    )

    trans_func = functools.partial(
        preprocess_function,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        label_nums=len(label_list),
        is_test=True,
    )

    data_ds = data_ds.map(trans_func)

    # batchify dataset
    collate_fn = DataCollatorWithPadding(tokenizer)
    data_batch_sampler = BatchSampler(data_ds, batch_size=args.batch_size, shuffle=False)

    data_data_loader = DataLoader(dataset=data_ds, batch_sampler=data_batch_sampler, collate_fn=collate_fn)

    results = []
    model.eval()
    for batch in data_data_loader:
        logits = model(**batch)
        probs = F.sigmoid(logits).numpy()
        for prob in probs:
            labels = []
            for i, p in enumerate(prob):
                if p > 0.5:
                    labels.append(label_list[i])
            results.append(labels)

    for t, labels in zip(data_ds.data, results):
        hierarchical_labels = {}
        logger.info("text: {}".format(t["sentence"]))
        logger.info("prediction result: {}".format(",".join(labels)))
        for label in labels:
            for i, l in enumerate(label.split("##")):
                if i not in hierarchical_labels:
                    hierarchical_labels[i] = []
                if l not in hierarchical_labels[i]:
                    hierarchical_labels[i].append(l)
        for d in range(len(hierarchical_labels)):
            logger.info("level {} : {}".format(d + 1, ",".join(hierarchical_labels[d])))
        logger.info("--------------------")
    return


if __name__ == "__main__":

    predict()
