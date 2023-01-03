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

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import BatchSampler, DataLoader
from sklearn.metrics import accuracy_score, classification_report

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.utils.log import logger

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', default="gpu", help="Select which device to evaluate model, defaults to gpu.")
parser.add_argument("--dataset_dir", required=True, type=str, help="Local dataset directory should include dev.txt and label.txt")
parser.add_argument("--params_path", default="../checkpoint/", type=str, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--dev_file", type=str, default="dev.txt", help="Dev dataset file name")
parser.add_argument("--label_file", type=str, default="label.txt", help="Label file name")
parser.add_argument("--bad_case_file", type=str, default="bad_case.txt", help="Bad case saving file name")
args = parser.parse_args()
# yapf: enable


def preprocess_function(examples, tokenizer, max_seq_length, label_nums, is_test=False):
    """
    Preprocess dataset
    """
    result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
    if not is_test:
        result["labels"] = [float(1) if i in examples["label"] else float(0) for i in range(label_nums)]
    return result


def read_local_dataset(path, label_list):
    """
    Read dataset file
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split("\t")
            if len(items) == 0:
                continue
            elif len(items) == 1:
                sentence = items[0]
                labels = []
                label = ""
            else:
                sentence = "".join(items[:-1])
                label = items[-1]
                labels = [label_list[l] for l in label.split(",")]
            yield {"text": sentence, "label": labels, "label_n": label}


@paddle.no_grad()
def evaluate():
    """
    Evaluate the model performance
    """
    paddle.set_device(args.device)
    # Define model & tokenizer
    if os.path.exists(args.params_path):
        model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
        tokenizer = AutoTokenizer.from_pretrained(args.params_path)
    else:
        raise ValueError("The {} should exist.".format(args.params_path))

    # load and preprocess dataset
    label_path = os.path.join(args.dataset_dir, args.label_file)
    dev_path = os.path.join(args.dataset_dir, args.dev_file)

    label_list = {}
    label_map = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            l = line.strip()
            label_list[l] = i
            label_map[i] = l
    dev_ds = load_dataset(read_local_dataset, path=dev_path, label_list=label_list, lazy=False)
    trans_func = functools.partial(
        preprocess_function, tokenizer=tokenizer, max_seq_length=args.max_seq_length, label_nums=len(label_list)
    )
    dev_ds = dev_ds.map(trans_func)

    # batchify dataset
    collate_fn = DataCollatorWithPadding(tokenizer)
    dev_batch_sampler = BatchSampler(dev_ds, batch_size=args.batch_size, shuffle=False)
    dev_data_loader = DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=collate_fn)

    model.eval()
    probs = []
    labels = []
    for batch in dev_data_loader:
        label = batch.pop("labels")
        logits = model(**batch)
        labels.extend(label.numpy())
        probs.extend(F.sigmoid(logits).numpy())
    probs = np.array(probs)
    labels = np.array(labels)
    preds = probs > 0.5
    report = classification_report(labels, preds, digits=4, output_dict=True)
    accuracy = accuracy_score(labels, preds)

    logger.info("-----Evaluate model-------")
    logger.info("Dev dataset size: {}".format(len(dev_ds)))
    logger.info("Accuracy in dev dataset: {:.2f}%".format(accuracy * 100))
    logger.info(
        "Micro avg in dev dataset: precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
            report["micro avg"]["precision"] * 100,
            report["micro avg"]["recall"] * 100,
            report["micro avg"]["f1-score"] * 100,
        )
    )
    logger.info(
        "Macro avg in dev dataset: precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
            report["macro avg"]["precision"] * 100,
            report["macro avg"]["recall"] * 100,
            report["macro avg"]["f1-score"] * 100,
        )
    )

    for i in label_map:
        logger.info("Class name: {}".format(label_map[i]))
        logger.info(
            "Evaluation examples in dev dataset: {}({:.1f}%) | precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
                report[str(i)]["support"],
                100 * report[str(i)]["support"] / len(dev_ds),
                report[str(i)]["precision"] * 100,
                report[str(i)]["recall"] * 100,
                report[str(i)]["f1-score"] * 100,
            )
        )
        logger.info("----------------------------")
    bad_case_path = os.path.join(args.dataset_dir, args.bad_case_file)
    with open(bad_case_path, "w", encoding="utf-8") as f:
        f.write("Text\tLabel\tPrediction\n")
        for i in range(len(preds)):
            for p, l in zip(preds[i], labels[i]):
                if (p and l == 0) or (not p and l == 1):
                    pred_n = [label_map[i] for i, pp in enumerate(preds[i]) if pp]
                    f.write(dev_ds.data[i]["text"] + "\t" + dev_ds.data[i]["label_n"] + "\t" + ",".join(pred_n) + "\n")
                    break

    f.close()
    logger.info("Bad case in dev dataset saved in {}".format(bad_case_path))

    return


if __name__ == "__main__":
    evaluate()
