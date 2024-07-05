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
from paddlenlp.dataaug import WordDelete, WordInsert, WordSubstitute, WordSwap
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.utils.log import logger

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--dataset_dir", required=True, type=str, help="The dataset directory should include train.txt,dev.txt and test.txt files.")
parser.add_argument("--aug_strategy", choices=["duplicate", "substitute", "insert", "delete", "swap"], default='substitute', help="Select data augmentation strategy")
parser.add_argument("--annotate", action='store_true', help="Select unlabeled data for annotation")
parser.add_argument("--params_path", default="../checkpoint/", type=str, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--seed", type=int, default=3, help="random seed for initialization")
parser.add_argument("--rationale_num_sparse", type=int, default=3, help="Number of rationales per example for sparse data.")
parser.add_argument("--rationale_num_support", type=int, default=6, help="Number of rationales per example for support data.")
parser.add_argument("--sparse_num", type=int, default=100, help="Number of sparse data.")
parser.add_argument("--support_threshold", type=float, default="0.7", help="The threshold to select support data.")
parser.add_argument("--support_num", type=int, default=100, help="Number of support data.")
parser.add_argument("--train_file", type=str, default="train.txt", help="Train dataset file name")
parser.add_argument("--dev_file", type=str, default="dev.txt", help="Dev dataset file name")
parser.add_argument("--label_file", type=str, default="label.txt", help="Label file name")
parser.add_argument("--unlabeled_file", type=str, default="data.txt", help="Unlabeled data filename")
parser.add_argument("--sparse_file", type=str, default="sparse.txt", help="Sparse data file name.")
parser.add_argument("--support_file", type=str, default="support.txt", help="support data file name.")
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
            if len(items) == 2:
                yield {"text": items[0], "label": items[1]}
            elif len(items) == 1:
                yield {"text": items[0]}
            else:
                logger.info(line.strip())
                raise ValueError("{} should be in fixed format.".format(path))
    f.close()


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


def get_sparse_data(analysis_result, sparse_num):
    """
    Get sparse data
    """
    idx_scores = {}
    preds = []
    for i in range(len(analysis_result)):
        scores = analysis_result[i].pos_scores
        idx_scores[i] = sum(scores) / len(scores)
        preds.append(analysis_result[i].pred_label)

    idx_socre_list = list(sorted(idx_scores.items(), key=lambda x: x[1]))[:sparse_num]
    ret_idxs, ret_scores = list(zip(*idx_socre_list))
    return ret_idxs, ret_scores, preds


def find_sparse_data():
    """
    Find sparse data (lack of supports in train dataset) in dev dataset
    """
    set_seed(args.seed)
    paddle.set_device(args.device)

    # Define model & tokenizer
    if os.path.exists(args.params_path):
        model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
        tokenizer = AutoTokenizer.from_pretrained(args.params_path)
    else:
        raise ValueError("The {} should exist.".format(args.params_path))

    # Prepare & preprocess dataset
    label_path = os.path.join(args.dataset_dir, args.label_file)
    train_path = os.path.join(args.dataset_dir, args.train_file)
    dev_path = os.path.join(args.dataset_dir, args.dev_file)

    label_list = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            l = line.strip()
            label_list[l] = i
    f.close()

    train_ds = load_dataset(read_local_dataset, path=train_path, lazy=False)
    dev_ds = load_dataset(read_local_dataset, path=dev_path, lazy=False)
    trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    train_ds = train_ds.map(trans_func)
    dev_ds = dev_ds.map(trans_func)

    # Batchify dataset
    collate_fn = LocalDataCollatorWithPadding(tokenizer)
    train_batch_sampler = BatchSampler(train_ds, batch_size=args.batch_size, shuffle=False)
    dev_batch_sampler = BatchSampler(dev_ds, batch_size=args.batch_size, shuffle=False)
    train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    dev_data_loader = DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=collate_fn)

    # Classifier_layer_name is the layer name of the last output layer
    feature_sim = FeatureSimilarityModel(model, train_data_loader, classifier_layer_name="classifier")
    # Feature similarity analysis & select sparse data
    analysis_result = []
    for batch in dev_data_loader:
        analysis_result += feature_sim(batch, sample_num=args.rationale_num_sparse)
    sparse_indexs, sparse_scores, preds = get_sparse_data(analysis_result, args.sparse_num)

    # Save the sparse data
    is_true = []
    with open(os.path.join(args.dataset_dir, args.sparse_file), "w") as f:
        for idx in sparse_indexs:
            data = dev_ds.data[idx]
            f.write(data["text"] + "\t" + str(data["label"]) + "\n")
            is_true.append(1 if str(preds[idx]) == str(label_list[data["label"]]) else 0)
    f.close()
    logger.info("Sparse data saved in {}".format(os.path.join(args.dataset_dir, args.sparse_file)))
    logger.info("Accuracy in sparse data: {:.2f}%".format(100 * sum(is_true) / len(is_true)))
    logger.info("Average score in sparse data: {:.4f}".format(sum(sparse_scores) / len(sparse_scores)))
    return os.path.join(args.dataset_dir, args.sparse_file)


def get_support_data(analysis_result, support_num, support_threshold=0.7):
    """
    get support data
    """
    ret_idxs = []
    ret_scores = []
    rationale_idx = 0
    try:
        while len(ret_idxs) < support_num:
            for n in range(len(analysis_result)):
                score = analysis_result[n].pos_scores[rationale_idx]
                if score > support_threshold:
                    idx = analysis_result[n].pos_indexes[rationale_idx]
                    if idx not in ret_idxs:
                        ret_idxs.append(idx)
                        ret_scores.append(score)
                    if len(ret_idxs) >= support_num:
                        break

            rationale_idx += 1
    except IndexError:
        logger.error(
            f"The index is out of range, please reduce support_num or increase support_threshold. Got {len(ret_idxs)} now."
        )

    return ret_idxs, ret_scores


def find_support_data():
    """
    Find support data (which supports sparse data) from candidate dataset
    """
    set_seed(args.seed)
    paddle.set_device(args.device)

    # Define model & tokenizer
    if os.path.exists(args.params_path):
        model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
        tokenizer = AutoTokenizer.from_pretrained(args.params_path)
    else:
        raise ValueError("The {} should exist.".format(args.params_path))

    # Prepare & preprocess dataset
    if args.annotate:
        candidate_path = os.path.join(args.dataset_dir, args.unlabeled_file)
    else:
        candidate_path = os.path.join(args.dataset_dir, args.train_file)

    sparse_path = os.path.join(args.dataset_dir, args.sparse_file)
    support_path = os.path.join(args.dataset_dir, args.support_file)
    candidate_ds = load_dataset(read_local_dataset, path=candidate_path, lazy=False)
    sparse_ds = load_dataset(read_local_dataset, path=sparse_path, lazy=False)
    trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    candidate_ds = candidate_ds.map(trans_func)
    sparse_ds = sparse_ds.map(trans_func)

    # Batchify dataset
    collate_fn = LocalDataCollatorWithPadding(tokenizer)
    candidate_batch_sampler = BatchSampler(candidate_ds, batch_size=args.batch_size, shuffle=False)
    sparse_batch_sampler = BatchSampler(sparse_ds, batch_size=args.batch_size, shuffle=False)
    candidate_data_loader = DataLoader(
        dataset=candidate_ds, batch_sampler=candidate_batch_sampler, collate_fn=collate_fn
    )
    sparse_data_loader = DataLoader(dataset=sparse_ds, batch_sampler=sparse_batch_sampler, collate_fn=collate_fn)

    # Classifier_layer_name is the layer name of the last output layer
    feature_sim = FeatureSimilarityModel(model, candidate_data_loader, classifier_layer_name="classifier")
    # Feature similarity analysis
    analysis_result = []
    for batch in sparse_data_loader:
        analysis_result += feature_sim(batch, sample_num=args.rationale_num_support)

    support_indexs, support_scores = get_support_data(analysis_result, args.support_num, args.support_threshold)

    # Save the support data
    if args.annotate or args.aug_strategy == "duplicate":
        with open(support_path, "w") as f:
            for idx in list(support_indexs):
                data = candidate_ds.data[idx]
                if "label" in data:
                    f.write(data["text"] + "\t" + data["label"] + "\n")
                else:
                    f.write(data["text"] + "\n")
        f.close()
    else:
        create_n = 1
        aug_percent = 0.1
        if args.aug_strategy == "substitute":
            aug = WordSubstitute("embedding", create_n=create_n, aug_percent=aug_percent)
        elif args.aug_strategy == "insert":
            aug = WordInsert("embedding", create_n=create_n, aug_percent=aug_percent)
        elif args.aug_strategy == "delete":
            aug = WordDelete(create_n=create_n, aug_percent=aug_percent)
        elif args.aug_strategy == "swap":
            aug = WordSwap(create_n=create_n, aug_percent=aug_percent)

        with open(support_path, "w") as f:
            for idx in list(support_indexs):
                data = candidate_ds.data[idx]
                augs = aug.augment(data["text"])
                if not isinstance(augs[0], str):
                    augs = augs[0]
                for a in augs:
                    f.write(a + "\t" + data["label"] + "\n")
        f.close()
    logger.info("support data saved in {}".format(support_path))
    logger.info("support average scores: {:.4f}".format(float(sum(support_scores)) / len(support_scores)))


if __name__ == "__main__":
    find_sparse_data()
    find_support_data()
