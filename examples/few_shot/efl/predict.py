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

import argparse
import os
import sys
import random
import time
import json
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset

from data import create_dataloader, convert_example, processor_dict
from task_label_description import TASK_LABELS_DESC


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True, type=str, help="The task_name to be evaluated")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--output_dir", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument('--save_steps', type=int, default=10000, help="Inteval steps to save checkpoint")

    return parser.parse_args()

# yapf: enable


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def do_predict(model, tokenizer, data_loader, task_label_description):
    model.eval()

    index2label = {
        idx: label
        for idx, label in enumerate(task_label_description.keys())
    }

    class_num = len(task_label_description)

    all_prediction_probs = []

    for batch in data_loader:
        src_ids, token_type_ids = batch

        # Prediction_probs:[bs, 2]
        prediction_probs = model(input_ids=src_ids,
                                 token_type_ids=token_type_ids).numpy()

        all_prediction_probs.append(prediction_probs)

    all_prediction_probs = np.concatenate(all_prediction_probs, axis=0)

    all_prediction_probs = np.reshape(all_prediction_probs, (-1, class_num, 2))

    prediction_pos_probs = all_prediction_probs[:, :, 1]
    prediction_pos_probs = np.reshape(prediction_pos_probs, (-1, class_num))
    y_pred_index = np.argmax(prediction_pos_probs, axis=-1)

    y_preds = [index2label[idx] for idx in y_pred_index]

    model.train()
    return y_preds


def write_iflytek(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = str(pred_labels[idx])

            str_test_example = json.dumps(test_example) + "\n"
            f.write(str_test_example)


def write_bustm(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]
            str_test_example = json.dumps(test_example) + "\n"
            f.write(str_test_example)


def write_csldcp(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]
            str_test_example = "\"{}\": {}, \"{}\": \"{}\"".format(
                "id", test_example['id'], "label", test_example["label"])
            f.write("{" + str_test_example + "}\n")


def write_tnews(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]

            str_test_example = json.dumps(test_example) + "\n"
            f.write(str_test_example)


def write_cluewsc(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]
            str_test_example = "\"{}\": {}, \"{}\": \"{}\"".format(
                "id", test_example['id'], "label", test_example["label"])
            f.write("{" + str_test_example + "}\n")


def write_eprstmt(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]

            str_test_example = json.dumps(test_example)
            f.write(str_test_example + "\n")


def write_ocnli(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]
            str_test_example = json.dumps(test_example)
            f.write(str_test_example + "\n")


def write_csl(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]
            str_test_example = json.dumps(test_example)
            f.write(str_test_example + "\n")


def write_chid(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["answer"] = pred_labels[idx]
            str_test_example = "\"{}\": {}, \"{}\": {}".format(
                "id", test_example['id'], "answer", test_example["answer"])
            f.write("{" + str_test_example + "}\n")


predict_file = {
    "bustm": "bustm_predict.json",
    "chid": "chidf_predict.json",
    "cluewsc": "cluewscf_predict.json",
    "csldcp": "csldcp_predict.json",
    "csl": "cslf_predict.json",
    "eprstmt": "eprstmt_predict.json",
    "iflytek": "iflytekf_predict.json",
    "ocnli": "ocnlif_predict.json",
    "tnews": "tnewsf_predict.json"
}

write_fn = {
    "bustm": write_bustm,
    "iflytek": write_iflytek,
    "csldcp": write_csldcp,
    "tnews": write_tnews,
    "cluewsc": write_cluewsc,
    "eprstmt": write_eprstmt,
    "ocnli": write_ocnli,
    "csl": write_csl,
    "chid": write_chid
}

if __name__ == "__main__":
    args = parse_args()
    paddle.set_device(args.device)
    set_seed(args.seed)

    processor = processor_dict[args.task_name]()
    # Load test_ds for FewCLUE leaderboard
    test_ds = load_dataset("fewclue", name=args.task_name, splits=("test"))
    test_ds = processor.get_test_datasets(test_ds,
                                          TASK_LABELS_DESC[args.task_name])

    model = AutoModelForSequenceClassification.from_pretrained(
        'ernie-3.0-medium-zh', num_classes=2)
    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')

    # [src_ids, token_type_ids]
    predict_batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    ): [data for data in fn(samples)]

    predict_trans_func = partial(convert_example,
                                 tokenizer=tokenizer,
                                 max_seq_length=args.max_seq_length,
                                 is_test=True)

    test_data_loader = create_dataloader(test_ds,
                                         mode='eval',
                                         batch_size=args.batch_size,
                                         batchify_fn=predict_batchify_fn,
                                         trans_fn=predict_trans_func)

    # Load parameters of best model on test_public.json of current task
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.init_from_ckpt)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    y_pred_labels = do_predict(
        model,
        tokenizer,
        test_data_loader,
        task_label_description=TASK_LABELS_DESC[args.task_name])
    output_file = os.path.join(args.output_dir, predict_file[args.task_name])
    write_fn[args.task_name](args.task_name, output_file, y_pred_labels)
