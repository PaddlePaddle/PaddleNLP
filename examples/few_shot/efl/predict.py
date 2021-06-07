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

import paddlenlp as ppnlp
from model import Classifier
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset

from data import create_dataloader
from data import convert_example, convert_chid_example
from evaluate import do_evaluate, do_evaluate_chid


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

    labels = [
        int(label)
        for label, label_description in task_label_description.items()
    ]

    class_num = len(labels)
    #print("class_num:{}".format(class_num))

    all_prediction_probs = []

    for batch in data_loader:
        src_ids, token_type_ids = batch

        # Prediction_probs:[bs, 2]
        prediction_probs = model(
            input_ids=src_ids, token_type_ids=token_type_ids).numpy()

        #print("prediciton_probs shape:{}".format(prediction_probs.shape))
        all_prediction_probs.append(prediction_probs)

    all_prediction_probs = np.concatenate(all_prediction_probs, axis=0)
    #print("all_prediciton_probs shape:{}".format(all_prediction_probs.shape))

    all_prediction_probs = np.reshape(all_prediction_probs, (-1, class_num, 2))
    #print("prediciton_probs reshape:{}".format(all_prediction_probs.shape))

    prediction_pos_probs = all_prediction_probs[:, :, 1]
    prediction_pos_probs = np.reshape(prediction_pos_probs, (-1, class_num))
    y_pred_index = np.argmax(prediction_pos_probs, axis=-1)

    y_preds = [labels[idx] for idx in y_pred_index]
    #print("y_preds:{}".format(y_preds))

    model.train()
    return y_preds


@paddle.no_grad()
def do_predict_chid(model, tokenizer, data_loader, label_normalize_dict):
    """
        FewCLUE `chid` dataset is specical when evaluate: input slots have 
        additional `candidate_label_ids`, so need to customize the
        evaluate function.
    """

    model.eval()

    normed_labels = [
        normalized_lable
        for origin_lable, normalized_lable in label_normalize_dict.items()
    ]

    label_length = len(normed_labels[0])

    y_pred_all = []
    for batch in data_loader:
        src_ids, token_type_ids, masked_positions, candidate_label_ids = batch

        # [bs * label_length, vocab_size]
        prediction_probs = model.predict(
            input_ids=src_ids,
            token_type_ids=token_type_ids,
            masked_positions=masked_positions)

        batch_size = len(src_ids)
        vocab_size = prediction_probs.shape[1]

        # prediction_probs: [batch_size, label_lenght, vocab_size]
        prediction_probs = paddle.reshape(
            prediction_probs, shape=[batch_size, -1, vocab_size]).numpy()

        candidate_num = candidate_label_ids.shape[1]

        # [batch_size, candidate_num(7)]
        y_pred = np.ones(shape=[batch_size, candidate_num])

        for label_idx in range(candidate_num):
            # [bathc_size, label_length(4)] 
            single_candidate_label_ids = candidate_label_ids[:, label_idx, :]
            # Calculate joint distribution of candidate labels
            for index in range(label_length):
                # [batch_size,]
                slice_word_ids = single_candidate_label_ids[:, index].numpy()

                batch_single_token_prob = []
                for bs_index in range(batch_size):
                    # [1, 1]
                    single_token_prob = prediction_probs[
                        bs_index, index, slice_word_ids[bs_index]]
                    batch_single_token_prob.append(single_token_prob)

                y_pred[:, label_idx] *= np.array(batch_single_token_prob)

        # Get max probs label's index
        y_pred_index = np.argmax(y_pred, axis=-1)
        y_pred_all.extend(y_pred_index)
    return y_pred_all


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


def write_iflytek(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    #test_ds = load_dataset("fewclue", name=task_name, data_files="/home/tianxin04/.paddlenlp/datasets/FewCLUE/fewclue_iflytek/test_demo.json")

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
            # {"id": 0, "label": "力学"}
            str_test_example = "\"{}\": {}, \"{}\": \"{}\"".format(
                "id", test_example['id'], "label", test_example["label"])
            f.write("{" + str_test_example + "}\n")


def write_tnews(task_name, output_file, pred_labels):
    test_ds, train_few_all = load_dataset(
        "fewclue", name=task_name, splits=("test", "train_few_all"))

    def label2id(train_few_all):
        label2id = {}
        for example in train_few_all:
            label = example["label_desc"]
            label_id = example["label"]
            if label not in label2id:
                label2id[label] = str(label_id)
        return label2id

    label2id_dict = label2id(train_few_all)

    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = label2id_dict[pred_labels[idx]]

            str_test_example = json.dumps(test_example) + "\n"
            f.write(str_test_example)


def write_cluewsc(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]
            # {"id": 0, "label": "力学"}
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

    label_normalize_json = os.path.join("./label_normalized",
                                        args.task_name + ".json")

    label_norm_dict = None
    with open(label_normalize_json) as f:
        label_norm_dict = json.load(f)

    convert_example_fn = convert_example if args.task_name != "chid" else convert_chid_example
    predict_fn = do_predict if args.task_name != "chid" else do_predict_chid

    # Load test_ds for FewCLUE leaderboard
    test_ds = load_dataset("fewclue", name=args.task_name, splits=("test"))

    # Task related transform operations, eg: numbert label -> text_label, english -> chinese
    transform_fn = partial(
        transform_fn_dict[args.task_name],
        label_normalize_dict=label_norm_dict,
        is_test=True)

    # Some fewshot_learning strategy is defined by transform_fn
    # Note: Set lazy=False to transform example inplace immediately,
    # because transform_fn should only be executed only once when 
    # iterate multi-times for train_ds
    test_ds = test_ds.map(transform_fn, lazy=False)

    pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(
        'ernie-1.0')
    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')
    model = Classifier(pretrained_model)

    # Load parameters of best model on test_public.json of current task
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.init_from_ckpt)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    if args.task_name != "chid":
        # [src_ids, token_type_ids, masked_positions, masked_lm_labels]
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64"),  # masked_positions
        ): [data for data in fn(samples)]
    else:
        # [src_ids, token_type_ids, masked_positions, masked_lm_labels, candidate_labels_ids]
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64"),  # masked_positions
            Stack(dtype="int64"),  # candidate_labels_ids [candidate_num, label_length]
        ): [data for data in fn(samples)]

    trans_func = partial(
        convert_example_fn,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        p_embedding_num=args.p_embedding_num,
        is_test=True)

    test_data_loader = create_dataloader(
        test_ds,
        mode='eval',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    y_pred_labels = predict_fn(model, tokenizer, test_data_loader,
                               label_norm_dict)
    output_file = os.path.join(args.output_dir, predict_file[args.task_name])

    write_fn[args.task_name](args.task_name, output_file, y_pred_labels)
