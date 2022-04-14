#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""
 This script includes code to calculating accuracy for results form textual similarity task
"""
import copy
import json
import time
import math
import argparse


def get_args():
    """
    get args
    """
    parser = argparse.ArgumentParser('Acc eval')
    parser.add_argument('--golden_path', required=True)
    parser.add_argument('--pred_path', required=True)
    parser.add_argument('--language', required=True, choices=['ch', 'en'])

    args = parser.parse_args()
    return args


def load_from_file(args):
    """
    load golden and pred data form file
    :return: golden_raw: {sent_id, rationales_lists}, pred_raw: {sent_id, rationales_list},
             golden_label: {sent_id, label}, pred_label: {sent_id, label}
    """
    golden_f = open(args.golden_path, 'r')
    pred_f = open(args.pred_path, 'r')

    golden_labels, pred_labels = {}, {}

    for golden_line in golden_f.readlines():
        golden_dict = json.loads(golden_line)
        id = golden_dict['sent_id']
        golden_labels[id] = int(golden_dict['sent_label'])

    for pred_line in pred_f.readlines():
        pred_dict = json.loads(pred_line)
        id = pred_dict['id']
        pred_labels[id] = int(pred_dict['pred_label'])

    result = {}
    result['golden_labels'] = golden_labels
    result['pred_labels'] = pred_labels

    return result


def cal_acc(golden_label, pred_label):
    """
    The function actually calculate the accuracy.
    """
    acc = 0.0
    for ids in pred_label:
        if ids not in golden_label:
            continue
        if pred_label[ids] == golden_label[ids]:
            acc += 1
    if len(golden_label):
        acc /= len(golden_label)
    return acc


def main(args):
    """
    main function
    """
    result = load_from_file(args)
    golden_label = result['golden_labels']
    pred_label = result['pred_labels']

    acc = cal_acc(golden_label, pred_label)
    return acc, len(pred_label)


if __name__ == '__main__':
    args = get_args()
    acc, num = main(args)
    print('total\tnum: %d\tacc: %.1f' \
        % (num, acc * 100))
