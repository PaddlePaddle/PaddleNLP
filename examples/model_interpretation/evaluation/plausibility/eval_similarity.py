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
 This script includes code to calculating F1 score for results form textual similarity task
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
    parser = argparse.ArgumentParser('F1 eval')
    parser.add_argument('--golden_path', required=True)
    parser.add_argument('--pred_path', required=True)
    parser.add_argument('--language', required=True, choices=['ch', 'en'])

    args = parser.parse_args()
    return args


def load_from_file(args):
    """
    Load golden and pred data form file
    :return: golden_raw: {sent_id, rationales_lists}, pred_raw: {sent_id, rationales_list},
             golden_label: {sent_id, label}, pred_label: {sent_id, label}
    """
    golden_f = open(args.golden_path, 'r')
    pred_f = open(args.pred_path, 'r')

    golden_q_rationales, golden_t_rationales = {}, {}
    pred_q_rationales, pred_t_rationales = {}, {}
    golden_labels, pred_labels = {}, {}

    for golden_line in golden_f.readlines():
        golden_dict = json.loads(golden_line)
        id = golden_dict['sent_id']
        # golden_rationale id
        golden_q_rationales[id] = [
            int(x) for x in golden_dict['rationale_q_idx']
        ]
        golden_t_rationales[id] = [
            int(x) for x in golden_dict['rationale_t_idx']
        ]
        golden_labels[id] = int(golden_dict['sent_label'])

    for pred_line in pred_f.readlines():
        pred_dict = json.loads(pred_line)
        id = pred_dict['id']
        pred_q_rationales[id] = pred_dict['rationale'][0]
        pred_t_rationales[id] = pred_dict['rationale'][1]
        pred_labels[id] = int(pred_dict['pred_label'])

    result = {}
    result['golden_q_rationales'] = golden_q_rationales
    result['golden_t_rationales'] = golden_t_rationales
    result['pred_q_rationales'] = pred_q_rationales
    result['pred_t_rationales'] = pred_t_rationales
    result['golden_labels'] = golden_labels
    result['pred_labels'] = pred_labels

    return result


def _f1(_p, _r):
    if _p == 0 or _r == 0:
        return 0
    return 2 * _p * _r / (_p + _r)


def calc_model_f1(golden_a_rationales, golden_b_rationales, pred_a_rationales,
                  pred_b_rationales):
    """
        :param golden_dict: dict
        :param pred_dict:   dict
        :return:    macro-f1, micro-f1
    """

    scores = {}

    for id in pred_a_rationales.keys():
        golden_a_ratioanl = golden_a_rationales[id]
        pred_a_rationale = pred_a_rationales[id]
        tp_a = set(golden_a_ratioanl) & set(pred_a_rationale)
        prec_a = len(tp_a) / len(pred_a_rationale) if len(
            pred_a_rationale) else 0
        rec_a = len(tp_a) / len(golden_a_ratioanl) if len(
            golden_a_ratioanl) else 0
        f1_a = _f1(prec_a, rec_a)

        golden_b_rationale = golden_b_rationales[id]
        pred_b_rationale = pred_b_rationales[id]
        tp_b = set(golden_b_rationale) & set(pred_b_rationale)
        prec_b = len(tp_b) / len(pred_b_rationale) if len(
            pred_b_rationale) else 0
        rec_b = len(tp_b) / len(golden_b_rationale) if len(
            golden_b_rationale) else 0
        f1_b = _f1(prec_b, rec_b)

        scores[id] = {
            'tp_count': (len(tp_a) + len(tp_b)) / 2,
            'pred_count': (len(pred_a_rationale) + len(pred_b_rationale)) / 2,
            'golden_count':
            (len(golden_a_ratioanl) + len(golden_b_rationale)) / 2,
            'prec': (prec_a + prec_b) / 2,
            'rec': (rec_a + rec_b) / 2,
            'f1': (f1_a + f1_b) / 2
        }

    macro_f1 = sum(score['f1'] for score in scores.values()) / len(golden_a_rationales) \
         if len(golden_a_rationales) else 0

    return macro_f1, scores


def main(args):
    result = load_from_file(args)
    golden_a_rationales = result['golden_q_rationales']
    golden_b_rationales = result['golden_t_rationales']
    pred_a_rationales = result['pred_q_rationales']
    pred_b_rationales = result['pred_t_rationales']
    golden_label = result['golden_labels']
    pred_label = result['pred_labels']

    macro_f1, scores = calc_model_f1(golden_a_rationales, golden_b_rationales,
                                     pred_a_rationales, pred_b_rationales)
    return macro_f1, len(scores)


if __name__ == '__main__':
    args = get_args()
    macro_f1, num = main(args)
    print('total\tnum: %d\tmacor_f1: %.1f' \
        % (num, macro_f1 * 100))
