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
 This script includes code to calculating F1 score for results form mrc task
"""
import copy
import json
import time
import math
import argparse


def get_args():
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

    golden_raw_rationale, pred_rationale = {}, {}

    for golden_line in golden_f.readlines():
        golden_dict = json.loads(golden_line)
        sent_id = golden_dict['sent_id']
        golden_raw_rationale[sent_id] = [
            int(x) for x in golden_dict['rationales']
        ]

    for pred_line in pred_f.readlines():
        pred_dict = json.loads(pred_line)
        senti_id = pred_dict['id']
        pred_rationale[senti_id] = pred_dict['rationale'][0]

    return golden_raw_rationale, pred_rationale


def _f1(_p, _r):
    if _p == 0 or _r == 0:
        return 0
    return 2 * _p * _r / (_p + _r)


def calc_f1(golden_evid, pred_evid):
    tp = set(pred_evid) & set(golden_evid)
    prec = len(tp) / len(pred_evid) if len(pred_evid) else 0
    rec = len(tp) / len(golden_evid) if len(golden_evid) else 0
    f1 = _f1(prec, rec)
    return f1


def calc_model_f1(golden_dict, pred_dict):
    """
        :param golden_dict: dict
        :param pred_dict:   dict
        :return:    macro-f1, micro-f1
    """

    scores = {}

    for s_id in pred_dict.keys():
        if s_id not in golden_dict:
            continue
        golden_evid = golden_dict[s_id]
        pred_evid = pred_dict[s_id]

        tp = set(golden_evid) & set(pred_evid)
        prec = len(tp) / len(pred_evid) if len(pred_evid) else 0
        rec = len(tp) / len(golden_evid) if len(golden_evid) else 0
        f1 = _f1(prec, rec)
        scores[s_id] = {
            'tp_count': len(tp),
            'pred_count': len(pred_evid),
            'golden_count': len(golden_evid),
            'prec': prec,
            'rec': rec,
            'f1': f1
        }

    macro_f1 = sum(score['f1']
                   for score in scores.values()) / len(golden_dict) if len(
                       golden_dict) else 0

    return macro_f1, scores


def main(args):
    golden_raw, pred_raw = load_from_file(args)
    macro_f1, scores = calc_model_f1(golden_raw, pred_raw)
    return macro_f1, len(golden_raw), scores


if __name__ == '__main__':
    args = get_args()
    macro_f1, num, scores = main(args)
    print('total\tnum: %d\tmacor_f1: %.1f' \
        % (num, macro_f1 * 100))
