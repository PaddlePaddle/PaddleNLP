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
 This script includes code to calculating F1 score for results form sentiment analysis task
"""
import copy
import json
import time
import math
import argparse


def get_args():
    parser = argparse.ArgumentParser('F1 eval')

    parser.add_argument('--language', required=True, choices=['en', 'ch'])
    parser.add_argument('--golden_path', required=True)
    parser.add_argument('--pred_path', required=True)

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

    golden_raw_rationale, golden_label, pred_rationale, pred_label = {}, {}, {}, {}

    for golden_line in golden_f.readlines():
        golden_dict = json.loads(golden_line)
        sent_id = golden_dict['sent_id']
        golden_raw_rationale[sent_id] = []
        for x in golden_dict['rationales']:
            temp = [int(y) for y in x]
            golden_raw_rationale[sent_id].append(temp)
        golden_label[sent_id] = int(golden_dict['sent_label'])

    for pred_line in pred_f.readlines():
        pred_dict = json.loads(pred_line)
        senti_id = pred_dict['id']
        pred_rationale[senti_id] = pred_dict['rationale'][0]
        pred_label[senti_id] = int(pred_dict['pred_label'])

    golden_f.close()
    pred_f.close()
    return golden_raw_rationale, pred_rationale, golden_label, pred_label


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


def combine(cur_max_f1, union_set, golden_evid, pred_evid):
    """
    Args:
        cur_max_f1  float:      当前最大f1
        union_set   set():      已合并集合
        golden_evid list():     标注证据
        pred_evid   list():     预测证据
    """
    if len(union_set & set(golden_evid)) < len(golden_evid) and calc_f1(
            golden_evid, pred_evid) > 0:
        new_union_set = union_set | set(golden_evid)
        new_f1 = calc_f1(new_union_set, pred_evid)
        if new_f1 > cur_max_f1:  # 若union_set合并golden_evid后f1未超过cur_max_f1，则不更新union_set
            cur_max_f1 = new_f1
            union_set = new_union_set

    return cur_max_f1, union_set


def pick_max_golden_evid(golden_raw, pred_raw):
    """
    从golden_evids中找出与pred_evid f1最大的golden_evid
    """
    golden_dict = {}
    err_rationale = []

    for s_id in pred_raw.keys():
        if s_id not in golden_raw:
            continue
        golden_evids = golden_raw[s_id]
        pred_evid = pred_raw[s_id]
        max_f1 = 0

        # 找f1最大的单条golden_evid
        for golden_evid in golden_evids:
            f1 = calc_f1(golden_evid, pred_evid)
            if f1 > max_f1:
                max_f1 = f1
                golden_dict[s_id] = golden_evid

        # 找f1最大的组合golden_evid
        for start_id in range(len(golden_evids) - 1):
            union_set = set()
            cur_max_f1 = 0
            for id in range(start_id, len(golden_evids)):
                golden_evid = golden_evids[id]
                cur_max_f1, union_set = combine(cur_max_f1, union_set,
                                                golden_evid, pred_evid)

            if cur_max_f1 > max_f1:
                max_f1 = cur_max_f1
                golden_dict[s_id] = list(union_set)

        if max_f1 == 0:
            golden_dict[s_id] = []
            err_rationale.append(s_id)

    return golden_dict


def calc_model_f1(golden_dict, pred_dict, golden_len):
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

    macro_prec = (sum(score['prec'] for score in scores.values()) /
                  golden_len) if golden_len else 0
    macro_rec = (sum(score['rec'] for score in scores.values()) /
                 golden_len) if golden_len else 0
    macro_f1 = (sum(score['f1'] for score in scores.values()) /
                golden_len) if golden_len else 0

    return macro_f1, scores


def main(args):
    golden_raw, pred_raw, golden_label, pred_label = load_from_file(args)
    golden_dict = pick_max_golden_evid(golden_raw, pred_raw)
    macro_f1, scores = calc_model_f1(golden_dict, pred_raw, len(golden_raw))
    return macro_f1, len(golden_raw)


if __name__ == '__main__':
    args = get_args()
    macro_f1, num = main(args)
    print('num\t%.2f\tmacor_f1: %.1f' \
        % (num, macro_f1 * 100))
