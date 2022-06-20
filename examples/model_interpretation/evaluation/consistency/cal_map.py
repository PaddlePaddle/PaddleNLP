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
	This script includes code to calculating MAP score for results form
	sentiment analysis, textual similarity, and mrc task
"""
import json
import re
import os
import math
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser('map eval')
    parser.add_argument('--pred_path', required=True)
    parser.add_argument('--golden_path', required=True)
    parser.add_argument('--language',
                        type=str,
                        required=True,
                        help='language that the model is built for')
    args = parser.parse_args()
    return args


def evids_load(args, path):
    golden_f = open(args.golden_path, 'r')
    golden = {}
    ins_num = 0
    for golden_line in golden_f.readlines():
        line = json.loads(golden_line)
        if line['sample_type'] == 'disturb':
            ins_num += 1
        golden[line['sent_id']] = line

    evids = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            dic = json.loads(line)
            dic['sample_type'] = golden[dic['id']]['sample_type']
            if 'rel_ids' in golden[dic['id']]:
                dic['rel_ids'] = golden[dic['id']]['rel_ids']
            evids[dic['id']] = dic
    return evids, ins_num


def _calc_MAP_by_bin(top_p, length_adv, adv_attriRank_list, ori_attriRank_list):
    """
    This is our old way to calculate MAP,
    which follows equation two in consistency section of README
    """
    hits = 0
    sum_precs = 0.0
    length_t = math.ceil(length_adv * top_p)
    adv_t = adv_attriRank_list[:length_t]
    for char_idx, char in enumerate(adv_t):
        if char in ori_attriRank_list[:char_idx + 1]:
            hits += 1
        sum_precs += hits / (char_idx + 1)
    if length_t > 0:
        sum_precs /= length_t
    return sum_precs


def _calc_MAP_by_bin_paper(top_p, length_adv, adv_attriRank_list,
                           ori_attriRank_list):
    """
    This function calculates MAP using the equation in our paper,
    which follows equation one in consistency section of README
    """
    total_precs = 0.0
    for i in range(length_adv):
        hits = 0.0
        i += 1
        adv_t = adv_attriRank_list[:i]
        for char_idx, char in enumerate(adv_t):
            if char in ori_attriRank_list[:i]:
                hits += 1
        hits = hits / i
        total_precs += hits
    if length_adv == 0:
        return 0
    return total_precs / length_adv


def _calc_map(evids, key, ins_num):
    t_map = 0.0

    adv_num = 0
    ori_num = 0
    sample_length = len(evids)
    for ori_idx in evids:
        if evids[ori_idx]['sample_type'] == 'ori':
            ori = evids[ori_idx]
            ori_num += 1
            # One original instance can be related to several disturbed instance
            for adv_idx in evids[ori_idx]['rel_ids']:
                if adv_idx in evids:
                    adv_num += 1
                    adv = evids[adv_idx]
                    ori_attriRank_list = list(ori['rationale_token'][key])
                    adv_attriRank_list = list(adv['rationale_token'][key])
                    length_adv = len(adv_attriRank_list)

                    sum_precs = _calc_MAP_by_bin_paper(1, length_adv,
                                                       adv_attriRank_list,
                                                       ori_attriRank_list)
                    t_map += sum_precs

    return t_map / ins_num, ori_num + adv_num


def cal_MAP(args, pred_path, la):
    evids, ins_num = evids_load(args, pred_path)
    if not evids:
        print(pred_path + " file empty!")
        return 0
    first_key = list(evids.keys())[0]
    t_map = 0
    num = 0
    for i in range(len(evids[first_key]['rationale'])):
        t_map_tmp, num_tmp = _calc_map(evids, i, ins_num)
        t_map += t_map_tmp
        num += num_tmp
    t_map /= len(evids[first_key]['rationale'])
    num /= len(evids[first_key]['rationale'])
    print('total\t%d\t%.1f' % \
        (num, 100 * t_map))
    return 0


if __name__ == '__main__':
    args = get_args()
    la = args.language
    pred_path = args.pred_path
    if os.path.exists(pred_path):
        cal_MAP(args, pred_path, la)
    else:
        print("Prediction file does not exists!")
