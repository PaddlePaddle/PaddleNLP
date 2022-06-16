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

import json
import re
import math
import numpy as np
import argparse
import os
import sys


def get_args():
    parser = argparse.ArgumentParser('generate data')

    parser.add_argument('--pred_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--language', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--ratio', type=str, required=True)

    args = parser.parse_args()
    return args


def evids_load(path):
    evids = []
    with open(path, 'r') as f:
        for line in f.readlines():
            dic = json.loads(line)
            evids.append(dic)
    return evids


def generate_for_senti(args, evid_dict, ratio):
    r = {}
    ex_r = {}

    raw_text = evid_dict['context']
    label = evid_dict['pred_label']
    char_attri = list(evid_dict['char_attri'].keys())
    length = len(char_attri)

    rationale_ratio = ratio[0]
    toprationale_text, toprationale_exclusive_text = [], []

    keys = [int(x) for x in char_attri[:math.ceil(length * rationale_ratio)]]
    keys.sort()
    for key in keys:
        toprationale_text.append(evid_dict['char_attri'][str(key)][0].strip())

    keys = [int(x) for x in char_attri[math.ceil(length * rationale_ratio):]]
    keys.sort()
    for key in keys:
        toprationale_exclusive_text.append(
            evid_dict['char_attri'][str(key)][0].strip())

    if args.language == 'en':
        toprationale_text = ' '.join(toprationale_text)
        toprationale_exclusive_text = ' '.join(toprationale_exclusive_text)
    else:
        toprationale_text = ''.join(toprationale_text)
        toprationale_exclusive_text = ''.join(toprationale_exclusive_text)

    if len(toprationale_text) == 0:
        toprationale_text = "['UNK']"
    if len(toprationale_exclusive_text) == 0:
        toprationale_exclusive_text = "['UNK']"

    r['id'] = evid_dict['id']
    r['context'] = toprationale_text
    r['context_idx'] = [[
        int(x) for x in char_attri[:math.ceil(length * rationale_ratio)]
    ]]
    r['context_token'] = [[
        evid_dict['char_attri'][x][0]
        for x in char_attri[:math.ceil(length * rationale_ratio)]
    ]]
    r['label'] = label
    ex_r['id'] = evid_dict['id']
    ex_r['context'] = toprationale_exclusive_text
    ex_r['context_idx'] = [[
        int(x) for x in char_attri[math.ceil(length * rationale_ratio):]
    ]]
    ex_r['context_token'] = [[
        evid_dict['char_attri'][x][0]
        for x in char_attri[math.ceil(length * rationale_ratio):]
    ]]
    ex_r['label'] = label
    return r, ex_r


def generate_for_similarity(args, evid_dict, ratio):
    r = {}
    ex_r = {}
    q_rationale_ratio = ratio[0]
    t_rationale_ratio = ratio[1]

    label = evid_dict['pred_label']
    # query
    q_text = evid_dict['query']
    q_char_attri = list(evid_dict['query_char_attri'].keys())
    q_length = len(q_char_attri)

    q_topR_Rtext, q_topR_noRtext = [], []
    keys = [
        int(x) for x in q_char_attri[:math.ceil(q_length * q_rationale_ratio)]
    ]
    keys.sort()
    for key in keys:
        q_topR_Rtext.append(evid_dict['query_char_attri'][str(key)][0].strip())

    keys = [
        int(x) for x in q_char_attri[math.ceil(q_length * q_rationale_ratio):]
    ]
    keys.sort()
    for key in keys:
        q_topR_noRtext.append(
            evid_dict['query_char_attri'][str(key)][0].strip())

    if args.language == 'ch':
        q_topR_Rtext = ''.join(q_topR_Rtext)
        q_topR_noRtext = ''.join(q_topR_noRtext)
    else:
        q_topR_Rtext = ' '.join(q_topR_Rtext)
        q_topR_noRtext = ' '.join(q_topR_noRtext)

    if len(q_topR_Rtext) == 0:
        q_topR_Rtext = "['UNK']"
    if len(q_topR_noRtext) == 0:
        q_topR_noRtext = "['UNK']"

    # title
    t_text = evid_dict['title']
    t_char_attri = list(evid_dict['title_char_attri'].keys())
    t_length = len(t_char_attri)

    t_topR_Rtext, t_topR_noRtext = [], []
    keys = [
        int(x) for x in t_char_attri[:math.ceil(t_length * t_rationale_ratio)]
    ]
    keys.sort()
    for key in keys:
        t_topR_Rtext.append(evid_dict['title_char_attri'][str(key)][0])

    keys = [
        int(x) for x in t_char_attri[math.ceil(t_length * t_rationale_ratio):]
    ]
    keys.sort()
    for key in keys:
        t_topR_noRtext.append(evid_dict['title_char_attri'][str(key)][0])

    if args.language == 'ch':
        t_topR_Rtext = ''.join(t_topR_Rtext)
        t_topR_noRtext = ''.join(t_topR_noRtext)
    else:
        t_topR_Rtext = ' '.join(t_topR_Rtext)
        t_topR_noRtext = ' '.join(t_topR_noRtext)

    if len(t_topR_Rtext) == 0:
        t_topR_Rtext = "['UNK']"
    if len(t_topR_noRtext) == 0:
        t_topR_noRtext = "['UNK']"

    r['id'] = evid_dict['id']
    r['context'] = [q_topR_Rtext, t_topR_Rtext]
    r['context_idx'] = [[
        int(x) for x in q_char_attri[:math.ceil(q_length * q_rationale_ratio)]
    ], [int(x) for x in t_char_attri[:math.ceil(t_length * t_rationale_ratio)]]]
    r['context_token'] = [
        [
            evid_dict['query_char_attri'][x][0]
            for x in q_char_attri[:math.ceil(q_length * q_rationale_ratio)]
        ],
        [
            evid_dict['title_char_attri'][x][0]
            for x in t_char_attri[:math.ceil(t_length * t_rationale_ratio)]
        ]
    ]
    r['label'] = label
    ex_r['id'] = evid_dict['id']
    ex_r['context'] = [q_topR_noRtext, t_topR_noRtext]
    ex_r['context_idx'] = [[
        int(x) for x in q_char_attri[math.ceil(q_length * q_rationale_ratio):]
    ], [int(x) for x in t_char_attri[math.ceil(t_length * t_rationale_ratio):]]]
    ex_r['context_token'] = [
        [
            evid_dict['query_char_attri'][x][0]
            for x in q_char_attri[math.ceil(q_length * q_rationale_ratio):]
        ],
        [
            evid_dict['title_char_attri'][x][0]
            for x in t_char_attri[math.ceil(t_length * t_rationale_ratio):]
        ]
    ]
    ex_r['label'] = label
    return r, ex_r


def generate_for_MRC(args, evid_dict, ratio):
    id = evid_dict['id']
    raw_text = evid_dict['context'] + evid_dict['title']
    question = evid_dict['question']
    char_attri = list(evid_dict['char_attri'].keys())
    length = len(char_attri)

    rationale_ratio = ratio[0]
    toprationale_text, toprationale_exclusive_text = [], []
    keys = [int(x) for x in char_attri[:math.ceil(length * rationale_ratio)]]
    keys.sort()
    for key in keys:
        toprationale_text.append(evid_dict['char_attri'][str(key)][0].strip())

    keys = [int(x) for x in char_attri[math.ceil(length * rationale_ratio):]]
    keys.sort()
    for key in keys:
        toprationale_exclusive_text.append(
            evid_dict['char_attri'][str(key)][0].strip())

    if args.language == 'en':
        toprationale_text = ' '.join(toprationale_text)
        toprationale_exclusive_text = ' '.join(toprationale_exclusive_text)
    else:
        toprationale_text = ''.join(toprationale_text)
        toprationale_exclusive_text = ''.join(toprationale_exclusive_text)

    if len(toprationale_text) == 0:
        toprationale_text = "['UNK']"
    if len(toprationale_exclusive_text) == 0:
        toprationale_exclusive_text = "['UNK']"

    data_R_dict, Rdata_noR_dict = {}, {}

    data_R_dict['id'] = id
    data_R_dict['title'] = ""
    data_R_dict['context'] = toprationale_text
    data_R_dict['question'] = question
    data_R_dict['answers'] = ['']
    data_R_dict['answer_starts'] = [-1]
    data_R_dict['is_impossible'] = False
    data_R_dict['context_idx'] = [[
        int(x) for x in char_attri[:math.ceil(length * rationale_ratio)]
    ]]
    data_R_dict['context_token'] = [[
        evid_dict['char_attri'][x][0]
        for x in char_attri[:math.ceil(length * rationale_ratio)]
    ]]

    Rdata_noR_dict['id'] = id
    Rdata_noR_dict['title'] = ""
    Rdata_noR_dict['context'] = toprationale_exclusive_text
    Rdata_noR_dict['question'] = question
    Rdata_noR_dict['answers'] = ['']
    Rdata_noR_dict['answer_starts'] = [-1]
    Rdata_noR_dict['is_impossible'] = False
    Rdata_noR_dict['context_idx'] = [[
        int(x) for x in char_attri[math.ceil(length * rationale_ratio):]
    ]]
    Rdata_noR_dict['context_token'] = [[
        evid_dict['char_attri'][x][0]
        for x in char_attri[math.ceil(length * rationale_ratio):]
    ]]

    return data_R_dict, Rdata_noR_dict


def r_text_generation(evids, args):
    print('num: {}'.format(len(evids)))

    f_rationale_path = os.path.join(args.save_path, 'rationale_text/dev')
    f_rationale_exclusive_path = os.path.join(args.save_path,
                                              'rationale_exclusive_text/dev')

    if not os.path.exists(f_rationale_path):
        os.makedirs(f_rationale_path)
    if not os.path.exists(f_rationale_exclusive_path):
        os.makedirs(f_rationale_exclusive_path)

    f_rationale = open(os.path.join(f_rationale_path, 'dev'), 'w')
    f_rationale_exclusive = open(
        os.path.join(f_rationale_exclusive_path, 'dev'), 'w')

    rationale_ratio = json.loads(args.ratio)
    for id, evid_dict in enumerate(evids):
        if args.task == 'senti':
            data_R_dict, Rdata_noR_dict = generate_for_senti(
                args, evid_dict, rationale_ratio)
        elif args.task == 'similarity':
            data_R_dict, Rdata_noR_dict = generate_for_similarity(
                args, evid_dict, rationale_ratio)
        elif args.task == 'mrc':
            data_R_dict, Rdata_noR_dict = generate_for_MRC(
                args, evid_dict, rationale_ratio)
        f_rationale.write(json.dumps(data_R_dict, ensure_ascii=False) + '\n')
        f_rationale_exclusive.write(
            json.dumps(Rdata_noR_dict, ensure_ascii=False) + '\n')

    f_rationale.close()
    f_rationale_exclusive.close()


if __name__ == '__main__':
    args = get_args()

    evids = evids_load(args.pred_path)
    r_text_generation(evids, args)
