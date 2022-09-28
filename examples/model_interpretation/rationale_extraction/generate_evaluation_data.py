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

import argparse
import json


def get_args():
    parser = argparse.ArgumentParser('generate data')

    parser.add_argument('--pred_path', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--data_dir2', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--inter_mode', required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--language', required=True)

    args = parser.parse_args()
    return args


def evids_load(path):
    evids = []
    with open(path, 'r') as f:
        for line in f.readlines():
            dic = json.loads(line)
            evids.append(dic)
    return evids


def dataLoad(args):
    base_path = args.data_dir + '/'
    text_path = base_path + 'rationale_text/dev/dev'
    text_exclusive_path = base_path + 'rationale_exclusive_text/dev/dev'

    with open(text_path, 'r') as f_text:
        text_dict_list = {}
        for line in f_text.readlines():
            line_dict = json.loads(line)
            text_dict_list[line_dict['id']] = line_dict

    with open(text_exclusive_path, 'r') as f_exclusive_text:
        text_exclusive_dict_list = {}
        for line in f_exclusive_text.readlines():
            line_dict = json.loads(line)
            text_exclusive_dict_list[line_dict['id']] = line_dict

    base_path = args.data_dir2 + '/'
    text_path = base_path + 'rationale_text/dev/dev'
    text_exclusive_path = base_path + 'rationale_exclusive_text/dev/dev'

    with open(text_path, 'r') as f_text:
        text_dict_list2 = {}
        for line in f_text.readlines():
            line_dict = json.loads(line)
            text_dict_list2[line_dict['id']] = line_dict

    with open(text_exclusive_path, 'r') as f_exclusive_text:
        text_exclusive_dict_list2 = {}
        for line in f_exclusive_text.readlines():
            line_dict = json.loads(line)
            text_exclusive_dict_list2[line_dict['id']] = line_dict

    return text_dict_list, text_exclusive_dict_list, text_dict_list2, text_exclusive_dict_list2


def r_data_generation(args, evids, text_dict_list, text_exclusive_dict_list,
                      text_dict_list2, text_exclusive_dict_list2):
    save_path = args.save_path
    f_save = open(save_path, 'w')

    res_data = []
    for ins in evids:
        temp = {}
        temp['id'] = ins['id']
        temp['pred_label'] = ins['pred_label']
        temp['rationale'] = text_dict_list2[ins['id']]['context_idx']
        temp['no_rationale'] = text_exclusive_dict_list2[
            ins['id']]['context_idx']
        if len(temp['rationale']) > 1 and \
            args.inter_mode != 'lime' and \
            not (args.base_model.startswith('roberta')):
            for i in range(len(temp['rationale'][1])):
                temp['rationale'][1][i] -= len(temp['rationale'][0]) + len(
                    temp['no_rationale'][0])
            for i in range(len(temp['no_rationale'][1])):
                temp['no_rationale'][1][i] -= len(temp['rationale'][0]) + len(
                    temp['no_rationale'][0])
        temp['rationale_pred'] = text_dict_list[ins['id']]['pred_label']
        temp['no_rationale_pred'] = text_exclusive_dict_list[
            ins['id']]['pred_label']
        temp['rationale_token'] = text_dict_list2[ins['id']]['context_token']

        res_data.append(temp)

        f_save.write(json.dumps(temp, ensure_ascii=False) + '\n')
    f_save.close()


if __name__ == '__main__':
    args = get_args()
    text_dict_list, text_exclusive_dict_list, text_dict_list2, text_exclusive_dict_list2 = dataLoad(
        args)
    evids = evids_load(args.pred_path)
    r_data_generation(args, evids, text_dict_list, text_exclusive_dict_list,
                      text_dict_list2, text_exclusive_dict_list2)
