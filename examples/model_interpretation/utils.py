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
"""This file contains some public functions
"""


def convert_tokenizer_res_to_old_version(tokenized_res):
    if isinstance(tokenized_res, list):
        return tokenized_res
    if isinstance(tokenized_res, dict):
        if len(tokenized_res['input_ids']) == 0 or not isinstance(
                tokenized_res['input_ids'][0], list):
            return tokenized_res
        else:
            res = []
            for idx in range(len(tokenized_res['input_ids'])):
                temp_dict = {}
                key_list = list(tokenized_res.keys())
                for key in key_list:
                    temp_dict[key] = tokenized_res[key][idx]
                res.append(temp_dict)
            return res
    else:
        raise ValueError('unsupported result type')


def cal_score(match_list, sorted_token):
    over_all = []
    miss = 0
    for i in match_list:
        over_all.extend(i[0])

    score_dic = {}
    for i in sorted_token:
        split_time = over_all.count(i[0])
        if split_time:
            score_dic[i[0]] = i[2] / split_time
        else:
            score_dic[i[0]] = 0.0
    if miss != 0:
        print(miss)

    score = []
    for i in range(len(match_list)):
        cur_score = 0.0
        for j in match_list[i][0]:
            if j == -1:
                continue
            cur_score += score_dic[j]
        score.append([str(match_list[i][1]), match_list[i][2], cur_score])
    return score


def match(context, context_seg, sorted_token):
    result = []
    pointer1 = 0  # point at the context
    pointer2 = 0  # point at the sorted_token array
    for i in range(len(context_seg)):
        seg_start_idx = context.find(context_seg[i], pointer1)
        if seg_start_idx < 0:
            print("Error: token not in context")
        seg_end_idx = seg_start_idx + len(context_seg[i])

        cur_set = []
        while pointer2 < len(sorted_token):
            while pointer2 < len(sorted_token) and sorted_token[pointer2][1][
                    1] <= seg_start_idx:
                pointer2 += 1
            if pointer2 >= len(sorted_token):
                break
            if sorted_token[pointer2][1][0] >= seg_end_idx:
                break
            cur_set.append(sorted_token[pointer2][0])
            pointer2 += 1
        result.append([cur_set, i, context_seg[i]])
        pointer2 -= 1
        pointer1 = seg_end_idx
    score = cal_score(result, sorted_token)
    return score
