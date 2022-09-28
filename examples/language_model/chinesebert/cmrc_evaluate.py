#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Evaluation script for CMRC 2018
version: v5 - special
Note: 
v5 - special: Evaluate on SQuAD-style CMRC 2018 Datasets
v5: formatted output, add usage description
v4: fixed segmentation issues
'''

import argparse
import json
import re
import sys
from collections import OrderedDict
import nltk


# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = [
        '-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：',
        '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、', '「', '」', '（',
        '）', '－', '～', '『', '』'
    ]
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


# remove punctuation
def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = [
        '-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：',
        '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、', '「', '」', '（',
        '）', '－', '～', '『', '』'
    ]
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


# find longest common string
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


#
def evaluate(ground_truth_file, prediction_file):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    for instance in ground_truth_file["data"]:
        # context_id   = instance['context_id'].strip()
        # context_text = instance['context_text'].strip()
        for para in instance["paragraphs"]:
            for qas in para['qas']:
                total_count += 1
                query_id = qas['id'].strip()
                query_text = qas['question'].strip()
                answers = [x["text"] for x in qas['answers']]

                if query_id not in prediction_file:
                    sys.stderr.write(
                        'Unanswered question: {}\n'.format(query_id))
                    skip_count += 1
                    continue

                prediction = str(prediction_file[query_id])
                f1 += calc_f1_score(answers, prediction)
                em += calc_em_score(answers, prediction)

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score, total_count, skip_count


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


def get_result(ground_truth_file, prediction_file):
    ground_truth_file = json.load(open(ground_truth_file, 'rb'))
    prediction_file = json.load(open(prediction_file, 'rb'))
    F1, EM, TOTAL, SKIP = evaluate(ground_truth_file, prediction_file)
    AVG = (EM + F1) * 0.5
    output_result = OrderedDict()
    output_result['AVERAGE'] = '%.3f' % AVG
    output_result['F1'] = '%.3f' % F1
    output_result['EM'] = '%.3f' % EM
    output_result['TOTAL'] = TOTAL
    output_result['SKIP'] = SKIP
    print(json.dumps(output_result))
    return output_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation Script for CMRC 2018')
    parser.add_argument('--dataset_file',
                        default="cmrc2018_public/dev.json",
                        help='Official dataset file')
    parser.add_argument('--prediction_file',
                        default="all_predictions.json",
                        help='Your prediction File')
    args = parser.parse_args()
    ground_truth_file = json.load(open(args.dataset_file, 'rb'))
    prediction_file = json.load(open(args.prediction_file, 'rb'))
    F1, EM, TOTAL, SKIP = evaluate(ground_truth_file, prediction_file)
    AVG = (EM + F1) * 0.5
    output_result = OrderedDict()
    output_result['AVERAGE'] = '%.3f' % AVG
    output_result['F1'] = '%.3f' % F1
    output_result['EM'] = '%.3f' % EM
    output_result['TOTAL'] = TOTAL
    output_result['SKIP'] = SKIP
    output_result['FILE'] = args.prediction_file
    print(json.dumps(output_result))
