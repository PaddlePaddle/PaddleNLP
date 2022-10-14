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

import hashlib
import json
import sys
import argparse
from collections import defaultdict

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--q2id_map", default='', type=str, help="")
parser.add_argument("--p2id_map", default='', type=str, help="")
parser.add_argument("--recall_result", default='', type=str, help="")
parser.add_argument("--outputf", default='output/dual_res.json', type=str, help="")
parser.add_argument("--score_f", default='', type=str, help="")
parser.add_argument("--id_f", default='', type=str, help="")
parser.add_argument('--mode', choices=['recall', 'rank'], default="recall", help="Select which device to run dense_qa system, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def recall_res_to_json():
    q2id_map = args.q2id_map
    p2id_map = args.p2id_map
    recall_result = args.recall_result

    # map query to its origianl ID
    with open(q2id_map, "r") as fr:
        q2qid = json.load(fr)

    # map para line number to its original ID
    with open(p2id_map, "r") as fr:
        pcid2pid = json.load(fr)

    qprank = defaultdict(list)
    with open(recall_result, 'r') as f:
        for line in f.readlines():
            q, pcid, rank, score = line.strip().split('\t')
            qprank[q2qid[q]].append(pcid2pid[pcid])

    # check for length
    for key in list(qprank.keys()):
        assert len(qprank[key]) == 50

    with open(args.outputf, 'w', encoding='utf-8') as fp:
        json.dump(qprank, fp, ensure_ascii=False, indent='\t')


def rerank_res_to_json():

    score_f = args.score_f
    id_f = args.id_f

    scores = []
    q_ids = []
    p_ids = []
    q_dic = defaultdict(list)

    with open(score_f, 'r') as f:
        for line in f:
            scores.append(float(line.strip()))

    with open(id_f, 'r') as f:
        for line in f:
            v = line.strip().split('\t')
            q_ids.append((v[0]))
            p_ids.append((v[1]))

    for q, p, s in zip(q_ids, p_ids, scores):
        q_dic[q].append((s, p))

    output = []
    for q in q_dic:
        rank = 0
        cands = q_dic[q]
        cands.sort(reverse=True)
        for cand in cands:
            rank += 1
            output.append([q, cand[1], rank])
            if rank > 49:
                break

    with open(args.outputf, 'w') as f:
        res = dict()
        for line in output:
            qid, pid, rank = line
            if qid not in res:
                res[qid] = [0] * 50
            res[qid][int(rank) - 1] = pid
        json.dump(res, f, ensure_ascii=False, indent='\t')


if __name__ == "__main__":
    if (args.mode == 'recall'):
        recall_res_to_json()
    else:
        rerank_res_to_json()
