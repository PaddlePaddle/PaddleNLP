# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--similar_text_pair",
    type=str,
    default="",
    help="The full path of similar pair file",
)
parser.add_argument(
    "--recall_result_file",
    type=str,
    default="",
    help="The full path of recall result file",
)
parser.add_argument(
    "--recall_num",
    type=int,
    default=10,
    help="Most similar number of doc recalled from corpus per query",
)
args = parser.parse_args()


def recall(rs, N=10):
    """
    Ratio of recalled Ground Truth at topN Recalled Docs
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> recall(rs, N=1)
    0.333333
    >>> recall(rs, N=2)
    >>> 0.6666667
    >>> recall(rs, N=3)
    >>> 1.0
    Args:
        rs: Iterator of recalled flag()
    Returns:
        Recall@N
    """

    recall_flags = [np.sum(r[0:N]) for r in rs]
    return np.mean(recall_flags)


if __name__ == "__main__":
    text2similar = {}
    with open(args.similar_text_pair, "r", encoding="utf-8") as f:
        for line in f:
            text, similar_text = line.rstrip().rsplit("\t", 1)
            text2similar[text] = similar_text

    rs = []
    with open(args.recall_result_file, "r", encoding="utf-8") as f:
        relevance_labels = []
        for index, line in enumerate(f):
            text_arr = line.rstrip().split("\t")
            (
                text_title,
                text_para,
                recalled_title,
                recalled_para,
                label,
                cosine_sim,
            ) = text_arr
            if text2similar["\t".join([text_title, text_para])] == label:
                relevance_labels.append(1)
            else:
                relevance_labels.append(0)

            if (index + 1) % args.recall_num == 0:
                rs.append(relevance_labels)
                relevance_labels = []

    recall_N = []
    recall_num = [1, 5, 10, 20, 50]
    for topN in recall_num:
        R = round(100 * recall(rs, N=topN), 3)
        recall_N.append(str(R))
    result = open("result.tsv", "a")
    res = []
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    res.append(timestamp)
    for key, val in zip(recall_num, recall_N):
        print("recall@{}={}".format(key, val))
        res.append(str(val))
    result.write("\t".join(res) + "\n")
