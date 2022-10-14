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

import os
import argparse
import numpy as np

from paddlenlp.transformers import BasicTokenizer
from paddlenlp.metrics import BLEU

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--similar_text_pair", type=str, default='', help="The full path of similat pair file")
parser.add_argument("--recall_result_file", type=str, default='', help="The full path of recall result file")
parser.add_argument("--recall_num", type=int, default=10, help="Most similair number of doc recalled from corpus per query")
parser.add_argument("--query_answer_file", type=str, default='', help="The full path of test query with answer file")
parser.add_argument("--question_answer_file", type=str, default='', help="The full path of true question with answer file")
parser.add_argument("--bleu_threshold", type=float, default=None, help="The bleu_threshold to determine whether the two answers are the same, if None The two are required to be identical")



def calc_bleu_n(preds, targets, n_size=4):
    assert len(preds) == len(targets), (
        'The length of pred_responses should be equal to the length of '
        'target_responses. But received {} and {}.'.format(
            len(preds), len(targets)))
    bleu = BLEU(n_size=n_size)
    tokenizer = BasicTokenizer()

    for pred, target in zip(preds, targets):
        pred_tokens = tokenizer.tokenize(pred)
        target_token = tokenizer.tokenize(target)
        bleu.add_inst(pred_tokens, [target_token])
    return bleu.score()

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

    recall_flags = [1 if np.sum(r[0:N])>0 else 0 for r in rs]
    # recall_flags = [np.sum(r[0:N]) else 0 for r in rs]
    return np.mean(recall_flags)


if __name__ == "__main__":
    args = parser.parse_args()
    # yapf: enable

    if args.query_answer_file and args.question_answer_file:
        query2answer = {}
        with open(args.query_answer_file, 'r', encoding='utf-8') as f:
            for line in f:
                text, answer = line.rstrip(' ').split("\t")
                query2answer[text] = answer.strip()
        question2answer = {}
        with open(args.question_answer_file, 'r', encoding='utf-8') as f:
            for line in f:
                text, answer = line.rstrip(' ').split("\t")
                question2answer[text] = answer.strip()

    text2similar = {}
    with open(args.similar_text_pair, 'r', encoding='utf-8') as f:
        for line in f:
            text, similar_text = line.rstrip().split("\t")
            text2similar[text] = similar_text

    rs = []

    with open(args.recall_result_file, 'r', encoding='utf-8') as f:
        relevance_labels = []
        for index, line in enumerate(f):

            if index % args.recall_num == 0 and index != 0:
                rs.append(relevance_labels)
                relevance_labels = []

            text, recalled_text, cosine_sim = line.rstrip().split("\t")
            if args.query_answer_file and args.question_answer_file:
                answer_query = query2answer[text]
                answer_question = question2answer[recalled_text]
                if args.bleu_threshold:
                    if not answer_query or not answer_question:
                        score = 0
                    else:
                        score = calc_bleu_n([answer_query], [answer_question],
                                            1)
                    if score >= args.bleu_threshold:
                        relevance_labels.append(1)
                    else:
                        relevance_labels.append(0)
                else:
                    if answer_query == answer_question:
                        relevance_labels.append(1)
                    else:
                        relevance_labels.append(0)
            else:
                if text2similar[text] == recalled_text:
                    relevance_labels.append(1)
                else:
                    relevance_labels.append(0)
    recall_N = []
    recall_num = [1, 5, 10]
    result = open('result.tsv', 'a')
    res = []
    for topN in recall_num:
        R = round(100 * recall(rs, N=topN), 3)
        recall_N.append(str(R))
    for key, val in zip(recall_num, recall_N):
        print('recall@{}={}'.format(key, val))
        res.append(str(val))
    result.write('\t'.join(res) + '\n')
