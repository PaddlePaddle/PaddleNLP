# -*- coding: utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.                                                                                                      
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
"""evaluation metrics"""

import io
import os
import sys
import numpy as np

import ade.evaluate as evaluate
from ade.utils.configure import PDConfig


def do_eval(args):
    """evaluate metrics"""
    labels = []
    fr = io.open(args.evaluation_file, 'r', encoding="utf8")
    for line in fr:
        tokens = line.strip().split('\t')
        assert len(tokens) == 3
        label = int(tokens[2])
        labels.append(label)

    scores = []
    fr = io.open(args.output_prediction_file, 'r', encoding="utf8")
    for line in fr:
        tokens = line.strip().split('\t')
        assert len(tokens) == 2
        score = tokens[1].strip("[]").split()
        score = np.array(score)
        score = score.astype(np.float64)
        scores.append(score)

    if args.loss_type == 'CLS':
        recall_dict = evaluate.evaluate_Recall(list(zip(scores, labels)))
        mean_score = sum(scores) / len(scores)
        print('mean score: %.6f' % mean_score)
        print('evaluation recall result:')
        print('1_in_2: %.6f\t1_in_10: %.6f\t2_in_10: %.6f\t5_in_10: %.6f' %
              (recall_dict['1_in_2'], recall_dict['1_in_10'],
               recall_dict['2_in_10'], recall_dict['5_in_10']))
    elif args.loss_type == 'L2':
        scores = [x[0] for x in scores]
        mean_score = sum(scores) / len(scores)
        cor = evaluate.evaluate_cor(scores, labels)
        print('mean score: %.6f\nevaluation cor results:%.6f' %
              (mean_score, cor))
    else:
        raise ValueError


if __name__ == "__main__":
    args = PDConfig(yaml_file="./data/config/ade.yaml")
    args.build()

    do_eval(args)
