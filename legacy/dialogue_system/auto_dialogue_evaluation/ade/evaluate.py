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
"""Evaluation for auto dialogue evaluation"""

import sys
import numpy as np
import pandas as pd


def get_p_at_n_in_m(data, n, m, ind):
    """
    Get n in m
    """
    pos_score = data[ind][0]
    curr = data[ind:ind + m]
    curr = sorted(curr, key=lambda x: x[0], reverse=True)

    if curr[n - 1][0] <= pos_score:
        return 1
    return 0


def evaluate_Recall(data):
    """
    Evaluate Recall
    """
    p_at_1_in_2 = 0.0
    p_at_1_in_10 = 0.0
    p_at_2_in_10 = 0.0
    p_at_5_in_10 = 0.0

    length = len(data) // 10
    print('length=%s' % length)

    for i in range(0, length):
        ind = i * 10
        assert data[ind][1] == 1

        p_at_1_in_2 += get_p_at_n_in_m(data, 1, 2, ind)
        p_at_1_in_10 += get_p_at_n_in_m(data, 1, 10, ind)
        p_at_2_in_10 += get_p_at_n_in_m(data, 2, 10, ind)
        p_at_5_in_10 += get_p_at_n_in_m(data, 5, 10, ind)

    recall_dict = {
        '1_in_2': p_at_1_in_2 / length,
        '1_in_10': p_at_1_in_10 / length,
        '2_in_10': p_at_2_in_10 / length,
        '5_in_10': p_at_5_in_10 / length
    }

    return recall_dict


def evaluate_cor(pred, true):
    """
    Evaluate cor
    """
    df = pd.DataFrame({'pred': pred, 'true': true})
    cor_matrix = df.corr('spearman')
    return cor_matrix['pred']['true']
