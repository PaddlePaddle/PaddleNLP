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
"""define prediction results"""

import re
import sys
import numpy as np

import paddle.fluid as fluid


class DefinePredict(object):
    """
    Packaging Prediction Results
    """

    def __init__(self):
        """
        init
        """
        self.task_map = {
            'udc': 'get_matching_res',
            'swda': 'get_cls_res',
            'mrda': 'get_cls_res',
            'atis_intent': 'get_cls_res',
            'atis_slot': 'get_sequence_tagging',
            'dstc2': 'get_multi_cls_res',
            'dstc2_asr': 'get_multi_cls_res',
            'multi-woz': 'get_multi_cls_res'
        }

    def get_matching_res(self, probs, params=None):
        """
        get matching score
        """
        probs = list(probs)
        return probs[1]

    def get_cls_res(self, probs, params=None):
        """
        get da classify tag
        """
        probs = list(probs)
        max_prob = max(probs)
        tag = probs.index(max_prob)
        return tag

    def get_sequence_tagging(self, probs, params=None):
        """
        get sequence tagging tag
        """
        labels = []
        batch_labels = np.array(probs).reshape(-1, params)
        labels = [" ".join([str(l) for l in list(l_l)]) for l_l in batch_labels]
        return labels

    def get_multi_cls_res(self, probs, params=None):
        """
        get dst classify tag
        """
        labels = []
        probs = list(probs)
        for i in range(len(probs)):
            if probs[i] >= 0.5:
                labels.append(i)
        if not labels:
            max_prob = max(probs)
            label_str = str(probs.index(max_prob))
        else:
            label_str = " ".join([str(l) for l in sorted(labels)])

        return label_str
