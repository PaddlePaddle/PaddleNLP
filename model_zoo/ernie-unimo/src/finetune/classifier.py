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
"""Model for classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

import paddle
import paddle.nn as nn
from paddle import ParamAttr


class UNIMOClassifier(nn.Layer):
    def __init__(self, unimo_model, num_labels):
        super(UNIMOClassifier, self).__init__()
        self.d_model = unimo_model._emb_size
        self.encoder = unimo_model.encoder
        self.fc0 = nn.Linear(self.d_model, self.d_model, weight_attr=ParamAttr(initializer=nn.initializer.TruncatedNormal(std=0.02)), 
            bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.0)))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1, mode="upscale_in_train")
        self.fc1 = nn.Linear(self.d_model, num_labels, weight_attr=ParamAttr(initializer=nn.initializer.TruncatedNormal(std=0.02)), 
            bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.0)))

            

    def forward(self, emb_ids, input_mask):
        features = self.encoder(emb_ids=emb_ids, input_mask=input_mask)
        features = paddle.reshape(features[:,0,:], [-1, self.d_model])
        cls_feats = self.relu(self.fc0(features))
        cls_feats = self.dropout(cls_feats)
        logits = self.fc1(cls_feats)
        return logits