#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
softmax loss
"""

import sys
import paddle.fluid as fluid

sys.path.append("../../../")
import models.matching.paddle_layers as layers


class SoftmaxCrossEntropyLoss(object):
    """
    Softmax with Cross Entropy Loss Calculate
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        pass

    def compute(self, input, label):
        """
        compute loss
        """
        reduce_mean = layers.ReduceMeanLayer()
        cost = fluid.layers.cross_entropy(input=input, label=label)
        avg_cost = reduce_mean.ops(cost)
        return avg_cost
