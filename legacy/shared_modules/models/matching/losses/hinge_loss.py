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
hinge loss
"""

import sys

sys.path.append("../../../")
import models.matching.paddle_layers as layers


class HingeLoss(object):
    """
    Hing Loss Calculate class
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        self.margin = conf_dict["loss"]["margin"]

    def compute(self, pos, neg):
        """
        compute loss
        """
        elementwise_max = layers.ElementwiseMaxLayer()
        elementwise_add = layers.ElementwiseAddLayer()
        elementwise_sub = layers.ElementwiseSubLayer()
        constant = layers.ConstantLayer()
        reduce_mean = layers.ReduceMeanLayer()
        loss = reduce_mean.ops(
            elementwise_max.ops(
                constant.ops(neg, neg.shape, "float32", 0.0),
                elementwise_add.ops(
                    elementwise_sub.ops(neg, pos),
                    constant.ops(neg, neg.shape, "float32", self.margin))))
        return loss
