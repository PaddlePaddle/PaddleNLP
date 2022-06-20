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
import os

from .ema import ExponentialMovingAverage

__all__ = ['layerwise_lr_decay', 'ExponentialMovingAverage']


# Layerwise decay
def layerwise_lr_decay(decay_rate, name_dict, n_layers, param):
    """
    Args:
        decay_rate (float): 
            The layer-wise decay ratio.
        name_dict (dict): 
            The keys of name_dict is dynamic name of model while the value
            of name_dict is static name.
            Use model.named_parameters() to get name_dict.
        n_layers (int):
            Total number of layers in the transformer encoder.
    """
    ratio = 1.0
    static_name = name_dict[param.name]
    if "encoder.layers" in static_name:
        idx = static_name.find("encoder.layers.")
        layer = int(static_name[idx:].split(".")[2])
        ratio = decay_rate**(n_layers - layer)
    elif "embedding" in static_name:
        ratio = decay_rate**(n_layers + 1)
    return ratio
