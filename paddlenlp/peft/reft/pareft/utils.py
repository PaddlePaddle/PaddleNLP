# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import enum

import paddle

from .reft_model import ReftModel


class TaskType(str, enum.Enum):
    """
    Enum class for the different types of tasks supported by REFT.

    Overview of the supported task types:
    - SEQ_CLS: Text classification.
    - CAUSAL_LM: Causal language modeling.
    """

    SEQ_CLS = "SEQ_CLS"
    CAUSAL_LM = "CAUSAL_LM"


# Create an instance of ReFT model.
def get_reft_model(model, reft_config, set_device=True, disable_model_grads=True):
    reft_model = ReftModel(reft_config, model)
    if set_device:
        device = paddle.set_device("gpu")
        reft_model = reft_model.to(device)
    if disable_model_grads:
        reft_model.disable_model_gradients()
    return reft_model
