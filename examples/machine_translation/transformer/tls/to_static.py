#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import copy
import importlib

import paddle.nn as nn
from paddle.jit import to_static
from paddle.static import InputSpec
import paddle


def create_input_specs():
    src_word = paddle.static.InputSpec(name="src_word",
                                       shape=[None, None],
                                       dtype="int64")
    trg_word = paddle.static.InputSpec(name="trg_word",
                                       shape=[None, None],
                                       dtype="int64")
    return [src_word, trg_word]


def apply_to_static(config, model):
    support_to_static = config.get('to_static', False)
    if support_to_static:
        specs = create_input_specs()
        model = to_static(model, input_spec=specs)
    return model
