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

from __future__ import print_function
from __future__ import division
from __future__ import print_function

import os
import six
import ast
import copy

import numpy as np
import paddle.fluid as fluid


class InputField(object):
    def __init__(self, input_field):
        """init inpit field"""
        self.context_wordseq = input_field[0]
        self.response_wordseq = input_field[1]
        self.labels = input_field[2]
