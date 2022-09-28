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

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys
import argparse
import logging
import paddle


class UnpackDataLoader(paddle.io.DataLoader):

    def __init__(self, *args, **kwargs):
        super(UnpackDataLoader, self).__init__(*args, batch_size=1, **kwargs)

    def __iter__(self):
        return ([yy[0] for yy in y]
                for y in super(UnpackDataLoader, self).__iter__())


def create_if_not_exists(dir):
    try:
        dir.mkdir(parents=True)
    except:
        pass
    return dir


def get_warmup_and_linear_decay(max_steps, warmup_steps):
    return lambda step: min(
        step / warmup_steps, 1. - (step - warmup_steps) /
        (max_steps - warmup_steps))
