# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import distutils.util
import os

import paddle


def strtobool(s):
    return True if distutils.util.strtobool(s) else False


def get_env(env_name, default_value=False):
    return strtobool(os.getenv(env_name, str(default_value)))


def mock_layers():
    if get_env("USE_FUSED_LN"):
        paddle.nn.Linear = paddle.incubate.nn.FusedLinear
