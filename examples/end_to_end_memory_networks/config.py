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

import yaml


class Config(object):
    """
    A simple waper for configs
    """

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.d = yaml.load(f.read(), Loader=yaml.SafeLoader)

    def __getattribute__(self, key):
        d = super(Config, self).__getattribute__('d')
        if key in d:
            return d[key]
        else:
            return super(Config, self).__getattribute__(key)
