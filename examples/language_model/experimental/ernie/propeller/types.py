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
"""Basic types"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import json
from collections import namedtuple


class RunMode(object):
    """model_fn will be called in 3 modes"""
    TRAIN = 1
    PREDICT = 2
    EVAL = 3


class HParams(object):
    """Hyper paramerter"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        if key not in self.__dict__:
            raise ValueError('key(%s) not in HParams.' % key)
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.to_dict())

    def __setitem__(self, key, val):
        self.__dict__[key] = val

    @classmethod
    def from_json(cls, json_str):
        """doc"""
        d = json.loads(json_str)
        if type(d) != dict:
            raise ValueError('json object must be dict.')
        return HParams.from_dict(d)

    def get(self, key, default=None):
        """doc"""
        return self.__dict__.get(key, default)

    @classmethod
    def from_dict(cls, d):
        """doc"""
        if type(d) != dict:
            raise ValueError('input must be dict.')
        hp = HParams(**d)
        return hp

    def to_json(self):
        """doc"""
        return json.dumps(self.__dict__)

    def to_dict(self):
        """doc"""
        return self.__dict__

    def join(self, other):
        """doc"""
        if not isinstance(other, HParams):
            raise ValueError('input must be HParams instance. got %s' %
                             type(other))
        self.__dict__.update(**other.__dict__)
        return self


SummaryRecord = namedtuple('SummaryRecord', ['scalar', 'histogram'])

WarmStartSetting = namedtuple('WarmStartSetting', ['predicate_fn', 'from_dir'])
TextoneWarmStartSetting = namedtuple('TextoneWarmStartSetting', ['from_dir'])

RunConfig = namedtuple('RunConfig', [
    'model_dir', 'run_steps', 'max_steps', 'save_steps', 'eval_steps',
    'eval_max_steps', 'skip_steps', 'log_steps', 'max_ckpt', 'shit',
    'log_prefix'
])
RunConfig.__new__.__defaults__ = (None, ) * len(RunConfig._fields)

ProgramPair = namedtuple('ProgramPair', ['train_program', 'startup_program'])

InferenceSpec = namedtuple('InferenceSpec', ['inputs', 'outputs'])

ModelSpec = namedtuple('ModelSpec', [
    'loss',
    'predictions',
    'metrics',
    'mode',
    'inference_spec',
    'train_hooks',
    'eval_hooks',
])
ModelSpec.__new__.__defaults__ = (None, ) * len(ModelSpec._fields)


class StopException(Exception):
    """doc"""
    pass
