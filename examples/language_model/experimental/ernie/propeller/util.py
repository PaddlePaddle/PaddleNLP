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
"""global utils"""

import os
import six
import re
import json
import argparse
import itertools
import logging
from functools import reduce

from .types import RunConfig
from .types import HParams

log = logging.getLogger(__name__)


def ArgumentParser(name):
    """predefined argparser"""
    parser = argparse.ArgumentParser('propeller model')
    parser.add_argument('--run_config', type=str, default='')
    parser.add_argument(
        '--hparam', type=str, nargs='*', action='append', default=[['']])
    return parser


def _get_dict_from_environ_or_json_or_file(args, env_name):
    if args == '':
        return None
    if args is None:
        s = os.environ.get(env_name)
    else:
        s = args
        if os.path.exists(s):
            s = open(s).read()
    if isinstance(s, six.string_types):
        try:
            r = json.loads(s)
        except ValueError:
            try:
                r = eval(s)
            except SyntaxError as e:
                raise ValueError('json parse error: %s \n>Got json: %s' %
                                 (repr(e), s))
        return r
    else:
        return s  #None


def parse_file(filename):
    """useless api"""
    d = _get_dict_from_environ_or_json_or_file(filename, None)
    if d is None:
        raise ValueError('file(%s) not found' % filename)
    return d


def parse_runconfig(args=None):
    """get run_config from env or file"""
    d = _get_dict_from_environ_or_json_or_file(args.run_config,
                                               'PROPELLER_RUNCONFIG')
    if d is None:
        raise ValueError('run_config not found')
    return RunConfig(**d)


def parse_hparam(args=None):
    """get hparam from env or file"""
    if args is not None:
        hparam_strs = reduce(list.__add__, args.hparam)
    else:
        hparam_strs = [None]

    hparams = [
        _get_dict_from_environ_or_json_or_file(hp, 'PROPELLER_HPARAMS')
        for hp in hparam_strs
    ]
    hparams = [HParams(**h) for h in hparams if h is not None]
    if len(hparams) == 0:
        return HParams()
    else:
        hparam = reduce(lambda x, y: x.join(y), hparams)
        return hparam


def flatten(s):
    """doc"""
    assert is_struture(s)
    schema = [len(ss) for ss in s]
    flt = list(itertools.chain(*s))
    return flt, schema


def unflatten(structure, schema):
    """doc"""
    start = 0
    res = []
    for _range in schema:
        res.append(structure[start:start + _range])
        start += _range
    return res


def is_struture(s):
    """doc"""
    return isinstance(s, list) or isinstance(s, tuple)


def map_structure(func, s):
    """same sa tf.map_structure"""
    if isinstance(s, list) or isinstance(s, tuple):
        return [map_structure(func, ss) for ss in s]
    elif isinstance(s, dict):
        return {k: map_structure(func, v) for k, v in six.iteritems(s)}
    else:
        return func(s)
