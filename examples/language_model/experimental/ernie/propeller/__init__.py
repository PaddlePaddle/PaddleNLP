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
"""Propeller"""
import os
import sys
import logging
import six
from time import time

__version__ = '0.2'

log = logging.getLogger(__name__)
stream_hdl = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
    fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s')

try:
    from colorlog import ColoredFormatter
    fancy_formatter = ColoredFormatter(
        fmt='%(log_color)s[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s'
    )
    stream_hdl.setFormatter(fancy_formatter)
except ImportError:
    stream_hdl.setFormatter(formatter)

log.setLevel(logging.INFO)
log.addHandler(stream_hdl)
log.propagate = False

#from propeller.types import *
from propeller.util import ArgumentParser, parse_hparam, parse_runconfig, parse_file
