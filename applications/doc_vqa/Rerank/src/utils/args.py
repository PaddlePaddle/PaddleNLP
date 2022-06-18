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
"""Arguments for configuration."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import six
import os
import sys
import argparse
import logging

import paddle.fluid as fluid

log = logging.getLogger(__name__)


def prepare_logger(logger, debug=False, save_to_file=None):
    formatter = logging.Formatter(
        fmt=
        '[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s'
    )
    console_hdl = logging.StreamHandler()
    console_hdl.setFormatter(formatter)
    logger.addHandler(console_hdl)
    if save_to_file is not None and not os.path.exists(save_to_file):
        file_hdl = logging.FileHandler(save_to_file)
        file_hdl.setFormatter(formatter)
        logger.addHandler(file_hdl)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def str2bool(v):
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):

    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self,
                name,
                type,
                default,
                help,
                positional_arg=False,
                **kwargs):
        prefix = "" if positional_arg else "--"
        type = str2bool if type == bool else type
        self._group.add_argument(prefix + name,
                                 default=default,
                                 type=type,
                                 help=help + ' Default: %(default)s.',
                                 **kwargs)


def print_arguments(args):
    log.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        log.info('%s: %s' % (arg, value))
    log.info('------------------------------------------------')


def check_cuda(use_cuda, err = \
    "\nYou can not set use_cuda = True in the model because you are using paddlepaddle-cpu.\n \
    Please: 1. Install paddlepaddle-gpu to run your models on GPU or 2. Set use_cuda = False to run models on CPU.\n"
                                                                                                                     ):
    try:
        if use_cuda == True and fluid.is_compiled_with_cuda() == False:
            log.error(err)
            sys.exit(1)
    except Exception as e:
        pass
