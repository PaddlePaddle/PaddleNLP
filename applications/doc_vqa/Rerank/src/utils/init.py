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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import six
import ast
import copy
import logging
import numpy as np

import paddle.fluid as fluid

log = logging.getLogger(__name__)


def init_checkpoint(exe, init_checkpoint_path, main_program):
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        if not fluid.io.is_persistable(var):
            return False
        if not os.path.exists(os.path.join(init_checkpoint_path, var.name)):
            print("Var not exists: [%s]\t%s" %
                  (var.name, os.path.join(init_checkpoint_path, var.name)))
        #else:
        #    print ("Var exists: [%s]" % (var.name))
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    fluid.io.load_vars(exe,
                       init_checkpoint_path,
                       main_program=main_program,
                       predicate=existed_persitables)
    log.info("Load model from {}".format(init_checkpoint_path))


def init_pretraining_params(exe, pretraining_params_path, main_program):
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        if not os.path.exists(os.path.join(pretraining_params_path, var.name)):
            print("Var not exists: [%s]\t%s" %
                  (var.name, os.path.join(pretraining_params_path, var.name)))
        #else:
        #    print ("Var exists: [%s]" % (var.name))
        return os.path.exists(os.path.join(pretraining_params_path, var.name))

    fluid.io.load_vars(exe,
                       pretraining_params_path,
                       main_program=main_program,
                       predicate=existed_params)
    log.info(
        "Load pretraining parameters from {}.".format(pretraining_params_path))
