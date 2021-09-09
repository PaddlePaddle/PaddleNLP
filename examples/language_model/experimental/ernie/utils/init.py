#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import six
import ast
import copy

import numpy as np
import paddle.fluid as fluid


def init_pretraining_params(
        exe,
        pretraining_params_path,
        main_program, ):
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    def existed_params(var):
        var_path = os.path.join(pretraining_params_path, var.name)
        var_exists = os.path.exists(var_path)
        if not isinstance(var, fluid.framework.Parameter):
            return False
        else:
            if var_exists:
                print('loading {} for {}'.format(var_path, var.name))
            else:
                print('do not detecting param {} for {}'.format(var.name,
                                                                var_path))
        return var_exists

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=existed_params)
    print("Load pretraining parameters from {}.".format(
        pretraining_params_path))


def init_checkpoint(exe, init_checkpoint_path, main_program):
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        if not fluid.io.is_persistable(var):
            return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables)
    print("Load model from {}".format(init_checkpoint_path))
