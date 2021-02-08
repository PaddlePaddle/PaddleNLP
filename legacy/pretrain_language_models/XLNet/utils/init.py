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

from __future__ import print_function

import os
import six
import ast
import copy

import numpy as np
import paddle.fluid as fluid


def cast_fp32_to_fp16(exe, main_program):
    print("Cast parameters to float16 data format.")
    for param in main_program.global_block().all_parameters():
        if not param.name.endswith(".master"):
            param_t = fluid.global_scope().find_var(param.name).get_tensor()
            data = np.array(param_t)
            if param.name.find("layer_norm") == -1 and param.name.find(
                    "embedding") == -1:
                print("shkip params", param.name)
                param_t.set(np.float16(data), exe.place)
            master_param_var = fluid.global_scope().find_var(param.name +
                                                             ".master")
            if master_param_var is not None:
                master_param_var.get_tensor().set(data, exe.place)


def init_checkpoint(exe, init_checkpoint_path, main_program, use_fp16=False):
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path
    print("Load checkpoint from {}".format(init_checkpoint_path))

    def existed_persitables(var):
        if not fluid.io.is_persistable(var):
            return False
        if os.path.exists(os.path.join(init_checkpoint_path, var.name)):
            print("INIT %s" % var.name)
            return True
        else:
            print("SKIP %s" % var.name)
            return False

    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables)

    if use_fp16:
        cast_fp32_to_fp16(exe, main_program)


def init_pretraining_params(exe,
                            pretraining_params_path,
                            main_program,
                            use_fp16=False):
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path
    print("Load pretraining parameters from {}.".format(
        pretraining_params_path))

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        if os.path.exists(os.path.join(pretraining_params_path, var.name)):
            print("INIT %s" % var.name)
            return True
        else:
            print("SKIP %s" % var.name)
            return False

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=existed_params)

    if use_fp16:
        cast_fp32_to_fp16(exe, main_program)
