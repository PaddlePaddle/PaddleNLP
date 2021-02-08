# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.                                                                                                      
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
"""save inference model for auto dialogue evaluation"""

import os
import sys
import six
import numpy as np
import time
import paddle
import paddle.fluid as fluid

import ade.reader as reader
from ade_net import create_net

from ade.utils.configure import PDConfig
from ade.utils.input_field import InputField
from ade.utils.model_check import check_cuda


def do_save_inference_model(args):

    test_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(test_prog, startup_prog):
        test_prog.random_seed = args.random_seed
        startup_prog.random_seed = args.random_seed

        with fluid.unique_name.guard():

            context_wordseq = fluid.data(
                name='context_wordseq',
                shape=[-1, 1],
                dtype='int64',
                lod_level=1)
            response_wordseq = fluid.data(
                name='response_wordseq',
                shape=[-1, 1],
                dtype='int64',
                lod_level=1)
            labels = fluid.data(name='labels', shape=[-1, 1], dtype='int64')

            input_inst = [context_wordseq, response_wordseq, labels]
            input_field = InputField(input_inst)
            data_reader = fluid.io.DataLoader.from_generator(
                feed_list=input_inst, capacity=4, iterable=False)

            logits = create_net(
                is_training=False, model_input=input_field, args=args)

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    assert (args.init_from_params) or (args.init_from_pretrain_model)

    if args.init_from_params:
        fluid.load(test_prog, args.init_from_params)
    elif args.init_from_pretrain_model:
        fluid.load(test_prog, args.init_from_pretrain_model)

    # saving inference model
    fluid.io.save_inference_model(
        args.inference_model_dir,
        feeded_var_names=[
            input_field.context_wordseq.name,
            input_field.response_wordseq.name,
        ],
        target_vars=[logits, ],
        executor=exe,
        main_program=test_prog,
        model_filename="model.pdmodel",
        params_filename="params.pdparams")

    print("save inference model at %s" % (args.inference_model_dir))


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = PDConfig(yaml_file="./data/config/ade.yaml")
    args.build()

    check_cuda(args.use_cuda)

    do_save_inference_model(args)
