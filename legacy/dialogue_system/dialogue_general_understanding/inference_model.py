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
"""save inference model"""

import os
import sys
import argparse
import collections
import numpy as np

import paddle
import paddle.fluid as fluid

from dgu.utils.configure import PDConfig
from dgu.utils.input_field import InputField
from dgu.utils.model_check import check_cuda

import dgu.reader as reader
from dgu_net import create_net
import dgu.define_paradigm as define_paradigm


def do_save_inference_model(args):
    """save inference model function"""

    task_name = args.task_name.lower()
    paradigm_inst = define_paradigm.Paradigm(task_name)

    processors = {
        'udc': reader.UDCProcessor,
        'swda': reader.SWDAProcessor,
        'mrda': reader.MRDAProcessor,
        'atis_slot': reader.ATISSlotProcessor,
        'atis_intent': reader.ATISIntentProcessor,
        'dstc2': reader.DSTC2Processor,
    }

    test_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(test_prog, startup_prog):
        test_prog.random_seed = args.random_seed
        startup_prog.random_seed = args.random_seed

        with fluid.unique_name.guard():

            # define inputs of the network
            num_labels = len(processors[task_name].get_labels())

            src_ids = fluid.data(
                name='src_ids', shape=[-1, args.max_seq_len], dtype='int64')
            pos_ids = fluid.data(
                name='pos_ids', shape=[-1, args.max_seq_len], dtype='int64')
            sent_ids = fluid.data(
                name='sent_ids', shape=[-1, args.max_seq_len], dtype='int64')
            input_mask = fluid.data(
                name='input_mask',
                shape=[-1, args.max_seq_len],
                dtype='float32')
            if args.task_name == 'atis_slot':
                labels = fluid.data(
                    name='labels', shape=[-1, args.max_seq_len], dtype='int64')
            elif args.task_name in ['dstc2', 'dstc2_asr', 'multi-woz']:
                labels = fluid.data(
                    name='labels', shape=[-1, num_labels], dtype='int64')
            else:
                labels = fluid.data(name='labels', shape=[-1, 1], dtype='int64')

            input_inst = [src_ids, pos_ids, sent_ids, input_mask, labels]
            input_field = InputField(input_inst)

            results = create_net(
                is_training=False,
                model_input=input_field,
                num_labels=num_labels,
                paradigm_inst=paradigm_inst,
                args=args)
            probs = results.get("probs", None)

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    assert (args.init_from_params)

    if args.init_from_params:
        fluid.load(test_prog, args.init_from_params)

    # saving inference model
    fluid.io.save_inference_model(
        args.inference_model_dir,
        feeded_var_names=[
            input_field.src_ids.name, input_field.pos_ids.name,
            input_field.sent_ids.name, input_field.input_mask.name
        ],
        target_vars=[probs],
        executor=exe,
        main_program=test_prog,
        model_filename="model.pdmodel",
        params_filename="params.pdparams")

    print("save inference model at %s" % (args.inference_model_dir))


if __name__ == "__main__":
    import paddle
    paddle.enable_static()

    args = PDConfig(yaml_file="./data/config/dgu.yaml")
    args.build()

    check_cuda(args.use_cuda)

    do_save_inference_model(args)
