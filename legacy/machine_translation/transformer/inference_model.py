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

import logging
import os
import six
import sys
import time

import numpy as np
import paddle
import paddle.fluid as fluid

from utils.input_field import InputField
from utils.configure import PDConfig
from utils.load import load

# include task-specific libs
import desc
import reader
from transformer import create_net


def do_save_inference_model(args):
    if args.use_cuda:
        dev_count = fluid.core.get_cuda_device_count()
        place = fluid.CUDAPlace(0)
    else:
        dev_count = int(os.environ.get('CPU_NUM', 1))
        place = fluid.CPUPlace()

    src_vocab = reader.DataProcessor.load_dict(args.src_vocab_fpath)
    trg_vocab = reader.DataProcessor.load_dict(args.trg_vocab_fpath)
    args.src_vocab_size = len(src_vocab)
    args.trg_vocab_size = len(trg_vocab)

    test_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():

            # define input and reader

            input_field_names = desc.encoder_data_input_fields + desc.fast_decoder_data_input_fields
            input_descs = desc.get_input_descs(args.args)
            input_slots = [{
                "name": name,
                "shape": input_descs[name][0],
                "dtype": input_descs[name][1]
            } for name in input_field_names]

            input_field = InputField(input_slots)
            input_field.build(build_pyreader=True)

            # define the network

            predictions = create_net(
                is_training=False, model_input=input_field, args=args)
            out_ids, out_scores = predictions

    # This is used here to set dropout to the test mode.
    test_prog = test_prog.clone(for_test=True)

    # prepare predicting

    ## define the executor and program for training

    exe = fluid.Executor(place)

    exe.run(startup_prog)
    assert (
        args.init_from_params), "must set init_from_params to load parameters"
    load(test_prog, os.path.join(args.init_from_params, "transformer"), exe)
    print("finish initing model from params from %s" % (args.init_from_params))

    # saving inference model

    fluid.io.save_inference_model(
        args.inference_model_dir,
        feeded_var_names=list(input_field_names),
        target_vars=[out_ids, out_scores],
        executor=exe,
        main_program=test_prog,
        model_filename="model.pdmodel",
        params_filename="params.pdparams")

    print("save inference model at %s" % (args.inference_model_dir))


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()

    do_save_inference_model(args)
