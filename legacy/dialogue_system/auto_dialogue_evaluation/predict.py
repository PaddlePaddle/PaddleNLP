# -*- coding: utf-8 -*-
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
"""predict auto dialogue evaluation task"""
import io
import os
import sys
import six
import time
import numpy as np

import paddle
import paddle.fluid as fluid

import ade.reader as reader
from ade_net import create_net

from ade.utils.configure import PDConfig
from ade.utils.input_field import InputField
from ade.utils.model_check import check_cuda


def do_predict(args):
    """
    predict function
    """
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

            fetch_list = [logits.name]
    #for_test is True if change the is_test attribute of operators to True
    test_prog = test_prog.clone(for_test=True)
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    assert (args.init_from_params) or (args.init_from_pretrain_model)
    if args.init_from_params:
        fluid.load(test_prog, args.init_from_params, executor=exe)
    if args.init_from_pretrain_model:
        fluid.load(test_prog, args.init_from_pretrain_model, executor=exe)

    compiled_test_prog = fluid.CompiledProgram(test_prog)

    processor = reader.DataProcessor(
        data_path=args.predict_file,
        max_seq_length=args.max_seq_len,
        batch_size=args.batch_size)

    batch_generator = processor.data_generator(
        place=place, phase="test", shuffle=False, sample_pro=1)
    num_test_examples = processor.get_num_examples(phase='test')

    data_reader.set_batch_generator(batch_generator, places=place)
    data_reader.start()

    scores = []
    while True:
        try:
            results = exe.run(compiled_test_prog, fetch_list=fetch_list)
            scores.extend(results[0])
        except fluid.core.EOFException:
            data_reader.reset()
            break

    scores = scores[:num_test_examples]
    print("Write the predicted results into the output_prediction_file")
    fw = io.open(args.output_prediction_file, 'w', encoding="utf8")
    for index, score in enumerate(scores):
        fw.write(u"%s\t%s\n" % (index, score[0]))
    print("finish........................................")


if __name__ == "__main__":
    import paddle
    paddle.enable_static()

    args = PDConfig(yaml_file="./data/config/ade.yaml")
    args.build()
    args.Print()

    check_cuda(args.use_cuda)

    do_predict(args)
