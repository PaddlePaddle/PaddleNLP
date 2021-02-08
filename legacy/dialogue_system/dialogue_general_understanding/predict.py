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

import io
import os
import sys
import numpy as np
import argparse
import collections
import paddle.fluid as fluid

import dgu.reader as reader
from dgu_net import create_net
import dgu.define_paradigm as define_paradigm
import dgu.define_predict_pack as define_predict_pack

from dgu.utils.configure import PDConfig
from dgu.utils.input_field import InputField
from dgu.utils.model_check import check_cuda
from dgu.utils.py23 import tab_tok, rt_tok


def do_predict(args):
    """predict function"""

    task_name = args.task_name.lower()
    paradigm_inst = define_paradigm.Paradigm(task_name)
    pred_inst = define_predict_pack.DefinePredict()
    pred_func = getattr(pred_inst, pred_inst.task_map[task_name])

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
                shape=[-1, args.max_seq_len, 1],
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
            data_reader = fluid.io.DataLoader.from_generator(
                feed_list=input_inst, capacity=4, iterable=False)

            results = create_net(
                is_training=False,
                model_input=input_field,
                num_labels=num_labels,
                paradigm_inst=paradigm_inst,
                args=args)

            probs = results.get("probs", None)
            fetch_list = [probs.name]

    #for_test is True if change the is_test attribute of operators to True
    test_prog = test_prog.clone(for_test=True)

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    assert (args.init_from_params)

    if args.init_from_params:
        fluid.load(test_prog, args.init_from_params)

    compiled_test_prog = fluid.CompiledProgram(test_prog)

    processor = processors[task_name](data_dir=args.data_dir,
                                      vocab_path=args.vocab_path,
                                      max_seq_len=args.max_seq_len,
                                      do_lower_case=args.do_lower_case,
                                      in_tokens=args.in_tokens,
                                      task_name=task_name,
                                      random_seed=args.random_seed)
    batch_generator = processor.data_generator(
        batch_size=args.batch_size, phase='test', shuffle=False)

    data_reader.set_batch_generator(batch_generator, places=place)
    data_reader.start()

    all_results = []
    while True:
        try:
            results = exe.run(compiled_test_prog, fetch_list=fetch_list)
            all_results.extend(results[0])
        except fluid.core.EOFException:
            data_reader.reset()
            break

    np.set_printoptions(precision=4, suppress=True)
    print("Write the predicted results into the output_prediction_file")

    fw = io.open(args.output_prediction_file, 'w', encoding="utf8")
    if task_name not in ['atis_slot']:
        for index, result in enumerate(all_results):
            tags = pred_func(result)
            fw.write("%s%s%s%s" % (index, tab_tok, tags, rt_tok))
    else:
        tags = pred_func(all_results, args.max_seq_len)
        for index, tag in enumerate(tags):
            fw.write("%s%s%s%s" % (index, tab_tok, tag, rt_tok))


if __name__ == "__main__":
    import paddle
    paddle.enable_static()

    args = PDConfig(yaml_file="./data/config/dgu.yaml")
    args.build()
    args.Print()

    check_cuda(args.use_cuda)

    do_predict(args)
