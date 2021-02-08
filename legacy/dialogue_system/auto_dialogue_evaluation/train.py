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
"""train auto dialogue evaluation task"""
import io
import os
import sys
import six
import time
import numpy as np

import paddle
import paddle.fluid as fluid

import ade.reader as reader
from ade_net import create_net, set_word_embedding

from ade.utils.configure import PDConfig
from ade.utils.input_field import InputField
from ade.utils.model_check import check_cuda

try:
    import cPickle as pickle  #python 2
except ImportError as e:
    import pickle  #python 3


def do_train(args):
    """train function"""

    train_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(train_prog, startup_prog):
        train_prog.random_seed = args.random_seed
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

            loss = create_net(
                is_training=True, model_input=input_field, args=args)

            # gradient clipping
            optimizer = fluid.optimizer.AdamOptimizer(
                learning_rate=args.learning_rate,
                grad_clip=fluid.clip.GradientClipByValue(
                    max=1.0, min=-1.0))
            optimizer.minimize(loss)

            if args.use_cuda:
                places = fluid.cuda_places()
                dev_count = len(places)
                place = fluid.CUDAPlace(
                    int(os.getenv('FLAGS_selected_gpus', '0')))
            else:
                places = fluid.cpu_places()
                dev_count = len(places)
                place = fluid.CPUPlace()

            processor = reader.DataProcessor(
                data_path=args.training_file,
                max_seq_length=args.max_seq_len,
                batch_size=args.batch_size)

            batch_generator = processor.data_generator(
                place=place,
                phase="train",
                shuffle=True,
                sample_pro=args.sample_pro)

            num_train_examples = processor.get_num_examples(phase='train')
            max_train_steps = args.epoch * num_train_examples // dev_count // args.batch_size

            print("Num train examples: %d" % num_train_examples)
            print("Max train steps: %d" % max_train_steps)

    data_reader.set_batch_generator(batch_generator, places=place)

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    assert (args.init_from_params == "") or (
        args.init_from_pretrain_model == "")

    #init from some checkpoint, to resume the previous training
    if args.init_from_params:
        fluid.load(train_prog, args.init_from_params, exe)
    #init from some pretrain models, to better solve the current task
    if args.init_from_pretrain_model:
        fluid.load(train_prog, args.init_from_pretrain_model, exe)

    if args.word_emb_init:
        print("start loading word embedding init ...")
        if six.PY2:
            word_emb = np.array(
                pickle.load(io.open(args.word_emb_init, 'rb'))).astype(
                    'float32')
        else:
            word_emb = np.array(
                pickle.load(
                    io.open(args.word_emb_init, 'rb'),
                    encoding="bytes")).astype('float32')
        set_word_embedding(word_emb, place)
        print("finish init word embedding  ...")

    build_strategy = fluid.compiler.BuildStrategy()
    build_strategy.enable_inplace = True

    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=loss.name, build_strategy=build_strategy)

    steps = 0
    begin_time = time.time()
    time_begin = time.time()

    for epoch_step in range(args.epoch):
        data_reader.start()
        sum_loss = 0.0
        ce_loss = 0.0
        while True:
            try:
                fetch_list = [loss.name]
                outputs = exe.run(compiled_train_prog, fetch_list=fetch_list)
                np_loss = outputs
                sum_loss += np.array(np_loss).mean()
                ce_loss = np.array(np_loss).mean()

                if steps % args.print_steps == 0:
                    time_end = time.time()
                    used_time = time_end - time_begin
                    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(time.time()))
                    print(
                        '%s epoch: %d, step: %s, avg loss %s, speed: %f steps/s'
                        % (current_time, epoch_step, steps, sum_loss /
                           args.print_steps, args.print_steps / used_time))
                    sum_loss = 0.0
                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    model_path = os.path.join(args.save_model_path,
                                              "step_" + str(steps))
                    fluid.save(train_prog, model_path)

                steps += 1
            except fluid.core.EOFException:
                data_reader.reset()
                break

    model_path = os.path.join(args.save_model_path, "step_final")
    fluid.save(train_prog, model_path)

    def get_cards():
        num = 0
        cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cards != '':
            num = len(cards.split(","))
        return num

    if args.enable_ce:
        card_num = get_cards()
        pass_time_cost = time.time() - begin_time
        print("test_card_num", card_num)
        print("kpis\ttrain_duration_card%s\t%s" % (card_num, pass_time_cost))
        print("kpis\ttrain_loss_card%s\t%f" % (card_num, ce_loss))


if __name__ == '__main__':
    import paddle
    paddle.enable_static()

    args = PDConfig(yaml_file="./data/config/ade.yaml")
    args.build()
    args.Print()

    check_cuda(args.use_cuda)

    do_train(args)
