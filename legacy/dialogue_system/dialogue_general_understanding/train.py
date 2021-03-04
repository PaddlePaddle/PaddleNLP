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
"""Finetuning on dialogue tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np

import paddle.fluid as fluid

from dgu_net import create_net
import dgu.reader as reader
from dgu.optimization import optimization
import dgu.define_paradigm as define_paradigm
from dgu.utils.configure import PDConfig
from dgu.utils.input_field import InputField
from dgu.utils.model_check import check_cuda


def do_train(args):
    """train function"""

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

    train_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(train_prog, startup_prog):
        train_prog.random_seed = args.random_seed
        startup_prog.random_seed = args.random_seed
        with fluid.unique_name.guard():
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
            elif args.task_name in ['dstc2']:
                labels = fluid.data(
                    name='labels', shape=[-1, num_labels], dtype='int64')
            else:
                labels = fluid.data(name='labels', shape=[-1, 1], dtype='int64')

            input_inst = [src_ids, pos_ids, sent_ids, input_mask, labels]
            input_field = InputField(input_inst)

            data_reader = fluid.io.DataLoader.from_generator(
                feed_list=input_inst, capacity=4, iterable=False)

            processor = processors[task_name](data_dir=args.data_dir,
                                              vocab_path=args.vocab_path,
                                              max_seq_len=args.max_seq_len,
                                              do_lower_case=args.do_lower_case,
                                              in_tokens=args.in_tokens,
                                              task_name=task_name,
                                              random_seed=args.random_seed)

            results = create_net(
                is_training=True,
                model_input=input_field,
                num_labels=num_labels,
                paradigm_inst=paradigm_inst,
                args=args)

            loss = results.get("loss", None)
            probs = results.get("probs", None)
            accuracy = results.get("accuracy", None)
            num_seqs = results.get("num_seqs", None)

            places = fluid.cuda_places() if args.use_cuda else fluid.cpu_places(
            )
            dev_count = len(places)

            batch_generator = processor.data_generator(
                batch_size=args.batch_size, phase='train', shuffle=True)
            num_train_examples = processor.get_num_examples(phase='train')

            if args.in_tokens:
                max_train_steps = args.epoch * num_train_examples // (
                    args.batch_size // args.max_seq_len) // dev_count
            else:
                max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

            warmup_steps = int(max_train_steps * args.warmup_proportion)
            print("Num train examples: %d" % num_train_examples)
            print("Max train steps: %d" % max_train_steps)
            print("Num warmup steps: %d" % warmup_steps)

            optimizor = optimization(
                loss=loss,
                warmup_steps=warmup_steps,
                num_train_steps=max_train_steps,
                learning_rate=args.learning_rate,
                train_program=train_prog,
                startup_prog=startup_prog,
                weight_decay=args.weight_decay,
                scheduler=args.lr_scheduler,
                use_fp16=False,
                loss_scaling=args.loss_scaling)

    data_reader.set_batch_generator(batch_generator, places=places)

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    assert args.init_from_params or args.init_from_pretrain_model

    # init from some checkpoint, to resume the previous training
    if args.init_from_params:
        fluid.load(train_prog, args.init_from_params, exe)
    if args.init_from_pretrain_model:
        fluid.load(train_prog, args.init_from_pretrain_model, exe)

    build_strategy = fluid.compiler.BuildStrategy()
    build_strategy.enable_inplace = True

    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=loss.name, build_strategy=build_strategy)

    # start training
    steps = 0
    time_begin = time.time()
    ce_info = []
    for epoch_step in range(args.epoch):
        data_reader.start()
        while True:
            try:
                steps += 1
                if steps % args.print_steps == 0:
                    if warmup_steps <= 0:
                        if accuracy is not None:
                            fetch_list = [
                                loss.name, accuracy.name, num_seqs.name
                            ]
                        else:
                            fetch_list = [loss.name, num_seqs.name]
                    else:
                        if accuracy is not None:
                            fetch_list = [
                                loss.name, accuracy.name, optimizor.name,
                                num_seqs.name
                            ]
                        else:
                            fetch_list = [
                                loss.name, optimizor.name, num_seqs.name
                            ]
                else:
                    fetch_list = []

                outputs = exe.run(compiled_train_prog, fetch_list=fetch_list)

                if steps % args.print_steps == 0:
                    if warmup_steps <= 0:
                        if accuracy is not None:
                            np_loss, np_acc, np_num_seqs = outputs
                        else:
                            np_loss, np_num_seqs = outputs
                    else:
                        if accuracy is not None:
                            np_loss, np_acc, np_lr, np_num_seqs = outputs
                        else:
                            np_loss, np_lr, np_num_seqs = outputs

                    time_end = time.time()
                    used_time = time_end - time_begin
                    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(time.time()))
                    if accuracy is not None:
                        print("%s epoch: %d, step: %d, ave loss: %f, "
                              "ave acc: %f, speed: %f steps/s" %
                              (current_time, epoch_step, steps,
                               np.mean(np_loss), np.mean(np_acc),
                               args.print_steps / used_time))
                        ce_info.append([
                            np.mean(np_loss), np.mean(np_acc),
                            args.print_steps / used_time
                        ])
                    else:
                        print("%s epoch: %d, step: %d, ave loss: %f, "
                              "speed: %f steps/s" %
                              (current_time, epoch_step, steps,
                               np.mean(np_loss), args.print_steps / used_time))
                        ce_info.append(
                            [np.mean(np_loss), args.print_steps / used_time])
                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    model_path = os.path.join(args.save_model_path,
                                              "step_" + str(steps))
                    fluid.save(train_prog, model_path)

            except fluid.core.EOFException:
                data_reader.reset()
                break

    model_path = os.path.join(args.save_model_path, "step_final")
    fluid.save(train_prog, model_path)

    def get_cards():
        num = 0
        cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        print("test_cards", cards)
        if cards != '':
            num = len(cards.split(","))
        return num

    if args.enable_ce:
        card_num = get_cards()
        print("test_card_num", card_num)
        ce_loss = 0
        ce_acc = 0
        ce_time = 0
        try:
            ce_loss = ce_info[-2][0]
            ce_acc = ce_info[-2][1]
            ce_time = ce_info[-2][2]
        except:
            print("ce info error")
        print("kpis\teach_step_duration_%s_card%s\t%s" %
              (task_name, card_num, ce_time))
        print("kpis\ttrain_loss_%s_card%s\t%f" % (task_name, card_num, ce_loss))
        print("kpis\ttrain_acc_%s_card%s\t%f" % (task_name, card_num, ce_acc))


if __name__ == '__main__':
    import paddle
    paddle.enable_static()

    args = PDConfig(yaml_file="./data/config/dgu.yaml")
    args.build()
    args.Print()

    check_cuda(args.use_cuda)

    do_train(args)
