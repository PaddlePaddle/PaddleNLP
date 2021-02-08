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
"""
Baidu's open-source Lexical Analysis tool for Chinese, including:
    1. Word Segmentation,
    2. Part-of-Speech Tagging
    3. Named Entity Recognition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import multiprocessing
import sys
from collections import namedtuple

import paddle.fluid as fluid

import creator
import utils
sys.path.append("../shared_modules/")
from models.representation.ernie import ErnieConfig
from models.model_check import check_cuda
from models.model_check import check_version


def evaluate(exe, test_program, test_pyreader, test_ret):
    """
    Evaluation Function
    """
    test_ret["chunk_evaluator"].reset()
    total_loss = []
    start_time = time.time()
    for data in test_pyreader():
        loss, nums_infer, nums_label, nums_correct = exe.run(
            test_program,
            fetch_list=[
                test_ret["avg_cost"],
                test_ret["num_infer_chunks"],
                test_ret["num_label_chunks"],
                test_ret["num_correct_chunks"],
            ],
            feed=data[0])
        total_loss.append(loss)

        test_ret["chunk_evaluator"].update(nums_infer, nums_label, nums_correct)

    precision, recall, f1 = test_ret["chunk_evaluator"].eval()
    end_time = time.time()

    print(
        "\t[test] loss: %.5f, P: %.5f, R: %.5f, F1: %.5f, elapsed time: %.3f s"
        % (np.mean(total_loss), precision, recall, f1, end_time - start_time))


def do_train(args):
    """
    Main Function
    """
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = 1
    else:
        dev_count = min(multiprocessing.cpu_count(), args.cpu_num)
        if (dev_count < args.cpu_num):
            print(
                "WARNING: The total CPU NUM in this machine is %d, which is less than cpu_num parameter you set. "
                "Change the cpu_num from %d to %d" %
                (dev_count, args.cpu_num, dev_count))
        os.environ['CPU_NUM'] = str(dev_count)
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    train_program = fluid.Program()
    with fluid.program_guard(train_program, startup_prog):
        with fluid.unique_name.guard():
            # user defined model based on ernie embeddings
            train_ret = creator.create_ernie_model(args, ernie_config)

            # ernie pyreader
            train_pyreader = creator.create_pyreader(
                args,
                file_name=args.train_data,
                feed_list=train_ret['feed_list'],
                model="ernie",
                place=place)

            test_program = train_program.clone(for_test=True)
            test_pyreader = creator.create_pyreader(
                args,
                file_name=args.test_data,
                feed_list=train_ret['feed_list'],
                model="ernie",
                place=place)
            
            clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0)
            optimizer = fluid.optimizer.Adam(
                learning_rate=args.base_learning_rate, 
                grad_clip=clip)
            optimizer.minimize(train_ret["avg_cost"])

    lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
        program=train_program, batch_size=args.batch_size)
    print("Theoretical memory usage in training: %.3f - %.3f %s" %
          (lower_mem, upper_mem, unit))
    print("Device count: %d" % dev_count)

    exe.run(startup_prog)
    # load checkpoints
    if args.init_checkpoint and args.init_pretraining_params:
        print("WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
              "both are set! Only arg 'init_checkpoint' is made valid.")
    if args.init_checkpoint:
        utils.init_checkpoint(exe, args.init_checkpoint, startup_prog)
    elif args.init_pretraining_params:
        utils.init_pretraining_params(exe, args.init_pretraining_params,
                                      startup_prog)

    if dev_count > 1 and not args.use_cuda:
        device = "GPU" if args.use_cuda else "CPU"
        print("%d %s are used to train model" % (dev_count, device))

        # multi cpu/gpu config
        exec_strategy = fluid.ExecutionStrategy()
        build_strategy = fluid.BuildStrategy()
        compiled_prog = fluid.compiler.CompiledProgram(
            train_program).with_data_parallel(
                loss_name=train_ret['avg_cost'].name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)
    else:
        compiled_prog = fluid.compiler.CompiledProgram(train_program)

    # start training
    steps = 0
    for epoch_id in range(args.epoch):
        for data in train_pyreader():
            steps += 1
            if steps % args.print_steps == 0:
                fetch_list = [
                    train_ret["avg_cost"],
                    train_ret["precision"],
                    train_ret["recall"],
                    train_ret["f1_score"],
                ]
            else:
                fetch_list = []

            start_time = time.time()

            outputs = exe.run(program=compiled_prog,
                              feed=data[0],
                              fetch_list=fetch_list)
            end_time = time.time()
            if steps % args.print_steps == 0:
                loss, precision, recall, f1_score = [
                    np.mean(x) for x in outputs
                ]
                print(
                    "[train] batch_id = %d, loss = %.5f, P: %.5f, R: %.5f, F1: %.5f, elapsed time %.5f, "
                    "pyreader queue_size: %d " %
                    (steps, loss, precision, recall, f1_score,
                     end_time - start_time, train_pyreader.queue.size()))

            if steps % args.save_steps == 0:
                save_path = os.path.join(args.model_save_dir,
                                         "step_" + str(steps), "checkpoint")
                print("\tsaving model as %s" % (save_path))
                fluid.save(train_program, save_path)

            if steps % args.validation_steps == 0:
                evaluate(exe, test_program, test_pyreader, train_ret)

    save_path = os.path.join(args.model_save_dir, "step_" + str(steps),
                             "checkpoint")
    fluid.save(train_program, save_path)


def do_eval(args):
    # init executor
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()

    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()
    test_program = fluid.Program()
    with fluid.program_guard(test_program, fluid.default_startup_program()):
        with fluid.unique_name.guard():
            test_ret = creator.create_ernie_model(args, ernie_config)
    test_program = test_program.clone(for_test=True)

    pyreader = creator.create_pyreader(
        args,
        file_name=args.test_data,
        feed_list=test_ret['feed_list'],
        model="ernie",
        place=place,
        mode='test', )

    print('program startup')

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    print('program loading')
    # load model
    if not args.init_checkpoint:
        raise ValueError(
            "args 'init_checkpoint' should be set if only doing test or infer!")
    utils.init_checkpoint(exe, args.init_checkpoint, test_program)

    evaluate(exe, test_program, pyreader, test_ret)


def do_infer(args):
    # init executor
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()

    # define network and reader
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()
    infer_program = fluid.Program()
    with fluid.program_guard(infer_program, fluid.default_startup_program()):
        with fluid.unique_name.guard():
            infer_ret = creator.create_ernie_model(args, ernie_config)
    infer_program = infer_program.clone(for_test=True)
    print(args.test_data)
    pyreader, reader = creator.create_pyreader(
        args,
        file_name=args.test_data,
        feed_list=infer_ret['feed_list'],
        model="ernie",
        place=place,
        return_reader=True,
        mode='test')

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load model
    if not args.init_checkpoint:
        raise ValueError(
            "args 'init_checkpoint' should be set if only doing test or infer!")
    utils.init_checkpoint(exe, args.init_checkpoint, infer_program)

    # create dict
    id2word_dict = dict(
        [(str(word_id), word) for word, word_id in reader.vocab.items()])
    id2label_dict = dict([(str(label_id), label)
                          for label, label_id in reader.label_map.items()])
    Dataset = namedtuple("Dataset", ["id2word_dict", "id2label_dict"])
    dataset = Dataset(id2word_dict, id2label_dict)

    # make prediction
    for data in pyreader():
        (words, crf_decode, seq_lens) = exe.run(infer_program,
                                                fetch_list=[
                                                    infer_ret["words"],
                                                    infer_ret["crf_decode"],
                                                    infer_ret["seq_lens"]
                                                ],
                                                feed=data[0],
                                                return_numpy=True)
        # User should notice that words had been clipped if long than args.max_seq_len
        results = utils.parse_padding_result(words, crf_decode, seq_lens,
                                             dataset)
        for sent, tags in results:
            result_list = [
                '(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tags)
            ]
            print(''.join(result_list))


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    parser = argparse.ArgumentParser(__doc__)
    utils.load_yaml(parser, './conf/ernie_args.yaml')
    args = parser.parse_args()
    check_cuda(args.use_cuda)
    check_version()
    utils.print_arguments(args)

    if args.mode == 'train':
        do_train(args)
    elif args.mode == 'eval':
        do_eval(args)
    elif args.mode == 'infer':
        do_infer(args)
    else:
        print("Usage: %s --mode train|eval|infer " % sys.argv[0])
