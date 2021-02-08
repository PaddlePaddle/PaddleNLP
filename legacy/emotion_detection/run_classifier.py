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
Emotion Detection Task
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing
import sys
sys.path.append("../shared_modules/")

import paddle
import paddle.fluid as fluid
import numpy as np

from models.classification import nets
from models.model_check import check_cuda
from models.model_check import check_version
from config import PDConfig
import reader
import utils


def create_model(args, num_labels, is_prediction=False):
    """
    Create Model for Emotion Detection
    """
    data = fluid.data(name="words", shape=[-1, args.max_seq_len], dtype="int64")
    label = fluid.data(name="label", shape=[-1, 1], dtype="int64")
    seq_len = fluid.data(name="seq_len", shape=[-1], dtype="int64")

    if is_prediction:
        loader = fluid.io.DataLoader.from_generator(
            feed_list=[data, seq_len],
            capacity=16,
            iterable=False,
            return_list=False)
    else:
        loader = fluid.io.DataLoader.from_generator(
            feed_list=[data, label, seq_len],
            capacity=16,
            iterable=False,
            return_list=False)

    if args.model_type == "cnn_net":
        network = nets.cnn_net
    elif args.model_type == "bow_net":
        network = nets.bow_net
    elif args.model_type == "lstm_net":
        network = nets.lstm_net
    elif args.model_type == "bilstm_net":
        network = nets.bilstm_net
    elif args.model_type == "gru_net":
        network = nets.gru_net
    elif args.model_type == "textcnn_net":
        network = nets.textcnn_net
    else:
        raise ValueError("Unknown network type!")

    if is_prediction:
        probs = network(
            data,
            seq_len,
            None,
            args.vocab_size,
            class_dim=num_labels,
            is_prediction=True)
        return loader, probs, [data.name, seq_len.name]

    avg_loss, probs = network(
        data, seq_len, label, args.vocab_size, class_dim=num_labels)
    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=label, total=num_seqs)
    return loader, avg_loss, accuracy, num_seqs


def evaluate(exe, test_program, test_loader, fetch_list, eval_phase):
    """
    Evaluation Function
    """
    test_loader.start()
    total_cost, total_acc, total_num_seqs = [], [], []
    time_begin = time.time()
    while True:
        try:
            np_loss, np_acc, np_num_seqs = exe.run(program=test_program,
                                                   fetch_list=fetch_list,
                                                   return_numpy=False)
            np_loss = np.array(np_loss)
            np_acc = np.array(np_acc)
            np_num_seqs = np.array(np_num_seqs)
            total_cost.extend(np_loss * np_num_seqs)
            total_acc.extend(np_acc * np_num_seqs)
            total_num_seqs.extend(np_num_seqs)
        except fluid.core.EOFException:
            test_loader.reset()
            break
    time_end = time.time()
    print("[%s evaluation] avg loss: %f, avg acc: %f, elapsed time: %f s" %
          (eval_phase, np.sum(total_cost) / np.sum(total_num_seqs),
           np.sum(total_acc) / np.sum(total_num_seqs), time_end - time_begin))


def infer(exe, infer_program, infer_loader, fetch_list, infer_phase):
    infer_loader.start()
    time_begin = time.time()
    while True:
        try:
            batch_probs = exe.run(program=infer_program,
                                  fetch_list=fetch_list,
                                  return_numpy=True)
            for probs in batch_probs[0]:
                print("%d\t%f\t%f\t%f" %
                      (np.argmax(probs), probs[0], probs[1], probs[2]))
        except fluid.core.EOFException as e:
            infer_loader.reset()
            break
    time_end = time.time()
    print("[%s] elapsed time: %f s" % (infer_phase, time_end - time_begin))


def main(args):
    """
    Main Function
    """
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    task_name = args.task_name.lower()
    processor = reader.EmoTectProcessor(
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        random_seed=args.random_seed)
    #num_labels = len(processor.get_labels())
    num_labels = args.num_labels

    if not (args.do_train or args.do_val or args.do_infer):
        raise ValueError("For args `do_train`, `do_val` and `do_infer`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train:
        train_data_generator = processor.data_generator(
            batch_size=args.batch_size, phase='train', epoch=args.epoch)

        num_train_examples = processor.get_num_examples(phase="train")
        max_train_steps = args.epoch * num_train_examples // args.batch_size + 1

        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)

        train_program = fluid.Program()
        if args.random_seed is not None:
            train_program.random_seed = args.random_seed

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_loader, loss, accuracy, num_seqs = create_model(
                    args, num_labels=num_labels, is_prediction=False)

                sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=args.lr)
                sgd_optimizer.minimize(loss)

        if args.verbose:
            lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                program=train_program, batch_size=args.batch_size)
            print("Theoretical memory usage in training: %.3f - %.3f %s" %
                  (lower_mem, upper_mem, unit))

    if args.do_val:
        if args.do_train:
            test_data_generator = processor.data_generator(
                batch_size=args.batch_size, phase='dev', epoch=1)
        else:
            test_data_generator = processor.data_generator(
                batch_size=args.batch_size, phase='test', epoch=1)

        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_loader, loss, accuracy, num_seqs = create_model(
                    args, num_labels=num_labels, is_prediction=False)
        test_prog = test_prog.clone(for_test=True)

    if args.do_infer:
        infer_data_generator = processor.data_generator(
            batch_size=args.batch_size, phase='infer', epoch=1)

        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                infer_loader, probs, _ = create_model(
                    args, num_labels=num_labels, is_prediction=True)
        test_prog = test_prog.clone(for_test=True)

    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint:
            utils.init_checkpoint(
                exe, args.init_checkpoint, main_program=startup_prog)
    elif args.do_val or args.do_infer:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or infer!")
        utils.init_checkpoint(exe, args.init_checkpoint, main_program=test_prog)

    if args.do_train:
        train_exe = exe
        train_loader.set_sample_list_generator(train_data_generator)
    else:
        train_exe = None
    if args.do_val:
        test_exe = exe
        test_loader.set_sample_list_generator(test_data_generator)
    if args.do_infer:
        test_exe = exe
        infer_loader.set_sample_list_generator(infer_data_generator)

    if args.do_train:
        train_loader.start()
        steps = 0
        total_cost, total_acc, total_num_seqs = [], [], []
        time_begin = time.time()
        ce_info = []
        while True:
            try:
                steps += 1
                if steps % args.skip_steps == 0:
                    fetch_list = [loss.name, accuracy.name, num_seqs.name]
                else:
                    fetch_list = []

                outputs = train_exe.run(program=train_program,
                                        fetch_list=fetch_list,
                                        return_numpy=False)
                if steps % args.skip_steps == 0:
                    np_loss, np_acc, np_num_seqs = outputs
                    np_loss = np.array(np_loss)
                    np_acc = np.array(np_acc)
                    np_num_seqs = np.array(np_num_seqs)
                    total_cost.extend(np_loss * np_num_seqs)
                    total_acc.extend(np_acc * np_num_seqs)
                    total_num_seqs.extend(np_num_seqs)

                    if args.verbose:
                        verbose = "train loader queue size: %d, " % train_loader.queue.size(
                        )
                        print(verbose)

                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("step: %d, avg loss: %f, "
                          "avg acc: %f, speed: %f steps/s" %
                          (steps, np.sum(total_cost) / np.sum(total_num_seqs),
                           np.sum(total_acc) / np.sum(total_num_seqs),
                           args.skip_steps / used_time))
                    ce_info.append([
                        np.sum(total_cost) / np.sum(total_num_seqs),
                        np.sum(total_acc) / np.sum(total_num_seqs), used_time
                    ])
                    total_cost, total_acc, total_num_seqs = [], [], []
                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.save_checkpoint_dir,
                                             "step_" + str(steps))
                    fluid.save(train_program, save_path)

                if steps % args.validation_steps == 0:
                    # evaluate on dev set
                    if args.do_val:
                        evaluate(test_exe, test_prog, test_loader,
                                 [loss.name, accuracy.name, num_seqs.name],
                                 "dev")

            except fluid.core.EOFException:
                print("final step: %d " % steps)
                if args.do_val:
                    evaluate(test_exe, test_prog, test_loader,
                             [loss.name, accuracy.name, num_seqs.name], "dev")

                save_path = os.path.join(args.save_checkpoint_dir,
                                         "step_" + str(steps))
                fluid.save(train_program, save_path)
                train_loader.reset()
                break

    if args.do_train and args.enable_ce:
        card_num = get_cards()
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

    # evaluate on test set
    if not args.do_train and args.do_val:
        print("Final test result:")
        evaluate(test_exe, test_prog, test_loader,
                 [loss.name, accuracy.name, num_seqs.name], "test")

    # infer
    if args.do_infer:
        print("Final infer result:")
        infer(test_exe, test_prog, infer_loader, [probs.name], "infer")


def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = PDConfig('config.json')
    args.build()
    args.print_arguments()
    check_cuda(args.use_cuda)
    check_version()
    main(args)
