"""
Sentiment Classification Task
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
sys.path.append("../shared_modules/models/classification/")
sys.path.append("../shared_modules/")

from nets import bow_net
from nets import lstm_net
from nets import cnn_net
from nets import bilstm_net
from nets import gru_net
from models.model_check import check_cuda
from models.model_check import check_version
from config import PDConfig

import paddle
import paddle.fluid as fluid

import reader
from utils import init_checkpoint


def create_model(args, pyreader_name, num_labels, is_prediction=False):
    """
    Create Model for sentiment classification
    """

    data = fluid.data(
        name="src_ids", shape=[None, args.max_seq_len], dtype='int64')
    label = fluid.data(name="label", shape=[None, 1], dtype="int64")
    seq_len = fluid.data(name="seq_len", shape=[None], dtype="int64")

    data_reader = fluid.io.DataLoader.from_generator(
        feed_list=[data, label, seq_len], capacity=4, iterable=False)

    if args.model_type == "bilstm_net":
        network = bilstm_net
    elif args.model_type == "bow_net":
        network = bow_net
    elif args.model_type == "cnn_net":
        network = cnn_net
    elif args.model_type == "lstm_net":
        network = lstm_net
    elif args.model_type == "gru_net":
        network = gru_net
    else:
        raise ValueError("Unknown network type!")

    if is_prediction:
        probs = network(
            data, seq_len, None, args.vocab_size, is_prediction=is_prediction)
        print("create inference model...")
        return data_reader, probs, [data.name, seq_len.name]

    ce_loss, probs = network(
        data, seq_len, label, args.vocab_size, is_prediction=is_prediction)
    loss = fluid.layers.mean(x=ce_loss)
    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=label, total=num_seqs)
    return data_reader, loss, accuracy, num_seqs


def evaluate(exe, test_program, test_pyreader, fetch_list, eval_phase):
    """
    Evaluation Function
    """
    test_pyreader.start()
    total_cost, total_acc, total_num_seqs = [], [], []
    time_begin = time.time()
    while True:
        #print("===============")
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
            test_pyreader.reset()
            break
    time_end = time.time()
    print("[%s evaluation] ave loss: %f, ave acc: %f, elapsed time: %f s" %
          (eval_phase, np.sum(total_cost) / np.sum(total_num_seqs),
           np.sum(total_acc) / np.sum(total_num_seqs), time_end - time_begin))


def inference(exe, test_program, test_pyreader, fetch_list, infer_phrase):
    """
    Inference Function
    """
    test_pyreader.start()
    time_begin = time.time()
    while True:
        try:
            np_props = exe.run(program=test_program,
                               fetch_list=fetch_list,
                               return_numpy=True)
            for probs in np_props[0]:
                print("%d\t%f\t%f" % (np.argmax(probs), probs[0], probs[1]))
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()
    print("[%s] elapsed time: %f s" % (infer_phrase, time_end - time_begin))


def main(args):
    """
    Main Function
    """
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = 1
    exe = fluid.Executor(place)

    task_name = args.task_name.lower()
    processor = reader.SentaProcessor(
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        random_seed=args.random_seed,
        max_seq_len=args.max_seq_len)
    num_labels = len(processor.get_labels())

    if not (args.do_train or args.do_val or args.do_infer):
        raise ValueError("For args `do_train`, `do_val` and `do_infer`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train:
        train_data_generator = processor.data_generator(
            batch_size=args.batch_size / dev_count,
            phase='train',
            epoch=args.epoch,
            shuffle=True)

        num_train_examples = processor.get_num_examples(phase="train")

        max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

        print("Device count: %d" % dev_count)
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)

        train_program = fluid.Program()
        if args.enable_ce and args.random_seed is not None:
            train_program.random_seed = args.random_seed

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_reader, loss, accuracy, num_seqs = create_model(
                    args,
                    pyreader_name='train_reader',
                    num_labels=num_labels,
                    is_prediction=False)

                sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=args.lr)
                sgd_optimizer.minimize(loss)

        if args.verbose:
            lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                program=train_program, batch_size=args.batch_size)
            print("Theoretical memory usage in training: %.3f - %.3f %s" %
                  (lower_mem, upper_mem, unit))

    if args.do_val:
        test_data_generator = processor.data_generator(
            batch_size=args.batch_size / dev_count,
            phase='dev',
            epoch=1,
            shuffle=False)
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_reader, loss, accuracy, num_seqs = create_model(
                    args,
                    pyreader_name='test_reader',
                    num_labels=num_labels,
                    is_prediction=False)

        test_prog = test_prog.clone(for_test=True)

    if args.do_infer:
        infer_data_generator = processor.data_generator(
            batch_size=args.batch_size / dev_count,
            phase='infer',
            epoch=1,
            shuffle=False)
        infer_prog = fluid.Program()
        with fluid.program_guard(infer_prog, startup_prog):
            with fluid.unique_name.guard():
                infer_reader, prop, _ = create_model(
                    args,
                    pyreader_name='infer_reader',
                    num_labels=num_labels,
                    is_prediction=True)
        infer_prog = infer_prog.clone(for_test=True)

    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint:
            init_checkpoint(
                exe, args.init_checkpoint, main_program=startup_prog)

    elif args.do_val or args.do_infer:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or testing!")
        init_checkpoint(exe, args.init_checkpoint, main_program=startup_prog)

    if args.do_train:
        train_exe = exe
        train_reader.set_sample_list_generator(train_data_generator)
    else:
        train_exe = None
    if args.do_val:
        test_exe = exe
        test_reader.set_sample_list_generator(test_data_generator)
    if args.do_infer:
        test_exe = exe
        infer_reader.set_sample_list_generator(infer_data_generator)

    if args.do_train:
        train_reader.start()
        steps = 0
        total_cost, total_acc, total_num_seqs = [], [], []
        time_begin = time.time()
        while True:
            try:
                steps += 1
                #print("steps...")
                if steps % args.skip_steps == 0:
                    fetch_list = [loss.name, accuracy.name, num_seqs.name]
                else:
                    fetch_list = []

                outputs = train_exe.run(program=train_program,
                                        fetch_list=fetch_list,
                                        return_numpy=False)
                #print("finished one step")
                if steps % args.skip_steps == 0:
                    np_loss, np_acc, np_num_seqs = outputs
                    np_loss = np.array(np_loss)
                    np_acc = np.array(np_acc)
                    np_num_seqs = np.array(np_num_seqs)
                    total_cost.extend(np_loss * np_num_seqs)
                    total_acc.extend(np_acc * np_num_seqs)
                    total_num_seqs.extend(np_num_seqs)

                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size(
                        )
                        print(verbose)

                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("step: %d, ave loss: %f, "
                          "ave acc: %f, speed: %f steps/s" %
                          (steps, np.sum(total_cost) / np.sum(total_num_seqs),
                           np.sum(total_acc) / np.sum(total_num_seqs),
                           args.skip_steps / used_time))
                    total_cost, total_acc, total_num_seqs = [], [], []
                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoints,
                                             "step_" + str(steps), "checkpoint")
                    fluid.save(train_program, save_path)

                if steps % args.validation_steps == 0:
                    # evaluate dev set
                    if args.do_val:
                        print("do evalatation")
                        evaluate(exe, test_prog, test_reader,
                                 [loss.name, accuracy.name, num_seqs.name],
                                 "dev")

            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps),
                                         "checkpoint")
                fluid.save(train_program, save_path)
                train_reader.reset()
                break

    # final eval on dev set
    if args.do_val:
        print("Final validation result:")
        evaluate(exe, test_prog, test_reader,
                 [loss.name, accuracy.name, num_seqs.name], "dev")

    # final eval on test set
    if args.do_infer:
        print("Final test result:")
        inference(exe, infer_prog, infer_reader, [prop.name], "infer")


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = PDConfig('senta_config.json')
    args.build()
    args.print_arguments()
    check_cuda(args.use_cuda)
    main(args)
