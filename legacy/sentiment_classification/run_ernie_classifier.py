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

import paddle
import paddle.fluid as fluid

sys.path.append("../shared_modules/models/classification/")
sys.path.append("../shared_modules/")
print(sys.path)

from nets import bow_net
from nets import lstm_net
from nets import cnn_net
from nets import bilstm_net
from nets import gru_net
from nets import ernie_base_net
from nets import ernie_bilstm_net
from preprocess.ernie import task_reader
from models.representation.ernie import ErnieConfig
from models.representation.ernie import ernie_encoder, ernie_encoder_with_paddle_hub
#from models.representation.ernie import ernie_pyreader
from models.model_check import check_cuda
from config import PDConfig

from utils import init_checkpoint


def ernie_pyreader(args, pyreader_name):
    src_ids = fluid.layers.data(
        name="src_ids", shape=[None, args.max_seq_len, 1], dtype="int64")
    sent_ids = fluid.layers.data(
        name="sent_ids", shape=[None, args.max_seq_len, 1], dtype="int64")
    pos_ids = fluid.layers.data(
        name="pos_ids", shape=[None, args.max_seq_len, 1], dtype="int64")
    input_mask = fluid.layers.data(
        name="input_mask", shape=[None, args.max_seq_len, 1], dtype="float32")
    labels = fluid.layers.data(name="labels", shape=[None, 1], dtype="int64")
    seq_lens = fluid.layers.data(name="seq_lens", shape=[None], dtype="int64")

    pyreader = fluid.io.DataLoader.from_generator(
        feed_list=[src_ids, sent_ids, pos_ids, input_mask, labels, seq_lens],
        capacity=50,
        iterable=False,
        use_double_buffer=True)

    ernie_inputs = {
        "src_ids": src_ids,
        "sent_ids": sent_ids,
        "pos_ids": pos_ids,
        "input_mask": input_mask,
        "seq_lens": seq_lens
    }

    return pyreader, ernie_inputs, labels


def create_model(args, embeddings, labels, is_prediction=False):
    """
    Create Model for sentiment classification based on ERNIE encoder
    """
    sentence_embeddings = embeddings["sentence_embeddings"]
    token_embeddings = embeddings["token_embeddings"]

    if args.model_type == "ernie_base":
        ce_loss, probs = ernie_base_net(sentence_embeddings, labels,
                                        args.num_labels)

    elif args.model_type == "ernie_bilstm":
        ce_loss, probs = ernie_bilstm_net(token_embeddings, labels,
                                          args.num_labels)

    else:
        raise ValueError("Unknown network type!")

    if is_prediction:
        return probs
    loss = fluid.layers.mean(x=ce_loss)

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)

    return loss, accuracy, num_seqs


def evaluate(exe, test_program, test_pyreader, fetch_list, eval_phase):
    """
    Evaluation Function
    """
    test_pyreader.start()
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
            test_pyreader.reset()
            break
    time_end = time.time()
    print("[%s evaluation] ave loss: %f, ave acc: %f, elapsed time: %f s" %
          (eval_phase, np.sum(total_cost) / np.sum(total_num_seqs),
           np.sum(total_acc) / np.sum(total_num_seqs), time_end - time_begin))


def infer(exe, infer_program, infer_pyreader, fetch_list, infer_phase):
    """
    Inference Function
    """
    infer_pyreader.start()
    time_begin = time.time()
    while True:
        try:
            batch_probs = exe.run(program=infer_program,
                                  fetch_list=fetch_list,
                                  return_numpy=True)
            for probs in batch_probs[0]:
                print("%d\t%f\t%f" % (np.argmax(probs), probs[0], probs[1]))
        except fluid.core.EOFException:
            infer_pyreader.reset()
            break
    time_end = time.time()
    print("[%s] elapsed time: %f s" % (infer_phase, time_end - time_begin))


def main(args):
    """
    Main Function
    """
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    reader = task_reader.ClassifyReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        random_seed=args.random_seed)

    if not (args.do_train or args.do_val or args.do_infer):
        raise ValueError("For args `do_train`, `do_val` and `do_infer`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train:
        train_data_generator = reader.data_generator(
            input_file=args.train_set,
            batch_size=args.batch_size,
            epoch=args.epoch,
            shuffle=True,
            phase="train")

        num_train_examples = reader.get_num_examples(args.train_set)

        max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

        print("Device count: %d" % dev_count)
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)

        train_program = fluid.Program()

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                # create ernie_pyreader
                train_pyreader, ernie_inputs, labels = ernie_pyreader(
                    args, pyreader_name='train_pyreader')

                # get ernie_embeddings
                if args.use_paddle_hub:
                    embeddings = ernie_encoder_with_paddle_hub(ernie_inputs,
                                                               args.max_seq_len)
                else:
                    embeddings = ernie_encoder(
                        ernie_inputs, ernie_config=ernie_config)

                # user defined model based on ernie embeddings
                loss, accuracy, num_seqs = create_model(
                    args, embeddings, labels=labels, is_prediction=False)

                optimizer = fluid.optimizer.Adam(learning_rate=args.lr)
                optimizer.minimize(loss)

        if args.verbose:
            lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                program=train_program, batch_size=args.batch_size)
            print("Theoretical memory usage in training: %.3f - %.3f %s" %
                  (lower_mem, upper_mem, unit))

    if args.do_val:
        test_data_generator = reader.data_generator(
            input_file=args.dev_set,
            batch_size=args.batch_size,
            phase='dev',
            epoch=1,
            shuffle=False)
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                # create ernie_pyreader
                test_pyreader, ernie_inputs, labels = ernie_pyreader(
                    args, pyreader_name='eval_reader')

                # get ernie_embeddings
                if args.use_paddle_hub:
                    embeddings = ernie_encoder_with_paddle_hub(ernie_inputs,
                                                               args.max_seq_len)
                else:
                    embeddings = ernie_encoder(
                        ernie_inputs, ernie_config=ernie_config)

                # user defined model based on ernie embeddings
                loss, accuracy, num_seqs = create_model(
                    args, embeddings, labels=labels, is_prediction=False)

        test_prog = test_prog.clone(for_test=True)

    if args.do_infer:
        infer_data_generator = reader.data_generator(
            input_file=args.test_set,
            batch_size=args.batch_size,
            phase='infer',
            epoch=1,
            shuffle=False)
        infer_prog = fluid.Program()
        with fluid.program_guard(infer_prog, startup_prog):
            with fluid.unique_name.guard():
                infer_pyreader, ernie_inputs, labels = ernie_pyreader(
                    args, pyreader_name="infer_pyreader")

                # get ernie_embeddings
                if args.use_paddle_hub:
                    embeddings = ernie_encoder_with_paddle_hub(ernie_inputs,
                                                               args.max_seq_len)
                else:
                    embeddings = ernie_encoder(
                        ernie_inputs, ernie_config=ernie_config)

                probs = create_model(
                    args, embeddings, labels=labels, is_prediction=True)

        infer_prog = infer_prog.clone(for_test=True)

    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint:
            init_checkpoint(
                exe, args.init_checkpoint, main_program=train_program)
    elif args.do_val:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or testing!")
        init_checkpoint(exe, args.init_checkpoint, main_program=test_prog)
    elif args.do_infer:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or testing!")
        init_checkpoint(exe, args.init_checkpoint, main_program=infer_prog)

    if args.do_train:
        train_exe = exe
        train_pyreader.set_batch_generator(train_data_generator)
    else:
        train_exe = None
    if args.do_val:
        test_exe = exe
        test_pyreader.set_batch_generator(test_data_generator)
    if args.do_infer:
        test_exe = exe
        infer_pyreader.set_batch_generator(infer_data_generator)

    if args.do_train:
        train_pyreader.start()
        steps = 0
        total_cost, total_acc, total_num_seqs = [], [], []
        time_begin = time.time()
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
                        evaluate(exe, test_prog, test_pyreader,
                                 [loss.name, accuracy.name, num_seqs.name],
                                 "dev")

            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps),
                                         "checkpoint")
                fluid.save(train_program, save_path)
                train_pyreader.reset()
                break

    # final eval on dev set
    if args.do_val:
        print("Final validation result:")
        evaluate(exe, test_prog, test_pyreader,
                 [loss.name, accuracy.name, num_seqs.name], "dev")

    # final eval on test set
    if args.do_infer:
        print("Final test result:")
        infer(exe, infer_prog, infer_pyreader, [probs.name], "infer")


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = PDConfig()
    args.build()
    args.print_arguments()
    check_cuda(args.use_cuda)
    main(args)
