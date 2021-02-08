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
SimNet Task
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import multiprocessing
import sys

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

sys.path.append("../shared_modules/")

import paddle
import paddle.fluid as fluid
import numpy as np
import config
import utils
import reader
import models.matching.paddle_layers as layers
import io
import logging

from utils import ArgConfig
from models.model_check import check_version
from models.model_check import check_cuda


def create_model(args, is_inference=False, is_pointwise=False):
    """
    Create Model for simnet
    """
    if is_inference:
        left = fluid.data(name='left', shape=[None], dtype='int64', lod_level=1)
        pos_right = fluid.data(
            name='pos_right', shape=[None], dtype='int64', lod_level=1)
        inf_loader = fluid.io.DataLoader.from_generator(
            capacity=16,
            feed_list=[left, pos_right],
            iterable=False,
            use_double_buffer=False)

        return inf_loader, left, pos_right

    else:
        if is_pointwise:
            left = fluid.data(
                name='left', shape=[None], dtype='int64', lod_level=1)
            right = fluid.data(
                name='right', shape=[None], dtype='int64', lod_level=1)
            label = fluid.data(name='label', shape=[None], dtype='int64')
            pointwise_loader = fluid.io.DataLoader.from_generator(
                capacity=16,
                feed_list=[left, right, label],
                iterable=False,
                use_double_buffer=False)

            return pointwise_loader, left, right, label

        else:
            left = fluid.data(
                name='left', shape=[None], dtype='int64', lod_level=1)
            pos_right = fluid.data(
                name='pos_right', shape=[None], dtype='int64', lod_level=1)
            neg_right = fluid.data(
                name='neg_right', shape=[None], dtype='int64', lod_level=1)
            pairwise_loader = fluid.io.DataLoader.from_generator(
                capacity=16,
                feed_list=[left, pos_right, neg_right],
                iterable=False,
                use_double_buffer=False)

            return pairwise_loader, left, pos_right, neg_right


def train(conf_dict, args):
    """
    train processic
    """
    # loading vocabulary
    vocab = utils.load_vocab(args.vocab_path)
    # get vocab size
    conf_dict['dict_size'] = len(vocab)
    # Load network structure dynamically
    net = utils.import_class("../shared_modules/models/matching",
                             conf_dict["net"]["module_name"],
                             conf_dict["net"]["class_name"])(conf_dict)
    # Load loss function dynamically
    loss = utils.import_class("../shared_modules/models/matching/losses",
                              conf_dict["loss"]["module_name"],
                              conf_dict["loss"]["class_name"])(conf_dict)
    # Load Optimization method
    optimizer = utils.import_class(
        "../shared_modules/models/matching/optimizers", "paddle_optimizers",
        conf_dict["optimizer"]["class_name"])(conf_dict)
    # load auc method
    metric = fluid.metrics.Auc(name="auc")
    # Get device
    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    startup_prog = fluid.Program()
    train_program = fluid.Program()

    # used for continuous evaluation 
    if args.enable_ce:
        SEED = 102
        startup_prog.random_seed = SEED
        train_program.random_seed = SEED

    simnet_process = reader.SimNetProcessor(args, vocab)
    if args.task_mode == "pairwise":
        # Build network
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_loader, left, pos_right, neg_right = create_model(args)
                left_feat, pos_score = net.predict(left, pos_right)
                pred = pos_score
                _, neg_score = net.predict(left, neg_right)
                avg_cost = loss.compute(pos_score, neg_score)
                avg_cost.persistable = True
                optimizer.ops(avg_cost)

        # Get Reader
        get_train_examples = simnet_process.get_reader(
            "train", epoch=args.epoch)
        if args.do_valid:
            test_prog = fluid.Program()
            with fluid.program_guard(test_prog, startup_prog):
                with fluid.unique_name.guard():
                    test_loader, left, pos_right = create_model(
                        args, is_inference=True)
                    left_feat, pos_score = net.predict(left, pos_right)
                    pred = pos_score
            test_prog = test_prog.clone(for_test=True)

    else:
        # Build network
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_loader, left, right, label = create_model(
                    args, is_pointwise=True)
                left_feat, pred = net.predict(left, right)
                avg_cost = loss.compute(pred, label)
                avg_cost.persistable = True
                optimizer.ops(avg_cost)

        # Get Feeder and Reader
        get_train_examples = simnet_process.get_reader(
            "train", epoch=args.epoch)
        if args.do_valid:
            test_prog = fluid.Program()
            with fluid.program_guard(test_prog, startup_prog):
                with fluid.unique_name.guard():
                    test_loader, left, right = create_model(
                        args, is_inference=True)
                    left_feat, pred = net.predict(left, right)
            test_prog = test_prog.clone(for_test=True)

    if args.init_checkpoint is not "":
        utils.init_checkpoint(exe, args.init_checkpoint, startup_prog)

    def valid_and_test(test_program, test_loader, get_valid_examples, process,
                       mode, exe, fetch_list):
        """
        return auc and acc
        """
        # Get Batch Data
        batch_data = fluid.io.batch(
            get_valid_examples, args.batch_size, drop_last=False)
        test_loader.set_sample_list_generator(batch_data)
        test_loader.start()
        pred_list = []
        while True:
            try:
                _pred = exe.run(program=test_program, fetch_list=[pred.name])
                pred_list += list(_pred)
            except fluid.core.EOFException:
                test_loader.reset()
                break
        pred_list = np.vstack(pred_list)
        if mode == "test":
            label_list = process.get_test_label()
        elif mode == "valid":
            label_list = process.get_valid_label()
        if args.task_mode == "pairwise":
            pred_list = (pred_list + 1) / 2
            pred_list = np.hstack(
                (np.ones_like(pred_list) - pred_list, pred_list))
        metric.reset()
        metric.update(pred_list, label_list)
        auc = metric.eval()
        if args.compute_accuracy:
            acc = utils.get_accuracy(pred_list, label_list, args.task_mode,
                                     args.lamda)
            return auc, acc
        else:
            return auc

    # run train
    logging.info("start train process ...")
    # set global step
    global_step = 0
    ce_info = []
    train_exe = exe
    #for epoch_id in range(args.epoch):
    # used for continuous evaluation
    if args.enable_ce:
        train_batch_data = fluid.io.batch(
            get_train_examples, args.batch_size, drop_last=False)
    else:
        train_batch_data = fluid.io.batch(
            fluid.io.shuffle(
                get_train_examples, buf_size=10000),
            args.batch_size,
            drop_last=False)
    train_loader.set_sample_list_generator(train_batch_data)
    train_loader.start()
    exe.run(startup_prog)
    losses = []
    start_time = time.time()
    while True:
        try:
            global_step += 1
            fetch_list = [avg_cost.name]
            avg_loss = train_exe.run(program=train_program,
                                     fetch_list=fetch_list)
            losses.append(np.mean(avg_loss[0]))
            if args.do_valid and global_step % args.validation_steps == 0:
                get_valid_examples = simnet_process.get_reader("valid")
                valid_result = valid_and_test(
                    test_prog, test_loader, get_valid_examples, simnet_process,
                    "valid", exe, [pred.name])
                if args.compute_accuracy:
                    valid_auc, valid_acc = valid_result
                    logging.info(
                        "global_steps: %d, valid_auc: %f, valid_acc: %f, valid_loss: %f"
                        % (global_step, valid_auc, valid_acc, np.mean(losses)))
                else:
                    valid_auc = valid_result
                    logging.info(
                        "global_steps: %d, valid_auc: %f, valid_loss: %f" %
                        (global_step, valid_auc, np.mean(losses)))
            if global_step % args.save_steps == 0:
                model_save_dir = os.path.join(args.output_dir,
                                              conf_dict["model_path"])
                model_path = os.path.join(model_save_dir, str(global_step))

                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                if args.task_mode == "pairwise":
                    feed_var_names = [left.name, pos_right.name]
                    target_vars = [left_feat, pos_score]
                else:
                    feed_var_names = [
                        left.name,
                        right.name,
                    ]
                    target_vars = [left_feat, pred]
                fluid.io.save_inference_model(model_path, feed_var_names,
                                              target_vars, exe, test_prog)
                logging.info("saving infer model in %s" % model_path)

        except fluid.core.EOFException:
            train_loader.reset()
            break
    end_time = time.time()
    #logging.info("epoch: %d, loss: %f, used time: %d sec" %
    #(epoch_id, np.mean(losses), end_time - start_time))
    ce_info.append([np.mean(losses), end_time - start_time])
    #final save
    logging.info("the final step is %s" % global_step)
    model_save_dir = os.path.join(args.output_dir, conf_dict["model_path"])
    model_path = os.path.join(model_save_dir, str(global_step))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if args.task_mode == "pairwise":
        feed_var_names = [left.name, pos_right.name]
        target_vars = [left_feat, pos_score]
    else:
        feed_var_names = [
            left.name,
            right.name,
        ]
        target_vars = [left_feat, pred]
    fluid.io.save_inference_model(model_path, feed_var_names, target_vars, exe,
                                  test_prog)
    logging.info("saving infer model in %s" % model_path)
    # used for continuous evaluation
    if args.enable_ce:
        card_num = get_cards()
        ce_loss = 0
        ce_time = 0
        try:
            ce_loss = ce_info[-1][0]
            ce_time = ce_info[-1][1]
        except:
            logging.info("ce info err!")
        print("kpis\teach_step_duration_%s_card%s\t%s" %
              (args.task_name, card_num, ce_time))
        print("kpis\ttrain_loss_%s_card%s\t%f" %
              (args.task_name, card_num, ce_loss))

    if args.do_test:
        if args.task_mode == "pairwise":
            # Get Feeder and Reader
            get_test_examples = simnet_process.get_reader("test")
        else:
            # Get Feeder and Reader
            get_test_examples = simnet_process.get_reader("test")
        test_result = valid_and_test(test_prog, test_loader, get_test_examples,
                                     simnet_process, "test", exe, [pred.name])
        if args.compute_accuracy:
            test_auc, test_acc = test_result
            logging.info("AUC of test is %f, Accuracy of test is %f" %
                         (test_auc, test_acc))
        else:
            test_auc = test_result
            logging.info("AUC of test is %f" % test_auc)


def test(conf_dict, args):
    """
    Evaluation Function
    """
    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    vocab = utils.load_vocab(args.vocab_path)
    simnet_process = reader.SimNetProcessor(args, vocab)

    startup_prog = fluid.Program()

    get_test_examples = simnet_process.get_reader("test")
    batch_data = fluid.io.batch(
        get_test_examples, args.batch_size, drop_last=False)
    test_prog = fluid.Program()

    conf_dict['dict_size'] = len(vocab)

    net = utils.import_class("../shared_modules/models/matching",
                             conf_dict["net"]["module_name"],
                             conf_dict["net"]["class_name"])(conf_dict)

    metric = fluid.metrics.Auc(name="auc")

    with io.open("predictions.txt", "w", encoding="utf8") as predictions_file:
        if args.task_mode == "pairwise":
            with fluid.program_guard(test_prog, startup_prog):
                with fluid.unique_name.guard():
                    test_loader, left, pos_right = create_model(
                        args, is_inference=True)
                    left_feat, pos_score = net.predict(left, pos_right)
                    pred = pos_score
            test_prog = test_prog.clone(for_test=True)

        else:
            with fluid.program_guard(test_prog, startup_prog):
                with fluid.unique_name.guard():
                    test_loader, left, right = create_model(
                        args, is_inference=True)
                    left_feat, pred = net.predict(left, right)
            test_prog = test_prog.clone(for_test=True)

        exe.run(startup_prog)

        utils.init_checkpoint(exe, args.init_checkpoint, main_program=test_prog)

        test_exe = exe
        test_loader.set_sample_list_generator(batch_data)

        logging.info("start test process ...")
        test_loader.start()
        pred_list = []
        fetch_list = [pred.name]
        output = []
        while True:
            try:
                output = test_exe.run(program=test_prog, fetch_list=fetch_list)
                if args.task_mode == "pairwise":
                    pred_list += list(
                        map(lambda item: float(item[0]), output[0]))
                    predictions_file.write(u"\n".join(
                        map(lambda item: str((item[0] + 1) / 2), output[0])) +
                                           "\n")
                else:
                    pred_list += map(lambda item: item, output[0])
                    predictions_file.write(u"\n".join(
                        map(lambda item: str(np.argmax(item)), output[0])) +
                                           "\n")
            except fluid.core.EOFException:
                test_loader.reset()
                break
        if args.task_mode == "pairwise":
            pred_list = np.array(pred_list).reshape((-1, 1))
            pred_list = (pred_list + 1) / 2
            pred_list = np.hstack(
                (np.ones_like(pred_list) - pred_list, pred_list))
        else:
            pred_list = np.array(pred_list)
        labels = simnet_process.get_test_label()

        metric.update(pred_list, labels)
        if args.compute_accuracy:
            acc = utils.get_accuracy(pred_list, labels, args.task_mode,
                                     args.lamda)
            logging.info("AUC of test is %f, Accuracy of test is %f" %
                         (metric.eval(), acc))
        else:
            logging.info("AUC of test is %f" % metric.eval())

    if args.verbose_result:
        utils.get_result_file(args)
        logging.info("test result saved in %s" %
                     os.path.join(os.getcwd(), args.test_result_path))


def infer(conf_dict, args):
    """
    run predict
    """
    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    vocab = utils.load_vocab(args.vocab_path)
    simnet_process = reader.SimNetProcessor(args, vocab)

    startup_prog = fluid.Program()

    get_infer_examples = simnet_process.get_infer_reader
    batch_data = fluid.io.batch(
        get_infer_examples, args.batch_size, drop_last=False)

    test_prog = fluid.Program()

    conf_dict['dict_size'] = len(vocab)

    net = utils.import_class("../shared_modules/models/matching",
                             conf_dict["net"]["module_name"],
                             conf_dict["net"]["class_name"])(conf_dict)

    if args.task_mode == "pairwise":
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                infer_loader, left, pos_right = create_model(
                    args, is_inference=True)
                left_feat, pos_score = net.predict(left, pos_right)
                pred = pos_score
        test_prog = test_prog.clone(for_test=True)
    else:
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                infer_loader, left, right = create_model(
                    args, is_inference=True)
                left_feat, pred = net.predict(left, right)
        test_prog = test_prog.clone(for_test=True)

    exe.run(startup_prog)

    utils.init_checkpoint(exe, args.init_checkpoint, main_program=test_prog)

    test_exe = exe
    infer_loader.set_sample_list_generator(batch_data)

    logging.info("start test process ...")
    preds_list = []
    fetch_list = [pred.name]
    output = []
    infer_loader.start()
    while True:
        try:
            output = test_exe.run(program=test_prog, fetch_list=fetch_list)
            if args.task_mode == "pairwise":
                preds_list += list(
                    map(lambda item: str((item[0] + 1) / 2), output[0]))
            else:
                preds_list += map(lambda item: str(np.argmax(item)), output[0])
        except fluid.core.EOFException:
            infer_loader.reset()
            break
    with io.open(args.infer_result_path, "w", encoding="utf8") as infer_file:
        for _data, _pred in zip(simnet_process.get_infer_data(), preds_list):
            infer_file.write(_data + "\t" + _pred + "\n")
    logging.info("infer result saved in %s" %
                 os.path.join(os.getcwd(), args.infer_result_path))


def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


if __name__ == "__main__":
    import paddle
    paddle.enable_static()

    args = ArgConfig()
    args = args.build_conf()

    utils.print_arguments(args)
    check_cuda(args.use_cuda)
    check_version()
    utils.init_log("./log/TextSimilarityNet")
    conf_dict = config.SimNetConfig(args)
    if args.do_train:
        train(conf_dict, args)
    elif args.do_test:
        test(conf_dict, args)
    elif args.do_infer:
        infer(conf_dict, args)
    else:
        raise ValueError(
            "one of do_train and do_test and do_infer must be True")
