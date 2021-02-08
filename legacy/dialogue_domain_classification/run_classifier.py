#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import multiprocessing
import sys
# sys.path.append("../models/classification/")
from nets import textcnn_net_multi_label
import paddle
import paddle.fluid as fluid
from utils import ArgumentGroup, print_arguments, DataProcesser, DataReader, ConfigReader
from utils import init_checkpoint, check_version, logger
import random
import codecs
import logging
import math
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(__doc__)
DEV_COUNT = 1
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("init_checkpoint", str, None,
                "Init checkpoint to resume training from.")
model_g.add_arg("checkpoints", str, "./checkpoints",
                "Path to save checkpoints.")
model_g.add_arg("config_path", str, "./data/input/model.conf", "Model conf.")
model_g.add_arg("build_dict", bool, False, "Build dict.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("cpu_num", int, 3, "Number of Threads.")
train_g.add_arg("epoch", int, 100, "Number of epoches for training.")
train_g.add_arg("learning_rate", float, 0.1,
                "Learning rate used to train with warmup.")
train_g.add_arg("save_steps", int, 1000,
                "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps", int, 100,
                "The steps interval to evaluate model performance.")
train_g.add_arg("random_seed", int, 7, "random seed")
train_g.add_arg(
    "threshold", float, 0.1,
    "When the confidence exceeds the threshold, the corresponding label is given."
)

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")

data_g = ArgumentGroup(parser, "data",
                       "Data paths, vocab paths and data processing options")
data_g.add_arg("data_dir", str, "./data/input/", "Path to training data.")
data_g.add_arg("save_dir", str, "./data/output/", "Path to save.")
data_g.add_arg("max_seq_len", int, 50,
               "Tokens' number of the longest seqence allowed.")
data_g.add_arg("batch_size", int, 64,
               "The total number of examples in one batch for training.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, False, "If set, use GPU for training.")
# run_type_g.add_arg("use_fast_executor", bool, False, "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("do_train", bool, True,
                   "Whether to perform evaluation on test data set.")
run_type_g.add_arg("do_eval", bool, True,
                   "Whether to perform evaluation on test data set.")
run_type_g.add_arg("do_test", bool, True,
                   "Whether to perform evaluation on test data set.")
args = parser.parse_args()


def get_score(pred_result, label, eval_phase):
    """[get precision recall and f-score]
    
    Arguments:
        pred_result {[type]} -- [pred labels]
        label {[type]} -- [origin labels]
    """
    tp = 0
    total = 0
    true_cnt = 0
    pred_pos_num = 0
    pos_num = 0
    for i in range(len(pred_result)):
        total += 1
        pred_labels = []
        actual_labels = []
        for j in range(1, len(pred_result[0])):  # the 0 one is background
            if pred_result[i][j] == 1:
                pred_labels.append(j)
            if label[i][j] == 1:
                actual_labels.append(j)
        if len(pred_labels) > 0:
            pred_pos_num += 1
        if len(actual_labels) > 0:
            pos_num += 1
            if set(actual_labels).issubset(set(pred_labels)):
                tp += 1
                true_cnt += 1
        elif len(pred_labels) == 0 and len(actual_labels) == 0:
            true_cnt += 1
    try:
        precision = tp * 1.0 / pred_pos_num
        recall = tp * 1.0 / pos_num
        f1 = 2 * precision * recall / (recall + precision)
    except Exception as e:
        precision = 0
        recall = 0
        f1 = 0
    acc = true_cnt * 1.0 / total
    logger.info("tp, pred_pos_num, pos_num, total")
    logger.info("%d, %d, %d, %d" % (tp, pred_pos_num, pos_num, total))
    logger.info("%s result is : precision is %f, recall is %f, f1_score is %f, acc is %f" % (eval_phase, precision, \
                recall, f1, acc))


def train(args, train_exe, build_res, place):
    """[train the net]
    
    Arguments:
        args {[type]} -- [description]
        train_exe {[type]} -- [description]
        compiled_prog{[type]} -- [description]
        build_res {[type]} -- [description]
        place {[type]} -- [description]
    """
    global DEV_COUNT
    compiled_prog = build_res["compiled_prog"]
    cost = build_res["cost"]
    prediction = build_res["prediction"]
    pred_label = build_res["pred_label"]
    label = build_res["label"]
    fetch_list = [cost.name, prediction.name, pred_label.name, label.name]
    train_data_loader = build_res["train_data_loader"]
    train_prog = build_res["train_prog"]
    steps = 0
    time_begin = time.time()
    test_exe = train_exe
    logger.info("Begin training")
    for i in range(args.epoch):
        try:
            for data in train_data_loader():
                avg_cost_np, avg_pred_np, pred_label, label = train_exe.run(feed=data, program=compiled_prog, \
                                                                            fetch_list=fetch_list)
                steps += 1
                if steps % int(args.skip_steps) == 0:
                    time_end = time.time()
                    used_time = time_end - time_begin
                    get_score(pred_label, label, eval_phase="Train")
                    logger.info('loss is {}'.format(avg_cost_np))
                    logger.info("epoch: %d, step: %d, speed: %f steps/s" %
                                (i, steps, args.skip_steps / used_time))
                    time_begin = time.time()
                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoints,
                                             "step_" + str(steps))
                    fluid.io.save(train_prog, save_path)
                    logger.info("[save]step %d : save at %s" %
                                (steps, save_path))
                if steps % args.validation_steps == 0:
                    if args.do_eval:
                        evaluate(args, test_exe, build_res, "eval")
                    if args.do_test:
                        evaluate(args, test_exe, build_res, "test")
        except Exception as e:
            logger.exception(str(e))
            logger.error("Train error : %s" % str(e))
            exit(1)
    save_path = os.path.join(args.checkpoints, "step_" + str(steps))
    fluid.io.save(train_prog, save_path)
    logger.info("[save]step %d : save at %s" % (steps, save_path))


def evaluate(args,
             test_exe,
             build_res,
             eval_phase,
             save_result=False,
             id2intent=None):
    """[evaluate on dev/test dataset]
    
    Arguments:
        args {[type]} -- [description]
        test_exe {[type]} -- [description]
        test_prog {[type]} -- [description]
        build_res {[type]} -- [description]
        place {[type]} -- [description]
        eval_phase {[type]} -- [description]
    
    Keyword Arguments:
        threshold {float} -- [description] (default: {0.5})
        save_result {bool} -- [description] (default: {False})
        id2intent {[type]} -- [description] (default: {None})
    """
    place = build_res["test_place"]
    threshold = args.threshold
    cost = build_res["cost"]
    prediction = build_res["prediction"]
    pred_label = build_res["pred_label"]
    label = build_res["label"]
    fetch_list = [cost.name, prediction.name, pred_label.name, label.name]
    total_cost, total_acc, pred_prob_list, pred_label_list, label_list = [], [], [], [], []
    if eval_phase == "eval":
        test_prog = build_res["eval_compiled_prog"]
        test_data_loader = build_res["eval_data_loader"]
    elif eval_phase == "test":
        test_prog = build_res["test_compiled_prog"]
        test_data_loader = build_res["test_data_loader"]
    else:
        exit(1)
    logger.info("-----------------------------------------------------------")
    for data in test_data_loader():
        avg_cost_np, avg_pred_np, pred_label, label= test_exe.run(program=test_prog, fetch_list=fetch_list, feed=data, \
            return_numpy=True)
        total_cost.append(avg_cost_np)
        pred_prob_list.extend(avg_pred_np)
        pred_label_list.extend(pred_label)
        label_list.extend(label)

    if save_result:
        logger.info("save result at : %s" % args.save_dir + "/" + eval_phase +
                    ".rst")
        save_dir = args.save_dir
        if not os.path.exists(save_dir):
            logger.warning("save dir not exists, and create it")
            os.makedirs(save_dir)
        fin = codecs.open(
            os.path.join(args.data_dir, eval_phase + ".txt"),
            "r",
            encoding="utf8")
        fout = codecs.open(
            args.save_dir + "/" + eval_phase + ".rst", "w", encoding="utf8")
        for line in pred_prob_list:
            query = fin.readline().rsplit("\t", 1)[0]
            res = []
            for i in range(1, len(line)):
                if line[i] > threshold:
                    #res.append(id2intent[i]+":"+str(line[i]))
                    res.append(id2intent[i])
            if len(res) == 0:
                res.append(id2intent[0])
            fout.write("%s\t%s\n" % (query, "\2".join(sorted(res))))
        fout.close()
        fin.close()

    logger.info("[%s] result: " % eval_phase)
    get_score(pred_label_list, label_list, eval_phase)
    logger.info('loss is {}'.format(sum(total_cost) * 1.0 / len(total_cost)))
    logger.info("-----------------------------------------------------------")


def create_net(args,
               flow_data,
               class_dim,
               dict_dim,
               place,
               model_name="textcnn_net",
               is_infer=False):
    """[create network and loader]
    
    Arguments:
        flow_data {[type]} -- [description]
        class_dim {[type]} -- [description]
        dict_dim {[type]} -- [description]
        place {[type]} -- [description]
    
    Keyword Arguments:
        model_name {str} -- [description] (default: {"textcnn_net"})
        is_infer {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- [description]
    """
    if model_name == "textcnn_net":
        model = textcnn_net_multi_label
    else:
        return
    char_list = fluid.data(
        name="char",
        shape=[None, args.max_seq_len, 1],
        dtype="int64",
        lod_level=0)
    label = fluid.data(
        name="label", shape=[None, class_dim], dtype="float32",
        lod_level=0)  # label data
    data_loader = fluid.io.DataLoader.from_generator(
        feed_list=[char_list, label],
        capacity=args.batch_size * 10,
        iterable=True,
        return_list=False)
    output = model(
        char_list,
        label,
        dict_dim,
        emb_dim=flow_data["model"]["emb_dim"],
        hid_dim=flow_data["model"]["hid_dim"],
        hid_dim2=flow_data["model"]["hid_dim2"],
        class_dim=class_dim,
        win_sizes=flow_data["model"]["win_sizes"],
        is_infer=is_infer,
        threshold=args.threshold,
        max_seq_len=args.max_seq_len)
    if is_infer:
        prediction = output
        return [data_loader, prediction]
    else:
        avg_cost, prediction, pred_label, label = output[0], output[1], output[
            2], output[3]
        return [data_loader, avg_cost, prediction, pred_label, label]


def build_data_loader(args, char_dict, intent_dict):
    """[decorate samples for dataloader]
    
    Arguments:
        args {[type]} -- [description]
        char_dict {[type]} -- [description]
        intent_dict {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    loader_res = {}
    if args.do_train:
        train_processor = DataReader(char_dict, intent_dict, args.max_seq_len)
        train_data_generator = train_processor.prepare_data(
            data_path=args.data_dir + "train.txt",
            batch_size=args.batch_size,
            mode='train')
        loader_res["train_data_generator"] = train_data_generator
        num_train_examples = train_processor._get_num_examples()
        logger.info("Num train examples: %d" % num_train_examples)
        logger.info("Num train steps: %d" % (math.ceil(num_train_examples * 1.0 / args.batch_size) * \
                                            args.epoch // DEV_COUNT))
        if math.ceil(num_train_examples * 1.0 /
                     args.batch_size) // DEV_COUNT <= 0:
            logger.error(
                "Num of train steps is less than 0  or equals to 0, exit")
            exit(1)
    if args.do_eval:
        eval_processor = DataReader(char_dict, intent_dict, args.max_seq_len)
        eval_data_generator = eval_processor.prepare_data(
            data_path=args.data_dir + "eval.txt",
            batch_size=args.batch_size,
            mode='eval')
        loader_res["eval_data_generator"] = eval_data_generator
        num_eval_examples = eval_processor._get_num_examples()
        logger.info("Num eval examples: %d" % num_eval_examples)
    if args.do_test:
        test_processor = DataReader(char_dict, intent_dict, args.max_seq_len)
        test_data_generator = test_processor.prepare_data(
            data_path=args.data_dir + "test.txt",
            batch_size=args.batch_size,
            mode='test')
        loader_res["test_data_generator"] = test_data_generator
    return loader_res


def build_graph(args, model_config, num_labels, dict_dim, place, test_place,
                loader_res):
    """[build paddle graph]
    
    Arguments:
        args {[type]} -- [description]
        model_config {[type]} -- [description]
        num_labels {[type]} -- [description]
        dict_dim {[type]} -- [description]
        place {[type]} -- [description]
        loader_res {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    res = {}
    cost, prediction, pred_label, label = None, None, None, None
    train_prog = fluid.default_main_program()

    startup_prog = fluid.default_startup_program()
    eval_prog = train_prog.clone(for_test=True)
    test_prog = train_prog.clone(for_test=True)
    train_prog.random_seed = args.random_seed
    startup_prog.random_seed = args.random_seed
    if args.do_train:
        with fluid.program_guard(train_prog, startup_prog):
            with fluid.unique_name.guard():
                train_data_loader, cost, prediction, pred_label, label = create_net(args, model_config, num_labels, \
                                                            dict_dim, place, model_name="textcnn_net")
                train_data_loader.set_sample_list_generator(
                    loader_res['train_data_generator'], places=place)
                res["train_data_loader"] = train_data_loader
                sgd_optimizer = fluid.optimizer.SGD(
                    learning_rate=fluid.layers.exponential_decay(
                        learning_rate=args.learning_rate,
                        decay_steps=1000,
                        decay_rate=0.5,
                        staircase=True))
                sgd_optimizer.minimize(cost)
    if args.do_eval:
        with fluid.program_guard(eval_prog, startup_prog):
            with fluid.unique_name.guard():
                eval_data_loader, cost, prediction, pred_label, label = create_net(args, model_config, num_labels, \
                                                             dict_dim, test_place, model_name="textcnn_net")
                eval_data_loader.set_sample_list_generator(
                    loader_res['eval_data_generator'], places=test_place)
                res["eval_data_loader"] = eval_data_loader
    if args.do_test:
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_data_loader, cost, prediction, pred_label, label = create_net(args, model_config, num_labels, \
                                                            dict_dim, test_place, model_name="textcnn_net")
                test_data_loader.set_sample_list_generator(
                    loader_res['test_data_generator'], places=test_place)
                res["test_data_loader"] = test_data_loader
    res["cost"] = cost
    res["prediction"] = prediction
    res["label"] = label
    res["pred_label"] = pred_label
    res["train_prog"] = train_prog
    res["eval_prog"] = eval_prog
    res["test_prog"] = test_prog

    return res


def main(args):
    """
    Main Function
    """
    global DEV_COUNT
    startup_prog = fluid.default_startup_program()
    random.seed(args.random_seed)
    model_config = ConfigReader.read_conf(args.config_path)
    if args.use_cuda:
        test_place = fluid.cuda_places(0)
        place = fluid.cuda_places()
        DEV_COUNT = len(place)
    else:
        test_place = fluid.cpu_places(1)
        os.environ['CPU_NUM'] = str(args.cpu_num)
        place = fluid.cpu_places()
        DEV_COUNT = args.cpu_num
    logger.info("Dev Num is %s" % str(DEV_COUNT))
    exe = fluid.Executor(place[0])
    if args.do_train and args.build_dict:
        DataProcesser.build_dict(args.data_dir + "train.txt", args.data_dir)
    # read dict
    char_dict = DataProcesser.read_dict(args.data_dir + "char.dict")
    dict_dim = len(char_dict)
    intent_dict = DataProcesser.read_dict(args.data_dir + "domain.dict")
    id2intent = {}
    for key, value in intent_dict.items():
        id2intent[int(value)] = key
    num_labels = len(intent_dict)
    # build model
    loader_res = build_data_loader(args, char_dict, intent_dict)
    build_res = build_graph(args, model_config, num_labels, dict_dim, place,
                            test_place, loader_res)
    build_res["place"] = place
    build_res["test_place"] = test_place
    if not (args.do_train or args.do_eval or args.do_test):
        raise ValueError("For args `do_train`, `do_eval` and `do_test`, at "
                         "least one of them must be True.")

    exe.run(startup_prog)
    if args.init_checkpoint and args.init_checkpoint != "None":
        try:
            init_checkpoint(
                exe, args.init_checkpoint, main_program=startup_prog)
            logger.info("Load model from %s" % args.init_checkpoint)
        except Exception as e:
            logger.exception(str(e))
            logger.error("Faild load model from %s [%s]" %
                         (args.init_checkpoint, str(e)))
    build_strategy = fluid.compiler.BuildStrategy()
    build_strategy.fuse_all_reduce_ops = False
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 1
    # add compiled prog
    if args.do_train:
        compiled_prog = fluid.compiler.CompiledProgram(build_res["train_prog"]).with_data_parallel( \
                                                                    loss_name=build_res["cost"].name, \
                                                                    build_strategy=build_strategy, \
                                                                    exec_strategy=exec_strategy)
        build_res["compiled_prog"] = compiled_prog
    if args.do_test:
        test_compiled_prog = fluid.compiler.CompiledProgram(build_res[
            "test_prog"])
        build_res["test_compiled_prog"] = test_compiled_prog
    if args.do_eval:
        eval_compiled_prog = fluid.compiler.CompiledProgram(build_res[
            "eval_prog"])
        build_res["eval_compiled_prog"] = eval_compiled_prog

    if args.do_train:
        train(args, exe, build_res, place)
    if args.do_eval:
        evaluate(args, exe, build_res, "eval", \
                 save_result=True, id2intent=id2intent)
    if args.do_test:
        evaluate(args, exe, build_res, "test",\
                  save_result=True, id2intent=id2intent)


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    logger.info("the paddle version is %s" % paddle.__version__)
    check_version('1.6.0')
    print_arguments(args)
    main(args)
