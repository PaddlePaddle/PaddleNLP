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
"""Fine-tuning on regression/classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import sys
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')

import os
import time
import json
import argparse
import numpy as np
import subprocess
import multiprocessing
from scipy.stats import pearsonr

import paddle
import paddle.fluid as fluid

import reader.cls as reader
from model.xlnet import XLNetConfig
from model.classifier import create_model
from optimization import optimization
from utils.args import ArgumentGroup, print_arguments, check_cuda
from utils.init import init_pretraining_params, init_checkpoint
from utils.cards import get_cards

num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("model_config_path",        str,  None,           "Path to the json file for bert model config.")
model_g.add_arg("dropout",                  float,  0.1,          "Dropout rate.")
model_g.add_arg("dropatt",                  float,  0.1,          "Attention dropout rate.")
model_g.add_arg("clamp_len",                int,    -1,           "Clamp length.")
model_g.add_arg("summary_type",             str, "last",
                "Method used to summarize a sequence into a vector.", choices=['last'])
model_g.add_arg("use_summ_proj",            bool, True,
                "Whether to use projection for summarizing sequences.")
model_g.add_arg("spiece_model_file",        str,  None,           "Sentence Piece model path.")
model_g.add_arg("init_checkpoint",          str,  None,           "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params",  str,  None,
                "Init pre-training params which preforms fine-tuning from. If the "
                 "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints",              str,  "checkpoints",  "Path to save checkpoints.")

init_g = ArgumentGroup(parser, "init", "parameter initialization options.")
init_g.add_arg("init",       str, "normal",    "Initialization method.", choices=["normal", "uniform"])
init_g.add_arg("init_std",   str, 0.02,    "Initialization std when init is normal.")
init_g.add_arg("init_range", str, 0.1,   "Initialization std when init is uniform.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch",             int,    1000,    "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate",     float,  5e-5,    "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay",      float,  0.01,    "Weight decay rate for L2 regularizer.")
train_g.add_arg("lr_layer_decay_rate", float,  1.0, "Top layer: lr[L] = args.learning_rate. "
                                                     "Lower layers: lr[l-1] = lr[l] * lr_layer_decay_rate.")
train_g.add_arg("save_steps",        int,    10000,   "The steps interval to save checkpoints.")
train_g.add_arg("train_batch_size",    int,    8,       "Total examples' number in batch for training.")
train_g.add_arg("eval_batch_size",     int,    128,     "Total examples' number in batch for development.")
train_g.add_arg("predict_batch_size",  int,    128,     "Total examples' number in batch for prediction.")
train_g.add_arg("train_steps",       int,    1000,   "The total steps for training.")
train_g.add_arg("warmup_steps",      int,    1000,   "The steps for warmup.")
train_g.add_arg("validation_steps",  int,    1000,    "The steps interval to evaluate model performance.")

log_g = ArgumentGroup(parser,     "logging", "logging related.")
log_g.add_arg("skip_steps",          int,    10,    "The steps interval to print loss.")
log_g.add_arg("verbose",             bool,   False, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("data_dir",           str,   None,  "Path to training data.")
data_g.add_arg("predict_dir",        str,   None,  "Path to write predict results.")
data_g.add_arg("predict_threshold",  float, 0.0,   "Threshold for binary prediction.")
data_g.add_arg("max_seq_length",     int,   512,   "Number of words of the longest seqence.")
data_g.add_arg("uncased", bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
data_g.add_arg("random_seed",   int,  0,     "Random seed.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",                     bool,   True,  "If set, use GPU for training.")
run_type_g.add_arg("use_fast_executor",            bool,   False, "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("shuffle",                      bool,   True,  "")
run_type_g.add_arg("task_name",                    str,    None,
                   "The name of task to perform fine-tuning, should be in {'xnli', 'mnli', 'cola', 'mrpc'}.")
run_type_g.add_arg("is_regression",                str,    None,  "Whether it's a regression task.")
run_type_g.add_arg("do_train",                     bool,   True,  "Whether to perform training.")
run_type_g.add_arg("do_eval",                      bool,   True,  "Whether to perform evaluation on dev data set.")
run_type_g.add_arg("do_predict",                   bool,   True,  "Whether to perform evaluation on test data set.")
run_type_g.add_arg("eval_split",                   str,    "dev", "Could be dev or test")

parser.add_argument("--enable_ce", action='store_true', help="The flag indicating whether to run the task for continuous evaluation.")

args = parser.parse_args()
# yapf: enable.


def evaluate(exe, predict_program, test_data_loader, fetch_list, eval_phase, num_examples):
    test_data_loader.start()
    total_cost, total_num_seqs = [], []
    all_logits, all_labels = [], []
    time_begin = time.time()
    total_steps = int(num_examples / args.eval_batch_size)
    steps = 0
    while True:
        try:
            np_loss, np_num_seqs, np_logits, np_labels = exe.run(program=predict_program,
                                                   fetch_list=fetch_list)
            total_cost.extend(np_loss * np_num_seqs)
            total_num_seqs.extend(np_num_seqs)
            all_logits.extend(np_logits)
            all_labels.extend(np_labels)
            if steps % (int(total_steps / 10)) == 0:
                print("Evaluation [{}/{}]".format(steps, total_steps))
            steps += 1
        except fluid.core.EOFException:
            test_data_loader.reset()
            break
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    if args.is_regression:
        key = "eval_pearsonr"
        eval_result, _ = pearsonr(all_logits, all_labels)
    else:
        key = "eval_accuracy"
        pred = np.argmax(all_logits, axis=1).reshape(all_labels.shape)
        eval_result = np.sum(pred == all_labels) / float(all_labels.size)
    time_end = time.time()
    print("[%s evaluation] ave loss: %f, %s: %f, elapsed time: %f s" %
          (eval_phase, np.sum(total_cost) / np.sum(total_num_seqs), key, eval_result,
           time_end - time_begin))

def predict(exe, predict_program, test_data_loader, task_name, label_list, fetch_list):
    test_data_loader.start()
    pred_cnt = 0
    predict_results = []
    with open(os.path.join(args.predict_dir, "{}.tsv".format(
        task_name)), "w") as fout:
        fout.write("index\tprediction\n")
        while True:
            try:
                np_logits = exe.run(program=predict_program,
                                           fetch_list=fetch_list)
                for result in np_logits[0]:
                    if pred_cnt % 1000 == 0:
                        print("Predicting submission for example: {}".format(
                     pred_cnt))

                    logits = [float(x) for x in result.flat]
                    predict_results.append(logits)

                    if len(logits) == 1:
                        label_out = logits[0]
                    elif len(logits) == 2:
                        if logits[1] - logits[0] > args.predict_threshold:
                            label_out = label_list[1]
                        else:
                            label_out = label_list[0]
                    elif len(logits) > 2:
                        max_index = np.argmax(np.array(logits, dtype=np.float32))
                        label_out = label_list[max_index]
                    else:
                        raise NotImplementedError

                    fout.write("{}\t{}\n".format(pred_cnt, label_out))
                    pred_cnt += 1

            except fluid.core.EOFException:
                test_data_loader.reset()
                break

    predict_json_path = os.path.join(args.predict_dir, "{}.logits.json".format(
        task_name))

    with open(predict_json_path, "w") as fp:
        json.dump(predict_results, fp, indent=4)

def get_device_num():
    # NOTE(zcd): for multi-processe training, each process use one GPU card.
    if num_trainers > 1 : return 1
    visible_device = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible_device:
        device_num = len(visible_device.split(','))
    else:
        device_num = subprocess.check_output(['nvidia-smi','-L']).decode().count('\n')
    return device_num

def main(args):
    if not (args.do_train or args.do_eval or args.do_predict):
        raise ValueError("For args `do_train`, `do_eval` and `do_predict`, at "
                         "least one of them must be True.")
    if args.do_predict and not args.predict_dir:
        raise ValueError("args 'predict_dir' should be given when doing predict")

    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)

    xlnet_config = XLNetConfig(args.model_config_path)
    xlnet_config.print_config()

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = get_device_num()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    task_name = args.task_name.lower()
    processors = {
      "mnli_matched": reader.MnliMatchedProcessor,
      "mnli_mismatched": reader.MnliMismatchedProcessor,
      'sts-b': reader.StsbProcessor,
      'imdb': reader.ImdbProcessor,
      "yelp5": reader.Yelp5Processor
    }

    processor = processors[task_name](args)

    label_list = processor.get_labels() if not args.is_regression else None
    num_labels = len(label_list) if label_list is not None else None
    train_program = fluid.Program()
    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed
        train_program.random_seed = args.random_seed

    if args.do_train:
        # NOTE: If num_trainers > 1, the shuffle_seed must be set, because
        # the order of batch data generated by reader
        # must be the same in the respective processes.
        shuffle_seed = 1 if num_trainers > 1 else None
        train_data_generator = processor.data_generator(
            batch_size=args.train_batch_size,
            is_regression=args.is_regression,
            phase='train',
            epoch=args.epoch,
            dev_count=dev_count,
            shuffle=args.shuffle)

        num_train_examples = processor.get_num_examples(phase='train')
        print("Device count: %d" % dev_count)
        print("Max num of epoches: %d" % args.epoch)
        print("Num of train examples: %d" % num_train_examples)
        print("Num of train steps: %d" % args.train_steps)
        print("Num of warmup steps: %d" % args.warmup_steps)

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_data_loader, loss, logits, num_seqs, label_ids = create_model(
                    args,
                    xlnet_config=xlnet_config,
                    n_class=num_labels)
                scheduled_lr = optimization(
                    loss=loss,
                    warmup_steps=args.warmup_steps,
                    num_train_steps=args.train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    lr_layer_decay_rate=args.lr_layer_decay_rate,
                    scheduler=args.lr_scheduler)

    if args.do_eval:
        dev_prog = fluid.Program()
        with fluid.program_guard(dev_prog, startup_prog):
            with fluid.unique_name.guard():
                dev_data_loader, loss, logits, num_seqs, label_ids = create_model(
                    args,
                    xlnet_config=xlnet_config,
                    n_class=num_labels)

        dev_prog = dev_prog.clone(for_test=True)
        dev_data_loader.set_batch_generator(
            processor.data_generator(
                batch_size=args.eval_batch_size,
                is_regression=args.is_regression,
                phase=args.eval_split,
                epoch=1,
                dev_count=1,
                shuffle=False), place)

    if args.do_predict:
        predict_prog = fluid.Program()
        with fluid.program_guard(predict_prog, startup_prog):
            with fluid.unique_name.guard():
                predict_data_loader, loss, logits, num_seqs, label_ids = create_model(
                    args,
                    xlnet_config=xlnet_config,
                    n_class=num_labels)

        predict_prog = predict_prog.clone(for_test=True)
        predict_data_loader.set_batch_generator(
            processor.data_generator(
                batch_size=args.predict_batch_size,
                is_regression=args.is_regression,
                phase=args.eval_split,
                epoch=1,
                dev_count=1,
                shuffle=False), place)

    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint and args.init_pretraining_params:
            print(
                "WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
                "both are set! Only arg 'init_checkpoint' is made valid.")
        if args.init_checkpoint:
            init_checkpoint(
                exe,
                args.init_checkpoint,
                main_program=startup_prog)
        elif args.init_pretraining_params:
            init_pretraining_params(
                exe,
                args.init_pretraining_params,
                main_program=startup_prog)
    elif args.do_eval or args.do_predict:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or testing!")
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=startup_prog)

    if args.do_train:
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = args.use_fast_executor
        exec_strategy.num_threads = dev_count
        build_strategy = fluid.BuildStrategy()

        if args.use_cuda and num_trainers > 1:
            assert shuffle_seed is not None
            dist_utils.prepare_for_multi_process(exe, build_strategy, train_program)
            train_data_generator = fluid.contrib.reader.distributed_batch_reader(
                  train_data_generator)

        train_compiled_program = fluid.CompiledProgram(train_program).with_data_parallel(
                 loss_name=loss.name, build_strategy=build_strategy)

        train_data_loader.set_batch_generator(train_data_generator, place)


    if args.do_train:
        train_data_loader.start()
        steps = 0
        total_cost, total_num_seqs, total_time = [], [], 0.0
        throughput = []
        ce_info = []
        while steps < args.train_steps:
            try:
                time_begin = time.time()
                steps += 1
                if steps % args.skip_steps == 0:
                    fetch_list = [loss.name, scheduled_lr.name, num_seqs.name]
                else:
                    fetch_list = []

                outputs = exe.run(train_compiled_program, fetch_list=fetch_list)

                time_end = time.time()
                used_time = time_end - time_begin
                total_time += used_time

                if steps % args.skip_steps == 0:
                    np_loss, np_lr, np_num_seqs = outputs

                    total_cost.extend(np_loss * np_num_seqs)
                    total_num_seqs.extend(np_num_seqs)

                    if args.verbose:
                        verbose = "train data_loader queue size: %d, " % train_data_loader.queue.size(
                        )
                        verbose += "learning rate: %f" % np_lr[0]
                        print(verbose)

                    current_example, current_epoch = processor.get_train_progress(
                    )

                    log_record = "epoch: {}, progress: {}/{}, step: {}, ave loss: {}".format(
                           current_epoch, current_example, num_train_examples,
                           steps, np.sum(total_cost) / np.sum(total_num_seqs))
                    ce_info.append([np.sum(total_cost) / np.sum(total_num_seqs), used_time])
                    if steps > 0 :
                        throughput.append( args.skip_steps / total_time)
                        log_record = log_record + ", speed: %f steps/s" % (args.skip_steps / total_time)
                        print(log_record)
                    else:
                        print(log_record)
                    total_cost, total_num_seqs, total_time = [], [], 0.0

                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoints,
                                             "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)

                if steps % args.validation_steps == 0:
                    print("Average throughtput: %s" % (np.average(throughput)))
                    throughput = []
                    # evaluate dev set
                    if args.do_eval:
                        evaluate(exe, dev_prog, dev_data_loader,
                                 [loss.name,  num_seqs.name, logits.name, label_ids.name],
                                 args.eval_split, processor.get_num_examples(phase=args.eval_split))
            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)
                train_data_loader.reset()
                break
        if args.enable_ce:
            card_num = get_cards()
            ce_cost = 0
            ce_time = 0
            try:
                ce_cost = ce_info[-2][0]
                ce_time = ce_info[-2][1]
            except:
                print("ce info error")
            print("kpis\ttrain_duration_%s_card%s\t%s" %
                (args.task_name.replace("-", "_"), card_num, ce_time))
            print("kpis\ttrain_cost_%s_card%s\t%f" %
                (args.task_name.replace("-", "_"), card_num, ce_cost))


    # final eval on dev set
    if args.do_eval:
        evaluate(exe, dev_prog, dev_data_loader,
                 [loss.name, num_seqs.name, logits.name, label_ids], args.eval_split,
                 processor.get_num_examples(phase=args.eval_split))

    # final eval on test set
    if args.do_predict:
        predict(exe, predict_prog, predict_data_loader, task_name, label_list, [logits.name])


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    print_arguments(args)
    check_cuda(args.use_cuda)
    main(args)
