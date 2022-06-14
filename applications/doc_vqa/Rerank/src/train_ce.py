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
"""Finetuning on classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import time
import logging
import warnings
import multiprocessing

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle
import paddle.fluid as fluid
if hasattr(paddle, 'enable_static'):
    paddle.enable_static()
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker

import reader_ce as reader_ce
from model.ernie import ErnieConfig
from cross_encoder import create_model, evaluate, predict
from optimization import optimization
from utils.args import print_arguments, check_cuda, prepare_logger
from utils.init import init_pretraining_params, init_checkpoint
from finetune_args import parser

warnings.filterwarnings("ignore")
args = parser.parse_args()
log = logging.getLogger()


def main(args):
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    if args.use_cuda:
        dev_list = fluid.cuda_places()
        place = dev_list[0]
        dev_count = len(dev_list)
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    reader = reader_ce.ClassifyReader(vocab_path=args.vocab_path,
                                      label_map_config=args.label_map_config,
                                      max_seq_len=args.max_seq_len,
                                      total_num=args.train_data_size,
                                      do_lower_case=args.do_lower_case,
                                      in_tokens=args.in_tokens,
                                      random_seed=args.random_seed,
                                      tokenizer=args.tokenizer,
                                      for_cn=args.for_cn,
                                      task_id=args.task_id)

    if not (args.do_train or args.do_val or args.do_test):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")

    if args.do_test:
        assert args.test_save is not None
    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.predict_batch_size == None:
        args.predict_batch_size = args.batch_size

    if args.do_train:
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        dev_count = fleet.worker_num()

        train_data_generator = reader.data_generator(
            input_file=args.train_set,
            batch_size=args.batch_size,
            epoch=args.epoch,
            dev_count=1,
            trainer_id=fleet.worker_index(),
            trainer_num=fleet.worker_num(),
            shuffle=True,
            phase="train")

        num_train_examples = reader.get_num_examples(args.train_set)

        if args.in_tokens:
            max_train_steps = args.epoch * num_train_examples // (
                args.batch_size // args.max_seq_len) // dev_count
        else:
            max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

        warmup_steps = int(max_train_steps * args.warmup_proportion)
        log.info("Device count: %d" % dev_count)
        log.info("Num train examples: %d" % num_train_examples)
        log.info("Max train steps: %d" % max_train_steps)
        log.info("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()

        # use fleet api
        exec_strategy = fluid.ExecutionStrategy()
        if args.use_fast_executor:
            exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = dev_count
        if args.is_distributed:
            exec_strategy.num_threads = 3

        exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

        dist_strategy = DistributedStrategy()
        dist_strategy.exec_strategy = exec_strategy
        dist_strategy.nccl_comm_num = 1
        if args.is_distributed:
            dist_strategy.nccl_comm_num = 2
        dist_strategy.use_hierarchical_allreduce = True

        if args.use_mix_precision:
            dist_strategy.use_amp = True

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, graph_vars = create_model(
                    args,
                    pyreader_name='train_reader',
                    ernie_config=ernie_config)
                scheduled_lr = optimization(
                    loss=graph_vars["loss"],
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    use_dynamic_loss_scaling=args.use_dynamic_loss_scaling,
                    incr_every_n_steps=args.incr_every_n_steps,
                    decr_every_n_nan_or_inf=args.decr_every_n_nan_or_inf,
                    incr_ratio=args.incr_ratio,
                    decr_ratio=args.decr_ratio,
                    dist_strategy=dist_strategy)

        if args.verbose:
            if args.in_tokens:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program,
                    batch_size=args.batch_size // args.max_seq_len)
            else:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program, batch_size=args.batch_size)
            log.info("Theoretical memory usage in training: %.3f - %.3f %s" %
                     (lower_mem, upper_mem, unit))

    if args.do_val or args.do_test:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, graph_vars = create_model(
                    args,
                    pyreader_name='test_reader',
                    ernie_config=ernie_config,
                    is_prediction=True)

        test_prog = test_prog.clone(for_test=True)

    train_program = fleet.main_program

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint and args.init_pretraining_params:
            log.warning(
                "WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
                "both are set! Only arg 'init_checkpoint' is made valid.")
        if args.init_checkpoint:
            init_checkpoint(exe,
                            args.init_checkpoint,
                            main_program=startup_prog)
        elif args.init_pretraining_params:
            init_pretraining_params(exe,
                                    args.init_pretraining_params,
                                    main_program=startup_prog)
    elif args.do_val or args.do_test:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or testing!")
        init_checkpoint(exe, args.init_checkpoint, main_program=startup_prog)

    if args.do_train:
        train_exe = exe
        train_pyreader.decorate_tensor_provider(train_data_generator)
    else:
        train_exe = None

    test_exe = exe

    current_epoch = 0
    steps = 0
    if args.do_train:
        train_pyreader.start()
        if warmup_steps > 0:
            graph_vars["learning_rate"] = scheduled_lr

        ce_info = []
        time_begin = time.time()
        last_epoch = 0
        while True:
            try:
                steps += 1

                if fleet.worker_index() != 0:
                    train_exe.run(fetch_list=[], program=train_program)
                    continue

                if steps % args.skip_steps != 0:
                    train_exe.run(fetch_list=[], program=train_program)

                else:
                    outputs = evaluate(train_exe,
                                       train_program,
                                       train_pyreader,
                                       graph_vars,
                                       "train",
                                       metric=args.metric)

                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size(
                        )
                        verbose += "learning rate: %f" % (
                            outputs["learning_rate"]
                            if warmup_steps > 0 else args.learning_rate)
                        log.info(verbose)

                    current_example, current_epoch = reader.get_train_progress()
                    time_end = time.time()
                    used_time = time_end - time_begin

                    log.info(
                        "epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                        "ave acc: %f, speed: %f steps/s" %
                        (current_epoch, current_example * dev_count,
                         num_train_examples, steps, outputs["loss"],
                         outputs["accuracy"], args.skip_steps / used_time))
                    ce_info.append(
                        [outputs["loss"], outputs["accuracy"], used_time])

                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoints,
                                             "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path,
                                               fleet._origin_program)

                if steps % args.validation_steps == 0:
                    # evaluate dev set
                    if args.do_val:
                        evaluate_wrapper(args, reader, exe, test_prog,
                                         test_pyreader, graph_vars,
                                         current_epoch, steps)

                    if args.do_test:
                        predict_wrapper(args, reader, exe, test_prog,
                                        test_pyreader, graph_vars,
                                        current_epoch, steps)

                if last_epoch != current_epoch:
                    last_epoch = current_epoch

            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path,
                                           fleet._origin_program)
                train_pyreader.reset()
                break

    # final eval on dev set
    if args.do_val:
        evaluate_wrapper(args, reader, exe, test_prog, test_pyreader,
                         graph_vars, current_epoch, steps)

    # final eval on test set
    if args.do_test:
        predict_wrapper(args, reader, exe, test_prog, test_pyreader, graph_vars)

    # final eval on dianostic, hack for glue-ax
    if args.diagnostic:
        test_pyreader.decorate_tensor_provider(
            reader.data_generator(args.diagnostic,
                                  batch_size=args.batch_size,
                                  epoch=1,
                                  dev_count=1,
                                  shuffle=False))

        log.info("Final diagnostic")
        qids, preds, probs = predict(test_exe, test_prog, test_pyreader,
                                     graph_vars)
        assert len(qids) == len(preds), '{} v.s. {}'.format(
            len(qids), len(preds))
        with open(args.diagnostic_save, 'w') as f:
            for id, s, p in zip(qids, preds, probs):
                f.write('{}\t{}\t{}\n'.format(id, s, p))

        log.info("Done final diagnostic, saving to {}".format(
            args.diagnostic_save))


def evaluate_wrapper(args, reader, exe, test_prog, test_pyreader, graph_vars,
                     epoch, steps):
    # evaluate dev set
    for ds in args.dev_set.split(','):
        test_pyreader.decorate_tensor_provider(
            reader.data_generator(ds,
                                  batch_size=args.predict_batch_size,
                                  epoch=1,
                                  dev_count=1,
                                  shuffle=False))
        log.info("validation result of dataset {}:".format(ds))
        evaluate_info = evaluate(exe,
                                 test_prog,
                                 test_pyreader,
                                 graph_vars,
                                 "dev",
                                 metric=args.metric)
        log.info(evaluate_info +
                 ', file: {}, epoch: {}, steps: {}'.format(ds, epoch, steps))


def predict_wrapper(args,
                    reader,
                    exe,
                    test_prog,
                    test_pyreader,
                    graph_vars,
                    epoch=None,
                    steps=None):
    test_sets = args.test_set.split(',')
    save_dirs = args.test_save.split(',')
    assert len(test_sets) == len(save_dirs)

    for test_f, save_f in zip(test_sets, save_dirs):
        test_pyreader.decorate_tensor_provider(
            reader.data_generator(test_f,
                                  batch_size=args.predict_batch_size,
                                  epoch=1,
                                  dev_count=1,
                                  shuffle=False))

        if epoch is not None or steps is not None:
            save_path = save_f + '.' + str(epoch) + '.' + str(steps)
        else:
            save_path = save_f
        log.info("testing {}, save to {}".format(test_f, save_path))
        qids, preds, probs = predict(exe, test_prog, test_pyreader, graph_vars)

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            log.warning('save dir exsits: %s, will skip saving' % save_dir)

        with open(save_path, 'w') as f:
            for p in probs:
                f.write('{}\n'.format(p[1]))


if __name__ == '__main__':
    prepare_logger(log)
    print_arguments(args)
    check_cuda(args.use_cuda)
    main(args)
