# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import os
import random
import math
import contextlib

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.profiler as profiler
from paddle.fluid.executor import Executor
import reader

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import os

from args import *
from model import VAE
import logging
import pickle


@contextlib.contextmanager
def profile_context(profile=True):
    if profile:
        with profiler.profiler('All', 'total', 'seq2seq.profile'):
            yield
    else:
        yield


def main():
    args = parse_args()
    print(args)
    num_layers = args.num_layers
    src_vocab_size = args.vocab_size
    tar_vocab_size = args.vocab_size
    batch_size = args.batch_size
    init_scale = args.init_scale
    max_grad_norm = args.max_grad_norm
    hidden_size = args.hidden_size
    attr_init = args.attr_init
    latent_size = 32

    main_program = fluid.Program()
    startup_program = fluid.Program()
    if args.enable_ce:
        fluid.default_main_program().random_seed = 123
        framework.default_startup_program().random_seed = 123

    # Training process
    with fluid.program_guard(main_program, startup_program):
        with fluid.unique_name.guard():
            model = VAE(hidden_size,
                        latent_size,
                        src_vocab_size,
                        tar_vocab_size,
                        batch_size,
                        num_layers=num_layers,
                        init_scale=init_scale,
                        attr_init=attr_init)

            loss, kl_loss, rec_loss = model.build_graph()
            # clone from default main program and use it as the validation program
            main_program = fluid.default_main_program()
            inference_program = fluid.default_main_program().clone(
                for_test=True)

            clip=fluid.clip.GradientClipByGlobalNorm(
                    clip_norm=max_grad_norm)

            learning_rate = fluid.layers.create_global_var(
                name="learning_rate",
                shape=[1],
                value=float(args.learning_rate),
                dtype="float32",
                persistable=True)

            opt_type = args.optimizer
            if opt_type == "sgd":
                optimizer = fluid.optimizer.SGD(learning_rate, grad_clip=clip)
            elif opt_type == "adam":
                optimizer = fluid.optimizer.Adam(learning_rate, grad_clip=clip)
            else:
                print("only support [sgd|adam]")
                raise Exception("opt type not support")

            optimizer.minimize(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = Executor(place)
    exe.run(startup_program)

    train_program = fluid.compiler.CompiledProgram(main_program)

    dataset_prefix = args.dataset_prefix
    print("begin to load data")
    raw_data = reader.raw_data(dataset_prefix, args.max_len)
    print("finished load data")
    train_data, valid_data, test_data, _ = raw_data

    anneal_r = 1.0 / (args.warm_up * len(train_data) / args.batch_size)

    def prepare_input(batch, kl_weight=1.0, lr=None):
        src_ids, src_mask = batch
        res = {}
        src_ids = src_ids.reshape((src_ids.shape[0], src_ids.shape[1]))
        in_tar = src_ids[:, :-1]
        label_tar = src_ids[:, 1:]

        in_tar = in_tar.reshape((in_tar.shape[0], in_tar.shape[1]))
        label_tar = label_tar.reshape(
            (label_tar.shape[0], label_tar.shape[1], 1))

        res['src'] = src_ids
        res['tar'] = in_tar
        res['label'] = label_tar
        res['src_sequence_length'] = src_mask
        res['tar_sequence_length'] = src_mask - 1
        res['kl_weight'] = np.array([kl_weight]).astype(np.float32)
        if lr is not None:
            res['learning_rate'] = np.array([lr]).astype(np.float32)

        return res, np.sum(src_mask), np.sum(src_mask - 1)

    # get train epoch size
    def eval(data):
        eval_data_iter = reader.get_data_iter(data, batch_size, mode='eval')
        total_loss = 0.0
        word_count = 0.0
        batch_count = 0.0
        for batch_id, batch in enumerate(eval_data_iter):
            input_data_feed, src_word_num, dec_word_sum = prepare_input(batch)
            fetch_outs = exe.run(inference_program,
                                 feed=input_data_feed,
                                 fetch_list=[loss.name],
                                 use_program_cache=False)

            cost_train = np.array(fetch_outs[0])

            total_loss += cost_train * batch_size
            word_count += dec_word_sum
            batch_count += batch_size

        nll = total_loss / batch_count
        ppl = np.exp(total_loss / word_count)

        return nll, ppl

    def train():
        ce_time = []
        ce_ppl = []
        max_epoch = args.max_epoch
        kl_w = args.kl_start
        lr_w = args.learning_rate
        best_valid_nll = 1e100  # +inf
        best_epoch_id = -1
        decay_cnt = 0
        max_decay = args.max_decay
        decay_factor = 0.5
        decay_ts = 2
        steps_not_improved = 0
        for epoch_id in range(max_epoch):
            start_time = time.time()
            if args.enable_ce:
                train_data_iter = reader.get_data_iter(
                    train_data,
                    batch_size,
                    args.sort_cache,
                    args.cache_num,
                    enable_ce=True)
            else:
                train_data_iter = reader.get_data_iter(
                    train_data, batch_size, args.sort_cache, args.cache_num)

            total_loss = 0
            total_rec_loss = 0
            total_kl_loss = 0
            word_count = 0.0
            batch_count = 0.0
            batch_times = []
            for batch_id, batch in enumerate(train_data_iter):
                batch_start_time = time.time()
                kl_w = min(1.0, kl_w + anneal_r)
                kl_weight = kl_w
                input_data_feed, src_word_num, dec_word_sum = prepare_input(
                    batch, kl_weight, lr_w)
                fetch_outs = exe.run(
                    program=train_program,
                    feed=input_data_feed,
                    fetch_list=[loss.name, kl_loss.name, rec_loss.name],
                    use_program_cache=False)

                cost_train = np.array(fetch_outs[0])
                kl_cost_train = np.array(fetch_outs[1])
                rec_cost_train = np.array(fetch_outs[2])

                total_loss += cost_train * batch_size
                total_rec_loss += rec_cost_train * batch_size
                total_kl_loss += kl_cost_train * batch_size
                word_count += dec_word_sum
                batch_count += batch_size
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)

                if batch_id > 0 and batch_id % 200 == 0:
                    print("-- Epoch:[%d]; Batch:[%d]; Time: %.4f s; "
                          "kl_weight: %.4f; kl_loss: %.4f; rec_loss: %.4f; "
                          "nll: %.4f; ppl: %.4f" %
                          (epoch_id, batch_id, batch_time, kl_w, total_kl_loss /
                           batch_count, total_rec_loss / batch_count, total_loss
                           / batch_count, np.exp(total_loss / word_count)))
                    ce_ppl.append(np.exp(total_loss / word_count))

            end_time = time.time()
            epoch_time = end_time - start_time
            ce_time.append(epoch_time)
            print(
                "\nTrain epoch:[%d]; Epoch Time: %.4f; avg_time: %.4f s/step\n"
                % (epoch_id, epoch_time, sum(batch_times) / len(batch_times)))

            val_nll, val_ppl = eval(valid_data)
            print("dev ppl", val_ppl)
            test_nll, test_ppl = eval(test_data)
            print("test ppl", test_ppl)

            if val_nll < best_valid_nll:
                best_valid_nll = val_nll
                steps_not_improved = 0
                best_nll = test_nll
                best_ppl = test_ppl
                best_epoch_id = epoch_id
                save_path = os.path.join(args.model_path,
                                         "epoch_" + str(best_epoch_id),
                                         "checkpoint")
                print("save model {}".format(save_path))
                fluid.save(main_program, save_path)
            else:
                steps_not_improved += 1
                if steps_not_improved == decay_ts:
                    old_lr = lr_w
                    lr_w *= decay_factor
                    steps_not_improved = 0
                    new_lr = lr_w

                    print('-----\nchange lr, old lr: %f, new lr: %f\n-----' %
                          (old_lr, new_lr))

                    dir_name = args.model_path + "/epoch_" + str(best_epoch_id)
                    fluid.load(main_program, dir_name, exe)

                    decay_cnt += 1
                    if decay_cnt == max_decay:
                        break

        print('\nbest testing nll: %.4f, best testing ppl %.4f\n' %
              (best_nll, best_ppl))

        if args.enable_ce:
            card_num = get_cards()
            _ppl = 0
            _time = 0
            try:
                _time = ce_time[-1]
                _ppl = ce_ppl[-1]
            except:
                print("ce info error")
            print("kpis\ttrain_duration_card%s\t%s" % (card_num, _time))
            print("kpis\ttrain_ppl_card%s\t%f" % (card_num, _ppl))

    with profile_context(args.profile):
        train()


def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


def check_version():
    """
    Log error and exit when the installed version of paddlepaddle is
    not satisfied.
    """
    err = "PaddlePaddle version 1.6 or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code." \

    try:
        fluid.require_version('1.6.0')
    except Exception as e:
        logger.error(err)
        sys.exit(1)


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    check_version()
    main()
