# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import time

import argparse
from pprint import pprint
import numpy as np
import yaml
from attrdict import AttrDict

import paddle
import paddle.distributed as dist
from paddlenlp.utils.log import logger

import reader
from model import SimultaneousTransformer, CrossEntropyCriterion
from utils.record import AverageStatistical


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="./config/transformer.yaml",
                        type=str,
                        help="Path of the config file. ")
    args = parser.parse_args()
    return args


def do_train(args):
    paddle.set_device(args.device)
    trainer_count = dist.get_world_size()
    rank = dist.get_rank()

    if trainer_count > 1:
        dist.init_parallel_env()

    # Set seed for CE
    random_seed = eval(str(args.random_seed))
    if random_seed is not None:
        paddle.seed(random_seed)

    # Define data loader
    (train_loader), (eval_loader) = reader.create_data_loader(
        args, places=paddle.get_device())

    # Define model
    transformer = SimultaneousTransformer(
        args.src_vocab_size, args.trg_vocab_size, args.max_length + 1,
        args.n_layer, args.n_head, args.d_model, args.d_inner_hid, args.dropout,
        args.weight_sharing, args.bos_idx, args.eos_idx, args.waitk)

    print('waitk=', args.waitk)

    # Define loss
    criterion = CrossEntropyCriterion(args.label_smooth_eps, args.bos_idx)

    # Define optimizer
    scheduler = paddle.optimizer.lr.NoamDecay(args.d_model, args.warmup_steps,
                                              args.learning_rate)

    optimizer = paddle.optimizer.Adam(learning_rate=scheduler,
                                      beta1=args.beta1,
                                      beta2=args.beta2,
                                      epsilon=float(args.eps),
                                      parameters=transformer.parameters())

    # Init from some checkpoint, to resume the previous training
    if args.init_from_checkpoint:
        model_dict = paddle.load(
            os.path.join(args.init_from_checkpoint, "transformer.pdparams"))
        opt_dict = paddle.load(
            os.path.join(args.init_from_checkpoint, "transformer.pdopt"))
        transformer.set_state_dict(model_dict)
        optimizer.set_state_dict(opt_dict)
        print("loaded from checkpoint.")
    # Init from some pretrain models, to better solve the current task
    if args.init_from_pretrain_model:
        model_dict = paddle.load(
            os.path.join(args.init_from_pretrain_model, "transformer.pdparams"))
        transformer.set_state_dict(model_dict)
        print("loaded from pre-trained model.")

    if trainer_count > 1:
        transformer = paddle.DataParallel(transformer)

    # The best cross-entropy value with label smoothing
    loss_normalizer = -(
        (1. - args.label_smooth_eps) * np.log((1. - args.label_smooth_eps)) +
        args.label_smooth_eps * np.log(args.label_smooth_eps /
                                       (args.trg_vocab_size - 1) + 1e-20))

    step_idx = 0

    # For logging
    reader_cost_avg = AverageStatistical()
    batch_cost_avg = AverageStatistical()
    batch_ips_avg = AverageStatistical()

    # Train loop
    for pass_id in range(args.epoch):
        epoch_start = time.time()
        batch_id = 0
        batch_start = time.time()
        for input_data in train_loader:
            train_reader_cost = time.time() - batch_start
            (src_word, trg_word, lbl_word) = input_data
            if args.use_amp:
                scaler = paddle.amp.GradScaler(
                    init_loss_scaling=args.scale_loss)
                with paddle.amp.auto_cast():
                    logits = transformer(src_word=src_word, trg_word=trg_word)
                    sum_cost, avg_cost, token_num = criterion(logits, lbl_word)

                scaled_loss = scaler.scale(avg_cost)  # scale the loss
                scaled_loss.backward()  # do backward

                scaler.minimize(optimizer, scaled_loss)  # update parameters
                optimizer.clear_grad()
            else:
                logits = transformer(src_word=src_word, trg_word=trg_word)
                sum_cost, avg_cost, token_num = criterion(logits, lbl_word)

                avg_cost.backward()

                optimizer.step()
                optimizer.clear_grad()

            if args.max_iter and step_idx + 1 == args.max_iter:
                return

            tokens_per_cards = token_num.numpy()

            train_batch_cost = time.time() - batch_start
            reader_cost_avg.record(train_reader_cost)
            batch_cost_avg.record(train_batch_cost)
            batch_ips_avg.record(train_batch_cost, tokens_per_cards)

            if step_idx % args.print_step == 0:
                total_avg_cost = avg_cost.numpy()
                if step_idx == 0:
                    logger.info(
                        "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f " %
                        (step_idx, pass_id, batch_id, total_avg_cost,
                         total_avg_cost - loss_normalizer,
                         np.exp([min(total_avg_cost, 100)])))
                else:
                    train_avg_batch_cost = args.print_step / \
                                           batch_cost_avg.get_total_time()
                    logger.info(
                        "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f, avg_speed: %.2f step/sec, "
                        "batch_cost: %.5f sec, reader_cost: %.5f sec, tokens: %d, "
                        "ips: %.5f words/sec" %
                        (step_idx, pass_id, batch_id, total_avg_cost,
                         total_avg_cost - loss_normalizer,
                         np.exp([min(total_avg_cost, 100)]),
                         train_avg_batch_cost, batch_cost_avg.get_average(),
                         reader_cost_avg.get_average(),
                         batch_ips_avg.get_total_cnt(),
                         batch_ips_avg.get_average_per_sec()))
                reader_cost_avg.reset()
                batch_cost_avg.reset()
                batch_ips_avg.reset()

            if step_idx % args.save_step == 0 and step_idx != 0:
                # Validation
                transformer.eval()
                total_sum_cost = 0
                total_token_num = 0
                with paddle.no_grad():
                    for input_data in eval_loader:
                        (src_word, trg_word, lbl_word) = input_data
                        logits = transformer(src_word=src_word,
                                             trg_word=trg_word)
                        sum_cost, avg_cost, token_num = criterion(
                            logits, lbl_word)
                        total_sum_cost += sum_cost.numpy()
                        total_token_num += token_num.numpy()
                        total_avg_cost = total_sum_cost / total_token_num
                    logger.info(
                        "validation, step_idx: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f" %
                        (step_idx, total_avg_cost, total_avg_cost -
                         loss_normalizer, np.exp([min(total_avg_cost, 100)])))
                transformer.train()

                if args.save_model and rank == 0:
                    model_dir = os.path.join(args.save_model,
                                             "step_" + str(step_idx))
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    paddle.save(transformer.state_dict(),
                                os.path.join(model_dir, "transformer.pdparams"))
                    paddle.save(optimizer.state_dict(),
                                os.path.join(model_dir, "transformer.pdopt"))

            batch_id += 1
            step_idx += 1
            scheduler.step()
            batch_start = time.time()

        train_epoch_cost = time.time() - epoch_start
        logger.info("train epoch: %d, epoch_cost: %.5f s" %
                    (pass_id, train_epoch_cost))

    if args.save_model and rank == 0:
        model_dir = os.path.join(args.save_model, "step_final")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        paddle.save(transformer.state_dict(),
                    os.path.join(model_dir, "transformer.pdparams"))
        paddle.save(optimizer.state_dict(),
                    os.path.join(model_dir, "transformer.pdopt"))


if __name__ == "__main__":
    args = parse_args()
    yaml_file = args.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)
    do_train(args)
