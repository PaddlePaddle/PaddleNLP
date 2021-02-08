import os
import time
import sys

import logging
import argparse
import numpy as np
import yaml
from attrdict import AttrDict
from pprint import pprint

import paddle
import paddle.distributed as dist

from paddlenlp.transformers import TransformerModel, CrossEntropyCriterion, position_encoding_init

sys.path.append("../")
import reader
from utils.record import AverageStatistical

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../configs/transformer.big.yaml",
        type=str,
        help="Path of the config file. ")
    args = parser.parse_args()
    return args


def do_train(args):
    if args.use_gpu:
        rank = dist.get_rank()
        trainer_count = dist.get_world_size()
    else:
        rank = 0
        trainer_count = 1

    if trainer_count > 1:
        dist.init_parallel_env()

    # Set seed for CE
    random_seed = eval(str(args.random_seed))
    if random_seed is not None:
        paddle.seed(random_seed)

    # Define data loader
    (train_loader, train_steps_fn), (eval_loader,
                                     eval_steps_fn) = reader.create_data_loader(
                                         args, trainer_count, rank)

    # Define model
    transformer = TransformerModel(
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
        max_length=args.max_length + 1,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        weight_sharing=args.weight_sharing,
        bos_id=args.bos_idx,
        eos_id=args.eos_idx)

    # Define loss
    criterion = CrossEntropyCriterion(args.label_smooth_eps, args.bos_idx)

    scheduler = paddle.optimizer.lr.NoamDecay(
        args.d_model, args.warmup_steps, args.learning_rate, last_epoch=0)

    # Define optimizer
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler,
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
        (1. - args.label_smooth_eps) * np.log(
            (1. - args.label_smooth_eps)) + args.label_smooth_eps *
        np.log(args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20))

    step_idx = 0

    # For benchmark
    reader_cost_avg = AverageStatistical()
    batch_cost_avg = AverageStatistical()
    batch_ips_avg = AverageStatistical()

    # Train loop
    for pass_id in range(args.epoch):
        epoch_start = time.time()

        batch_id = 0
        batch_start = time.time()
        for input_data in train_loader:
            #NOTE: Used for benchmark and use None as default. 
            if args.max_iter and step_idx == args.max_iter:
                return
            train_reader_cost = time.time() - batch_start
            (src_word, trg_word, lbl_word) = input_data

            logits = transformer(src_word=src_word, trg_word=trg_word)

            sum_cost, avg_cost, token_num = criterion(logits, lbl_word)

            avg_cost.backward()

            optimizer.step()
            optimizer.clear_grad()

            tokens_per_cards = token_num.numpy()

            train_batch_cost = time.time() - batch_start
            reader_cost_avg.record(train_reader_cost)
            batch_cost_avg.record(train_batch_cost)
            batch_ips_avg.record(train_batch_cost, tokens_per_cards)

            # NOTE: For benchmark, loss infomation on all cards will be printed.
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
                    train_avg_batch_cost = args.print_step / batch_cost_avg.get_total_time(
                    )
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
                if args.validation_file:
                    transformer.eval()
                    total_sum_cost = 0
                    total_token_num = 0
                    with paddle.no_grad():
                        for input_data in eval_loader:
                            (src_word, trg_word, lbl_word) = input_data
                            logits = transformer(
                                src_word=src_word, trg_word=trg_word)
                            sum_cost, avg_cost, token_num = criterion(logits,
                                                                      lbl_word)
                            total_sum_cost += sum_cost.numpy()
                            total_token_num += token_num.numpy()
                            total_avg_cost = total_sum_cost / total_token_num
                        logger.info("validation, step_idx: %d, avg loss: %f, "
                                    "normalized loss: %f, ppl: %f" %
                                    (step_idx, total_avg_cost,
                                     total_avg_cost - loss_normalizer,
                                     np.exp([min(total_avg_cost, 100)])))
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
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)

    do_train(args)
