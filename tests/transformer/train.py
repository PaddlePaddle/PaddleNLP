import os
import sys
import time

import yaml
import argparse
import numpy as np
from pprint import pprint
from attrdict import AttrDict

import paddle
import paddle.distributed as dist

from modeling import TransformerModel, CrossEntropyCriterion
from paddlenlp.utils.log import logger

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                     "examples", "machine_translation", "transformer")))
import reader
from tls.record import AverageStatistical

paddle.set_default_dtype("float64")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="./configs/transformer.big.yaml",
                        type=str,
                        help="Path of the config file. ")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=
        "Whether to print logs on each cards and use benchmark vocab. Normally, not necessary to set --benchmark. "
    )
    parser.add_argument("--max_iter",
                        default=None,
                        type=int,
                        help="The maximum iteration for training. ")
    parser.add_argument(
        "--train_file",
        nargs='+',
        default=None,
        type=str,
        help=
        "The files for training, including [source language file, target language file]. Normally, it shouldn't be set and in this case, the default WMT14 dataset will be used to train. "
    )
    parser.add_argument(
        "--dev_file",
        nargs='+',
        default=None,
        type=str,
        help=
        "The files for validation, including [source language file, target language file]. Normally, it shouldn't be set and in this case, the default WMT14 dataset will be used to do validation. "
    )
    parser.add_argument(
        "--vocab_file",
        default=None,
        type=str,
        help=
        "The vocab file. Normally, it shouldn't be set and in this case, the default WMT14 dataset will be used."
    )
    parser.add_argument(
        "--unk_token",
        default=None,
        type=str,
        help=
        "The unknown token. It should be provided when use custom vocab_file. ")
    parser.add_argument(
        "--bos_token",
        default=None,
        type=str,
        help="The bos token. It should be provided when use custom vocab_file. "
    )
    parser.add_argument(
        "--eos_token",
        default=None,
        type=str,
        help="The eos token. It should be provided when use custom vocab_file. "
    )
    args = parser.parse_args()
    return args


def transfer_param(state_dict):
    for item in state_dict:
        state_dict[item] = paddle.cast(state_dict[item], dtype="float32")
    return state_dict


def do_train(args):
    if args.device == "gpu":
        rank = dist.get_rank()
        trainer_count = dist.get_world_size()
    else:
        rank = 0
        trainer_count = 1
        paddle.set_device("cpu")

    if trainer_count > 1:
        dist.init_parallel_env()

    # Set seed for CE
    random_seed = eval(str(args.random_seed))
    if random_seed is not None:
        paddle.seed(random_seed)

    # Define data loader
    (train_loader), (eval_loader) = reader.create_data_loader(args)

    # Define model
    transformer = TransformerModel(src_vocab_size=args.src_vocab_size,
                                   trg_vocab_size=args.trg_vocab_size,
                                   max_length=args.max_length + 1,
                                   num_encoder_layers=args.n_layer,
                                   num_decoder_layers=args.n_layer,
                                   n_head=args.n_head,
                                   d_model=args.d_model,
                                   d_inner_hid=args.d_inner_hid,
                                   dropout=args.dropout,
                                   weight_sharing=args.weight_sharing,
                                   bos_id=args.bos_idx,
                                   eos_id=args.eos_idx)

    # Define loss
    criterion = CrossEntropyCriterion(args.label_smooth_eps, args.bos_idx)

    scheduler = paddle.optimizer.lr.NoamDecay(args.d_model,
                                              args.warmup_steps,
                                              args.learning_rate,
                                              last_epoch=0)

    # Define optimizer
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
            train_reader_cost = time.time() - batch_start
            (src_word, trg_word, lbl_word) = input_data

            if args.use_amp:
                scaler = paddle.amp.GradScaler(
                    init_loss_scaling=args.scale_loss)
                with paddle.amp.auto_cast():
                    logits = transformer(src_word=src_word, trg_word=trg_word)
                    sum_cost, avg_cost, token_num = criterion(logits, lbl_word)

                scaled = scaler.scale(avg_cost)  # scale the loss
                scaled.backward()  # do backward

                scaler.minimize(optimizer, scaled)  # update parameters
                optimizer.clear_grad()
            else:
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
            if step_idx % args.print_step == 0 and (args.benchmark
                                                    or rank == 0):
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

            #NOTE: Used for benchmark and use None as default.
            if args.max_iter and step_idx == args.max_iter:
                break
            batch_id += 1
            step_idx += 1
            scheduler.step()
            batch_start = time.time()

        #NOTE: Used for benchmark and use None as default.
        if args.max_iter and step_idx == args.max_iter:
            break

        train_epoch_cost = time.time() - epoch_start
        logger.info("train epoch: %d, epoch_cost: %.5f s" %
                    (pass_id, train_epoch_cost))

    if args.save_model and rank == 0:
        model_dir = os.path.join(args.save_model, "step_final")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # Transform dtype from float64 to float32,
        # since some pass during inference doesn't
        # support float64 kernel.
        param_sd = transfer_param(transformer.state_dict())
        paddle.save(param_sd, os.path.join(model_dir, "transformer.pdparams"))

        optim_sd = transfer_param(transformer.state_dict())
        paddle.save(optim_sd, os.path.join(model_dir, "transformer.pdopt"))


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
    args.benchmark = ARGS.benchmark
    if ARGS.max_iter:
        args.max_iter = ARGS.max_iter
    args.train_file = ARGS.train_file
    args.dev_file = ARGS.dev_file
    args.vocab_file = ARGS.vocab_file
    args.unk_token = ARGS.unk_token
    args.bos_token = ARGS.bos_token
    args.eos_token = ARGS.eos_token
    pprint(args)

    do_train(args)
