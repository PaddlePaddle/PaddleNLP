import os
import time

import yaml
import argparse
import numpy as np
from pprint import pprint
from attrdict import AttrDict
import inspect

import paddle
import paddle.distributed as dist

import reader
from paddlenlp.transformers import TransformerModel, CrossEntropyCriterion
from paddlenlp.utils.log import logger
from paddlenlp.utils import profiler

from tls.record import AverageStatistical
from tls.to_static import apply_to_static


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
    parser.add_argument("--batch_size",
                        default=None,
                        type=int,
                        help="The maximum tokens per batch. ")
    parser.add_argument("--use_amp",
                        default=None,
                        type=str,
                        choices=['true', 'false', 'True', 'False'],
                        help="Whether to use amp to train Transformer. ")
    parser.add_argument(
        "--amp_level",
        default=None,
        type=str,
        choices=['O1', 'O2'],
        help="The amp level if --use_amp is on. Can be one of [O1, O2]. ")

    # For benchmark.
    parser.add_argument(
        '--profiler_options',
        type=str,
        default=None,
        help=
        'The option of profiler, which should be in format \"key1=value1;key2=value2;key3=value3\".'
    )
    args = parser.parse_args()
    return args


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

    transformer = apply_to_static(args, transformer)

    # Define loss
    criterion = CrossEntropyCriterion(args.label_smooth_eps, args.bos_idx)

    scheduler = paddle.optimizer.lr.NoamDecay(args.d_model,
                                              args.warmup_steps,
                                              args.learning_rate,
                                              last_epoch=0)

    # Define optimizer
    if 'use_multi_tensor' not in inspect.getfullargspec(
            paddle.optimizer.Adam.__init__).args:
        optimizer = paddle.optimizer.Adam(learning_rate=scheduler,
                                          beta1=args.beta1,
                                          beta2=args.beta2,
                                          epsilon=float(args.eps),
                                          parameters=transformer.parameters())
    else:
        optimizer = paddle.optimizer.Adam(learning_rate=scheduler,
                                          beta1=args.beta1,
                                          beta2=args.beta2,
                                          epsilon=float(args.eps),
                                          parameters=transformer.parameters(),
                                          use_multi_tensor=True)

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

    # for amp training
    if args.use_amp:
        amp_level = 'O2' if args.use_pure_fp16 else 'O1'
        scaler = paddle.amp.GradScaler(enable=True,
                                       init_loss_scaling=args.scale_loss)
        transformer = paddle.amp.decorate(models=transformer,
                                          level=amp_level,
                                          save_dtype='float32')

    # for distributed training
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
                with paddle.amp.auto_cast(custom_black_list={
                        'scale', 'reduce_sum', 'elementwise_div'
                } if amp_level == 'O2' else {},
                                          level=amp_level):
                    logits = transformer(src_word=src_word, trg_word=trg_word)
                    sum_cost, avg_cost, token_num = criterion(logits, lbl_word)

                tokens_per_cards = token_num.numpy()
                scaled = scaler.scale(avg_cost)  # scale the loss
                scaled.backward()  # do backward

                scaler.minimize(optimizer, scaled)  # update parameters
                if 'set_to_zero' in inspect.getfullargspec(
                        optimizer.clear_grad).args:
                    optimizer.clear_grad(set_to_zero=False)
                else:
                    optimizer.clear_grad()
            else:
                logits = transformer(src_word=src_word, trg_word=trg_word)
                sum_cost, avg_cost, token_num = criterion(logits, lbl_word)
                tokens_per_cards = token_num.numpy()

                avg_cost.backward()

                optimizer.step()
                optimizer.clear_grad()

            train_batch_cost = time.time() - batch_start
            reader_cost_avg.record(train_reader_cost)
            batch_cost_avg.record(train_batch_cost)
            batch_ips_avg.record(train_batch_cost, tokens_per_cards)

            # Profile for model benchmark
            if args.profiler_options is not None:
                profiler.add_profiler_step(args.profiler_options)

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
                        if args.use_amp:
                            with paddle.amp.auto_cast(custom_black_list={
                                    'scale', 'reduce_sum', 'elementwise_div'
                            } if amp_level == 'O2' else {},
                                                      level=amp_level):
                                logits = transformer(src_word=src_word,
                                                     trg_word=trg_word)
                                sum_cost, avg_cost, token_num = criterion(
                                    logits, lbl_word)

                        else:
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
        paddle.save(transformer.state_dict(),
                    os.path.join(model_dir, "transformer.pdparams"))
        paddle.save(optimizer.state_dict(),
                    os.path.join(model_dir, "transformer.pdopt"))


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
    args.benchmark = ARGS.benchmark
    if ARGS.max_iter:
        args.max_iter = ARGS.max_iter
    if ARGS.batch_size:
        args.batch_size = ARGS.batch_size
    if ARGS.use_amp:
        ARGS.use_amp = ARGS.use_amp.lower()
        if ARGS.use_amp == "true":
            args.use_amp = True
        else:
            args.use_amp = False
    if ARGS.amp_level:
        args.use_pure_fp16 = ARGS.amp_level == 'O2'
    args.train_file = ARGS.train_file
    args.dev_file = ARGS.dev_file
    args.vocab_file = ARGS.vocab_file
    args.unk_token = ARGS.unk_token
    args.bos_token = ARGS.bos_token
    args.eos_token = ARGS.eos_token
    pprint(args)

    args.profiler_options = ARGS.profiler_options

    do_train(args)
