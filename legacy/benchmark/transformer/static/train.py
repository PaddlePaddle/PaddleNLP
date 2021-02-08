import os
import time
import sys

import argparse
import logging
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


def batch_creator(loader, trainer_count):
    batch = []
    for data in loader:
        batch.append(data)
        if len(batch) == trainer_count:
            yield batch
            batch = []
    # DO NOT drop last.
    if len(batch) > 0:
        while len(batch) < trainer_count:
            batch.append(batch[-1])
        yield batch


def do_train(args):
    paddle.enable_static()
    if args.use_gpu:
        trainer_count = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        place = paddle.set_device("gpu:0")
    else:
        trainer_count = int(os.environ['CPU_NUM'])
        place = paddle.set_device("cpu")

    # Set seed for CE
    random_seed = eval(str(args.random_seed))
    if random_seed is not None:
        paddle.seed(random_seed)

    # Define data loader
    # NOTE: To guarantee all data is involved, use world_size=1 and rank=0. 
    (train_loader, train_steps_fn), (
        eval_loader, eval_steps_fn) = reader.create_data_loader(args)

    train_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(train_program, startup_program):
        src_word = paddle.static.data(
            name="src_word", shape=[None, None], dtype="int64")
        trg_word = paddle.static.data(
            name="trg_word", shape=[None, None], dtype="int64")
        lbl_word = paddle.static.data(
            name="lbl_word", shape=[None, None, 1], dtype="int64")

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

        logits = transformer(src_word=src_word, trg_word=trg_word)

        sum_cost, avg_cost, token_num = criterion(logits, lbl_word)

        scheduler = paddle.optimizer.lr.NoamDecay(
            args.d_model, args.warmup_steps, args.learning_rate, last_epoch=0)

        # Define optimizer
        optimizer = paddle.optimizer.Adam(
            learning_rate=scheduler,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=float(args.eps),
            parameters=transformer.parameters())

        optimizer.minimize(avg_cost)

    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    build_strategy = paddle.static.BuildStrategy()
    build_strategy.enable_inplace = True
    exec_strategy = paddle.static.ExecutionStrategy()

    compiled_train_program = paddle.static.CompiledProgram(
        train_program).with_data_parallel(
            loss_name=avg_cost.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    # the best cross-entropy value with label smoothing
    loss_normalizer = -(
        (1. - args.label_smooth_eps) * np.log(
            (1. - args.label_smooth_eps)) + args.label_smooth_eps *
        np.log(args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20))

    step_idx = 0

    # For benchmark
    reader_cost_avg = AverageStatistical()
    batch_cost_avg = AverageStatistical()
    batch_ips_avg = AverageStatistical()

    for pass_id in range(args.epoch):
        batch_id = 0
        batch_start = time.time()
        pass_start_time = batch_start
        for data in batch_creator(train_loader, trainer_count):
            # NOTE: used for benchmark and use None as default.
            if args.max_iter and step_idx == args.max_iter:
                return
            train_reader_cost = time.time() - batch_start

            outs = exe.run(compiled_train_program,
                           feed=[{
                               'src_word': data[i][0],
                               'trg_word': data[i][1],
                               'lbl_word': data[i][2],
                           } for i in range(trainer_count)],
                           fetch_list=[sum_cost.name, token_num.name])
            scheduler.step()

            train_batch_cost = time.time() - batch_start
            reader_cost_avg.record(train_reader_cost)
            batch_cost_avg.record(train_batch_cost)
            batch_ips_avg.record(train_batch_cost, np.asarray(outs[1]).sum())

            if step_idx % args.print_step == 0:
                sum_cost_val, token_num_val = np.array(outs[0]), np.array(outs[
                    1])
                # Sum the cost from multi-devices
                total_sum_cost = sum_cost_val.sum()
                total_token_num = token_num_val.sum()
                total_avg_cost = total_sum_cost / total_token_num

                if step_idx == 0:
                    logging.info(
                        "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f" %
                        (step_idx, pass_id, batch_id, total_avg_cost,
                         total_avg_cost - loss_normalizer,
                         np.exp([min(total_avg_cost, 100)])))
                else:
                    train_avg_batch_cost = args.print_step / batch_cost_avg.get_total_time(
                    )
                    logging.info(
                        "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f, avg_speed: %.2f step/s, "
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
                if args.save_model:
                    model_path = os.path.join(
                        args.save_model, "step_" + str(step_idx), "transformer")
                    paddle.io.save(train_program, model_path)

            batch_id += 1
            step_idx += 1
            batch_start = time.time()

    paddle.disable_static()


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)

    do_train(args)
