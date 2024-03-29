# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import sys
import time
from pprint import pprint

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import yaml
from easydict import EasyDict as AttrDict

from paddlenlp.transformers import CrossEntropyCriterion, TransformerModel
from paddlenlp.utils import profiler
from paddlenlp.utils.log import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import reader  # noqa: E402
from tls.record import AverageStatistical  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="../configs/transformer.big.yaml", type=str, help="Path of the config file. "
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Whether to print logs on each cards and use benchmark vocab. Normally, not necessary to set --benchmark. ",
    )
    parser.add_argument("--distributed", action="store_true", help="Whether to use fleet to launch. ")
    parser.add_argument("--max_iter", default=None, type=int, help="The maximum iteration for training. ")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The dir of train, dev and test datasets. If data_dir is given, train_file and dev_file and test_file will be replaced by data_dir/[train|dev|test].\{src_lang\}-\{trg_lang\}.[\{src_lang\}|\{trg_lang\}]. ",
    )
    parser.add_argument(
        "--train_file",
        nargs="+",
        default=None,
        type=str,
        help="The files for training, including [source language file, target language file]. If it's None, the default WMT14 en-de dataset will be used. ",
    )
    parser.add_argument(
        "--dev_file",
        nargs="+",
        default=None,
        type=str,
        help="The files for validation, including [source language file, target language file]. If it's None, the default WMT14 en-de dataset will be used. ",
    )
    parser.add_argument(
        "--vocab_file",
        default=None,
        type=str,
        help="The vocab file. Normally, it shouldn't be set and in this case, the default WMT14 dataset will be used.",
    )
    parser.add_argument(
        "--src_vocab",
        default=None,
        type=str,
        help="The vocab file for source language. If --vocab_file is given, the --vocab_file will be used. ",
    )
    parser.add_argument(
        "--trg_vocab",
        default=None,
        type=str,
        help="The vocab file for target language. If --vocab_file is given, the --vocab_file will be used. ",
    )
    parser.add_argument("-s", "--src_lang", default=None, type=str, help="Source language. ")
    parser.add_argument("-t", "--trg_lang", default=None, type=str, help="Target language. ")
    parser.add_argument(
        "--unk_token",
        default=None,
        type=str,
        help="The unknown token. It should be provided when use custom vocab_file. ",
    )
    parser.add_argument(
        "--bos_token", default=None, type=str, help="The bos token. It should be provided when use custom vocab_file. "
    )
    parser.add_argument(
        "--eos_token", default=None, type=str, help="The eos token. It should be provided when use custom vocab_file. "
    )
    parser.add_argument(
        "--pad_token",
        default=None,
        type=str,
        help="The pad token. It should be provided when use custom vocab_file. And if it's None, bos_token will be used. ",
    )
    parser.add_argument("--weight_decay", default=None, type=float, help="Weight Decay for optimizer. ")

    # For benchmark.
    parser.add_argument(
        "--profiler_options",
        type=str,
        default=None,
        help='The option of profiler, which should be in format "key1=value1;key2=value2;key3=value3".',
    )
    args = parser.parse_args()
    return args


def do_train(args):
    paddle.enable_static()
    if args.is_distributed:
        fleet.init(is_collective=True)
        assert args.device != "xpu", "xpu doesn't support distributed training"
        places = [paddle.set_device("gpu")] if args.device == "gpu" else paddle.static.cpu_places()
        trainer_count = len(places)
    else:
        if args.device == "gpu":
            places = paddle.static.cuda_places()
        elif args.device == "xpu":
            places = paddle.static.xpu_places()
            paddle.set_device("xpu")
        else:
            places = paddle.static.cpu_places()
            paddle.set_device("cpu")
        trainer_count = len(places)

    # Set seed for CE
    random_seed = eval(str(args.random_seed))
    if random_seed is not None:
        paddle.seed(random_seed)

    # Define data loader
    (train_loader), (eval_loader) = reader.create_data_loader(args, places=places)

    train_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(train_program, startup_program):
        src_word = paddle.static.data(name="src_word", shape=[None, None], dtype=args.input_dtype)
        trg_word = paddle.static.data(name="trg_word", shape=[None, None], dtype=args.input_dtype)
        lbl_word = paddle.static.data(name="lbl_word", shape=[None, None, 1], dtype=args.input_dtype)

        # Define model
        transformer = TransformerModel(
            src_vocab_size=args.src_vocab_size,
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
            eos_id=args.eos_idx,
            pad_id=args.pad_idx,
        )
        # Define loss
        criterion = CrossEntropyCriterion(args.label_smooth_eps, args.bos_idx)

        logits = transformer(src_word=src_word, trg_word=trg_word)

        sum_cost, avg_cost, token_num = criterion(logits, lbl_word)

        scheduler = paddle.optimizer.lr.NoamDecay(args.d_model, args.warmup_steps, args.learning_rate, last_epoch=0)

        # Define optimizer
        optimizer = paddle.optimizer.Adam(
            learning_rate=scheduler,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=float(args.eps),
            parameters=transformer.parameters(),
            weight_decay=args.weight_decay,
        )

        if args.is_distributed:
            build_strategy = paddle.static.BuildStrategy()
            exec_strategy = paddle.static.ExecutionStrategy()
            dist_strategy = fleet.DistributedStrategy()
            dist_strategy.build_strategy = build_strategy
            dist_strategy.execution_strategy = exec_strategy
            dist_strategy.fuse_grad_size_in_MB = 16

            if args.use_amp:
                dist_strategy.amp = True
                dist_strategy.amp_configs = {
                    "custom_white_list": ["softmax", "layer_norm"],
                    "init_loss_scaling": args.scale_loss,
                    "custom_black_list": ["lookup_table_v2"],
                    "use_pure_fp16": args.use_pure_fp16,
                }

            optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
        else:
            if args.use_amp:
                amp_list = paddle.static.amp.AutoMixedPrecisionLists(
                    custom_white_list=["softmax", "layer_norm"], custom_black_list=["lookup_table_v2"]
                )
                optimizer = paddle.static.amp.decorate(
                    optimizer,
                    amp_list,
                    init_loss_scaling=args.scale_loss,
                    use_dynamic_loss_scaling=True,
                    use_pure_fp16=args.use_pure_fp16,
                )
        optimizer.minimize(avg_cost)

    if args.is_distributed:
        exe = paddle.static.Executor(places[0])
    else:
        exe = paddle.static.Executor()
        build_strategy = paddle.static.BuildStrategy()
        exec_strategy = paddle.static.ExecutionStrategy()

        compiled_train_program = paddle.static.CompiledProgram(train_program, build_strategy=build_strategy)
    exe.run(startup_program)

    if args.use_amp:
        optimizer.amp_init(places[0])

    # the best cross-entropy value with label smoothing
    loss_normalizer = -(
        (1.0 - args.label_smooth_eps) * np.log((1.0 - args.label_smooth_eps))
        + args.label_smooth_eps * np.log(args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20)
    )

    step_idx = 0

    # For benchmark
    reader_cost_avg = AverageStatistical()
    batch_cost_avg = AverageStatistical()
    batch_ips_avg = AverageStatistical()

    for pass_id in range(args.epoch):
        batch_id = 0
        batch_start = time.time()
        for data in train_loader:
            # NOTE: used for benchmark and use None as default.
            if args.max_iter and step_idx == args.max_iter:
                break
            if trainer_count == 1:
                data = [data]
            train_reader_cost = time.time() - batch_start

            if args.is_distributed:
                outs = exe.run(
                    train_program,
                    feed=[
                        {
                            "src_word": data[i][0],
                            "trg_word": data[i][1],
                            "lbl_word": data[i][2],
                        }
                        for i in range(trainer_count)
                    ],
                    fetch_list=[sum_cost.name, token_num.name],
                )
                train_batch_cost = time.time() - batch_start
                batch_ips_avg.record(train_batch_cost, np.asarray(outs[1]).sum())
            else:
                outs = exe.run(
                    compiled_train_program,
                    feed=[
                        {
                            "src_word": data[i][0],
                            "trg_word": data[i][1],
                            "lbl_word": data[i][2],
                        }
                        for i in range(trainer_count)
                    ],
                    fetch_list=[sum_cost.name, token_num.name],
                )
                train_batch_cost = time.time() - batch_start
                batch_ips_avg.record(train_batch_cost, np.asarray(outs[1]).sum() / trainer_count)
            scheduler.step()

            reader_cost_avg.record(train_reader_cost)
            batch_cost_avg.record(train_batch_cost)

            # Profile for model benchmark
            if args.profiler_options is not None:
                profiler.add_profiler_step(args.profiler_options)

            if step_idx % args.print_step == 0 and (
                args.benchmark or (args.is_distributed and dist.get_rank() == 0) or not args.is_distributed
            ):
                sum_cost_val, token_num_val = np.array(outs[0]), np.array(outs[1])
                # Sum the cost from multi-devices
                total_sum_cost = sum_cost_val.sum()
                total_token_num = token_num_val.sum()
                total_avg_cost = total_sum_cost / total_token_num

                if step_idx == 0:
                    logger.info(
                        "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f"
                        % (
                            step_idx,
                            pass_id,
                            batch_id,
                            total_avg_cost,
                            total_avg_cost - loss_normalizer,
                            np.exp([min(total_avg_cost, 100)]),
                        )
                    )
                else:
                    train_avg_batch_cost = args.print_step / batch_cost_avg.get_total_time()
                    max_mem_reserved_msg = ""
                    max_mem_allocated_msg = ""
                    if paddle.device.is_compiled_with_cuda():
                        max_mem_reserved_msg = (
                            f"max_mem_reserved: {paddle.device.cuda.max_memory_reserved() // (1024 ** 2)} MB,"
                        )
                        max_mem_allocated_msg = (
                            f"max_mem_allocated: {paddle.device.cuda.max_memory_allocated() // (1024 ** 2)} MB"
                        )
                    logger.info(
                        "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f, avg_speed: %.2f step/s, "
                        "batch_cost: %.5f sec, reader_cost: %.5f sec, tokens: %d, "
                        "ips: %.5f words/sec, %s %s"
                        % (
                            step_idx,
                            pass_id,
                            batch_id,
                            total_avg_cost,
                            total_avg_cost - loss_normalizer,
                            np.exp([min(total_avg_cost, 100)]),
                            train_avg_batch_cost,
                            batch_cost_avg.get_average(),
                            reader_cost_avg.get_average(),
                            batch_ips_avg.get_total_cnt(),
                            batch_ips_avg.get_average_per_sec(),
                            max_mem_reserved_msg,
                            max_mem_allocated_msg,
                        )
                    )
                reader_cost_avg.reset()
                batch_cost_avg.reset()
                batch_ips_avg.reset()

            if step_idx % args.save_step == 0 and step_idx != 0:
                if args.save_model and dist.get_rank() == 0:
                    model_path = os.path.join(args.save_model, "step_" + str(step_idx), "transformer")
                    paddle.static.save(train_program, model_path)

            batch_id += 1
            step_idx += 1
            batch_start = time.time()

        # NOTE: used for benchmark and use None as default.
        if args.max_iter and step_idx == args.max_iter:
            break

    if args.save_model and dist.get_rank() == 0:
        model_path = os.path.join(args.save_model, "step_final", "transformer")
        paddle.static.save(train_program, model_path)

    paddle.disable_static()


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, "rt") as f:
        args = AttrDict(yaml.safe_load(f))
    args.benchmark = ARGS.benchmark
    args.is_distributed = ARGS.distributed
    if ARGS.max_iter:
        args.max_iter = ARGS.max_iter
    args.weight_decay = ARGS.weight_decay

    args.data_dir = ARGS.data_dir
    args.train_file = ARGS.train_file
    args.dev_file = ARGS.dev_file

    if ARGS.vocab_file is not None:
        args.src_vocab = ARGS.vocab_file
        args.trg_vocab = ARGS.vocab_file
        args.joined_dictionary = True
    elif ARGS.src_vocab is not None and ARGS.trg_vocab is None:
        args.vocab_file = args.trg_vocab = args.src_vocab = ARGS.src_vocab
        args.joined_dictionary = True
    elif ARGS.src_vocab is None and ARGS.trg_vocab is not None:
        args.vocab_file = args.trg_vocab = args.src_vocab = ARGS.trg_vocab
        args.joined_dictionary = True
    else:
        args.src_vocab = ARGS.src_vocab
        args.trg_vocab = ARGS.trg_vocab
        args.joined_dictionary = not (
            args.src_vocab is not None and args.trg_vocab is not None and args.src_vocab != args.trg_vocab
        )
    if args.weight_sharing != args.joined_dictionary:
        if args.weight_sharing:
            raise ValueError("The src_vocab and trg_vocab must be consistency when weight_sharing is True. ")
        else:
            raise ValueError(
                "The src_vocab and trg_vocab must be specified respectively when weight sharing is False. "
            )

    if ARGS.src_lang is not None:
        args.src_lang = ARGS.src_lang
    if ARGS.trg_lang is not None:
        args.trg_lang = ARGS.trg_lang

    args.unk_token = ARGS.unk_token
    args.bos_token = ARGS.bos_token
    args.eos_token = ARGS.eos_token
    args.pad_token = ARGS.pad_token
    pprint(args)
    args.profiler_options = ARGS.profiler_options

    do_train(args)
