# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import collections
import itertools
import os
import random
import time
import h5py
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import distutils.util

import paddle
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset

from paddlenlp.utils import profiler
from paddlenlp.utils.tools import TimeCostAverage
from paddlenlp.transformers import BertForPretraining, BertModel, BertPretrainingCriterion
from paddlenlp.transformers import BertTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from dataset import create_data_holder, create_pretraining_dataset

MODEL_CLASSES = {"bert": (BertForPretraining, BertTokenizer)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])),
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input directory where the data will be read from.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        default=80,
        type=int,
        help="The maximum total of masked tokens in input sequence")

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps",
                        type=int,
                        default=500,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for initialization")
    parser.add_argument("--use_amp",
                        type=distutils.util.strtobool,
                        default=False,
                        help="Enable mixed precision training.")
    parser.add_argument(
        "--enable_addto",
        type=distutils.util.strtobool,
        default=False,
        help=
        "Whether to enable the addto strategy for gradient accumulation or not. This is only used for AMP training."
    )
    parser.add_argument("--scale_loss",
                        type=float,
                        default=2**15,
                        help="The value of scale_loss for fp16.")
    parser.add_argument("--use_pure_fp16",
                        type=distutils.util.strtobool,
                        default=False,
                        help="Whether to use pure fp16 training.")
    parser.add_argument("--device",
                        type=str,
                        default="gpu",
                        help="Device for selecting for the training.")
    parser.add_argument(
        "--gradient_merge_steps",
        type=int,
        default=1,
        help="Number of merge steps before gradient update."
        "global_batch_size = gradient_merge_steps * batch_size.")

    # For benchmark.
    parser.add_argument(
        '--profiler_options',
        type=str,
        default=None,
        help=
        'The option of profiler, which should be in format \"key1=value1;key2=value2;key3=value3\".'
    )
    parser.add_argument(
        "--fuse_transformer",
        type=distutils.util.strtobool,
        default=False,
        help=
        "Whether to use FusedTransformerEncoderLayer to replace a TransformerEncoderLayer or not."
    )
    args = parser.parse_args()
    return args


def select_dataset_file_for_each_worker(files, f_start_id, worker_num,
                                        worker_index):
    """  
    Spliting the train file according to the worker index.
    """
    num_files = len(files)
    if worker_num > num_files:
        remainder = worker_num % num_files
        data_file = files[(f_start_id * worker_num + worker_index +
                           remainder * f_start_id) % num_files]
    else:
        data_file = files[(f_start_id * worker_num + worker_index) % num_files]
    return data_file


def reset_program_state_dict(model, state_dict):
    """
    Initialize the parameter from the bert config, and set the parameter by 
    reseting the state dict."
    """
    scale = model.initializer_range if hasattr(model, "initializer_range")\
        else model.bert.config["initializer_range"]

    new_state_dict = dict()
    for n, p in state_dict.items():
        if "layer_norm" not in p.name:
            dtype_str = "float32"
            if str(p.dtype) == "VarType.FP64":
                dtype_str = "float64"
            new_state_dict[p.name] = np.random.normal(
                loc=0.0, scale=scale, size=p.shape).astype(dtype_str)
    return new_state_dict


def create_strategy(args):
    """
    Create build strategy and exec strategy.
    """
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    build_strategy.enable_addto = args.enable_addto

    exec_strategy.num_threads = 1
    exec_strategy.num_iteration_per_drop_scope = 10000
    return build_strategy, exec_strategy


def dist_optimizer(args, optimizer):
    """
    Create a distributed optimizer based on a normal optimizer
    """
    build_strategy, exec_strategy = create_strategy(args)

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.build_strategy = build_strategy

    dist_strategy.fuse_grad_size_in_MB = 16
    if args.use_amp:
        dist_strategy.amp = True

        custom_black_list = ['lookup_table', 'lookup_table_v2'
                             ] if args.use_pure_fp16 else None
        dist_strategy.amp_configs = {
            'custom_white_list': ['softmax', 'layer_norm', 'gelu'],
            'init_loss_scaling': args.scale_loss,
            'custom_black_list': custom_black_list,
            'use_pure_fp16': args.use_pure_fp16
        }
    if args.gradient_merge_steps > 1:
        dist_strategy.gradient_merge = True
        dist_strategy.gradient_merge_configs = {
            'k_steps': args.gradient_merge_steps
        }

    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
    return optimizer


def set_seed(seed):
    """
    Use the same data seed(for data shuffle) for all procs to guarantee data
    consistency after sharding.
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


class WorkerInitObj(object):
    "Construct the object with different seed, and the Dataloader will generate the data "
    "with different seed in each worker."

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def do_train(args):
    # Initialize the paddle and paddle fleet execute enviroment
    paddle.enable_static()
    place = paddle.set_device(args.device)
    fleet.init(is_collective=True)

    worker_num = fleet.worker_num()
    worker_index = fleet.worker_index()

    # Create the random seed for the worker
    set_seed(args.seed)
    worker_init = WorkerInitObj(args.seed + worker_index)

    # Define the input data in the static mode
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    data_holders = create_data_holder(args)

    [
        input_ids, segment_ids, input_mask, masked_lm_positions,
        masked_lm_labels, next_sentence_labels, masked_lm_scale
    ] = data_holders

    # Define the model structure in static mode
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    config = model_class.pretrained_init_configuration[args.model_name_or_path]
    if config["vocab_size"] % 8 != 0:
        config["vocab_size"] += 8 - (config["vocab_size"] % 8)
    config['fuse'] = args.fuse_transformer
    model = BertForPretraining(BertModel(**config))
    criterion = BertPretrainingCriterion(model.bert.config["vocab_size"])
    prediction_scores, seq_relationship_score = model(
        input_ids=input_ids,
        token_type_ids=segment_ids,
        attention_mask=input_mask,
        masked_positions=masked_lm_positions)
    loss = criterion(prediction_scores, seq_relationship_score,
                     masked_lm_labels, next_sentence_labels, masked_lm_scale)

    # Define the dynamic learing_reate scheduler and optimizer
    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.num_train_epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_steps)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        multi_precision=args.use_pure_fp16)

    # Use the fleet api to compile the distributed optimizer
    optimizer = dist_optimizer(args, optimizer)
    optimizer.minimize(loss)

    # Define the Executor for running the static model
    exe = paddle.static.Executor(place)
    exe.run(startup_program)
    state_dict = model.state_dict()

    # Use the state dict to update the parameter
    reset_state_dict = reset_program_state_dict(model, state_dict)
    paddle.static.set_program_state(main_program, reset_state_dict)
    if args.use_amp:
        optimizer.amp_init(place)

    pool = ThreadPoolExecutor(1)
    global_step = 0
    tic_train = time.time()
    epoch = 0
    while True:
        files = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if os.path.isfile(os.path.join(args.input_dir, f))
            and "training" in f
        ]
        files.sort()
        num_files = len(files)
        random.Random(args.seed + epoch).shuffle(files)
        f_start_id = 0

        # Select one file for each worker and create the DataLoader for the file
        data_file = select_dataset_file_for_each_worker(files, f_start_id,
                                                        worker_num,
                                                        worker_index)
        train_data_loader, _ = create_pretraining_dataset(
            data_file, args.max_predictions_per_seq, args, data_holders,
            worker_init, paddle.static.cuda_places())

        for f_id in range(f_start_id + 1, len(files)):
            data_file = select_dataset_file_for_each_worker(
                files, f_id, worker_num, worker_index)
            dataset_future = pool.submit(create_pretraining_dataset, data_file,
                                         args.max_predictions_per_seq, args,
                                         data_holders, worker_init,
                                         paddle.static.cuda_places())

            train_cost_avg = TimeCostAverage()
            reader_cost_avg = TimeCostAverage()
            total_samples = 0
            batch_start = time.time()
            for step, batch in enumerate(train_data_loader):
                train_reader_cost = time.time() - batch_start
                reader_cost_avg.record(train_reader_cost)
                global_step += 1
                train_start = time.time()
                loss_return = exe.run(main_program,
                                      feed=batch,
                                      fetch_list=[loss])
                total_samples += args.batch_size
                # In the new 2.0 api, must call this function to change the learning_rate
                lr_scheduler.step()
                train_run_cost = time.time() - batch_start
                train_cost_avg.record(train_run_cost)

                # Profile for model benchmark
                if args.profiler_options is not None:
                    profiler.add_profiler_step(args.profiler_options)

                if global_step % args.logging_steps == 0:
                    print(
                        "total step: %d, epoch: %d, batch: %d, loss: %f, "
                        "avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sequences/sec"
                        % (global_step, epoch, step, loss_return[0],
                           reader_cost_avg.get_average(),
                           train_cost_avg.get_average(),
                           total_samples / args.logging_steps, args.batch_size /
                           (reader_cost_avg.get_average() +
                            train_cost_avg.get_average())))
                    total_samples = 0
                    train_cost_avg.reset()
                    reader_cost_avg.reset()
                if global_step % args.save_steps == 0:
                    if worker_index == 0:
                        output_dir = os.path.join(args.output_dir,
                                                  "model_%d" % global_step)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model.save_model_config(output_dir)
                        paddle.static.save(
                            main_program, os.path.join(output_dir,
                                                       "model_state"))
                        tokenizer.save_pretrained(output_dir)
                if global_step >= args.max_steps:
                    reader_start = time.time()
                    del train_data_loader
                    return
                batch_start = time.time()
            del train_data_loader
            train_data_loader, data_file = dataset_future.result(timeout=None)
        epoch += 1


if __name__ == "__main__":
    args = parse_args()
    print(args)
    do_train(args)
