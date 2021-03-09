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
import os
import random
import time
import h5py
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import paddle
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset

from paddlenlp.transformers import BertForPretraining, BertModel, BertPretrainingCriterion
from paddlenlp.transformers import BertTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from data import create_data_holder, create_pretraining_dataset

MODEL_CLASSES = {"bert": (BertForPretraining, BertTokenizer)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
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
            ], [])), )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input directory where the data will be read from.", )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
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
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization")
    args = parser.parse_args()
    return args


def select_dataset_file_for_each_worker(files, f_start_id, worker_num,
                                        worker_index):
    num_files = len(files)
    if worker_num > num_files:
        remainder = worker_num % num_files
        data_file = files[(
            f_start_id * worker_num + worker_index + remainder * f_start_id) %
                          num_files]
    else:
        data_file = files[(f_start_id * worker_num + worker_index) % num_files]
    return data_file


def reset_program_state_dict(model, state_dict):
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


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def do_train(args):
    # Initialize the paddle and paddle fleet execute enviroment
    paddle.enable_static()
    place = paddle.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    fleet.init(is_collective=True)

    # Create the random seed for the worker
    set_seed(args.seed)
    worker_init = WorkerInitObj(args.seed + fleet.worker_index())

    # Define the input data in the static mode
    data_holders = create_data_holder(args)

    [
        input_ids, segment_ids, input_mask, masked_lm_positions,
        masked_lm_labels, next_sentence_labels, masked_lm_scale
    ] = data_holders

    # Define the model structure in static mode
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = BertForPretraining(
        BertModel(**model_class.pretrained_init_configuration[
            args.model_name_or_path]))
    criterion = BertPretrainingCriterion(model.bert.config["vocab_size"])
    prediction_scores, seq_relationship_score = model(
        input_ids=input_ids,
        token_type_ids=segment_ids,
        attention_mask=input_mask,
        masked_positions=masked_lm_positions)
    loss = criterion(prediction_scores, seq_relationship_score,
                     masked_lm_labels, next_sentence_labels, masked_lm_scale)

    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.num_train_epochs
    # Define the dynamic learing_reate scheduler and optimizer
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
        apply_decay_param_fun=lambda x: x in decay_params)

    # Use the fleet api to compile the distributed optimizer
    strategy = fleet.DistributedStrategy()
    optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
    optimizer.minimize(loss)

    # Define the Executor for running the static model
    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())
    state_dict = model.state_dict()

    # Use the state dict to update the parameter
    reset_state_dict = reset_program_state_dict(model, state_dict)
    paddle.static.set_program_state(paddle.static.default_main_program(),
                                    reset_state_dict)

    pool = ThreadPoolExecutor(1)
    global_step = 0
    tic_train = time.time()
    worker_num = fleet.worker_num()
    worker_index = fleet.worker_index()
    epoch = 0
    while True:
        files = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if os.path.isfile(os.path.join(args.input_dir, f)) and "training" in
            f
        ]
        files.sort()
        num_files = len(files)
        random.Random(args.seed + epoch).shuffle(files)
        f_start_id = 0

        # Select one file for each worker and create the DataLoader for the file
        data_file = select_dataset_file_for_each_worker(
            files, f_start_id, worker_num, worker_index)
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

            for step, batch in enumerate(train_data_loader):
                global_step += 1
                loss_return = exe.run(paddle.static.default_main_program(),\
                    feed=batch,
                    fetch_list=[loss])
                # In the new 2.0 api, must call this function to change the learning_rate
                lr_scheduler.step()
                if global_step % args.logging_steps == 0:
                    time_cost = time.time() - tic_train
                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s, ips :%.2f sequences/s"
                        % (global_step, epoch, step, loss_return[0],
                           args.logging_steps / time_cost,
                           args.logging_steps * args.batch_size / time_cost))
                    tic_train = time.time()
                if global_step % args.save_steps == 0:
                    if worker_index == 0:
                        output_dir = os.path.join(args.output_dir,
                                                  "model_%d" % global_step)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # TODO(fangzeyang): Udpate the save_params to paddle.static
                        paddle.fluid.io.save_params(exe, output_dir)
                        tokenizer.save_pretrained(output_dir)
                if global_step >= args.max_steps:
                    del train_data_loader
                    return
            del train_data_loader
            train_data_loader, data_file = dataset_future.result(timeout=None)
        epoch += 1


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
