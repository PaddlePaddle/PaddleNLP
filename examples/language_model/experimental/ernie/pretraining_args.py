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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse

from utils.args import str2bool, print_arguments


def define_args():
    parser = argparse.ArgumentParser('ERNIE-en model with Paddle')
    parser.add_argument('--debug', type=str2bool, default=False)

    # Model Args
    parser.add_argument(
        '--ernie_config_file', type=str, default='./config/ernie_config.json')
    parser.add_argument('--vocab_file', type=str, default='./config/vocab.txt')
    parser.add_argument('--init_checkpoint', type=str, default="")
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--preln', type=str2bool, default=False)

    # Data Args
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--eval_data_path', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output')

    # Training Args
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--global_bsz', type=int, default=256)
    parser.add_argument('--micro_bsz', type=int, default=16)
    parser.add_argument('--do_eval', type=str2bool, default=True)
    parser.add_argument('--eval_batch_size', type=int, default=35)
    parser.add_argument('--num_train_steps', type=int, default=1500000)
    parser.add_argument('--global_steps', type=int, default=0)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--save_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=-1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--use_lamb', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--use_sop', type=str2bool, default=False)

    # Fleet Args
    parser.add_argument('--use_sharding', type=str2bool, default=False)
    parser.add_argument('--use_hybrid_dp', type=str2bool, default=True)
    parser.add_argument('--use_amp', type=str2bool, default=True)
    parser.add_argument('--use_recompute', type=str2bool, default=True)
    parser.add_argument('--use_offload', type=str2bool, default=False)
    parser.add_argument('--grad_merge', type=int, default=0)
    parser.add_argument(
        '--num_mp', type=int, default=1, help="num of model parallel")
    parser.add_argument('--num_pp', type=int, default=1, help="num of pipeline")
    parser.add_argument(
        '--num_sharding', type=int, default=1, help="num of sharding")
    parser.add_argument('--num_dp', type=int, default=1, help="num of dp")
    args = parser.parse_args()

    print_arguments(args)
    return args


if __name__ == '__main__':
    args = define_args()
