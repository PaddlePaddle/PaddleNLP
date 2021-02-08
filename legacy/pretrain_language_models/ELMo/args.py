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

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load_dir",
        type=str,
        default="",
        help="Specify the path to load trained models.")
    parser.add_argument(
        "--load_pretraining_params",
        type=str,
        default="",
        help="Specify the path to load pretrained model parameters, NOT including moment and learning_rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The sequence number of a mini-batch data. (default: %(default)d)")
    parser.add_argument(
        "--embed_size",
        type=int,
        default=512,
        help="The dimension of embedding table. (default: %(default)d)")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=4096,
        help="The size of rnn hidden unit. (default: %(default)d)")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="The size of rnn layers. (default: %(default)d)")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="The size of sequence len. (default: %(default)d)")
    parser.add_argument(
        "--all_train_tokens",
        type=int,
        default=35479,
        help="The size of all training tokens")
    parser.add_argument(
        "--data_path", type=str, help="all the data for train,valid,test")
    parser.add_argument("--vocab_path", type=str, help="vocab file path")
    # parser.add_argument(
    #    '--use_gpu', action='store_true',help='whether using gpu')
    parser.add_argument(
        "--use_gpu", type=str2bool, default='True', help="Activate nice mode.")
    parser.add_argument('--test_nccl', action='store_true')
    parser.add_argument('--optim', default='adagrad', help='optimizer type')
    parser.add_argument('--sample_softmax', action='store_true')
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.2,
        help="Learning rate used to train the model. (default: %(default)f)")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="log the train loss every n batches."
        "(default: %(default)d)")
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10000,
        help="log the train loss every n batches."
        "(default: %(default)d)")
    parser.add_argument(
        "--dev_interval",
        type=int,
        default=10000,
        help="cal dev loss every n batches."
        "(default: %(default)d)")
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=10.0)
    parser.add_argument('--proj_clip', type=float, default=3.0)
    parser.add_argument('--cell_clip', type=float, default=3.0)
    parser.add_argument('--max_epoch', type=float, default=10)
    parser.add_argument('--local', type=str2bool, default='False')
    parser.add_argument('--shuffle', type=str2bool, default='False')
    parser.add_argument('--use_custom_samples', type=str2bool, default='False')
    parser.add_argument('--para_save_dir', type=str, default='checkpoints')
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--update_method', type=str, default='nccl2')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--n_negative_samples_batch', type=int, default=8000)
    parser.add_argument(
        '--enable_ce', action='store_true', help='whether print log for ce')
    args = parser.parse_args()

    return args
