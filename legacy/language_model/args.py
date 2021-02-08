#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import distutils.util


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_type",
        type=str,
        default="small",
        help="model_type [test|small|medium|large]")
    parser.add_argument(
        "--rnn_model",
        type=str,
        default="static",
        help="model_type [static|padding|cudnn|basic_lstm]")
    parser.add_argument(
        "--data_path", type=str, help="all the data for train,valid,test")
    parser.add_argument('--para_init', action='store_true')
    parser.add_argument(
        '--use_gpu',
        type=str2bool,
        default=False,
        help='Whether using gpu [True|False]')
    parser.add_argument(
        '--parallel',
        type=str2bool,
        default=True,
        help='Whether using gpu in parallel [True|False]')
    parser.add_argument(
        '--profile',
        type=str2bool,
        default=False,
        help='Whether profiling the trainning [True|False]')
    parser.add_argument(
        '--enable_auto_fusion',
        type=str2bool,
        default=False,
        help='Whether enable fusion_group [True|False]. It is a experimental feature.'
    )
    parser.add_argument(
        '--use_dataloader',
        type=str2bool,
        default=False,
        help='Whether using dataloader to feed data [True|False]')
    parser.add_argument(
        '--log_path',
        help='path of the log file. If not set, logs are printed to console')
    parser.add_argument(
        '--save_model_dir',
        type=str,
        default="models",
        help='dir of the saved model.')
    parser.add_argument(
        '--init_from_pretrain_model',
        type=str,
        default=None,
        help='dir to init model.')
    parser.add_argument('--enable_ce', action='store_true')
    parser.add_argument('--batch_size', type=int, default=0, help='batch size')
    parser.add_argument('--max_epoch', type=int, default=0, help='max epoch')

    # NOTE: args for profiler, used for benchmark
    parser.add_argument(
        '--profiler_path',
        type=str,
        default='/tmp/paddingrnn.profile',
        help='the profiler output file path. used for benchmark')
    args = parser.parse_args()
    return args
