#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import six
import sys
import os
import time
import argparse
from functools import partial

import numpy as np
import paddle

import paddleslim
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieTokenizer

sys.path.append("../")
from data import convert_example, METRIC_CLASSES, MODEL_CLASSES

parser = argparse.ArgumentParser()

parser.add_argument(
    "--task_name", type=str, default="afqmc", required=False, help="task_name")
parser.add_argument(
    "--input_dir",
    type=str,
    default="afqmc",
    required=False,
    help="Input task model directory.")

parser.add_argument(
    "--save_model_filename",
    type=str,
    default="int8.pdmodel",
    required=False,
    help="File name of quantified model.")

parser.add_argument(
    "--save_params_filename",
    type=str,
    default="int8.pdiparams",
    required=False,
    help="File name of quantified model's parameters.")

parser.add_argument(
    "--input_model_filename",
    type=str,
    default="float.pdmodel",
    required=False,
    help="File name of float model.")

parser.add_argument(
    "--input_param_filename",
    type=str,
    default="float.pdiparams",
    required=False,
    help="File name of float model's parameters.")

parser.add_argument(
    "--model_name_or_path",
    default='ppminilm-6l-768h',
    type=str,
    help="Model name or the directory of model directory.", )

args = parser.parse_args()


def quant_post(args, batch_size=8, algo='avg'):
    place = paddle.set_device("gpu")
    exe = paddle.static.Executor(place)
    args.task_name = args.task_name.lower()

    train_ds = load_dataset("clue", args.task_name, splits="dev")

    tokenizer = ErnieTokenizer.from_pretrained(args.model_name_or_path)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=128,
        is_test=True)
    train_ds = train_ds.map(trans_func, lazy=True)

    def batch_generator_func():
        batch_data = [[], []]
        for data in train_ds:
            batch_data[0].append(data[0])
            batch_data[1].append(data[1])
            if len(batch_data[0]) == batch_size:
                input_ids = Pad(axis=0, pad_val=0)(batch_data[0])
                segment_ids = Pad(axis=0, pad_val=0)(batch_data[1])
                yield [input_ids, segment_ids]
                batch_data = [[], []]

    paddleslim.quant.quant_post_static(
        exe,
        args.input_dir,
        os.path.join(args.task_name + '_quant_models', algo + str(batch_size)),
        save_model_filename=args.save_model_filename,
        save_params_filename=args.save_params_filename,
        algo=algo,
        hist_percent=0.9999,
        batch_generator=batch_generator_func,
        model_filename=args.input_model_filename,
        params_filename=args.input_param_filename,
        quantizable_op_type=['matmul', 'matmul_v2'],
        weight_bits=8,
        weight_quantize_type='channel_wise_abs_max',
        batch_nums=1, )


if __name__ == '__main__':
    paddle.enable_static()
    for batch_size in [4, 8]:
        for algo in ['abs_max', 'avg', 'mse', 'hist']:
            quant_post(args, batch_size, algo)
