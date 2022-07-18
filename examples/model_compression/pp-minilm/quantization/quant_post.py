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
import distutils.util

import paddle
import paddleslim

from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import PPMiniLMTokenizer

sys.path.append("../")
from data import convert_example, METRIC_CLASSES, MODEL_CLASSES

parser = argparse.ArgumentParser()

parser.add_argument("--task_name", type=str, required=True, help="task_name")
parser.add_argument("--input_dir",
                    type=str,
                    default="../pruning/pruned_models/",
                    required=True,
                    help="Input task model directory.")
parser.add_argument("--output_dir",
                    type=str,
                    default="./",
                    required=False,
                    help="Output model directory.")

parser.add_argument("--save_model_filename",
                    type=str,
                    default="int8.pdmodel",
                    required=False,
                    help="File name of quantified model.")

parser.add_argument("--save_params_filename",
                    type=str,
                    default="int8.pdiparams",
                    required=False,
                    help="File name of quantified model's parameters.")

parser.add_argument("--input_model_filename",
                    type=str,
                    default="float.pdmodel",
                    required=False,
                    help="File name of float model.")

parser.add_argument("--input_param_filename",
                    type=str,
                    default="float.pdiparams",
                    required=False,
                    help="File name of float model's parameters.")
parser.add_argument(
    "--max_seq_length",
    default=128,
    type=int,
    help=
    "The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
)

parser.add_argument(
    "--use_faster_tokenizer",
    type=distutils.util.strtobool,
    default=True,
    help=
    "Whether to use FasterTokenizer to accelerate training or further inference."
)

parser.add_argument(
    "--model_name_or_path",
    default='ppminilm-6l-768h',
    type=str,
    help="Model name or the directory of model directory.",
)

args = parser.parse_args()


def quant_post(args, batch_size=8, algo='avg'):
    place = paddle.set_device("gpu")
    exe = paddle.static.Executor(place)
    args.task_name = args.task_name.lower()

    dev_ds = load_dataset("clue", args.task_name, splits="dev")
    if args.use_faster_tokenizer:
        trans_func = partial(convert_example, label_list=dev_ds.label_list)
    else:
        tokenizer = PPMiniLMTokenizer.from_pretrained("ppminilm-6l-768h")
        trans_func = partial(convert_example,
                             label_list=dev_ds.label_list,
                             tokenizer=tokenizer,
                             max_seq_length=128,
                             is_test=True)
    dev_ds = dev_ds.map(trans_func, lazy=True)

    def batch_generator_func():
        batch_data = [[], []]
        for data in dev_ds:
            batch_data[0].append(data[0])
            batch_data[1].append(data[1])
            if len(batch_data[0]) == batch_size:
                input_ids = Pad(axis=0, pad_val=0)(batch_data[0])
                segment_ids = Pad(axis=0, pad_val=0)(batch_data[1])
                yield [input_ids, segment_ids]
                batch_data = [[], []]

    def batch_generator_func_using_faster_tokenizer():
        if 'sentence' in dev_ds[0]:
            batch_data = []
        else:
            batch_data = [[], []]
        for data in dev_ds:
            if 'sentence' in data:
                batch_data.append(data['sentence'])
                if len(batch_data) == batch_size:
                    yield {"text": batch_data}
                    batch_data = []
            else:
                batch_data[0].append(data['sentence1'])
                batch_data[1].append(data['sentence2'])
                if len(batch_data[0]) == batch_size:
                    yield {"text": batch_data[0], "text_pair": batch_data[1]}
                    batch_data = [[], []]

    paddleslim.quant.quant_post_static(
        exe,
        args.input_dir,
        os.path.join(args.output_dir, args.task_name + '_quant_models',
                     algo + str(batch_size)),
        save_model_filename=args.save_model_filename,
        save_params_filename=args.save_params_filename,
        algo=algo,
        hist_percent=0.9999,
        batch_generator=batch_generator_func
        if not args.use_faster_tokenizer else None,
        data_loader=batch_generator_func_using_faster_tokenizer
        if args.use_faster_tokenizer else None,
        model_filename=args.input_model_filename,
        params_filename=args.input_param_filename,
        quantizable_op_type=['matmul', 'matmul_v2'],
        weight_bits=8,
        weight_quantize_type='channel_wise_abs_max',
        batch_nums=1,
    )


if __name__ == '__main__':
    paddle.enable_static()
    for batch_size in [4, 8]:
        for algo in ['abs_max', 'avg', 'mse', 'hist']:
            quant_post(args, batch_size, algo)
