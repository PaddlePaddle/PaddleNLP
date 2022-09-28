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

import sys
import os
import time
import argparse
import numpy as np
from functools import partial

import paddle
import paddleslim
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import PPMiniLMTokenizer
from data import convert_example_to_feature, read, load_dict


def quant_post(args):
    place = paddle.set_device("gpu")
    exe = paddle.static.Executor(place)

    label2id, id2label = load_dict(args.label_path)
    train_ds = load_dataset(read, data_path=args.dev_path, lazy=False)

    tokenizer = PPMiniLMTokenizer.from_pretrained(args.base_model_name)
    trans_func = partial(convert_example_to_feature,
                         tokenizer=tokenizer,
                         label2id=label2id,
                         max_seq_len=args.max_seq_len)
    train_ds = train_ds.map(trans_func, lazy=True)

    def batch_generator_func():
        batch_data = [[], []]
        for data in train_ds:
            batch_data[0].append(data[0])
            batch_data[1].append(data[1])
            if len(batch_data[0]) == args.batch_size:
                input_ids = Pad(axis=0, pad_val=0, dtype="int64")(batch_data[0])
                segment_ids = Pad(axis=0, pad_val=0,
                                  dtype="int64")(batch_data[1])
                yield [input_ids, segment_ids]
                batch_data = [[], []]

    paddleslim.quant.quant_post_static(
        exe,
        args.static_model_dir,
        args.quant_model_dir,
        save_model_filename=args.save_model_filename,
        save_params_filename=args.save_params_filename,
        algo=args.algorithm,
        hist_percent=0.9999,
        batch_generator=batch_generator_func,
        model_filename=args.input_model_filename,
        params_filename=args.input_param_filename,
        quantizable_op_type=['matmul', 'matmul_v2'],
        weight_bits=8,
        weight_quantize_type='channel_wise_abs_max',
        batch_nums=1)


if __name__ == '__main__':
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="ppminilm-6l-768h", help="The path of ppminilm model.")
    parser.add_argument("--static_model_dir", type=str, default="./checkpoints/static", help="Directory of static model that will be quantized.")
    parser.add_argument("--quant_model_dir", type=str, default=None, help="Directory of the quantized model that will be written.")
    parser.add_argument("--algorithm", type=str, default="avg", help="Quantize algorithm that you want to choice, such as abs_max, avg, mse, hist.")
    parser.add_argument('--dev_path', type=str, default=None, help="The path of dev set.")
    parser.add_argument("--label_path", type=str, default=None, help="The path of label dict.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--save_model_filename", type=str, default="infer.pdmodel", required=False, help="File name of quantified model.")
    parser.add_argument("--save_params_filename", type=str, default="infer.pdiparams", required=False, help="File name of quantified model's parameters.")
    parser.add_argument("--input_model_filename", type=str, default="infer.pdmodel", required=False, help="File name of float model.")
    parser.add_argument("--input_param_filename", type=str, default="infer.pdiparams", required=False, help="File name of float model's parameters.")

    args = parser.parse_args()
    # yapf: enable

    # start quantize model
    paddle.enable_static()
    quant_post(args)
    print(
        f"quantize model done. the quantized model has been saved to {args.quant_model_dir}"
    )
