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

import os
import argparse
import psutil

import paddle
from paddlenlp.utils.log import logger
from paddlenlp.datasets import load_dataset

from predictor import NERPredictor

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path_prefix", type=str, required=True, help="The path prefix of inference model to be used.")
parser.add_argument("--model_name_or_path", default="ernie-health-chinese", type=str, help="The directory or name of model.")
parser.add_argument("--dataset", default="CMeEE", type=str, help="Dataset for named entity recognition.")
parser.add_argument("--data_file", default=None, type=str, help="The data to predict with one sample per line.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization")
parser.add_argument("--use_fp16", action='store_true', help="Whether to use fp16 inference, only takes effect when deploying on gpu.")
parser.add_argument("--batch_size", default=200, type=int, help="Batch size per GPU/CPU for predicting.")
parser.add_argument("--num_threads", default=psutil.cpu_count(logical=False), type=int, help="Number of threads for cpu.")
parser.add_argument("--device", choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--device_id", default=0, help="Select which gpu device to train model.")
args = parser.parse_args()
# yapf: enable

LABEL_LIST = {
    'cmeee': [[
        'B-bod', 'I-bod', 'E-bod', 'S-bod', 'B-dis', 'I-dis', 'E-dis', 'S-dis',
        'B-pro', 'I-pro', 'E-pro', 'S-pro', 'B-dru', 'I-dru', 'E-dru', 'S-dru',
        'B-ite', 'I-ite', 'E-ite', 'S-ite', 'B-mic', 'I-mic', 'E-mic', 'S-mic',
        'B-equ', 'I-equ', 'E-equ', 'S-equ', 'B-dep', 'I-dep', 'E-dep', 'S-dep',
        'O'
    ], ['B-sym', 'I-sym', 'E-sym', 'S-sym', 'O']]
}

TEXT = {
    'cmeee': [
        "研究证实，细胞减少与肺内病变程度及肺内炎性病变吸收程度密切相关。",
        "可为不规则发热、稽留热或弛张热，但以不规则发热为多，可能与患儿应用退热药物导致热型不规律有关。"
    ]
}

if __name__ == "__main__":
    for arg_name, arg_value in vars(args).items():
        logger.info("{:20}: {}".format(arg_name, arg_value))

    dataset = args.dataset.lower()
    label_list = LABEL_LIST[dataset]
    if args.data_file is not None:
        with open(args.data_file, 'r') as fp:
            input_data = [x.strip() for x in fp.readlines()]
    else:
        input_data = TEXT[dataset]

    predictor = NERPredictor(args, label_list)
    predictor.predict(input_data)
