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
import json
import os

import psutil
from predictor import UTCPredictor

from paddlenlp.datasets import load_dataset
from paddlenlp.utils.log import logger

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path_prefix", type=str, required=True, help="The path prefix of inference model to be used.")
parser.add_argument("--model_name_or_path", default="utc-large", type=str, help="The name of pretrained model.")
parser.add_argument("--threshold", default=0.5, type=float, help="Probability threshold of predicted labels.")
parser.add_argument("--data_dir", default=None, type=str, help="The path to the prediction data, including label.txt and data.txt.")
parser.add_argument("--max_length", default=128, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--use_fp16", action='store_true', help="Whether to use fp16 inference, only takes effect when deploying on gpu.")
parser.add_argument("--batch_size", default=200, type=int, help="Batch size per GPU/CPU for predicting.")
parser.add_argument("--num_threads", default=psutil.cpu_count(logical=False), type=int, help="num_threads for cpu.")
parser.add_argument("--device", choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--device_id", default=0, help="Select which gpu device to train model.")
args = parser.parse_args()
# yapf: enable


def read_local_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield (json.loads(line.strip()))


if __name__ == "__main__":
    for arg_name, arg_value in vars(args).items():
        logger.info("{:20}: {}".format(arg_name, arg_value))

    predictor = UTCPredictor(args)

    text_dir = os.path.join(args.data_dir, "data.txt")
    input_data = load_dataset(read_local_dataset, path=text_dir, lazy=False)
    predictor.predict(input_data)
