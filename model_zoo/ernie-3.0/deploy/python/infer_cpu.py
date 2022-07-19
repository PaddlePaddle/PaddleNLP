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

import paddle
import argparse
from psutil import cpu_count
from ernie_predictor import ErniePredictor


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--task_name",
        default='seq_cls',
        type=str,
        help=
        "The name of the task to perform predict, selected in: seq_cls and token_cls"
    )
    parser.add_argument(
        "--model_name_or_path",
        default="ernie-3.0-medium-zh",
        type=str,
        help="The directory or name of model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path prefix of inference model to be used.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--precision_mode",
        type=str,
        default="fp32",
        choices=["fp32", "int8"],
        help=
        "Inference precision, set int8 to use dynamic quantization for acceleration.",
    )
    parser.add_argument(
        "--num_threads",
        default=cpu_count(logical=False),
        type=int,
        help="num_threads for cpu.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    args.task_name = args.task_name.lower()
    args.device = 'cpu'
    predictor = ErniePredictor(args)

    if args.task_name == 'seq_cls':
        text = ["未来自动驾驶真的会让酒驾和疲劳驾驶成历史吗？", "黄磊接受华少快问快答，不光智商逆天，情商也不逊黄渤"]
    elif args.task_name == 'token_cls':
        text = ["北京的涮肉，重庆的火锅，成都的小吃都是极具特色的美食。", "乔丹、科比、詹姆斯和姚明都是篮球界的标志性人物。"]

    outputs = predictor.predict(text)


if __name__ == "__main__":
    main()
