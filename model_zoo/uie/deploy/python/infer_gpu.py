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
from pprint import pprint

import paddle
from uie_predictor import UIEPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_path_prefix",
        type=str,
        required=True,
        help="The path prefix of inference model to be used.",
    )
    parser.add_argument(
        "--position_prob",
        default=0.5,
        type=float,
        help="Probability threshold for start/end index probabiliry.",
    )
    parser.add_argument(
        "--use_fp16",
        action='store_true',
        help=
        "Whether to use fp16 inference, only takes effect when deploying on gpu.",
    )
    parser.add_argument(
        "--max_seq_len",
        default=512,
        type=int,
        help=
        "The maximum input sequence length. Sequences longer than this will be split automatically.",
    )
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="Batch size per GPU for inference.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    texts = [
        '"北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。"',
        '原告赵六，2022年5月29日生\n委托代理人孙七，深圳市C律师事务所律师。\n被告周八，1990年7月28日出生\n委托代理人吴九，山东D律师事务所律师'
    ]
    schema1 = ['法院', {'原告': '委托代理人'}, {'被告': '委托代理人'}]
    schema2 = [{'原告': ['出生日期', '委托代理人']}, {'被告': ['出生日期', '委托代理人']}]

    args.device = 'gpu'
    args.schema = schema1
    predictor = UIEPredictor(args)

    print("-----------------------------")
    outputs = predictor.predict(texts)
    for text, output in zip(texts, outputs):
        print("1. Input text: ")
        print(text)
        print("2. Input schema: ")
        print(schema1)
        print("3. Result: ")
        pprint(output)
        print("-----------------------------")

    # Reset schema
    predictor.set_schema(schema2)
    outputs = predictor.predict(texts)
    for text, output in zip(texts, outputs):
        print("1. Input text: ")
        print(text)
        print("2. Input schema: ")
        print(schema2)
        print("3. Result: ")
        pprint(output)
        print("-----------------------------")


if __name__ == "__main__":
    main()
