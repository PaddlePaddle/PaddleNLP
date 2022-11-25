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

import sys

sys.path.append(".")
import argparse
import math
from pprint import pprint

import paddle
from paddlenlp.transformers import AutoTokenizer

from model import UIE
from predictor import UIEPredictor


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoint/model_best",
        help="The directory of inference model to be used.",
    )
    parser.add_argument(
        "--position_prob",
        default=0.5,
        type=float,
        help="Probability threshold for start/end index probabiliry.",
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
                        help="Batch size per CPU for inference.")

    parser.add_argument("--device_id",
                        default=0,
                        type=int,
                        help="The device ID.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load uie model
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
    model = UIE.from_pretrained(args.ckpt_dir)

    # define schema for pre-defined aspects, schema and initializing UIEPredictor
    schema = ["观点词", "情感倾向[正向,负向,未提及]"]
    # aspects = ["房间", "服务", "环境", "位置", "隔音", "价格","卫生"]
    # aspects = ["隔音", "隔声", "价格", "价钱", "费用", "位置", "所处位置"]
    aspects = ["服务", "价格"]

    predictor = UIEPredictor(args, model, schema)

    # test examples
    # texts = [
    #     '环境挺好的，房间很大，也很划算',
    #     '店面不太干净，服务员服务也不热情，地方也不太好找',
    #     '房间太小没有窗，隔音差，不过非常便宜，很划算',
    #     '环境好，位置佳，房间大，服务态度也很好，不过非常贵'
    # ]
    # texts = [
    #     "房间比较大，就是感觉贵了点，不太划算",
    #     "这次来荆州给我的房间小的无语了，所幸比较便宜",
    #     "不错的酒店，逛街都方便，酒店卫生，房内设施设备都不错，真是物有所值，就是不太好找",
    #     "酒店离虞山景点较近，房间不大，而且很脏，不建议去",
    #     "酒店不大，有点不干净"
    # ]
    # texts = [
    #     "酒店离虞山景点较近，房间不大，隔音也不好",
    #     "不错的酒店，逛街都方便，酒店卫生，房内设施设备都不错，就是地方不太好找",
    #     "房间比较小，就是只能看一场电影，胜在比较便宜"
    # ]
    texts = ['环境挺好的，卫生间不错，电视特别好，高清的', '店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队']

    # predict with uie and show corresponding result
    print("-----------------------------")
    outputs = predictor.predict(texts, aspects)
    for text, output in zip(texts, outputs):
        print("1. Input text: ")
        print(text)
        print("2. Input schema: ")
        print(schema)
        print("3. Result: ")
        pprint(output)
        print("-----------------------------")


if __name__ == "__main__":
    main()
