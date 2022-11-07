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

        
    # define schema for sentence classification
    # schema = ['情感倾向[正向，负向]']
    # define schema for AS Extraction 
    # schema = [{'评价维度': ['观点词']}]
    # define schema for AO Extraction 
    # schema = [{'评价维度': ['情感倾向[正向，负向]']}]
    # define schema for ASO Extraction 
    schema = [{'评价维度': ['观点词', '情感倾向[正向，负向]']}]
    
    # initializing UIEPredictor
    predictor = UIEPredictor(args, model, schema)

    # test examples
    texts = [
        '环境挺好的，卫生间不错，电视特别好，高清的',
        '店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队'
    ]

    # predict with uie and show corresponding result
    print("-----------------------------")
    outputs = predictor.predict(texts)
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