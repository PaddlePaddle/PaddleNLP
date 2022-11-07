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
import sys

sys.path.append(".")
import argparse
import math
from pprint import pprint

import paddle
from paddlenlp.transformers import AutoTokenizer

from model import UIE
from predictor import UIEPredictor
from utils import load_txt, write_json_file


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoint/model_best",
        help="The directory of inference model to be used.",
    )
    parser.add_argument(
        "--test_set_path",
        default="./data/test_hotel.txt",
        type=str,
        help="The Path of test file.",
    )
    parser.add_argument(
        "--save_path",
        default="./outputs/test_hotel.json",
        type=str,
        help="The saving Path of test results.",
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

    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
    schema = [{'评价维度': ['观点词', '情感倾向[正向,负向]']}]

    # initializing UIEPredictor
    predictor = UIEPredictor(args, model, schema)

    # load test examples
    texts = load_txt(args.test_set_path)

    # predict with uie and save predict result
    outputs = predictor.predict(texts)
    write_json_file(outputs, args.save_path)

    print("Infer results has been saved to {}".format(args.save_path))


if __name__ == "__main__":
    main()
