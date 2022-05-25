# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from multiprocessing import cpu_count
from uie_predictor import UIEPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_path_prefix",
        type=str,
        required=True,
        help="The path prefix of inference model to be used.", )
    parser.add_argument(
        "--infer_model_dir",
        type=str,
        default='./export',
        help="The path to model parameter in onnx to be saved.", )
    parser.add_argument(
        "--max_seq_len",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--use_quantize",
        action='store_true',
        help="Whether to use quantization for acceleration.", )
    parser.add_argument(
        "--num_threads",
        default=cpu_count(),
        type=int,
        help="num_threads for cpu.", )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    text = [
        '（右肝肿瘤）肝细胞性肝癌（II-III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵及周围肝组织，未见脉管内癌栓（MVI分级：M0级）及卫星子灶形成。（肿物1个，大小4.2×4.0×2.8cm）。'
    ]
    schema = ['肿瘤的大小', '肿瘤的个数', '肝癌级别', '脉管内癌栓分级']

    args.device = 'cpu'
    args.schema = schema
    predictor = UIEPredictor(args)

    outputs = predictor.predict(text)
    from pprint import pprint
    pprint(outputs)


if __name__ == "__main__":
    main()
