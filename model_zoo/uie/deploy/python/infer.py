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
from multiprocessing import cpu_count
import paddle
import numpy as np
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.metrics import SpanEvaluator
from utils import reader, map_offset
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
        "--position_prob",
        default=0.5,
        type=float,
        help="Probability threshold for start/end index probabiliry.", )
    parser.add_argument(
        "--use_fp16",
        action='store_true',
        help="Whether to use fp16 inference, only takes effect when deploying on gpu.",
    )
    parser.add_argument(
        "--use_quantize",
        action='store_true',
        help="Whether to use quantization for acceleration.", )
    parser.add_argument(
        "--num_threads",
        default=cpu_count(),
        type=int,
        help="num_threads for cpu.", )
    parser.add_argument(
        "--device",
        default='gpu',
        type=str,
        help="", )
    parser.add_argument(
        "--mode",
        default='onnx',
        type=str,
        help="", )
    parser.add_argument(
        "--enable_mkldnn",
        action='store_true',
        help="", )
    parser.add_argument(
        "--test_path",
        type=str,
        help="Whether to use fp16 inference, only takes effect when deploying on gpu.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.schema = []

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
    test_ds = load_dataset(
        reader,
        data_path=args.test_path,
        max_seq_len=args.max_seq_len,
        lazy=False)

    input_ids = []
    token_type_ids = []
    position_ids = []
    attention_mask = []
    all_start_ids = []
    all_end_ids = []
    for example in test_ds:
        encoded_inputs = tokenizer(
            text=[example["prompt"]],
            text_pair=[example["content"]],
            stride=len(example["prompt"]),
            truncation=True,
            max_seq_len=args.max_seq_len,
            pad_to_max_seq_len=True,
            return_attention_mask=True,
            return_position_ids=True,
            return_dict=False)
        encoded_inputs = encoded_inputs[0]
        offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"]]
        bias = 0
        for index in range(len(offset_mapping)):
            if index == 0:
                continue
            mapping = offset_mapping[index]
            if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                bias = index
            if mapping[0] == 0 and mapping[1] == 0:
                continue
            offset_mapping[index][0] += bias
            offset_mapping[index][1] += bias

        start_ids = [0 for x in range(args.max_seq_len)]
        end_ids = [0 for x in range(args.max_seq_len)]
        for item in example["result_list"]:
            start = map_offset(item["start"] + bias, offset_mapping)
            end = map_offset(item["end"] - 1 + bias, offset_mapping)
            start_ids[start] = 1.0
            end_ids[end] = 1.0

        input_ids.append(encoded_inputs["input_ids"])
        token_type_ids.append(encoded_inputs["token_type_ids"])
        position_ids.append(encoded_inputs["position_ids"])
        attention_mask.append(encoded_inputs["attention_mask"])
        all_start_ids.append(start_ids)
        all_end_ids.append(end_ids)

    input_dict = {
        "input_ids": np.array(
            input_ids, dtype="int64"),
        "token_type_ids": np.array(
            token_type_ids, dtype="int64"),
        "pos_ids": np.array(
            position_ids, dtype="int64"),
        "att_mask": np.array(
            attention_mask, dtype="int64")
    }
    all_start_ids = np.array(all_start_ids, dtype="int64")
    all_end_ids = np.array(all_end_ids, dtype="int64")

    metric = SpanEvaluator()
    predictor = UIEPredictor(args)

    start_prob, end_prob = predictor._infer(input_dict)
    num_correct, num_infer, num_label = metric.compute(
        start_prob, end_prob, all_start_ids, all_end_ids)

    metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" %
          (precision, recall, f1))


if __name__ == "__main__":
    main()
