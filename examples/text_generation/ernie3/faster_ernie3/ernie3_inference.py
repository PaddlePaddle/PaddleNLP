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

import time
import argparse
import numpy as np
from pprint import pprint

import paddle
import paddle.inference as paddle_infer

from paddlenlp.transformers import Ernie3Tokenizer, Ernie3ForGeneration
from paddlenlp.ops.ext_utils import load


def parse_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_model_dir",
        default="./infer_model/",
        type=str,
        help="Path to save inference model of ernie3. ")
    parser.add_argument(
        "--model_name_or_path",
        default="ernie3-10b",
        type=str,
        help="The model name to specify the ernie3 to use. ")
    args = parser.parse_args()

    return args


def tokenize_input(tokenizer, texts):
    input_ids = []
    max_len = 0
    for text in texts:
        ids = tokenizer(text)['input_ids']
        max_len = max(max_len, len(ids))
        input_ids.append(ids)
    for i in range(len(input_ids)):
        if len(input_ids[i]) < max_len:
            input_ids[i] += [tokenizer.pad_token_id] * (
                max_len - len(input_ids[i]))
    input_ids = np.asarray(input_ids, dtype="int32")
    return input_ids


def infer(args):
    tokenizer = Ernie3Tokenizer.from_pretrained(args.model_name_or_path)

    texts = ["中国的首都是哪里"]
    input_ids = tokenize_input(tokenizer, texts)

    # Load FasterTransformer lib. 
    load("FasterTransformer", verbose=True)

    config = paddle_infer.Config(args.inference_model_dir + "ernie3.pdmodel",
                                 args.inference_model_dir + "ernie3.pdiparams")
    config.enable_use_gpu(100, 0)
    # config.disable_glog_info()
    predictor = paddle_infer.create_predictor(config)

    input_handles = {}
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    input_handle.copy_from_cpu(input_ids)

    predictor.run()

    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()

    for sample in output_data.transpose([1, 0]).tolist():
        print(tokenizer.convert_ids_to_string(sample))


if __name__ == "__main__":
    args = parse_args()
    pprint(args)

    infer(args)
