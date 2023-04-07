# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import paddle.distributed.fleet as fleet

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../", "../")))

from ppfleetx.core.engine.inference_engine import InferenceEngine
from ppfleetx.data import tokenizers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp_degree", default=1, type=int, help="")
    parser.add_argument("--model_dir", default="output", type=str, help="model directory")

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    fleet.init(is_collective=True)
    infer_engine = InferenceEngine(args.model_dir, args.mp_degree)

    tokenizer = tokenizers.GPTTokenizer.from_pretrained("gpt2")
    input_text = "Hi, GPT2. Tell me where is Beijing?"
    ids = [tokenizer.encode(input_text)]

    # run test

    outs = infer_engine.predict([ids])

    ids = list(outs.values())[0]
    out_ids = [int(x) for x in ids[0]]
    result = tokenizer.decode(out_ids)
    result = input_text + result

    print("Prompt:", input_text)
    print("Generation:", result)


if __name__ == "__main__":
    main()
