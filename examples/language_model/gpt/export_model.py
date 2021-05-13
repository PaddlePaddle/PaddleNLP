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
import os

import paddle
from paddlenlp.transformers import GPTForGreedyGeneration, GPTChineseTokenizer, GPTTokenizer

MODEL_CLASSES = {
    "gpt-cn": (GPTForGreedyGeneration, GPTChineseTokenizer),
    "gpt": (GPTForGreedyGeneration, GPTTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path of the trained model to be exported.", )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="The output file prefix used to save the exported inference model.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Suild model and load trained parameters
    model = model_class.from_pretrained(args.model_path, max_predict_len=32)
    tokenizer = tokenizer_class.from_pretrained(args.model_path)
    # Switch to eval model
    model.eval()
    # Convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[1], dtype="int32"),  # end_id
        ])

    # Save converted static graph model
    paddle.jit.save(model, args.output_path)
    # Also save tokenizer for inference usage
    tokenizer.save_pretrained(os.path.dirname(args.output_path))


if __name__ == "__main__":
    main()
