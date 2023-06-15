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
import sys

import paddle

from paddlenlp.trainer.argparser import strtobool
from paddlenlp.transformers import PPMiniLMForSequenceClassification

sys.path.append("../")
from data import METRIC_CLASSES  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(METRIC_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_path",
        default="best_clue_model",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--save_inference_model_with_tokenizer",
        type=strtobool,
        default=True,
        help="Whether to save inference model with tokenizer.",
    )

    args = parser.parse_args()
    return args


def do_export(args):
    save_path = os.path.join(os.path.dirname(args.model_path), "inference")
    model = PPMiniLMForSequenceClassification.from_pretrained(args.model_path)
    args.task_name = args.task_name.lower()

    input_spec = [
        paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
        paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # token_type_ids
    ]
    model = paddle.jit.to_static(model, input_spec=input_spec)

    paddle.jit.save(model, save_path)


def print_arguments(args):
    """print arguments"""
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    do_export(args)
