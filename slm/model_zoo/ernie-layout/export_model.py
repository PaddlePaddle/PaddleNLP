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
import os

import paddle

from paddlenlp.transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, default='./ernie-layoutx-base-uncased/models/funsd/1e-5_2/', help="The path to model parameters to be loaded.")
parser.add_argument("--task_type", type=str, required=True, default="ner", choices=["ner", "cls", "mrc"], help="Select the task type.")
parser.add_argument("--output_path", type=str, default='./export', help="The path of model parameter in static graph to be saved.")
args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":
    if args.task_type == "ner":
        model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    elif args.task_type == "mrc":
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    elif args.task_type == "cls":
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    else:
        raise ValueError("Unsppoorted task type!")
    model.eval()

    # Convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            paddle.static.InputSpec(shape=[None, None, None], dtype="int64", name="bbox"),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype="int64", name="image"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="attention_mask"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
        ],
    )
    # Save in static graph model.
    save_path = os.path.join(args.output_path, "inference")
    paddle.jit.save(model, save_path)
