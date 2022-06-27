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
import paddlenlp as ppnlp
from paddlenlp.transformers import AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--params_path",
                    type=str,
                    default='./checkpoint/model_state.pdparams',
                    help="The path to model parameters to be loaded.")
parser.add_argument("--output_path",
                    type=str,
                    default='./export',
                    help="The path of model parameter in "
                    "static graph to be saved.")
parser.add_argument("--num_classes",
                    default=141,
                    type=int,
                    help="Number of classes for "
                    "hierarchical classfication tasks.")
parser.add_argument('--model_name',
                    default="ernie-2.0-base-en",
                    help="Select model to train, defaults "
                    "to ernie-2.0-base-en.")
args = parser.parse_args()

if __name__ == "__main__":

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_classes=args.num_classes)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    model.eval()

    # Convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64")  # segment_ids
        ])

    # Save in static graph model.
    save_path = os.path.join(args.output_path, "float32")
    paddle.jit.save(model, save_path)
