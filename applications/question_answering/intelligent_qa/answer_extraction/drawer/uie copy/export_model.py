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

from model import UIE

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, default='./checkpoint/model_best', help="The path to model parameters to be loaded.")
parser.add_argument("--output_path", type=str, default='./export', help="The path of model parameter in static graph to be saved.")
args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":
    model = UIE.from_pretrained(args.model_path)
    model.eval()

    # Convert to static graph with specific input description
    model = paddle.jit.to_static(model,
                                 input_spec=[
                                     paddle.static.InputSpec(shape=[None, None],
                                                             dtype="int64",
                                                             name='input_ids'),
                                     paddle.static.InputSpec(
                                         shape=[None, None],
                                         dtype="int64",
                                         name='token_type_ids'),
                                     paddle.static.InputSpec(shape=[None, None],
                                                             dtype="int64",
                                                             name='pos_ids'),
                                     paddle.static.InputSpec(shape=[None, None],
                                                             dtype="int64",
                                                             name='att_mask'),
                                 ])
    # Save in static graph model.
    save_path = os.path.join(args.output_path, "inference")
    paddle.jit.save(model, save_path)
