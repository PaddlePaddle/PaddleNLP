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
from paddlenlp.transformers import AutoModel

from model import PointwiseMatching

if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str, required=True, default='./checkpoint/model_900/model_state.pdparams', help="The path to model parameters to be loaded.")
    parser.add_argument("--output_path", type=str, default='./output', help="The path of model parameter in static graph to be saved.")
    args = parser.parse_args()
    # yapf: enable

    pretrained_model = AutoModel.from_pretrained('ernie-3.0-medium-zh')
    model = PointwiseMatching(pretrained_model)

    if args.params_path:
        if os.path.isfile(args.params_path):
            state_dict = paddle.load(args.params_path)
            model.set_dict(state_dict)
            print("Loaded parameters from %s" % args.params_path)
        elif os.path.isdir(args.params_path):
            path = os.path.join(args.params_path, "model_state.pdparams")
            state_dict = paddle.load(path)
            model.set_dict(state_dict)
            print("Loaded parameters from %s" % path)
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
    save_path = os.path.join(args.output_path, "inference")
    paddle.jit.save(model, save_path)
