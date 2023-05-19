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
from base_model import SemanticIndexBaseStatic

from paddlenlp.transformers import AutoModel, AutoTokenizer

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, required=True, default='./checkpoint/model_900/model_state.pdparams', help="The path to model parameters to be loaded.")
parser.add_argument("--output_path", type=str, default='./output', help="The path of model parameter in static graph to be saved.")
parser.add_argument("--output_emb_size", default=0, type=int, help="output_embedding_size")
parser.add_argument("--model_name_or_path", default='rocketqa-zh-dureader-query-encoder', type=str, help='The pretrained model used for training')
args = parser.parse_args()
# fmt: on

if __name__ == "__main__":
    pretrained_model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = SemanticIndexBaseStatic(pretrained_model, output_emb_size=args.output_emb_size)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError("Please set --params_path with correct pretrained model file")
    model.eval()
    # Convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # segment_ids
        ],
    )
    # Save in static graph model.
    save_path = os.path.join(args.output_path, "inference")
    paddle.jit.save(model, save_path)
