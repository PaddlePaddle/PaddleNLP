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
from paddlenlp.transformers import AutoModel

from model import TPLinkerPlus
from utils import get_label_dict

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, default='./checkpoint/model_best', help="The path to model parameters to be loaded.")
parser.add_argument("--output_path", type=str, default='./export', help="The path of model parameter in static graph to be saved.")
parser.add_argument("--label_dict_path", default="./ner_data/label_dict.json", type=str, help="The file path of the labels dictionary.")
parser.add_argument("--task_type", choices=['relation_extraction', 'event_extraction', 'entity_extraction', 'opinion_extraction'], default="entity_extraction", type=str, help="Select the training task type.")
args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":
    label_dict = get_label_dict(args.task_type, args.label_dict_path)
    num_tags = len(label_dict["id2tag"])

    encoder = AutoModel.from_pretrained("ernie-3.0-base-zh")
    model = TPLinkerPlus(encoder, num_tags, shaking_type="cln")
    state_dict = paddle.load(
        os.path.join(args.model_path, "model_state.pdparams"))
    model.set_dict(state_dict)
    model.eval()

    # Convert to static graph with specific input description
    model = paddle.jit.to_static(model,
                                 input_spec=[
                                     paddle.static.InputSpec(shape=[None, None],
                                                             dtype="int64",
                                                             name='input_ids'),
                                     paddle.static.InputSpec(shape=[None, None],
                                                             dtype="int64",
                                                             name='att_mask'),
                                 ])
    # Save in static graph model.
    save_path = os.path.join(args.output_path, "inference")
    paddle.jit.save(model, save_path)
