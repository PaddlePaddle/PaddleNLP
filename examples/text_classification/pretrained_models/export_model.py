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
import paddlenlp as ppnlp

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, required=True, default='./checkpoint/model_900/model_state.pdparams',
    help="The path to model parameters to be loaded.")
parser.add_argument("--output_path", type=str, default='./export',
    help="The path of model parameter in static graph to be saved.")
parser.add_argument('--accelerate_mode', default=False, type=eval,
     help="If true, it will use the FasterTokenizer to tokenize texts.")
args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":
    # The number of labels should be in accordance with the training dataset.
    label_map = {0: 'negative', 1: 'positive'}
    model = ppnlp.transformers.BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_classes=2,
        accelerate_mode=args.accelerate_mode)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    model.eval()

    model.to_static_model(args.output_path)
