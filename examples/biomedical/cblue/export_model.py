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
from paddlenlp.transformers import ElectraForSequenceClassification
from model import ElectraForBinaryTokenClassification, ElectraForSPO

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset', choices=['KUAKE-QIC', 'KUAKE-QQR', 'KUAKE-QTR', 'CHIP-STS', 'CHIP-CTC', 'CHIP-CDN-2C', 'CMeEE', 'CMeIE'],
                                       required=True, type=str, help='The name of dataset used for training.')
parser.add_argument('--params_path', type=str, required=True, default='./checkpoint/model_state.pdparams', help='The path to model parameters to be loaded.')
parser.add_argument('--output_path', type=str, default='./export', help='The path of model parameter in static graph to be saved.')
args = parser.parse_args()
# yapf: enable

NUM_CLASSES = {
    'CHIP-CDN-2C': 2,
    'CHIP-STS': 2,
    'CHIP-CTC': 44,
    'KUAKE-QQR': 3,
    'KUAKE-QTR': 4,
    'KUAKE-QIC': 11,
    'CMeEE': [33, 5],
    'CMeIE': 44
}

if __name__ == "__main__":
    if args.train_dataset == 'CMeEE':
        model = ElectraForBinaryTokenClassification.from_pretrained(
            'ernie-health-chinese', num_classes=NUM_CLASSES[args.train_dataset])
    elif args.train_dataset == 'CMeIE':
        model = ElectraForSPO.from_pretrained(
            'ernie-health-chinese', num_classes=NUM_CLASSES[args.train_dataset])
    else:
        model = ElectraForSequenceClassification.from_pretrained(
            'ernie-health-chinese',
            num_classes=NUM_CLASSES[args.train_dataset],
            activation='tanh')

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    model.eval()

    # Convert to static graph with specific input description
    input_spec = [
        paddle.static.InputSpec(
            shape=[None, None], dtype="int64"),  # input_ids
        paddle.static.InputSpec(
            shape=[None, None], dtype="int64"),  # token_type_ids
        paddle.static.InputSpec(
            shape=[None, None], dtype="int64")  # position_ids
    ]
    if args.train_dataset in ['CMeEE', 'CMeIE']:
        input_spec.append(
            paddle.static.InputSpec(
                shape=[None, None], dtype="float32"))  # masks

    model = paddle.jit.to_static(model, input_spec=input_spec)
    # Save in static graph model.
    save_path = os.path.join(args.output_path, "inference")
    paddle.jit.save(model, save_path)
