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
from paddlenlp.utils.log import logger
from paddlenlp.transformers import ElectraForSequenceClassification
from model import ElectraForBinaryTokenClassification, ElectraForSPO

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

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset', required=True, type=str, help='The name of dataset used for training.')
parser.add_argument('--params_path', type=str, required=True, default='./checkpoint/', help='The path to model parameters to be loaded.')
parser.add_argument('--output_path', type=str, default='./export', help='The path of model parameter in static graph to be saved.')
args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":
    # Load the model parameters.
    if args.train_dataset not in NUM_CLASSES:
        raise ValueError(f"Please modify the code to fit {args.dataset}")

    if args.train_dataset == 'CMeEE':
        model = ElectraForBinaryTokenClassification.from_pretrained(
            args.params_path, num_classes=NUM_CLASSES[args.train_dataset])
    elif args.train_dataset == 'CMeIE':
        model = ElectraForSPO.from_pretrained(
            args.params_path, num_classes=NUM_CLASSES[args.train_dataset])
    else:
        model = ElectraForSequenceClassification.from_pretrained(
            args.params_path,
            num_classes=NUM_CLASSES[args.train_dataset],
            activation='tanh')

    model.eval()

    # Convert to static graph with specific input description:
    # input_ids, token_type_ids and position_ids.
    input_spec = [
        paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        paddle.static.InputSpec(shape=[None, None], dtype='int64')
    ]
    model = paddle.jit.to_static(model, input_spec=input_spec)

    # Save in static graph model.
    save_path = os.path.join(args.output_path, 'inference')
    paddle.jit.save(model, save_path)
