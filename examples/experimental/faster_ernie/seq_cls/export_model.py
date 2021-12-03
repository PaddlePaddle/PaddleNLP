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
import paddlenlp
from paddlenlp.experimental import FasterErnieForSequenceClassification

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="ckpt/model_900", help="The path to model parameters to be loaded.")
parser.add_argument("--output_path", type=str, default='./export', help="The path of model parameter in static graph to be saved.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":
    # The number of labels should be in accordance with the training dataset.
    label_map = {0: 'negative', 1: 'positive'}
    model = FasterErnieForSequenceClassification.from_pretrained(args.save_dir)
    save_path = os.path.join(args.output_path, "inference")
    model.to_static(save_path)
