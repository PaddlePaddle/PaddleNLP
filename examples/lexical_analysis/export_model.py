# -*- coding: UTF-8 -*-
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.static import InputSpec
from paddlenlp.data import Vocab

from data import load_vocab
from model import BiGruCrf

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--data_dir", type=str, default=None, help="The folder where the dataset is located.")
parser.add_argument("--params_path", type=str, default='./checkpoints/final.pdparams', help="The path of model parameter to be loaded.")
parser.add_argument("--output_path", type=str, default='./infer_model/static_graph_params', help="The path of model parameter in static graph to be saved.")
parser.add_argument("--emb_dim", type=int, default=128, help="The dimension in which a word is embedded.")
parser.add_argument("--hidden_size", type=int, default=128, help="The number of hidden nodes in the GRU layer.")
args = parser.parse_args()
# yapf: enable


def main():
    word_vocab = load_vocab(os.path.join(args.data_dir, 'word.dic'))
    label_vocab = load_vocab(os.path.join(args.data_dir, 'tag.dic'))

    model = BiGruCrf(args.emb_dim, args.hidden_size, len(word_vocab),
                     len(label_vocab))

    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    model.eval()

    model = paddle.jit.to_static(model,
                                 input_spec=[
                                     InputSpec(shape=[None, None],
                                               dtype="int64",
                                               name='token_ids'),
                                     InputSpec(shape=[None],
                                               dtype="int64",
                                               name='length')
                                 ])
    # Save in static graph model.
    paddle.jit.save(model, args.output_path)


if __name__ == "__main__":
    main()
