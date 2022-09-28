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
from paddlenlp.data import Vocab
from model import TextCNNModel

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--vocab_path", type=str, default="./robot_chat_word_dict.txt", help="The path to vocabulary.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--params_path", type=str, default='./checkpoints/final.pdparams', help="The path of model parameter to be loaded.")
parser.add_argument("--output_path", type=str, default='./static_graph_params', help="The path of model parameter in static graph to be saved.")
args = parser.parse_args()
# yapf: enable


def main():
    # Load vocab.
    if not os.path.exists(args.vocab_path):
        raise RuntimeError('The vocab_path  can not be found in the path %s' %
                           args.vocab_path)

    vocab = Vocab.load_vocabulary(args.vocab_path)
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    # Construct the newtork.
    vocab_size = len(vocab)
    num_classes = len(label_map)
    pad_token_id = vocab.to_indices('[PAD]')

    model = TextCNNModel(vocab_size,
                         num_classes,
                         padding_idx=pad_token_id,
                         ngram_filter_sizes=(1, 2, 3))

    # Load model parameters.
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    model.eval()

    inputs = [paddle.static.InputSpec(shape=[None, None], dtype="int64")]

    model = paddle.jit.to_static(model, input_spec=inputs)
    # Save in static graph model.
    paddle.jit.save(model, args.output_path)


if __name__ == "__main__":
    main()
