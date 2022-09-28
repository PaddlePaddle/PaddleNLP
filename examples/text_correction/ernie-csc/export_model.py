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
from paddlenlp.transformers import ErnieModel
from model import ErnieForCSC

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--params_path", type=str, default='./checkpoints/final.pdparams', help="The path of model parameter to be loaded.")
parser.add_argument("--output_path", type=str, default='./infer_model/static_graph_params', help="The path of model parameter in static graph to be saved.")
parser.add_argument("--model_name_or_path", type=str, default="ernie-1.0", choices=["ernie-1.0"], help="Pretraining model name or path")
parser.add_argument("--pinyin_vocab_file_path", type=str, default="pinyin_vocab.txt", help="pinyin vocab file path")
args = parser.parse_args()
# yapf: enable


def main():
    pinyin_vocab = Vocab.load_vocabulary(args.pinyin_vocab_file_path,
                                         unk_token='[UNK]',
                                         pad_token='[PAD]')

    ernie = ErnieModel.from_pretrained(args.model_name_or_path)

    model = ErnieForCSC(ernie,
                        pinyin_vocab_size=len(pinyin_vocab),
                        pad_pinyin_id=pinyin_vocab[pinyin_vocab.pad_token])

    model_dict = paddle.load(args.params_path)
    model.set_dict(model_dict)
    model.eval()

    model = paddle.jit.to_static(model,
                                 input_spec=[
                                     InputSpec(shape=[None, None],
                                               dtype="int64",
                                               name='input_ids'),
                                     InputSpec(shape=[None, None],
                                               dtype="int64",
                                               name='pinyin_ids')
                                 ])

    paddle.jit.save(model, args.output_path)


if __name__ == "__main__":
    main()
