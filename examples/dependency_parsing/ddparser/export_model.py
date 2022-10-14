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
from paddlenlp.transformers import AutoModel, AutoTokenizer

from model.dep import BiAffineParser
from data import load_vocab

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--encoding_model", choices=["lstm-pe", "ernie-1.0", "ernie-3.0-medium-zh", "ernie-tiny", "ernie-gram-zh"], type=str, default="ernie-3.0-medium-zh", help="Select the encoding model.")
parser.add_argument("--params_path", type=str, required=True, default='./model_file/best.pdparams', help="The path to model parameters to be loaded.")
parser.add_argument("--output_path", type=str, default='./output', help="The path of model parameter in static graph to be saved.")
args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":

    # Load pretrained model if encoding model is ernie-3.0-medium-zh, ernie-1.0, ernie-tiny or ernie-gram-zh
    if args.encoding_model in [
            "ernie-3.0-medium-zh", "ernie-1.0", "ernie-tiny", "ernie-gram-zh"
    ]:
        pretrained_model = AutoModel.from_pretrained(args.encoding_model)
    else:
        pretrained_model = None

    # Load vocabs from model file path
    vocab_dir = os.path.split(args.params_path)[0]
    word_vocab, _, rel_vocab = load_vocab(vocab_dir)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Save vocabs to output path
    word_vocab.to_json(path=os.path.join(args.output_path, "word_vocab.json"))
    rel_vocab.to_json(path=os.path.join(args.output_path, "rel_vocab.json"))

    n_rels, n_words, n_feats = len(rel_vocab), len(word_vocab), None

    word_pad_index = word_vocab.to_indices("[PAD]")
    word_bos_index = word_vocab.to_indices("[CLS]")
    word_eos_index = word_vocab.to_indices("[SEP]")

    # Load ddparser model
    model = BiAffineParser(
        encoding_model=args.encoding_model,
        feat=None,
        n_rels=n_rels,
        n_feats=n_feats,
        n_words=n_words,
        pad_index=word_pad_index,
        eos_index=word_eos_index,
        pretrained_model=pretrained_model,
    )

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    model.eval()

    # Convert to static graph with specific input description
    model = paddle.jit.to_static(model,
                                 input_spec=[
                                     paddle.static.InputSpec(shape=[None, None],
                                                             dtype="int64"),
                                     paddle.static.InputSpec(shape=[None, None],
                                                             dtype="int64"),
                                 ])
    # Save in static graph model.
    save_path = os.path.join(args.output_path, "inference")
    paddle.jit.save(model, save_path)
