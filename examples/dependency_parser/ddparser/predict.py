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
import time
import copy
from functools import partial

import numpy as np
import paddle
import paddlenlp as ppnlp
from paddlenlp.data import Vocab
from paddlenlp.datasets import load_dataset

from data import create_dataloader, read_predict_data, convert_example
from model.dep import BiaffineDependencyModel
from utils import decode, index_sample

# yapf: disable
parser = argparse.ArgumentParser()
# Predict
parser.add_argument("--predict_data_file", type=str, default=None, required=True, help="The path of test dataset to be loaded.")
parser.add_argument("--model_file_path", type=str, default='model_file/best.pdparams', required=True, help="Directory to load model parameters.")
parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--encoding_model", choices=["lstm", "lstm-pe", "ernie-1.0", "ernie-tiny", "ernie-gram-zh"], type=str, default="ernie-1.0", help="Select the encoding model.")
parser.add_argument("--batch_size", type=int, default=1000, help="Numbers of examples a batch for training.")
parser.add_argument("--infer_output_file", type=str, default='infer_output.conll', help="The path to save infer results.")
# Preprocess
parser.add_argument("--n_buckets", type=int, default=15, help="Number of buckets to devide the dataset.")
parser.add_argument("--fix_len", type=int, default=20, help="The fixed length to pad the sequence")
# Postprocess
parser.add_argument("--tree", type=bool, default=True, help="Ensure the output conforms to the tree structure.")
# Lstm
parser.add_argument("--feat", choices=["char", "pos"], type=str, default=None, help="The feature representation to use.")
args = parser.parse_args()
# yapf: enable


@paddle.no_grad()
def batch_predict(
        model, 
        data_loader,
        rel_vocab,
        word_pad_index,
        word_bos_index,
        word_eos_index,
    ):
    arcs, rels = [], []
    for inputs in data_loader():
        if args.encoding_model.startswith("ernie") or args.encoding_model == "lstm-pe":
            words = inputs[0]
            s_arc, s_rel, words = model(words)
        else:
            words, feats = inputs
            s_arc, s_rel, words = model(words, feats)
        mask = paddle.logical_and(
            paddle.logical_and(words != word_pad_index, words != word_bos_index),
            words != word_eos_index,
        )
        lens = paddle.sum(paddle.cast(mask, "int32"), axis=-1)
        arc_preds, rel_preds = decode(s_arc, s_rel, mask)
        arcs.extend(paddle.split(paddle.masked_select(arc_preds, mask), lens.numpy().tolist()))
        rels.extend(paddle.split(paddle.masked_select(rel_preds, mask), lens.numpy().tolist()))

    arcs = [[str(s) for s in seq.numpy().tolist()] for seq in arcs]
    rels = [rel_vocab.to_tokens(seq.numpy().tolist()) for seq in rels]           

    return arcs, rels


def do_predict(args):
    paddle.set_device(args.device)

    if args.encoding_model == "ernie-gram-zh":
        tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained(args.encoding_model)
    elif args.encoding_model.startswith("ernie"):
        tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(args.encoding_model)
    elif args.encoding_model == "lstm-pe":
        tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained("ernie-1.0")
    else:
        tokenizer = None
    
    predict_ds = load_dataset(read_predict_data, filename=args.predict_data_file, lazy=False)
    predict_ds_copy = copy.deepcopy(predict_ds)

    # Load vocabs from model file path
    vocab_dir = os.path.split(args.model_file_path)[0]
    word_vocab = Vocab.from_json(os.path.join(vocab_dir, "word_vocab.json"))
    rel_vocab = Vocab.from_json(os.path.join(vocab_dir, "rel_vocab.json"))
    feat_vocab_path = os.path.join(vocab_dir, "feat_vocab.json")
    if os.path.exists(feat_vocab_path):
        feat_vocab = Vocab.from_json(os.path.join(feat_vocab_path))
    else:
        feat_vocab = None

    if feat_vocab:
        n_feats = len(feat_vocab)
        word_pad_index = word_vocab.to_indices("[PAD]")
        word_bos_index = word_vocab.to_indices("[BOS]")
        word_eos_index = word_vocab.to_indices("[EOS]")
    else:
        n_feats = None
        word_pad_index = word_vocab.to_indices("[PAD]")
        word_bos_index = word_vocab.to_indices("[CLS]")
        word_eos_index = word_vocab.to_indices("[SEP]")

    n_rels, n_words = len(rel_vocab), len(word_vocab)
    vocabs = [word_vocab, feat_vocab, rel_vocab]

    trans_fn = partial(
        convert_example,
        tokenizer=tokenizer,
        vocabs=vocabs,
        encoding_model=args.encoding_model,
        feat=args.feat,
    )

    predict_data_loader, buckets = create_dataloader(
        predict_ds,
        vocabs=vocabs,
        batch_size=args.batch_size,
        mode="predict",
        n_buckets=args.n_buckets,
        trans_fn=trans_fn,
    )

    # Load pretrained model if encoding model is ernie-1.0, ernie-tiny or ernie-gram-zh
    if args.encoding_model in ["ernie-1.0", "ernie-tiny"]:
        pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(args.encoding_model)
    elif args.encoding_model == "ernie-gram-zh":
        pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(args.encoding_model)
    else:
        pretrained_model = None

    # Load ddparser model
    model = BiaffineDependencyModel(
        encoding_model=args.encoding_model,
        feat=args.feat,
        n_rels=n_rels,
        n_feats=n_feats,
        n_words=n_words,
        pad_index=word_pad_index,
        eos_index=word_eos_index,
        pretrained_model=pretrained_model,
    )
    
    # Load saved model parameters
    if os.path.isfile(args.model_file_path):
        state_dict = paddle.load(args.model_file_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.model_file_path)
    else:
        raise ValueError("The parameters path is incorrect or not specified.")

    # Start predict
    pred_arcs, pred_rels = batch_predict(
        model,         
        predict_data_loader,
        rel_vocab, 
        word_pad_index,
        word_bos_index,
        word_eos_index,
    )

    # Restore the order of sentences in the buckets
    indices = np.argsort(np.array([i for bucket in buckets.values() for i in bucket]))
    pred_heads = [pred_arcs[i] for i in indices]
    pred_deprels = [pred_rels[i] for i in indices]

    with open(args.infer_output_file, 'w', encoding='utf-8') as out_file:
        for res, head, rel in zip(predict_ds_copy, pred_heads, pred_deprels):
            res["HEAD"] = tuple(head)
            res["DEPREL"] = tuple(rel)
            res = '\n'.join('\t'.join(map(str, line)) for line in zip(*res.values())) + '\n'
            out_file.write("{}\n".format(res)) 
    out_file.close()
    print("Results saved!")

if __name__ == "__main__":
    do_predict(args)
