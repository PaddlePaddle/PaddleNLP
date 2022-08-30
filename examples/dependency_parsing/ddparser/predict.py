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
from paddlenlp.transformers import AutoTokenizer, AutoModel
from paddlenlp.datasets import load_dataset

from data import create_dataloader, convert_example, load_vocab
from model.dep import BiAffineParser
from utils import decode, index_sample, flat_words

# yapf: disable
parser = argparse.ArgumentParser()
# Predict
parser.add_argument("--params_path", type=str, default='model_file/best.pdparams', required=True, help="Directory to load model parameters.")
parser.add_argument("--task_name", choices=["nlpcc13_evsam05_thu", "nlpcc13_evsam05_hit"], type=str, default="nlpcc13_evsam05_thu", help="Select the task.")
parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--encoding_model", choices=["lstm", "lstm-pe", "ernie-3.0-medium-zh", "ernie-1.0", "ernie-tiny", "ernie-gram-zh"], type=str, default="ernie-3.0-medium-zh", help="Select the encoding model.")
parser.add_argument("--batch_size", type=int, default=1000, help="Numbers of examples a batch for training.")
parser.add_argument("--infer_output_file", type=str, default='infer_output.conll', help="The path to save infer results.")
# Preprocess
parser.add_argument("--n_buckets", type=int, default=15, help="Number of buckets to devide the dataset.")
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

    model.eval()
    arcs, rels = [], []
    for inputs in data_loader():
        if args.encoding_model.startswith(
                "ernie") or args.encoding_model == "lstm-pe":
            words = inputs[0]
            words, feats = flat_words(words)
            s_arc, s_rel, words = model(words, feats)
        else:
            words, feats = inputs
            s_arc, s_rel, words = model(words, feats)

        mask = paddle.logical_and(
            paddle.logical_and(words != word_pad_index,
                               words != word_bos_index),
            words != word_eos_index,
        )

        lens = paddle.sum(paddle.cast(mask, "int32"), axis=-1)
        arc_preds, rel_preds = decode(s_arc, s_rel, mask)
        arcs.extend(
            paddle.split(paddle.masked_select(arc_preds, mask),
                         lens.numpy().tolist()))
        rels.extend(
            paddle.split(paddle.masked_select(rel_preds, mask),
                         lens.numpy().tolist()))

    arcs = [[str(s) for s in seq.numpy().tolist()] for seq in arcs]
    rels = [rel_vocab.to_tokens(seq.numpy().tolist()) for seq in rels]

    return arcs, rels


def do_predict(args):
    paddle.set_device(args.device)

    if args.encoding_model.startswith("ernie"):
        tokenizer = AutoTokenizer.from_pretrained(args.encoding_model)
    elif args.encoding_model == "lstm-pe":
        tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
    else:
        tokenizer = None

    # Load vocabs from model file path
    vocab_dir = os.path.split(args.params_path)[0]
    word_vocab, feat_vocab, rel_vocab = load_vocab(vocab_dir)

    n_rels, n_words = len(rel_vocab), len(word_vocab)
    if args.encoding_model == "lstm":
        n_feats = len(feat_vocab)
        word_pad_index = word_vocab.to_indices("[PAD]")
        word_bos_index = word_vocab.to_indices("[BOS]")
        word_eos_index = word_vocab.to_indices("[EOS]")
    else:
        n_feats = None
        word_pad_index = word_vocab.to_indices("[PAD]")
        word_bos_index = word_vocab.to_indices("[CLS]")
        word_eos_index = word_vocab.to_indices("[SEP]")

    test_ds = load_dataset(args.task_name, splits=["test"])
    test_ds_copy = copy.deepcopy(test_ds)

    trans_fn = partial(
        convert_example,
        vocabs=[word_vocab, feat_vocab, rel_vocab],
        encoding_model=args.encoding_model,
        feat=args.feat,
        mode="test",
    )

    test_data_loader, buckets = create_dataloader(
        test_ds,
        batch_size=args.batch_size,
        mode="test",
        n_buckets=args.n_buckets,
        trans_fn=trans_fn,
    )

    # Load pretrained model if encoding model is ernie-3.0-medium-zh, ernie-1.0, ernie-tiny or ernie-gram-zh
    if args.encoding_model in [
            "ernie-3.0-medium-zh", "ernie-1.0", "ernie-tiny"
    ]:
        pretrained_model = AutoModel.from_pretrained(args.encoding_model)
    elif args.encoding_model == "ernie-gram-zh":
        pretrained_model = AutoModel.from_pretrained(args.encoding_model)
    else:
        pretrained_model = None

    # Load model
    model = BiAffineParser(
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
    if os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError("The parameters path is incorrect or not specified.")

    # Start predict
    pred_arcs, pred_rels = batch_predict(
        model,
        test_data_loader,
        rel_vocab,
        word_pad_index,
        word_bos_index,
        word_eos_index,
    )

    # Restore the order of sentences in the buckets
    if buckets:
        indices = np.argsort(
            np.array([i for bucket in buckets.values() for i in bucket]))
    else:
        indices = range(len(pred_arcs))
    pred_heads = [pred_arcs[i] for i in indices]
    pred_deprels = [pred_rels[i] for i in indices]

    with open(args.infer_output_file, 'w', encoding='utf-8') as out_file:
        for res, head, rel in zip(test_ds_copy, pred_heads, pred_deprels):
            res["HEAD"] = tuple(head)
            res["DEPREL"] = tuple(rel)
            res = '\n'.join('\t'.join(map(str, line))
                            for line in zip(*res.values())) + '\n'
            out_file.write("{}\n".format(res))
    out_file.close()
    print("Results saved!")


if __name__ == "__main__":
    do_predict(args)
