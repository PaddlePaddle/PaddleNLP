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

import os
import argparse

import LAC
import paddle
import paddlenlp as ppnlp

from env import Environment
from data import batchify, TextDataset, Corpus
from model.dep import BiaffineDependencyModel
from utils import decode


@paddle.no_grad()
def predict(env, args, model, data_loader):
    arcs, rels = [], []
    for inputs in data_loader():
        if args.encoding_model.startswith("ernie") or args.encoding_model == "lstm-pe":
            words = inputs[0]
            s_arc, s_rel, words = model(words)
        else:
            words, feats = inputs
            s_arc, s_rel, words = model(words, feats)
        mask = paddle.logical_and(
            paddle.logical_and(words != args.pad_index, words != args.bos_index),
            words != args.eos_index,
        )
        lens = paddle.sum(paddle.cast(mask, "int32"), axis=-1)
        arc_preds, rel_preds = decode(args, s_arc, s_rel, mask)
        arcs.extend(paddle.split(paddle.masked_select(arc_preds, mask), lens.numpy().tolist()))
        rels.extend(paddle.split(paddle.masked_select(rel_preds, mask), lens.numpy().tolist()))

    arcs = [seq.numpy().tolist() for seq in arcs]
    rels = [env.REL.vocab[seq.numpy().tolist()] for seq in rels]           

    return arcs, rels


class Parser(object):
    def __init__(
        self,
        device="gpu",
        tree=True,
        n_buckets=15,
        batch_size=1000,
        encoding_model="ernie-1.0",
    ):
        paddle.set_device(device)

        args = argparse.ArgumentParser().parse_args()
        args.mode = "predict"
        args.batch_size = batch_size
        args.preprocess = False
        args.device = device
        args.tree = tree
        args.n_buckets = n_buckets
        args.encoding_model = encoding_model
        args.save_dir = encoding_model

        self.env = Environment(args)
        self.args = self.env.args

        if args.encoding_model in ["ernie-1.0", "ernie-tiny"]:
            self.pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(args.encoding_model)
        elif args.encoding_model == "ernie-gram-zh":
            self.pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(args.encoding_model)

        if args.encoding_model.startswith("ernie"):
            self.model = BiaffineDependencyModel(args=self.args, pretrained_model=self.pretrained_model)
        else:
            self.model = BiaffineDependencyModel(args=self.args)
    
        params_path = os.path.join(args.encoding_model, "model_state.pdparams")
        state_dict = paddle.load(params_path)
        self.model.set_dict(state_dict)
        self.model.eval()

        self.lac = LAC.LAC(mode="seg", use_cuda=True if args.device == "gpu" else False)


    def predict(self, inputs):

        if isinstance(inputs, str):
            inputs = [inputs]

        lac_results = []
        position = 0

        while position < len(inputs):
            lac_results += self.lac.run(inputs[position:position + self.args.batch_size])
            position += self.args.batch_size
        data = Corpus.load_lac_results(lac_results, self.env.fields)

        predict_ds = TextDataset(data, [self.env.WORD, self.env.FEAT], self.args.n_buckets)

        predict_data_loader = batchify(
            predict_ds, 
            self.args.batch_size, 
            use_multiprocess=False,
            sequential_sampler=True    
        )

        pred_arcs, pred_rels = predict(self.env, self.args, self.model, predict_data_loader)

        indices = range(len(pred_arcs))
        data.head = [pred_arcs[i] for i in indices]
        data.deprel = [pred_rels[i] for i in indices]
        outputs = data.get_result()
        return outputs
