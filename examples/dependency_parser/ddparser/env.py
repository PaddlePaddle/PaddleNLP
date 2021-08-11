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
import dill
import random

import numpy as np
import paddle
import paddlenlp as ppnlp

import utils
from data import CoNLL, Field, SubwordField, ErnieField, Corpus


class Environment(object):
    """initialize the enviroment"""
    def __init__(self, args):
        self.args = args
        
        if not args.mode == "train":
            random.seed(args.seed)
            np.random.seed(args.seed)
            paddle.seed(args.seed)

        if args.preprocess and args.mode == "train":
            if args.encoding_model.startswith("ernie") or args.encoding_model == "lstm-pe":
                if args.encoding_model == "lstm-pe":
                    self.tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained("ernie-1.0")
                else:
                    self.tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained(args.encoding_model)

                args.vocab_size = len(self.tokenizer.vocab)
                self.WORD = ErnieField(
                    "word",
                    pad='[PAD]',
                    unk='[UNK]',
                    bos='[CLS]',
                    eos='[SEP]',
                    fix_len=20,
                    tokenizer=self.tokenizer,
                )
                self.WORD.vocab = self.tokenizer.vocab
                args.feat = None
            else:
                self.WORD = Field(
                    "word",
                    pad='[PAD]',
                    unk='[UNK]',
                    bos='[BOS]',
                    eos='[EOS]',
                    lower=True,
                )
            if args.feat == "char":
                self.FEAT = SubwordField(
                    "chars",
                    pad='[PAD]',
                    unk='[UNK]',
                    bos='[BOS]',
                    eos='[EOS]',
                    fix_len=20,
                    tokenize=list,
                )
            elif args.feat == "pos":
                self.FEAT = Field("postag", bos='[BOS]', eos='[EOS]')
            else:
                self.FEAT = None
            self.ARC = Field(
                "head",
                bos='[BOS]',
                eos='[EOS]',
                use_vocab=False,
                fn=utils.numericalize,
            )
            self.REL = Field("deprel", bos='[BOS]', eos='[EOS]')
            if args.feat == "char":
                self.fields = CoNLL(FORM=(self.WORD, self.FEAT), HEAD=self.ARC, DEPREL=self.REL)
            else:
                self.fields = CoNLL(FORM=self.WORD, CPOS=self.FEAT, HEAD=self.ARC, DEPREL=self.REL)

            train = Corpus.load(args.train_data_path, self.fields)

            if not args.encoding_model.startswith("ernie") and not args.encoding_model == "lstm-pe":
                self.WORD.build(train, args.min_freq)
                self.FEAT.build(train)

            self.REL.build(train)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            with open(os.path.join(args.save_dir, "fields"), "wb") as f:
                dill.dump(self.fields, f)
        else:
            with open(os.path.join(args.save_dir, "fields"), "rb") as f:
                self.fields = dill.load(f)

            if isinstance(self.fields.FORM, tuple):
                self.WORD, self.FEAT = self.fields.FORM
            else:
                self.WORD, self.FEAT = self.fields.FORM, self.fields.CPOS
            self.ARC, self.REL = self.fields.HEAD, self.fields.DEPREL

        if self.WORD.tokenizer is not None:
            vocab_items = list(self.WORD.vocab.token_to_idx.items())
        else:
            vocab_items = self.WORD.vocab.stoi.items()
        self.puncts = np.array([i for s, i in vocab_items if utils.ispunct(s)], dtype=np.int64)

        self.args.n_words = len(self.WORD.vocab)
        self.args.n_feats = self.FEAT and len(self.FEAT.vocab)
        self.args.n_rels = len(self.REL.vocab)
        self.args.pad_index = self.WORD.pad_index
        self.args.unk_index = self.WORD.unk_index
        self.args.bos_index = self.WORD.bos_index
        self.args.eos_index = self.WORD.eos_index
        self.args.feat_pad_index = self.FEAT and self.FEAT.pad_index
