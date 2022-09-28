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
from functools import partial
import argparse
from pprint import pprint
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import ErnieTokenizer, ErnieModel
from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.ops import enable_faster_encoder, disable_faster_encoder
from paddlenlp.ops.faster_transformer.transformer.decoding import transfer_param

from data import read_text_pair, convert_example, create_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_pair_file",
                        type=str,
                        required=True,
                        help="The full path of input file")
    parser.add_argument("--output_emb_size",
                        default=None,
                        type=int,
                        help="output_embedding_size")
    parser.add_argument("--params_path",
                        type=str,
                        required=True,
                        help="The path to model parameters to be loaded.")
    parser.add_argument(
        "--max_seq_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded."
    )
    parser.add_argument("--dropout",
                        default=0.0,
                        type=float,
                        help="Dropout probability.")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--pad_to_max_seq_len",
                        action="store_true",
                        help="Whether to pad to max_seq_len.")
    parser.add_argument("--use_fp16",
                        action="store_true",
                        help="Whether to use fp16.")

    args = parser.parse_args()
    return args


class SemanticIndexingPredictor(nn.Layer):

    def __init__(self,
                 pretrained_model,
                 output_emb_size,
                 bos_id=0,
                 dropout=0,
                 use_fp16=False):
        super(SemanticIndexingPredictor, self).__init__()
        self.bos_id = bos_id
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.0)
        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            weight_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(768,
                                                      output_emb_size,
                                                      weight_attr=weight_attr)

        self.use_fp16 = use_fp16

    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None):
        src_mask = input_ids == self.bos_id
        src_mask = paddle.cast(src_mask, "float32")
        # [bs, 1, 1, max_len]
        src_mask = paddle.unsqueeze(src_mask, axis=[1, 2])
        src_mask.stop_gradient = True

        ones = paddle.ones_like(input_ids, dtype="int64")
        seq_length = paddle.cumsum(ones, axis=1)
        position_ids = seq_length - ones
        position_ids.stop_gradient = True

        embedding_output = self.ptm.embeddings(input_ids=input_ids,
                                               position_ids=position_ids,
                                               token_type_ids=token_type_ids)

        if self.use_fp16:
            embedding_output = paddle.cast(embedding_output, 'float16')

        sequence_output = self.ptm.encoder(embedding_output, src_mask)

        if self.use_fp16:
            sequence_output = paddle.cast(sequence_output, 'float32')

        cls_embedding = self.ptm.pooler(sequence_output)

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)

        return cls_embedding

    def forward(self,
                query_input_ids,
                title_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                title_token_type_ids=None,
                title_position_ids=None):
        query_cls_embedding = self.get_pooled_embedding(query_input_ids,
                                                        query_token_type_ids,
                                                        query_position_ids)
        title_cls_embedding = self.get_pooled_embedding(title_input_ids,
                                                        title_token_type_ids,
                                                        title_position_ids)

        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding,
                                axis=-1)
        return cosine_sim

    def load(self, init_from_params):
        if init_from_params and os.path.isfile(init_from_params):
            state_dict = paddle.load(init_from_params)
            self.set_state_dict(state_dict)
            print("Loaded parameters from %s" % init_from_params)
        else:
            raise ValueError(
                "Please set --params_path with correct pretrained model file")


def do_predict(args):
    place = paddle.set_device("gpu")
    paddle.seed(args.seed)
    tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length,
                         pad_to_max_seq_len=args.pad_to_max_seq_len)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # title_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tilte_segment
    ): [data for data in fn(samples)]

    valid_ds = load_dataset(read_text_pair,
                            data_path=args.text_pair_file,
                            lazy=False)

    valid_data_loader = create_dataloader(valid_ds,
                                          mode="predict",
                                          batch_size=args.batch_size,
                                          batchify_fn=batchify_fn,
                                          trans_fn=trans_func)

    pretrained_model = ErnieModel.from_pretrained("ernie-1.0")

    model = SemanticIndexingPredictor(pretrained_model,
                                      args.output_emb_size,
                                      dropout=args.dropout,
                                      use_fp16=args.use_fp16)
    model.eval()
    model.load(args.params_path)
    model = enable_faster_encoder(model, use_fp16=args.use_fp16)

    cosine_sims = []
    for batch_data in valid_data_loader:
        query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch_data
        query_input_ids = paddle.to_tensor(query_input_ids)
        query_token_type_ids = paddle.to_tensor(query_token_type_ids)
        title_input_ids = paddle.to_tensor(title_input_ids)
        title_token_type_ids = paddle.to_tensor(title_token_type_ids)
        batch_cosine_sim = model(
            query_input_ids=query_input_ids,
            title_input_ids=title_input_ids,
            query_token_type_ids=query_token_type_ids,
            title_token_type_ids=title_token_type_ids).numpy()
        cosine_sims.append(batch_cosine_sim)

    cosine_sims = np.concatenate(cosine_sims, axis=0)
    for cosine in cosine_sims:
        print('{}'.format(cosine))
    model = disable_faster_encoder(model)


if __name__ == "__main__":
    args = parse_args()
    pprint(args)
    do_predict(args)
