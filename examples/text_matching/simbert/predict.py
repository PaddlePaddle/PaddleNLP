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

from functools import partial
import argparse
import sys
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad

from data import create_dataloader, read_text_pair
from data import convert_example

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="The full path of input file")
# parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=64, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def predict(model, data_loader):
    """
    Predicts the similarity.

    Args:
        model (obj:`SemanticIndexBase`): A model to extract text embedding or calculate similarity of text pair.
        data_loaer (obj:`List(Example)`): The processed data ids of text pair: [query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids]
    Returns:
        results(obj:`List`): cosine similarity of text pairs.
    """
    results = []

    model.eval()

    with paddle.no_grad():
        for batch_data in data_loader:
            query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch_data
            query_input_ids = paddle.to_tensor(query_input_ids)
            query_token_type_ids = paddle.to_tensor(query_token_type_ids)
            title_input_ids = paddle.to_tensor(title_input_ids)
            title_token_type_ids = paddle.to_tensor(title_token_type_ids)

            vecs_query = model(input_ids=query_input_ids,
                               token_type_ids=query_token_type_ids)
            vecs_title = model(input_ids=title_input_ids,
                               token_type_ids=title_token_type_ids)
            vecs_query = vecs_query[1].numpy()
            vecs_title = vecs_title[1].numpy()

            vecs_query = vecs_query / (vecs_query**2).sum(axis=1,
                                                          keepdims=True)**0.5
            vecs_title = vecs_title / (vecs_title**2).sum(axis=1,
                                                          keepdims=True)**0.5
            sims = (vecs_query * vecs_title).sum(axis=1)

            results.extend(sims)

    return results


if __name__ == "__main__":
    paddle.set_device(args.device)

    model = AutoModel.from_pretrained('simbert-base-chinese', pool_act='linear')
    tokenizer = AutoTokenizer.from_pretrained('simbert-base-chinese')

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length,
                         phase="predict")

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # title_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tilte_segment
    ): [data for data in fn(samples)]

    valid_ds = load_dataset(read_text_pair,
                            data_path=args.input_file,
                            lazy=False)

    valid_data_loader = create_dataloader(valid_ds,
                                          mode='predict',
                                          batch_size=args.batch_size,
                                          batchify_fn=batchify_fn,
                                          trans_fn=trans_func)

    y_sims = predict(model, valid_data_loader)

    valid_ds = load_dataset(read_text_pair,
                            data_path=args.input_file,
                            lazy=False)

    for idx, prob in enumerate(y_sims):
        text_pair = valid_ds[idx]
        text_pair["similarity"] = y_sims[idx]
        print(text_pair)
