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

from data import read_text_pair, convert_example, create_dataloader
from model import SimCSE

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--text_pair_file", type=str, required=True, help="The full path of input file")
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=64, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--margin", default=0.0, type=float, help="Margin beteween pos_sample and neg_samples.")
parser.add_argument("--scale", default=20, type=int, help="Scale for pair-wise margin_rank_loss.")
parser.add_argument("--output_emb_size", default=0, type=int, help="Output_embedding_size, 0 means use hidden_size as output embedding size.")

args = parser.parse_args()
# yapf: enable


def predict(model, data_loader):
    """
    Predicts the data labels.

    Args:
        model (obj:`SimCSE`): A model to extract text embedding or calculate similarity of text pair.
        data_loaer (obj:`List(Example)`): The processed data ids of text pair: [query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids]
    Returns:
        results(obj:`List`): cosine similarity of text pairs.
    """

    cosine_sims = []

    model.eval()

    with paddle.no_grad():
        for batch_data in data_loader:
            query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch_data

            batch_cosine_sim = model.cosine_sim(
                query_input_ids=query_input_ids,
                title_input_ids=title_input_ids,
                query_token_type_ids=query_token_type_ids,
                title_token_type_ids=title_token_type_ids).numpy()

            cosine_sims.append(batch_cosine_sim)

        cosine_sims = np.concatenate(cosine_sims, axis=0)

        return cosine_sims


if __name__ == "__main__":
    paddle.set_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # title_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tilte_segment
    ): [data for data in fn(samples)]

    valid_ds = load_dataset(read_text_pair,
                            data_path=args.text_pair_file,
                            lazy=False,
                            is_test=True)

    valid_data_loader = create_dataloader(valid_ds,
                                          mode='predict',
                                          batch_size=args.batch_size,
                                          batchify_fn=batchify_fn,
                                          trans_fn=trans_func)

    pretrained_model = AutoModel.from_pretrained("ernie-3.0-medium-zh")

    model = SimCSE(pretrained_model,
                   margin=args.margin,
                   scale=args.scale,
                   output_emb_size=args.output_emb_size)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    cosin_sim = predict(model, valid_data_loader)

    for idx, cosine in enumerate(cosin_sim):
        print('{}'.format(cosine))
