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
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import AutoModel, AutoTokenizer

from data import create_dataloader, read_text_pair
from data import convert_pairwise_example as convert_example
from model import PairwiseMatching

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="The full path of input file")
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=64, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument('--model_name_or_path', default="ernie-3.0-medium-zh", help="The pretrained model used for training")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def predict(model, data_loader):
    """
    Predicts the data labels.

    Args:
        model (obj:`SemanticIndexBase`): A model to extract text embedding or calculate similarity of text pair.
        data_loaer (obj:`List(Example)`): The processed data ids of text pair: [query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids]
    Returns:
        results(obj:`List`): cosine similarity of text pairs.
    """
    batch_probs = []

    model.eval()

    with paddle.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data

            batch_prob = model.predict(input_ids=input_ids,
                                       token_type_ids=token_type_ids).numpy()

            batch_probs.append(batch_prob)
        if (len(batch_prob) == 1):
            batch_probs = np.array(batch_probs)
        else:
            batch_probs = np.concatenate(batch_probs, axis=0)

        return batch_probs


if __name__ == "__main__":
    paddle.set_device(args.device)

    pretrained_model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length,
                         phase="predict")

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"
            ),  # segment_ids
    ): [data for data in fn(samples)]

    valid_ds = load_dataset(read_text_pair,
                            data_path=args.input_file,
                            lazy=False)

    valid_data_loader = create_dataloader(valid_ds,
                                          mode='predict',
                                          batch_size=args.batch_size,
                                          batchify_fn=batchify_fn,
                                          trans_fn=trans_func)

    model = PairwiseMatching(pretrained_model)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    y_probs = predict(model, valid_data_loader)

    valid_ds = load_dataset(read_text_pair,
                            data_path=args.input_file,
                            lazy=False)

    for idx, prob in enumerate(y_probs):
        text_pair = valid_ds[idx]
        text_pair["pred_prob"] = prob[0]
        print(text_pair)
