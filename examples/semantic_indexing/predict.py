# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad

from model import SemanticIndexing
from utils.util import convert_example

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default='', help="The full path of input file")
parser.add_argument("--params_path", type=str, default='', help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=64, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def predict(model, data, tokenizer, batch_size=1):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `se_len`(sequence length).
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        batch_size(obj:`int`, defaults to 1): The number of batch.

    Returns:
        results(obj:`List`): cosine similarity of text pairs.
    """
    examples = []
    for text_pair in data:
        query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = convert_example(
            text_pair, tokenizer, max_seq_length=args.max_seq_length)

        examples.append((query_input_ids, query_token_type_ids, title_input_ids,
                         title_token_type_ids))

    # Seperates data into some batches.
    batches = [
        examples[idx:idx + batch_size]
        for idx in range(0, len(examples), batch_size)
    ]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # title_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tilte_segment
    ): [data for data in fn(samples)]

    cosin_sims = []
    model.eval()
    for batch in batches:
        query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batchify_fn(
            batch)

        query_input_ids = paddle.to_tensor(query_input_ids)
        query_token_type_ids = paddle.to_tensor(query_token_type_ids)
        title_input_ids = paddle.to_tensor(title_input_ids)
        title_token_type_ids = paddle.to_tensor(title_token_type_ids)

        query_embeddings = model.get_pooled_embedding(
            query_input_ids, token_type_ids=query_token_type_ids)

        title_embeddings = model.get_pooled_embedding(
            title_input_ids, token_type_ids=title_token_type_ids)

        cosin_sim = [
            local_cosin(query_embeddings[idx], title_embeddings[idx])
            for idx in range(0, len(query_embeddings))
        ]
        cosin_sims += list(cosin_sim)
    return cosin_sims


def local_cosin(u, v):
    u = u.numpy()
    v = v.numpy()
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


if __name__ == "__main__":
    paddle.set_device(args.device)

    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')

    def gen_data(input_file):
        all_lines = []
        with open(input_file) as f:
            for line in f:
                all_lines.append(line.rstrip().split("\t"))
        return all_lines

    data = gen_data(args.input_file)

    pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(
        "ernie-1.0")
    model = SemanticIndexing(pretrained_model)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    cosin_sim = predict(model, data, tokenizer, batch_size=args.batch_size)

    for idx, text in enumerate(data):
        text_a = text[0]
        text_b = text[1]
        print('{}\t{}\t{}'.format(text_a, text_b, cosin_sim[idx]))
