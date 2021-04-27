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

# coding=UTF-8

from functools import partial
import argparse
import os
import sys
import random
import time

import numpy as np
import hnswlib
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset, load_dataset
from paddlenlp.utils.log import logger

sys.path.append("../")
from data import convert_example, create_dataloader
from data import gen_id2corpus, gen_text_file
from model import SemanticIndexHardestNeg

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--corpus_file", type=str, required=True, help="The full path of input file")
parser.add_argument("--similar_text_pair_file", type=str, required=True, help="The full path of similar text pair file")
parser.add_argument("--recall_result", type=str, default='recall_result', help="The full path of recall result file")
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=64, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--recall_num", default=10, type=int, help="Recall number for each query from Ann index.")

parser.add_argument("--hnsw_m", default=100, type=int, help="Recall number for each query from Ann index.")
parser.add_argument("--hnsw_ef", default=100, type=int, help="Recall number for each query from Ann index.")
parser.add_argument("--hnsw_max_elements", default=1000000, type=int, help="Recall number for each query from Ann index.")

parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def build_index(data_loader, model):

    index = hnswlib.Index(space='ip', dim=768)

    # Initializing index
    # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
    # during insertion of an element.
    # The capacity can be increased by saving/loading the index, see below.
    #
    # ef_construction - controls index search speed/build speed tradeoff
    #
    # M - is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M)
    # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
    index.init_index(
        max_elements=args.hnsw_max_elements,
        ef_construction=args.hnsw_ef,
        M=args.hnsw_m)

    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    index.set_ef(args.hnsw_ef)

    # Set number of threads used during batch search/construction
    # By default using all available cores
    index.set_num_threads(16)

    logger.info("start build index..........")

    all_embeddings = []

    for text_embeddings in model.get_semantic_embedding(data_loader):
        all_embeddings.append(text_embeddings)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    index.add_items(all_embeddings)

    logger.info("Total index number:{}".format(index.get_current_count()))

    return index


if __name__ == "__main__":
    paddle.set_device(args.device)

    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_segment
    ): [data for data in fn(samples)]

    pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(
        "ernie-1.0")

    model = SemanticIndexHardestNeg(pretrained_model)

    # load pretrained semantic model
    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        logger.info("Loaded parameters from %s" % args.params_path)

    id2corpus = gen_id2corpus(args.corpus_file)

    # conver_example function's input must be dict
    corpus_list = [{idx: text} for idx, text in id2corpus.items()]
    corpus_ds = MapDataset(corpus_list)

    corpus_data_loader = create_dataloader(
        corpus_ds,
        mode='predict',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    final_index = build_index(corpus_data_loader, model)

    text_list, text2similar_text = gen_text_file(args.similar_text_pair_file)

    query_ds = MapDataset(text_list)

    query_data_loader = create_dataloader(
        query_ds,
        mode='predict',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    query_embedding = model.get_semantic_embedding(query_data_loader)

    recall_result_dir = "recall_result"
    if not os.path.exists(recall_result_dir):
        os.mkdir(recall_result_dir)

    recall_result_file = os.path.join(recall_result_dir, args.recall_result)
    with open(recall_result_file, 'w') as f:
        for batch_index, batch_query_embedding in enumerate(query_embedding):
            recalled_idx, cosine_sims = final_index.knn_query(
                batch_query_embedding, args.recall_num)

            batch_size = len(cosine_sims)

            for row_index in range(batch_size):
                text_index = args.batch_size * batch_index + row_index
                for idx, doc_idx in enumerate(recalled_idx[row_index]):
                    f.write("{}\t{}\t{}\n".format(text_list[text_index][
                        "text"], id2corpus[doc_idx], 1.0 - cosine_sims[
                            row_index][idx]))
