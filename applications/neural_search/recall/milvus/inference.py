from functools import partial
import argparse
import os
import sys
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset, load_dataset
from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.utils.log import logger

from base_model import SemanticIndexBaseStatic
from data import convert_example, create_dataloader
from data import gen_id2corpus, gen_text_file
from tqdm import tqdm
from milvus_recall import RecallByMilvus


def search_in_milvus(text_embedding):
    collection_name = 'literature_search'
    partition_tag = 'partition_2'
    client = RecallByMilvus()
    status, results = client.search(collection_name=collection_name,
                                    vectors=text_embedding.tolist(),
                                    partition_tag=partition_tag)
    corpus_file = "milvus/milvus_data.csv"
    id2corpus = gen_id2corpus(corpus_file)

    for line in results:
        for item in line:
            idx = item.id
            distance = item.distance
            text = id2corpus[idx]
            print(idx, text, distance)


if __name__ == "__main__":
    device = 'gpu'
    max_seq_length = 64
    output_emb_size = 256
    batch_size = 1
    params_path = 'checkpoints/model_40/model_state.pdparams'
    id2corpus = {0: '国有企业引入非国有资本对创新绩效的影响——基于制造业国有上市公司的经验证据'}
    paddle.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')
    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"
            ),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"
            ),  # text_segment
    ): [data for data in fn(samples)]

    pretrained_model = AutoModel.from_pretrained("ernie-3.0-medium-zh")

    model = SemanticIndexBaseStatic(pretrained_model,
                                    output_emb_size=output_emb_size)

    # Load pretrained semantic model
    if params_path and os.path.isfile(params_path):
        state_dict = paddle.load(params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % params_path)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    # conver_example function's input must be dict
    corpus_list = [{idx: text} for idx, text in id2corpus.items()]
    corpus_ds = MapDataset(corpus_list)

    corpus_data_loader = create_dataloader(corpus_ds,
                                           mode='predict',
                                           batch_size=batch_size,
                                           batchify_fn=batchify_fn,
                                           trans_fn=trans_func)

    # Need better way to get inner model of DataParallel

    all_embeddings = []
    model.eval()
    with paddle.no_grad():
        for batch_data in corpus_data_loader:
            input_ids, token_type_ids = batch_data

            text_embeddings = model.get_pooled_embedding(
                input_ids, token_type_ids)
            all_embeddings.append(text_embeddings)

    text_embedding = all_embeddings[0]
    print(text_embedding.shape)
    print(text_embedding)
    search_in_milvus(text_embedding)
