from functools import partial
import argparse
import os
import sys
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset, load_dataset
from paddlenlp.utils.log import logger

from base_model import SemanticIndexBase
from data import convert_example, create_dataloader
from data import gen_id2corpus, gen_text_file
from ann_util import build_index
from tqdm import tqdm 
from milvus_recall import RecallByMilvus


if __name__ == "__main__":
    device= 'gpu'
    max_seq_length=64
    output_emb_size=256
    batch_size=1
    params_path='checkpoints/train_0.001/model_40/model_state.pdparams'
    paddle.set_device(device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_segment
    ): [data for data in fn(samples)]

    pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(
        "ernie-1.0")

    model = SemanticIndexBase(
        pretrained_model, output_emb_size=output_emb_size)
    model = paddle.DataParallel(model)

    # Load pretrained semantic model
    if params_path and os.path.isfile(params_path):
        state_dict = paddle.load(params_path)
        model.set_dict(state_dict)
        logger.info("Loaded parameters from %s" % params_path)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    id2corpus={0:'外语阅读焦虑与英语成绩及性别的关系'}

    # id2corpus = gen_id2corpus(args.corpus_file)

    # conver_example function's input must be dict
    corpus_list = [{idx: text} for idx, text in id2corpus.items()]
    corpus_ds = MapDataset(corpus_list)

    corpus_data_loader = create_dataloader(
        corpus_ds,
        mode='predict',
        batch_size=batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    # Need better way to get inner model of DataParallel
    inner_model = model._layers

    # final_index = build_index(args, corpus_data_loader, inner_model)

    all_embeddings = []

    for text_embeddings in tqdm(inner_model.get_semantic_embedding(corpus_data_loader)):
        all_embeddings.append(text_embeddings.numpy())

    text_embedding=all_embeddings[0]
    print(text_embedding.shape)
    collection_name = 'wanfang1'
    partition_tag = 'partition_2'
    client = RecallByMilvus()
    status, resultes = client.search(collection_name=collection_name, vectors=text_embedding.tolist(), partition_tag=partition_tag)
    print(status)
    print(resultes)
