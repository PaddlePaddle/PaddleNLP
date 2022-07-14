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
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.utils.log import logger

from model import SimCSE
from data import create_dataloader
from tqdm import tqdm


def convert_example(example, tokenizer, max_seq_length=512, do_evalute=False):
    """
    Builds model inputs from a sequence.
        
    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``

    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
    """

    result = []

    for key, text in example.items():
        encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        result += [input_ids, token_type_ids]

    return result


if __name__ == "__main__":
    device = 'gpu'
    max_seq_length = 64
    output_emb_size = 256
    batch_size = 1
    params_path = 'checkpoints/model_20000/model_state.pdparams'
    id2corpus = {0: '国有企业引入非国有资本对创新绩效的影响——基于制造业国有上市公司的经验证据'}
    paddle.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')
    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_segment
    ): [data for data in fn(samples)]

    pretrained_model = AutoModel.from_pretrained("ernie-3.0-medium-zh")

    model = SimCSE(pretrained_model, output_emb_size=output_emb_size)

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
    print(text_embedding.numpy())
