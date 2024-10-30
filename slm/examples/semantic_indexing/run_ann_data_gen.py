# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import time
from functools import partial

import paddle
from ance.model import SemanticIndexANCE
from ann_util import build_index
from data import (
    convert_example,
    create_dataloader,
    gen_id2corpus,
    gen_text_file,
    get_latest_ann_data,
    get_latest_checkpoint,
)

from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.utils.log import logger

# yapf: disable
parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--similar_text_pair_file", default=None, type=str, required=True, help="The train_set tsv file that each line is simialr text pair")
parser.add_argument("--corpus_file", default=None, type=str, required=True, help="The corpus file that each line is a text for buinding indexing")
parser.add_argument("--save_dir", default=None, type=str, required=True, help="Saved model dir, will look for latest checkpoint dir in here")
parser.add_argument("--ann_data_dir", default=None, type=str, required=True, help="The output directory where the training data will be written")

parser.add_argument("--init_from_ckpt", default=None, type=str, help="Initial model dir, will use this if no checkpoint is found in model_dir")
parser.add_argument("--end_ann_step", default=1000000, type=int, help="Stop after this number of data versions has been generated, default run forever")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size for predicting embedding of texts")
parser.add_argument("--output_emb_size", default=None, type=int, help="output_embedding_size")

parser.add_argument("--max_seq_length", default=128, type=int, help="Batch size for predicting embedding of texts")
parser.add_argument("--topk_training", default=500, type=int, help="top k from which negative samples are collected")
parser.add_argument("--num_negative_sample", default=5, type=int, help="at each resample, how many negative samples per query do I use")

# hnsw argument
parser.add_argument("--hnsw_m", default=10, type=int, help="Recall number for each query from Ann index.")
parser.add_argument("--hnsw_ef", default=10, type=int, help="Recall number for each query from Ann index.")
parser.add_argument("--hnsw_max_elements", default=1000000, type=int, help="Recall number for each query from Ann index.")

args = parser.parse_args()
# yapf: enable


def generate_new_ann(args, data_loader_dict, checkpoint_path, latest_step_num):

    pretrained_model = AutoModel.from_pretrained("ernie-3.0-medium-zh")

    model = SemanticIndexANCE(pretrained_model, output_emb_size=args.output_emb_size)

    logger.info("checkpoint_path:{}".format(checkpoint_path))
    state_dict = paddle.load(checkpoint_path)

    model.set_dict(state_dict)
    logger.info("load params from:{}".format(checkpoint_path))

    logger.info("***** inference of corpus *****")
    final_index = build_index(args, data_loader_dict["corpus_data_loader"], model)

    logger.info("***** inference of query *****")
    query_embedding = model.get_semantic_embedding(data_loader_dict["text_data_loader"])

    text_list = data_loader_dict["text_list"]
    id2corpus = data_loader_dict["id2corpus"]
    text2similar_text = data_loader_dict["text2similar_text"]

    new_ann_data_path = os.path.join(args.ann_data_dir, str(latest_step_num))
    if not os.path.exists(new_ann_data_path):
        os.mkdir(new_ann_data_path)

    with open(os.path.join(new_ann_data_path, "new_ann_data"), "w") as f:
        for batch_index, batch_query_embedding in enumerate(query_embedding):
            recalled_idx, cosine_sims = final_index.knn_query(batch_query_embedding, args.topk_training)

            batch_size = len(cosine_sims)

            for row_index in range(batch_size):
                text_index = args.batch_size * batch_index + row_index

                hard_neg_samples = recalled_idx[row_index][-1 * args.num_negative_sample :]

                for idx, hard_neg_doc_idx in enumerate(hard_neg_samples):
                    text = text_list[text_index]["text"]
                    similar_text = text2similar_text[text]
                    hard_neg_sample = id2corpus[hard_neg_doc_idx]
                    f.write("{}\t{}\t{}\n".format(text, similar_text, hard_neg_sample))

    succeed_flag_file = os.path.join(new_ann_data_path, "succeed_flag_file")
    open(succeed_flag_file, "a").close()
    logger.info("finish generate ann data step:{}".format(latest_step_num))


def build_data_loader(args, tokenizer):
    """build corpus_data_loader and text_data_loader"""

    id2corpus = gen_id2corpus(args.corpus_file)

    # convert_example function's input must be dict
    corpus_list = [{idx: text} for idx, text in id2corpus.items()]
    corpus_ds = MapDataset(corpus_list)

    trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_segment
    ): [data for data in fn(samples)]

    corpus_data_loader = create_dataloader(
        corpus_ds, mode="predict", batch_size=args.batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
    )

    # build text data_loader
    text_list, text2similar_text = gen_text_file(args.similar_text_pair_file)

    text_ds = MapDataset(text_list)

    text_data_loader = create_dataloader(
        text_ds, mode="predict", batch_size=args.batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
    )

    d = {
        "text_data_loader": text_data_loader,
        "corpus_data_loader": corpus_data_loader,
        "id2corpus": id2corpus,
        "text2similar_text": text2similar_text,
        "text_list": text_list,
    }

    return d


def ann_data_gen(args):
    # use init_from_ckpt as last_checkpoint
    last_checkpoint = args.init_from_ckpt

    # get latest_ann_data_step to decide when stop gen_ann_data
    _, latest_ann_data_step = get_latest_ann_data(args.ann_data_dir)

    rank = paddle.distributed.get_rank()
    if rank == 0:
        if not os.path.exists(args.ann_data_dir):
            os.makedirs(args.ann_data_dir)

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")

    data_load_dict = build_data_loader(args, tokenizer)

    while latest_ann_data_step <= args.end_ann_step:
        next_checkpoint, latest_step_num = get_latest_checkpoint(args)
        logger.info("next_checkpoint:{}".format(next_checkpoint))

        if next_checkpoint == last_checkpoint:
            logger.info("next_checkpoint == lase_checkpoint:{}".format(next_checkpoint))
            logger.info("sleep 10s")
            time.sleep(10)
        else:
            logger.info("start generate ann data using checkpoint:{}".format(next_checkpoint))

            generate_new_ann(args, data_load_dict, next_checkpoint, latest_step_num)

            logger.info("finished generating ann data step {}".format(latest_step_num))

            last_checkpoint = next_checkpoint


def main():
    ann_data_gen(args)


if __name__ == "__main__":
    main()
