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
import argparse
import os
import random
import time
import numpy as np
from functools import partial
from collections import Counter

import paddle
import paddle.nn as nn
from paddlenlp.utils.log import logger
from paddlenlp.data import Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup

from base_model import SemanticIndexBase
from model import SemanticIndexBatchNeg
from data import read_text_pair, convert_example, create_dataloader, gen_id2corpus, gen_text_file, convert_corpus_example
from data import convert_label_example
from data import build_index, label2ids
from metric import MetricReport

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default='./checkpoint', type=str,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=512, type=int,
                    help="The maximum total input sequence length after tokenization. "
                    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--output_emb_size", default=256,
                    type=int, help="output_embedding_size")
parser.add_argument("--learning_rate", default=5E-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=10, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.0, type=float,
                    help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None,
                    help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000,
                    help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="cpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument('--save_steps', type=int, default=10000,
                    help="Inteval steps to save checkpoint")
parser.add_argument('--log_steps', type=int, default=10,
                    help="Inteval steps to print log")
parser.add_argument("--train_set_file", type=str,
                    default='./data/train.txt',
                    help="The full path of train_set_file.")
parser.add_argument("--margin", default=0.2, type=float,
                    help="Margin beteween pos_sample and neg_samples")
parser.add_argument("--scale", default=30, type=int,
                    help="Scale for pair-wise margin_rank_loss")
parser.add_argument("--corpus_file", type=str, default='./data/label.txt',
                    help="The full path of input file")
parser.add_argument("--similar_text_pair_file", type=str,
                    default='./data/dev.txt',
                    help="The full path of similar text pair file")
parser.add_argument("--recall_result_dir", type=str, default='./recall_result_dir',
                    help="The full path of recall result file to save")
parser.add_argument("--recall_result_file", type=str,
                    default='recall_result_init.txt', help="The file name of recall result")
parser.add_argument("--recall_num", default=50, type=int,
                    help="Recall number for each query from Ann index.")
parser.add_argument("--hnsw_m", default=100, type=int,
                    help="Recall number for each query from Ann index.")
parser.add_argument("--hnsw_ef", default=100, type=int,
                    help="Recall number for each query from Ann index.")
parser.add_argument("--hnsw_max_elements", default=1000000,
                    type=int, help="Recall number for each query from Ann index.")
parser.add_argument("--evaluate_result", type=str, default='evaluate_result.txt',
                    help="evaluate_result")
parser.add_argument('--evaluate', default=True, type=eval, choices=[True, False],
                    help='whether evaluate while training')
parser.add_argument("--model_name_or_path",default='rocketqa-zh-dureader-query-encoder',
                    type=str,help='The pretrained model used for training')
parser.add_argument("--threshold", default=0.5, type=float,
                    help="The threshold for selection the labels")
args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, corpus_data_loader, query_data_loader, recall_result_file,
             text_list, id2corpus, label2id):
    metric = MetricReport()
    # Load pretrained semantic model
    inner_model = model._layers
    final_index = build_index(corpus_data_loader,
                              inner_model,
                              output_emb_size=args.output_emb_size,
                              hnsw_max_elements=args.hnsw_max_elements,
                              hnsw_ef=args.hnsw_ef,
                              hnsw_m=args.hnsw_m)
    query_embedding = inner_model.get_semantic_embedding(query_data_loader)
    with open(recall_result_file, 'w', encoding='utf-8') as f:
        for batch_index, batch_query_embedding in enumerate(query_embedding):
            recalled_idx, cosine_sims = final_index.knn_query(
                batch_query_embedding.numpy(), args.recall_num)
            batch_size = len(cosine_sims)
            for row_index in range(batch_size):
                text_index = args.batch_size * batch_index + row_index
                for idx, doc_idx in enumerate(recalled_idx[row_index]):
                    f.write("{}\t{}\t{}\n".format(
                        text_list[text_index]["text"], id2corpus[doc_idx],
                        1.0 - cosine_sims[row_index][idx]))
    text2similar = {}
    with open(args.similar_text_pair_file, 'r', encoding='utf-8') as f:
        for line in f:
            text_arr = line.rstrip().rsplit("\t")
            text, similar_text = text_arr[0], text_arr[1]
            text2similar[text] = np.zeros(len(label2id))
            # One hot Encoding
            for label in similar_text.strip().split(','):
                text2similar[text][label2id[label]] = 1
    # Convert predicted labels into one hot encoding
    pred_labels = {}
    with open(recall_result_file, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            text_arr = line.rstrip().split("\t")
            text, labels, cosine_sim = text_arr
            # One hot Encoding
            if (text not in pred_labels):
                pred_labels[text] = np.zeros(len(label2id))
            if (float(cosine_sim) > args.threshold):
                for label in labels.split(','):
                    pred_labels[text][label2id[label]] = float(cosine_sim)

        for text, probs in pred_labels.items():
            metric.update(probs, text2similar[text])
        micro_f1_score, macro_f1_score = metric.accumulate()
    return macro_f1_score


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(args.seed)
    train_ds = load_dataset(read_text_pair,
                            data_path=args.train_set_file,
                            lazy=False)
    pretrained_model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'
            ),  # query_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'
            ),  # query_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'
            ),  # title_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'
            ),  # tilte_segment
    ): [data for data in fn(samples)]
    train_data_loader = create_dataloader(train_ds,
                                          mode='train',
                                          batch_size=args.batch_size,
                                          batchify_fn=batchify_fn,
                                          trans_fn=trans_func)
    model = SemanticIndexBatchNeg(pretrained_model,
                                  margin=args.margin,
                                  scale=args.scale,
                                  output_emb_size=args.output_emb_size)
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("warmup from:{}".format(args.init_from_ckpt))
    model = paddle.DataParallel(model)
    batchify_fn_dev = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'
            ),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'
            ),  # text_segment
    ): [data for data in fn(samples)]
    if (args.evaluate):
        eval_func = partial(convert_example,
                            tokenizer=tokenizer,
                            max_seq_length=args.max_seq_length)
        id2corpus = gen_id2corpus(args.corpus_file)
        label2id = label2ids(args.corpus_file)
        # conver_example function's input must be dict
        corpus_list = [{idx: text} for idx, text in id2corpus.items()]
        corpus_ds = MapDataset(corpus_list)
        corpus_data_loader = create_dataloader(corpus_ds,
                                               mode='predict',
                                               batch_size=args.batch_size,
                                               batchify_fn=batchify_fn_dev,
                                               trans_fn=eval_func)
        query_func = partial(convert_example,
                             tokenizer=tokenizer,
                             max_seq_length=args.max_seq_length)
        text_list, _ = gen_text_file(args.similar_text_pair_file)
        query_ds = MapDataset(text_list)
        query_data_loader = create_dataloader(query_ds,
                                              mode='predict',
                                              batch_size=args.batch_size,
                                              batchify_fn=batchify_fn_dev,
                                              trans_fn=query_func)
        if not os.path.exists(args.recall_result_dir):
            os.mkdir(args.recall_result_dir)
        recall_result_file = os.path.join(args.recall_result_dir,
                                          args.recall_result_file)
    num_training_steps = len(train_data_loader) * args.epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=nn.ClipGradByNorm(clip_norm=1.0))
    global_step = 0
    best_score = 0.0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch
            loss = model(query_input_ids=query_input_ids,
                         title_input_ids=title_input_ids,
                         query_token_type_ids=query_token_type_ids,
                         title_token_type_ids=title_token_type_ids)
            global_step += 1
            if global_step % args.log_steps == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, 10 /
                       (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if not args.evaluate and rank == 0:
                if global_step % args.save_steps == 0 and rank == 0:
                    save_dir = os.path.join(args.save_dir,
                                            "model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir,
                                                   'model_state.pdparams')
                    paddle.save(model.state_dict(), save_param_path)
                    tokenizer.save_pretrained(save_dir)
        if args.evaluate and rank == 0:
            print("evaluating")
            macro_f1_score = evaluate(model, corpus_data_loader,
                                      query_data_loader, recall_result_file,
                                      text_list, id2corpus, label2id)
            if macro_f1_score > best_score:
                best_score = macro_f1_score
                save_dir = os.path.join(args.save_dir, "model_best")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                tokenizer.save_pretrained(save_dir)
                with open(os.path.join(save_dir, "train_result.txt"),
                          'a',
                          encoding='utf-8') as fp:
                    fp.write('epoch=%d, global_step: %d, Macro f1: %s\n' %
                             (epoch, global_step, macro_f1_score))


if __name__ == "__main__":
    do_train()
