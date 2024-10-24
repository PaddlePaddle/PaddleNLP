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
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ann_util import build_index
from batch_negative.model import SemanticIndexBatchNeg, SemanticIndexCacheNeg
from data import (
    convert_example,
    create_dataloader,
    gen_id2corpus,
    gen_text_file,
    read_text_pair,
)

from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import MapDataset, load_dataset
from paddlenlp.transformers import AutoModel, AutoTokenizer, LinearDecayWithWarmup

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=512, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--model_name_or_path', default="rocketqa-zh-base-query-encoder", help="The pretrained model used for training")
parser.add_argument("--output_emb_size", default=256, type=int, help="output_embedding_size")
parser.add_argument("--learning_rate", default=5E-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proportion over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="cpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument('--save_steps', type=int, default=10000, help="Interval steps to save checkpoint")
parser.add_argument('--log_steps', type=int, default=10, help="Interval steps to print log")
parser.add_argument("--train_set_file", type=str, default='./recall/train.csv', help="The full path of train_set_file.")
parser.add_argument("--dev_set_file", type=str, default='./recall/dev.csv', help="The full path of dev_set_file.")
parser.add_argument("--margin", default=0.2, type=float, help="Margin between pos_sample and neg_samples")
parser.add_argument("--scale", default=30, type=int, help="Scale for pair-wise margin_rank_loss")
parser.add_argument("--corpus_file", type=str, default='./recall/corpus.csv', help="The full path of input file")
parser.add_argument("--similar_text_pair_file", type=str, default='./recall/dev.csv', help="The full path of similar text pair file")
parser.add_argument("--recall_result_dir", type=str, default='./recall_result_dir', help="The full path of recall result file to save")
parser.add_argument("--recall_result_file", type=str, default='recall_result_init.txt', help="The file name of recall result")
parser.add_argument("--recall_num", default=50, type=int, help="Recall number for each query from Ann index.")
parser.add_argument("--hnsw_m", default=100, type=int, help="Recall number for each query from Ann index.")
parser.add_argument("--hnsw_ef", default=100, type=int, help="Recall number for each query from Ann index.")
parser.add_argument("--hnsw_max_elements", default=1000000, type=int, help="Recall number for each query from Ann index.")
parser.add_argument("--evaluate_result", type=str, default='evaluate_result.txt', help="evaluate_result")
parser.add_argument('--evaluate', action='store_true', help='whether evaluate while training')
parser.add_argument("--max_grad_norm", type=float, default=5.0, help="max grad norm for global norm clip")
parser.add_argument("--use_amp", action="store_true", help="Whether to use AMP.")
parser.add_argument("--amp_loss_scale", default=32768, type=float, help="The value of scale_loss for fp16. This is only used for AMP training.")
parser.add_argument("--use_gradient_cache", action='store_true', help="Using the gradient cache to scale up the batch size and save the memory.")
parser.add_argument("--chunk_numbers", type=int, default=50, help="The number of the chunks for model")
args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def recall(rs, N=10):
    recall_flags = [np.sum(r[0:N]) for r in rs]
    return np.mean(recall_flags)


@paddle.no_grad()
def evaluate(model, corpus_data_loader, query_data_loader, recall_result_file, text_list, id2corpus):
    # Load pretrained semantic model
    inner_model = model._layers
    final_index = build_index(args, corpus_data_loader, inner_model)
    query_embedding = inner_model.get_semantic_embedding(query_data_loader)
    with open(recall_result_file, "w", encoding="utf-8") as f:
        for batch_index, batch_query_embedding in enumerate(query_embedding):
            recalled_idx, cosine_sims = final_index.knn_query(batch_query_embedding.numpy(), args.recall_num)
            batch_size = len(cosine_sims)
            for row_index in range(batch_size):
                text_index = args.batch_size * batch_index + row_index
                for idx, doc_idx in enumerate(recalled_idx[row_index]):
                    f.write(
                        "{}\t{}\t{}\n".format(
                            text_list[text_index]["text"], id2corpus[doc_idx], 1.0 - cosine_sims[row_index][idx]
                        )
                    )
    text2similar = {}
    with open(args.similar_text_pair_file, "r", encoding="utf-8") as f:
        for line in f:
            text, similar_text = line.rstrip().split("\t")
            text2similar[text] = similar_text
    rs = []
    with open(recall_result_file, "r", encoding="utf-8") as f:
        relevance_labels = []
        for index, line in enumerate(f):
            if index % args.recall_num == 0 and index != 0:
                rs.append(relevance_labels)
                relevance_labels = []
            text, recalled_text, cosine_sim = line.rstrip().split("\t")
            if text == recalled_text:
                continue
            if text2similar[text] == recalled_text:
                relevance_labels.append(1)
            else:
                relevance_labels.append(0)

    recall_N = []
    recall_num = [1, 5, 10, 20, 50]
    for topN in recall_num:
        R = round(100 * recall(rs, N=topN), 3)
        recall_N.append(str(R))
    evaluate_result_file = os.path.join(args.recall_result_dir, args.evaluate_result)
    result = open(evaluate_result_file, "a")
    res = []
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    res.append(timestamp)
    for key, val in zip(recall_num, recall_N):
        print("recall@{}={}".format(key, val))
        res.append(str(val))
    result.write("\t".join(res) + "\n")
    return float(recall_N[1])


def train(
    train_data_loader,
    model,
    optimizer,
    lr_scheduler,
    rank,
    corpus_data_loader,
    query_data_loader,
    recall_result_file,
    text_list,
    id2corpus,
    tokenizer,
):
    global_step = 0
    best_recall = 0.0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch

            loss = model(
                query_input_ids=query_input_ids,
                title_input_ids=title_input_ids,
                query_token_type_ids=query_token_type_ids,
                title_token_type_ids=title_token_type_ids,
            )

            global_step += 1
            if global_step % args.log_steps == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, args.log_steps / (time.time() - tic_train))
                )
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if not args.evaluate:
                if global_step % args.save_steps == 0 and rank == 0:
                    save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir, "model_state.pdparams")
                    paddle.save(model.state_dict(), save_param_path)
                    tokenizer.save_pretrained(save_dir)
        if args.evaluate and rank == 0:
            print("evaluating")
            recall_5 = evaluate(model, corpus_data_loader, query_data_loader, recall_result_file, text_list, id2corpus)
            if recall_5 > best_recall:
                best_recall = recall_5

                save_dir = os.path.join(args.save_dir, "model_best")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, "model_state.pdparams")
                paddle.save(model.state_dict(), save_param_path)
                tokenizer.save_pretrained(save_dir)
                with open(os.path.join(save_dir, "train_result.txt"), "a", encoding="utf-8") as fp:
                    fp.write("epoch=%d, global_step: %d, recall: %s\n" % (epoch, global_step, recall_5))


def gradient_cache_train(train_data_loader, model, optimizer, lr_scheduler, rank, tokenizer):

    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.amp_loss_scale)

    if args.batch_size % args.chunk_numbers == 0:
        chunk_numbers = args.chunk_numbers
    else:
        raise Exception(
            f" Batch_size {args.batch_size} must divides chunk_numbers {args.chunk_numbers} without producing a remainder "
        )

    def split(inputs, chunk_numbers, axis=0):
        if inputs.shape[0] % chunk_numbers == 0:
            return paddle.split(inputs, chunk_numbers, axis=0)
        else:
            return paddle.split(inputs, inputs.shape[0], axis=0)

    global_step = 0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            # Separate large batches into several sub batches
            chunked_x = [split(t, chunk_numbers, axis=0) for t in batch]
            sub_batchs = [list(s) for s in zip(*chunked_x)]

            all_grads = []
            all_CUDA_rnd_state = []
            all_query = []
            all_title = []

            for sub_batch in sub_batchs:
                all_reps = []
                all_labels = []
                (
                    sub_query_input_ids,
                    sub_query_token_type_ids,
                    sub_title_input_ids,
                    sub_title_token_type_ids,
                ) = sub_batch
                with paddle.amp.auto_cast(args.use_amp, custom_white_list=["layer_norm", "softmax", "gelu"]):

                    with paddle.no_grad():
                        sub_CUDA_rnd_state = paddle.framework.random.get_cuda_rng_state()
                        all_CUDA_rnd_state.append(sub_CUDA_rnd_state)
                        sub_cosine_sim, sub_label, query_embedding, title_embedding = model(
                            query_input_ids=sub_query_input_ids,
                            title_input_ids=sub_title_input_ids,
                            query_token_type_ids=sub_query_token_type_ids,
                            title_token_type_ids=sub_title_token_type_ids,
                        )
                        all_reps.append(sub_cosine_sim)
                        all_labels.append(sub_label)
                        all_title.append(title_embedding)
                        all_query.append(query_embedding)

                model_reps = paddle.concat(all_reps, axis=0)
                model_title = paddle.concat(all_title)
                model_query = paddle.concat(all_query)

                model_title = model_title.detach()
                model_query = model_query.detach()

                model_query.stop_gradient = False
                model_title.stop_gradient = False
                model_reps.stop_gradient = False

                model_label = paddle.concat(all_labels, axis=0)
                loss = F.cross_entropy(input=model_reps, label=model_label)
                loss.backward()
                # Store gradients
                all_grads.append(model_reps.grad)

            for sub_batch, CUDA_state, grad in zip(sub_batchs, all_CUDA_rnd_state, all_grads):

                (
                    sub_query_input_ids,
                    sub_query_token_type_ids,
                    sub_title_input_ids,
                    sub_title_token_type_ids,
                ) = sub_batch
                paddle.framework.random.set_cuda_rng_state(CUDA_state)
                # Recompute the forward propagation
                sub_cosine_sim, sub_label, query_embedding, title_embedding = model(
                    query_input_ids=sub_query_input_ids,
                    title_input_ids=sub_title_input_ids,
                    query_token_type_ids=sub_query_token_type_ids,
                    title_token_type_ids=sub_title_token_type_ids,
                )
                # Chain rule
                surrogate = paddle.dot(sub_cosine_sim, grad)
                # Backward propagation
                if args.use_amp:
                    scaled = scaler.scale(surrogate)
                    scaled.backward()
                else:
                    surrogate.backward()
            # Update model parameters
            if args.use_amp:
                scaler.minimize(optimizer, scaled)
            else:
                optimizer.step()

            global_step += 1
            if global_step % args.log_steps == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, args.log_steps / (time.time() - tic_train))
                )
                tic_train = time.time()

            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.save_steps == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, "model_state.pdparams")
                paddle.save(model.state_dict(), save_param_path)
                tokenizer.save_pretrained(save_dir)


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    train_ds = load_dataset(read_text_pair, data_path=args.train_set_file, lazy=False)

    pretrained_model = AutoModel.from_pretrained(args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(  # noqa: E731
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # query_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # query_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # title_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # title_segment
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(
        train_ds, mode="train", batch_size=args.batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
    )
    if args.use_gradient_cache:
        model = SemanticIndexCacheNeg(
            pretrained_model, margin=args.margin, scale=args.scale, output_emb_size=args.output_emb_size
        )
    else:
        model = SemanticIndexBatchNeg(
            pretrained_model, margin=args.margin, scale=args.scale, output_emb_size=args.output_emb_size
        )

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("warmup from:{}".format(args.init_from_ckpt))

    model = paddle.DataParallel(model)

    batchify_fn_dev = lambda samples, fn=Tuple(  # noqa: E731
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # text_segment
    ): [data for data in fn(samples)]

    id2corpus = gen_id2corpus(args.corpus_file)

    # convert_example function's input must be dict
    corpus_list = [{idx: text} for idx, text in id2corpus.items()]
    corpus_ds = MapDataset(corpus_list)

    corpus_data_loader = create_dataloader(
        corpus_ds, mode="predict", batch_size=args.batch_size, batchify_fn=batchify_fn_dev, trans_fn=trans_func
    )

    text_list, text2similar_text = gen_text_file(args.similar_text_pair_file)

    query_ds = MapDataset(text_list)

    query_data_loader = create_dataloader(
        query_ds, mode="predict", batch_size=args.batch_size, batchify_fn=batchify_fn_dev, trans_fn=trans_func
    )

    if not os.path.exists(args.recall_result_dir):
        os.mkdir(args.recall_result_dir)

    recall_result_file = os.path.join(args.recall_result_dir, args.recall_result_file)

    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm),
    )

    if args.use_gradient_cache:
        gradient_cache_train(train_data_loader, model, optimizer, lr_scheduler, rank, tokenizer)
    else:
        train(
            train_data_loader,
            model,
            optimizer,
            lr_scheduler,
            rank,
            corpus_data_loader,
            query_data_loader,
            recall_result_file,
            text_list,
            id2corpus,
            tokenizer,
        )


if __name__ == "__main__":
    do_train()
