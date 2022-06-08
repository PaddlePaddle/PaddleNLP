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
import os
import sys
import random
import time

from scipy import stats
import numpy as np
import paddle
import paddle.nn.functional as F

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import AutoTokenizer, AutoModel

from model import SimCSE
from data import read_simcse_text, read_text_pair, convert_example, create_dataloader
from data import word_repetition

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization."
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--output_emb_size", default=0, type=int, help="Output_embedding_size, 0 means use hidden_size as output embedding size.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=1, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument('--save_steps', type=int, default=10000, help="Step interval for saving checkpoint.")
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override ecpochs.")
parser.add_argument('--eval_steps', type=int, default=10000, help="Step interval for evaluation.")
parser.add_argument("--train_set_file", type=str, required=True, help="The full path of train_set_file.")
parser.add_argument("--margin", default=0.0, type=float, help="Margin beteween pos_sample and neg_samples.")
parser.add_argument("--scale", default=20, type=int, help="Scale for pair-wise margin_rank_loss.")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout for pretrained model encoder.")
parser.add_argument("--dup_rate", default=0.32, type=float, help="duplicate rate for word reptition.")
parser.add_argument("--infer_with_fc_pooler", action='store_true', help="Whether use fc layer after cls embedding or not for when infer.")
parser.add_argument("--rdrop_coef", default=0.0, type=float, help="The coefficient of KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")
args = parser.parse_args()

def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def do_evaluate(model, tokenizer, data_loader, with_pooler=False):
    model.eval()

    total_num = 0
    spearman_corr = 0.0
    sims = []
    labels = []

    for batch in data_loader:
        query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids, label = batch
        total_num += len(label)

        query_cls_embedding = model.get_pooled_embedding(
            query_input_ids, query_token_type_ids, with_pooler=with_pooler)

        title_cls_embedding = model.get_pooled_embedding(title_input_ids, title_token_type_ids, with_pooler=with_pooler)

        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding, axis=-1)

        sims.append(cosine_sim.numpy())
        labels.append(label.numpy())

    sims = np.concatenate(sims, axis=0)
    labels = np.concatenate(labels, axis=0)

    spearman_corr = stats.spearmanr(labels, sims).correlation
    model.train()
    return spearman_corr, total_num

def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)
    train_ds = load_dataset(
        read_text_pair, data_path=args.train_set_file,is_test=False, lazy=False)
    model_name_or_path='rocketqa-zh-dureader-query-encoder'
    pretrained_model = AutoModel.from_pretrained(
       model_name_or_path,
       hidden_dropout_prob=args.dropout,
       attention_probs_dropout_prob=args.dropout)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # query_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # query_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # title_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # tilte_segment
    ): [data for data in fn(samples)]


    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)


    model = SimCSE(
        pretrained_model,
        margin=args.margin,
        scale=args.scale,
        output_emb_size=args.output_emb_size)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("warmup from:{}".format(args.init_from_ckpt))

    model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.epochs

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
        apply_decay_param_fun=lambda x: x in decay_params)

    global_step = 0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch
            if(random.random()<0.2):
                title_input_ids,title_token_type_ids=query_input_ids,query_token_type_ids
                query_input_ids,query_token_type_ids=word_repetition(query_input_ids,query_token_type_ids,args.dup_rate)
                title_input_ids,title_token_type_ids=word_repetition(title_input_ids,title_token_type_ids,args.dup_rate)

            loss, kl_loss = model(
                query_input_ids=query_input_ids,
                title_input_ids=title_input_ids,
                query_token_type_ids=query_token_type_ids,
                title_token_type_ids=title_token_type_ids)

            loss = loss + kl_loss * args.rdrop_coef

            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % args.save_steps == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                tokenizer.save_pretrained(save_dir)

            if args.max_steps > 0 and global_step >= args.max_steps:
                return

    save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        save_param_path = os.path.join(save_dir, 'model_state.pdparams')
        paddle.save(model.state_dict(), save_param_path)
        tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    do_train()
