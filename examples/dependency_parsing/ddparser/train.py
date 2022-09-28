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
import time
import random
from functools import partial

import numpy as np
import paddle
from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.transformers.optimization import LinearDecayWithWarmup
from paddlenlp.datasets import load_dataset

from data import create_dataloader, build_vocab, convert_example
from model.dep import BiAffineParser
from metric import ParserEvaluator
from criterion import ParserCriterion
from utils import decode, flat_words

# yapf: disable
parser = argparse.ArgumentParser()
# Train
parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--task_name", choices=["nlpcc13_evsam05_thu", "nlpcc13_evsam05_hit"], type=str, default="nlpcc13_evsam05_thu", help="Select the task.")
parser.add_argument("--encoding_model", choices=["lstm", "lstm-pe", "ernie-3.0-medium-zh", "ernie-1.0", "ernie-tiny", "ernie-gram-zh"], type=str, default="ernie-3.0-medium-zh", help="Select the encoding model.")
parser.add_argument("--epochs", type=int, default=100, help="Number of epoches for training.")
parser.add_argument("--save_dir", type=str, default='model_file/', help="Directory to save model parameters.")
parser.add_argument("--batch_size", type=int, default=1000, help="Numbers of examples a batch for training.")
parser.add_argument("--init_from_params", type=str, default=None, help="The path of model parameters to be loaded.")
parser.add_argument("--clip", type=float, default=1.0, help="The threshold of gradient clip.")
parser.add_argument("--lstm_lr", type=float, default=0.002, help="The Learning rate of lstm encoding model.")
parser.add_argument("--ernie_lr", type=float, default=5e-05, help="The Learning rate of ernie encoding model.")
parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization.")
# Preprocess
parser.add_argument("--n_buckets", type=int, default=15, help="Number of buckets to devide the dataset.")
# Postprocess
parser.add_argument("--tree", type=bool, default=True, help="Ensure the output conforms to the tree structure.")
# Lstm
parser.add_argument("--feat", choices=["char", "pos"], type=str, default=None, help="The feature representation to use.")
# Ernie
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="Linear warmup proportion over total steps.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay if we apply some.")
args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def batch_evaluate(
    model,
    metric,
    criterion,
    data_loader,
    word_pad_index,
    word_bos_index,
    word_eos_index,
):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader():
        if args.encoding_model.startswith(
                "ernie") or args.encoding_model == "lstm-pe":
            words, arcs, rels = batch
            words, feats = flat_words(words)
            s_arc, s_rel, words = model(words, feats)
        else:
            words, feats, arcs, rels = batch
            s_arc, s_rel, words = model(words, feats)

        mask = paddle.logical_and(
            paddle.logical_and(words != word_pad_index,
                               words != word_bos_index),
            words != word_eos_index,
        )

        loss = criterion(s_arc, s_rel, arcs, rels, mask)

        losses.append(loss.numpy().item())

        arc_preds, rel_preds = decode(s_arc, s_rel, mask)
        metric.update(arc_preds, rel_preds, arcs, rels, mask)
    uas, las = metric.accumulate()
    total_loss = np.mean(losses)
    model.train()
    metric.reset()
    return total_loss, uas, las


def do_train(args):
    set_seed(args.seed)
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    if args.encoding_model == "ernie-gram-zh":
        tokenizer = AutoTokenizer.from_pretrained(args.encoding_model)
    elif args.encoding_model.startswith("ernie"):
        tokenizer = AutoTokenizer.from_pretrained(args.encoding_model)
    elif args.encoding_model == "lstm-pe":
        tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
    else:
        tokenizer = None

    train_ds, dev_ds = load_dataset(args.task_name, splits=["train", "dev"])

    # Build the vocabs based on train corpus
    word_examples = [seq["FORM"] for seq in train_ds]
    if args.feat == "pos":
        feat_examples = [seq["CPOS"] for seq in train_ds]
    elif args.feat == "char":
        feat_examples = [token for seq in train_ds for token in seq["FORM"]]
    else:
        feat_examples = None
    rel_examples = [seq["DEPREL"] for seq in train_ds]

    train_corpus = [word_examples, feat_examples, rel_examples]
    vocabs = build_vocab(
        train_corpus,
        tokenizer,
        encoding_model=args.encoding_model,
        feat=args.feat,
    )
    word_vocab, feat_vocab, rel_vocab = vocabs

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Save vocabs into json file
    word_vocab.to_json(path=os.path.join(args.save_dir, "word_vocab.json"))
    rel_vocab.to_json(path=os.path.join(args.save_dir, "rel_vocab.json"))

    if feat_vocab:
        n_feats = len(feat_vocab)
        feat_vocab.to_json(path=os.path.join(args.save_dir, "feat_vocab.json"))
        word_pad_index = word_vocab.to_indices("[PAD]")
        word_bos_index = word_vocab.to_indices("[BOS]")
        word_eos_index = word_vocab.to_indices("[EOS]")
    else:
        n_feats = None
        word_pad_index = word_vocab.to_indices("[PAD]")
        word_bos_index = word_vocab.to_indices("[CLS]")
        word_eos_index = word_vocab.to_indices("[SEP]")

    n_rels, n_words = len(rel_vocab), len(word_vocab)

    trans_fn = partial(
        convert_example,
        vocabs=vocabs,
        encoding_model=args.encoding_model,
        feat=args.feat,
    )

    train_data_loader, _ = create_dataloader(
        train_ds,
        batch_size=args.batch_size,
        mode="train",
        n_buckets=args.n_buckets,
        trans_fn=trans_fn,
    )
    dev_data_loader, _ = create_dataloader(
        dev_ds,
        batch_size=args.batch_size,
        mode="dev",
        n_buckets=args.n_buckets,
        trans_fn=trans_fn,
    )

    # Load pretrained model if encoding model is ernie-3.0-medium-zh, ernie-1.0, ernie-tiny or ernie-gram-zh
    if args.encoding_model in [
            "ernie-3.0-medium-zh", "ernie-1.0", "ernie-tiny", "ernie-gram-zh"
    ]:
        pretrained_model = AutoModel.from_pretrained(args.encoding_model)
    else:
        pretrained_model = None

    # Load ddparser model
    model = BiAffineParser(
        encoding_model=args.encoding_model,
        feat=args.feat,
        n_rels=n_rels,
        n_feats=n_feats,
        n_words=n_words,
        pad_index=word_pad_index,
        eos_index=word_eos_index,
        pretrained_model=pretrained_model,
    )

    # Define learning rate
    if args.encoding_model.startswith("ernie"):
        lr = args.ernie_lr
    else:
        lr = args.lstm_lr

    # Continue training from a pretrained model if the checkpoint is specified
    if args.init_from_params and os.path.isfile(args.init_from_params):
        state_dict = paddle.load(args.init_from_params)
        model.set_dict(state_dict)

    # Data parallel for distributed training
    model = paddle.DataParallel(model)
    trainer_num = paddle.distributed.get_world_size()

    num_training_steps = len(list(train_data_loader)) * args.epochs

    # Define the training strategy
    lr_scheduler = LinearDecayWithWarmup(lr, num_training_steps,
                                         args.warmup_proportion)
    grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.clip)
    if args.encoding_model.startswith("ernie"):
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            grad_clip=grad_clip,
        )
    else:
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr,
            beta1=0.9,
            beta2=0.9,
            epsilon=1e-12,
            parameters=model.parameters(),
            grad_clip=grad_clip,
        )

    # Load metric and criterion
    best_las = 0
    metric = ParserEvaluator()
    criterion = ParserCriterion()

    # Epoch train
    global_step = 0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for inputs in train_data_loader():
            if args.encoding_model.startswith(
                    "ernie") or args.encoding_model == "lstm-pe":
                words, arcs, rels = inputs
                words, feats = flat_words(words)
                s_arc, s_rel, words = model(words, feats)
            else:
                words, feats, arcs, rels = inputs
                s_arc, s_rel, words = model(words, feats)

            mask = paddle.logical_and(
                paddle.logical_and(words != word_pad_index,
                                   words != word_bos_index),
                words != word_eos_index,
            )

            loss = criterion(s_arc, s_rel, arcs, rels, mask)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, loss.numpy().item(), 10 /
                       (time.time() - tic_train)))
                tic_train = time.time()

        if rank == 0:
            # Evaluate on dev dataset
            loss, uas, las = batch_evaluate(
                model,
                metric,
                criterion,
                dev_data_loader,
                word_pad_index,
                word_bos_index,
                word_eos_index,
            )
            print("eval loss: %.5f, UAS: %.2f%%, LAS: %.2f%%" %
                  (loss, uas * 100, las * 100))
            # Save model parameter of last epoch
            save_param_path = os.path.join(args.save_dir, "last_epoch.pdparams")
            paddle.save(model.state_dict(), save_param_path)
            # Save the model if it get a higher score of las
            if las > best_las:
                save_param_path = os.path.join(args.save_dir, "best.pdparams")
                paddle.save(model.state_dict(), save_param_path)
                best_las = las


if __name__ == "__main__":
    do_train(args)
