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

import logging
import argparse
import os
import time

import numpy as np
import paddle
import paddlenlp as ppnlp
from paddlenlp.transformers.optimization import LinearDecayWithWarmup

from env import Environment
from data import batchify, TextDataset, Corpus
from model.dep import BiaffineDependencyModel
from metric import ParserEvaluator
from criterion import ParserCriterion
from utils import decode, index_sample

# yapf: disable
parser = argparse.ArgumentParser()
# Run
parser.add_argument("--mode", choices=["train", "evaluate", "predict"], type=str, default="train", help="Select the task mode.")
parser.add_argument("--device", choices=["cpu", "gpu", "xpu"], default="gpu", help="Select which device to train model, defaults to gpu.")
# Train
parser.add_argument("--encoding_model", choices=["lstm", "lstm-pe", "ernie-1.0", "ernie-tiny", "ernie-gram-zh"], type=str, default="ernie-1.0", help="Select the encoding model.")
parser.add_argument("--preprocess", type=bool, default=True, help="Whether to preprocess the dataset.")
parser.add_argument("--epochs", type=int, default=100, help="Number of epoches for training.")
parser.add_argument("--save_dir", type=str, default='model_file/', help="Directory to save model parameters.")
parser.add_argument("--train_data_path", type=str, default='./data/train.txt', help="The path of train dataset to be loaded.")
parser.add_argument("--dev_data_path", type=str, default='./data/dev.txt', help="The path of dev dataset to be loaded.")
parser.add_argument("--batch_size", type=int, default=1000, help="Numbers of examples a batch for training.")
parser.add_argument("--init_from_params", type=str, default=None, help="The path of model parameters to be loaded.")
parser.add_argument("--clip", type=float, default=1.0, help="The threshold of gradient clip.")
parser.add_argument("--lstm_lr", type=float, default=0.002, help="The Learning rate of lstm encoding model.")
parser.add_argument("--ernie_lr", type=float, default=5e-05, help="The Learning rate of ernie encoding model.")
parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization.")
# Evaluate & Predict
parser.add_argument("--test_data_path", type=str, default='./data/test.txt', help="The path of test dataset to be loaded.")
parser.add_argument("--model_file_path", type=str, default='model_file/', help="Directory to load model parameters.")
parser.add_argument("--infer_result_dir", type=str, default='infer_result/', help="The path to save infer results.")
# Preprocess
parser.add_argument("--min_freq", type=int, default=2, help="The minimum frequency of word when construct the vocabulary.")
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


@paddle.no_grad()
def evaluate(args, model, metric, criterion, data_loader):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader():
        if args.encoding_model.startswith("ernie") or args.encoding_model == "lstm-pe":
            words, arcs, rels = batch
            s_arc, s_rel, words = model(words)
        else:
            words, feats, arcs, rels = batch
            s_arc, s_rel, words = model(words, feats)   

        mask = paddle.logical_and(
                paddle.logical_and(words != args.pad_index, words != args.bos_index),
                words != args.eos_index,
        )

        loss = criterion(s_arc, s_rel, arcs, rels, mask)

        losses.append(loss.numpy().item())

        arc_preds, rel_preds = decode(args, s_arc, s_rel, mask)
        metric.update(arc_preds, rel_preds, arcs, rels, mask)
        uas, las = metric.accumulate()
    total_loss = np.mean(losses)
    model.train()
    metric.reset()
    return total_loss, uas, las


@paddle.no_grad()
def predict(env, args, model, data_loader):
    arcs, rels = [], []
    for inputs in data_loader():
        if args.encoding_model.startswith("ernie") or args.encoding_model == "lstm-pe":
            words = inputs[0]
            s_arc, s_rel, words = model(words)
        else:
            words, feats = inputs
            s_arc, s_rel, words = model(words, feats)
        mask = paddle.logical_and(
            paddle.logical_and(words != args.pad_index, words != args.bos_index),
            words != args.eos_index,
        )
        lens = paddle.sum(paddle.cast(mask, "int32"), axis=-1)
        arc_preds, rel_preds = decode(args, s_arc, s_rel, mask)
        arcs.extend(paddle.split(paddle.masked_select(arc_preds, mask), lens.numpy().tolist()))
        rels.extend(paddle.split(paddle.masked_select(rel_preds, mask), lens.numpy().tolist()))

    arcs = [seq.numpy().tolist() for seq in arcs]
    rels = [env.REL.vocab[seq.numpy().tolist()] for seq in rels]           

    return arcs, rels


def do_train(env):
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    # Load datasets
    train = Corpus.load(args.train_data_path, env.fields)
    dev = Corpus.load(args.dev_data_path, env.fields)

    train_ds = TextDataset(train, env.fields, args.n_buckets)
    dev_ds = TextDataset(dev, env.fields, args.n_buckets)

    train_data_loader = batchify(train_ds, args.batch_size, shuffle=True, use_multiprocess=True)
    dev_data_loader = batchify(dev_ds, args.batch_size)
    
    # Load pretrained model if encoding model is ernie-1.0, ernie-tiny or ernie-gram-zh
    if args.encoding_model in ["ernie-1.0", "ernie-tiny"]:
        pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(args.encoding_model)
    elif args.encoding_model == "ernie-gram-zh":
        pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(args.encoding_model)       

    # Define ddparser model and learning rate
    if args.encoding_model.startswith("ernie"):
        lr = args.ernie_lr
        model = BiaffineDependencyModel(args=args, pretrained_model=pretrained_model)
    else:
        lr = args.lstm_lr
        model = BiaffineDependencyModel(args=args)

    # Continue training from a pretrained model if the checkpoint is specified
    if args.init_from_params and os.path.isfile(args.init_from_params):
        state_dict = paddle.load(args.init_from_params)
        model.set_dict(state_dict)

    # Data parallel for distributed training
    model = paddle.DataParallel(model)
    num_training_steps = len(train_data_loader) * args.epochs

    # Define the training strategy
    lr_scheduler = LinearDecayWithWarmup(lr, num_training_steps, args.warmup_proportion)
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
            learning_rate=lr_scheduler,
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
            if args.encoding_model.startswith("ernie") or args.encoding_model == "lstm-pe":
                words, arcs, rels = inputs
                s_arc, s_rel, words = model(words)
            else:
                words, feats, arcs, rels = inputs
                s_arc, s_rel, words = model(words, feats)
        
            mask = paddle.logical_and(
                paddle.logical_and(words != args.pad_index, words != args.bos_index),
                words != args.eos_index,
            )

            loss = criterion(s_arc, s_rel, arcs, rels, mask)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            global_step += 1
            if global_step % 100 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, loss.numpy().item(), 10 / (time.time() - tic_train)))
                tic_train = time.time()

        if rank == 0:
            # Epoch evaluate
            loss, uas, las = evaluate(args, model, metric, criterion, dev_data_loader)
            print("eval loss: %.5f, UAS: %.2f%%, LAS: %.2f%%" % (loss, uas*100, las*100))
            # Save model parameter of last epoch
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_param_path = os.path.join(args.save_dir, "last_epoch.pdparams")
            paddle.save(model.state_dict(), save_param_path)       
            # Save the model if it get a higher las
            if las > best_las:
                save_param_path = os.path.join(args.save_dir, "best.pdparams")
                paddle.save(model.state_dict(), save_param_path)  
                best_las = las                 


def do_evaluate(env):
    args = env.args

    # Load dataset
    data = Corpus.load(args.test_data_path, env.fields)
    evaluate_ds = TextDataset(data, env.fields, args.n_buckets)
    evaluate_data_loader = batchify(evaluate_ds, args.batch_size)

    # Load pretrained model if encoding model is ernie-1.0, ernie-tiny or ernie-gram-zh
    if args.encoding_model in ["ernie-1.0", "ernie-tiny"]:
        pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(args.encoding_model)
    elif args.encoding_model == "ernie-gram-zh":
        pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(args.encoding_model)

    # Define ddparser model
    if args.encoding_model.startswith("ernie"):
        model = BiaffineDependencyModel(args=args, pretrained_model=pretrained_model)
    else:
        model = BiaffineDependencyModel(args=args)
    
    # Load saved model parameters
    if os.path.isfile(args.model_file_path):
        state_dict = paddle.load(args.model_file_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.model_file_path)
    else:
        raise ValueError("The parameters path is incorrect or not specified.")

    # Load metric and criterion
    metric = ParserEvaluator()
    criterion = ParserCriterion()

    # Start evaluate
    loss, uas, las = evaluate(args, model, metric, criterion, evaluate_data_loader)
    print("eval loss: %.5f, UAS: %.2f%%, LAS: %.2f%%" % (loss, uas*100, las*100))
    

def do_predict(env):
    args = env.args

    # Load dataset
    data = Corpus.load(args.test_data_path, env.fields)
    predict_ds = TextDataset(data, [env.WORD, env.FEAT], args.n_buckets)
    predict_data_loader = batchify(predict_ds, args.batch_size)

    # Load pretrained model if encoding model is ernie-1.0, ernie-tiny or ernie-gram-zh
    if args.encoding_model in ["ernie-1.0", "ernie-tiny"]:
        pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(args.encoding_model)
    elif args.encoding_model == "ernie-gram-zh":
        pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(args.encoding_model)

    # Define ddparser model
    if args.encoding_model.startswith("ernie"):
        model = BiaffineDependencyModel(args=args, pretrained_model=pretrained_model)
    else:
        model = BiaffineDependencyModel(args=args)
    
    # Load saved model parameters
    if os.path.isfile(args.model_file_path):
        state_dict = paddle.load(args.model_file_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.model_file_path)
    else:
        raise ValueError("The parameters path is incorrect or not specified.")

    # Start predict
    pred_arcs, pred_rels = predict(env, args, model, predict_data_loader)
    indices = np.argsort(np.array([i for bucket in predict_ds.buckets.values() for i in bucket]))
    data.head = [pred_arcs[i] for i in indices]
    data.deprel = [pred_rels[i] for i in indices]

    # Save results
    data.save(args.infer_result_dir)


if __name__ == "__main__":
    paddle.set_device(args.device)
    env = Environment(args)
    if args.mode == "train":   
        do_train(env)
    elif args.mode == "evaluate":
        do_evaluate(env)
    elif args.mode == "predict":
        do_predict(env)
