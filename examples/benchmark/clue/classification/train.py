from functools import partial
import numpy as np
import time
import os
import copy
import json
import random
from tqdm import tqdm
import argparse

import paddle
from paddlenlp.datasets import load_dataset
import paddle.nn.functional as F
import paddle.nn as nn
import paddlenlp as ppnlp
from paddlenlp.transformers import LinearDecayWithWarmup
from model import WSCModel, NLIModel, KeywordRecognitionModel, NLIModel
from model import LongTextClassification, ShortTextClassification, PointwiseMatching
from data import convert_wsc_example, convert_example, convert_csl_example
from data import convert_iflytek_example, convert_tnews_example
from paddlenlp.transformers import AutoTokenizer, AutoModel

model_dict = {
    "cluewsc2020": WSCModel,
    "ocnli": NLIModel,
    "csl": KeywordRecognitionModel,
    "cmnli": NLIModel,
    "iflytek": LongTextClassification,
    "tnews": ShortTextClassification,
    "afqmc": PointwiseMatching
}

data_process = {
    "cluewsc2020": convert_wsc_example,
    "ocnli": convert_example,
    "csl": convert_csl_example,
    "cmnli": convert_example,
    "iflytek": convert_iflytek_example,
    "tnews": convert_tnews_example,
    "afqmc": convert_example
}


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, phase="dev"):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(probs, labels)
        losses.append(loss.numpy())
        correct = metric.compute(probs, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval {} loss: {:.5}, accu: {:.5}".format(phase,
                                                    np.mean(losses), accu))
    model.train()
    metric.reset()
    return np.mean(losses), accu


def do_train(model, criterion, metric, dev_data_loader, train_data_loader):
    global_step = 0
    tic_train = time.time()
    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):

            input_ids, token_type_ids, labels = batch
            probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
            loss = criterion(probs, labels)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1

            # logging every 100 steps 
            if global_step % 100 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            # evaluate the model every 100 steps
            if global_step % 400 == 0:
                eval_loss, eval_accu = evaluate(model, criterion, metric,
                                                dev_data_loader, "dev")
                if (best_accuracy < eval_accu):
                    best_accuracy = eval_accu
                    # save model
                    save_param_path = os.path.join(save_dir,
                                                   'model_best.pdparams')
                    paddle.save(model.state_dict(), save_param_path)
                    # save tokenizer
                    tokenizer.save_pretrained(save_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to clue")

    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.")

    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )

    parser.add_argument(
        "--output_dir",
        default="checkpoint",
        type=str,
        help="The output directory where checkpoints will be written.", )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args.task_name)
    train_ds, dev_ds = load_dataset(
        'clue', args.task_name, splits=['train', 'dev'])

    # use ernie-gram-zh pretrained model
    pretrained_model = AutoModel.from_pretrained('ernie-gram-zh')
    tokenizer = AutoTokenizer.from_pretrained('ernie-gram-zh')

    train_ds = train_ds.map(
        partial(
            data_process[args.task_name], tokenizer=tokenizer))
    dev_ds = dev_ds.map(
        partial(
            data_process[args.task_name], tokenizer=tokenizer))

    train_batch_size = 32
    dev_batch_size = 32
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        dataset=train_ds, batch_size=train_batch_size, shuffle=True)
    train_data_loader = paddle.io.DataLoader(
        dataset=train_ds, batch_sampler=train_batch_sampler, return_list=True)

    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=dev_batch_size, shuffle=False)

    dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds, batch_sampler=dev_batch_sampler, return_list=True)

    model = model_dict[args.task_name](pretrained_model,
                                       len(train_ds.label_list))

    epochs = args.num_train_epochs
    num_training_steps = len(train_data_loader) * epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         0.0)

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=0.0,
        apply_decay_param_fun=lambda x: x in decay_params)
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()
    do_train(model, criterion, metric, dev_data_loader, train_data_loader)
