import json
import numpy as np
from tqdm import tqdm
import os
import pickle
import logging
import time
import random
import pandas as pd

import paddle
from paddle.io import TensorDataset
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Dict, Pad, Tuple
from paddlenlp.transformers import ErnieForMultipleChoice, ErnieTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from C3_preprocess import c3Processor, convert_examples_to_features


def process_validation_data(data_dir, processor, tokenizer, n_class,
                            max_seq_length):

    label_list = processor.get_labels()
    eval_examples = processor.get_dev_examples()
    feature_dir = os.path.join(data_dir,
                               'dev_features{}.pkl'.format(max_seq_length))

    if os.path.exists(feature_dir):
        eval_features = pickle.load(open(feature_dir, 'rb'))
    else:
        eval_features = convert_examples_to_features(eval_examples, label_list,
                                                     max_seq_length, tokenizer)
        with open(feature_dir, 'wb') as w:
            pickle.dump(eval_features, w)

    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []

    for f in eval_features:
        input_ids.append([])
        input_mask.append([])
        segment_ids.append([])
        for i in range(n_class):
            input_ids[-1].append(f[i].input_ids)
            input_mask[-1].append(f[i].input_mask)
            segment_ids[-1].append(f[i].segment_ids)
        label_id.append(f[0].label_id)

    all_input_ids = paddle.to_tensor(input_ids, dtype='int64')
    all_input_mask = paddle.to_tensor(input_mask, dtype='int64')
    all_segment_ids = paddle.to_tensor(segment_ids, dtype='int64')
    all_label_ids = paddle.to_tensor(label_id, dtype='int64')

    dev_data = TensorDataset(
        [all_input_ids, all_input_mask, all_segment_ids, all_label_ids])

    return dev_data


def process_train_data(data_dir, processor, tokenizer, n_class, max_seq_length):

    label_list = processor.get_labels()
    train_examples = processor.get_train_examples()

    feature_dir = os.path.join(data_dir,
                               'train_features{}.pkl'.format(max_seq_length))
    if os.path.exists(feature_dir):
        train_features = pickle.load(open(feature_dir, 'rb'))
    else:
        train_features = convert_examples_to_features(
            train_examples, label_list, max_seq_length, tokenizer)
        with open(feature_dir, 'wb') as w:
            pickle.dump(train_features, w)

    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []
    for f in train_features:
        input_ids.append([])
        input_mask.append([])
        segment_ids.append([])
        for i in range(n_class):
            input_ids[-1].append(f[i].input_ids)
            input_mask[-1].append(f[i].input_mask)
            segment_ids[-1].append(f[i].segment_ids)
        label_id.append(f[0].label_id)

    all_input_ids = paddle.to_tensor(input_ids, dtype='int64')
    all_input_mask = paddle.to_tensor(input_mask, dtype='int64')
    all_segment_ids = paddle.to_tensor(segment_ids, dtype='int64')
    all_label_ids = paddle.to_tensor(label_id, dtype='int64')

    train_data = TensorDataset(
        [all_input_ids, all_input_mask, all_segment_ids, all_label_ids])

    return train_data


@paddle.no_grad()
def evaluate(model, dev_data_loader, metric):
    all_loss = []
    metric.reset()
    criterion = paddle.nn.loss.CrossEntropyLoss()
    model.eval()
    for step, batch in enumerate(dev_data_loader):

        input_ids, input_mask, segment_ids, label_id = batch
        logits = model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask)

        loss = criterion(logits, label_id)
        correct = metric.compute(logits, label_id)
        metric.update(correct)
        all_loss.append(loss.numpy())

    acc = metric.accumulate()
    model.train()
    return np.mean(all_loss), acc


def do_train(model, metric, criterion, train_data_loader, dev_data_loader):
    model.train()
    global_step = 0
    tic_train = time.time()
    log_step = 100
    for epoch in range(EPOCH):
        metric.reset()
        for step, batch in enumerate(train_data_loader):
            input_ids, input_mask, segment_ids, label_id = batch

            logits = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask)

            loss = criterion(logits, label_id)
            correct = metric.compute(logits, label_id)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1

            # output logging every log_step 
            if global_step % log_step == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        loss, acc = evaluate(model, dev_data_loader, metric)
        print("epoch: %d,  eval loss: %.5f, accu: %.5f" % (epoch, loss, acc))
        model.save_pretrained("./checkpoint")


if __name__ == "__main__":

    data_dir = 'data'
    processor = c3Processor(data_dir)

    MODEL_NAME = "ernie-1.0"
    tokenizer = ErnieTokenizer.from_pretrained(MODEL_NAME)

    max_seq_length = 512
    n_class = 4
    batch_size = 4
    EPOCH = 8
    max_grad_norm = 1.0
    max_num_choices = 4

    output_dir = 'checkpoints'
    os.makedirs(output_dir, exist_ok=True)

    train_data = process_train_data(output_dir, processor, tokenizer, n_class,
                                    max_seq_length)
    train_data_loader = paddle.io.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        drop_last=True,
        num_workers=0)

    dev_data = process_validation_data(output_dir, processor, tokenizer,
                                       n_class, max_seq_length)

    dev_data_loader = paddle.io.DataLoader(
        dataset=dev_data, batch_size=batch_size, drop_last=True, num_workers=0)

    model = ErnieForMultipleChoice.from_pretrained(
        MODEL_NAME, num_choices=max_num_choices)

    num_training_steps = len(train_data_loader) * EPOCH

    lr_scheduler = LinearDecayWithWarmup(2e-5, num_training_steps, 0)
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    grad_clip = paddle.nn.ClipGradByGlobalNorm(max_grad_norm)

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=0.01,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=grad_clip)
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()
    do_train(model, metric, criterion, train_data_loader, dev_data_loader)
