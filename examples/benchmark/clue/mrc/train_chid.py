import pickle
from functools import partial
import collections
import time
import json
import inspect
import os
from tqdm import tqdm
import numpy as np
import random

import paddle
from paddle.io import TensorDataset, DataLoader
from paddlenlp.transformers import ErnieForMultipleChoice, ErnieTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Dict, Pad
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
from paddlenlp.transformers import LinearDecayWithWarmup
import paddlenlp as ppnlp
from CHID_preprocess import RawResult, get_final_predictions, write_predictions, generate_input, evaluate


def process_train_data(input_dir, tokenizer, max_seq_length, max_num_choices):

    train_file = 'work/train.json'
    train_ans_file = 'work/train_answer.json'

    train_example_file = os.path.join(
        input_dir, 'train_examples_{}.pkl'.format(str(max_seq_length)))
    train_feature_file = os.path.join(
        input_dir, 'train_features_{}.pkl'.format(str(max_seq_length)))

    train_features = generate_input(
        train_file,
        train_ans_file,
        train_example_file,
        train_feature_file,
        tokenizer,
        max_seq_length=max_seq_length,
        max_num_choices=max_num_choices,
        is_training=True)

    print("loaded train dataset")
    print("Num generate examples = {}".format(len(train_features)))

    all_input_ids = paddle.to_tensor(
        [f.input_ids for f in train_features], dtype='int64')
    all_input_masks = paddle.to_tensor(
        [f.input_masks for f in train_features], dtype='int64')
    all_segment_ids = paddle.to_tensor(
        [f.segment_ids for f in train_features], dtype='int64')
    all_choice_masks = paddle.to_tensor(
        [f.choice_masks for f in train_features], dtype='int64')
    all_labels = paddle.to_tensor(
        [f.label for f in train_features], dtype='int64')

    train_data = TensorDataset([
        all_input_ids, all_input_masks, all_segment_ids, all_choice_masks,
        all_labels
    ])

    return train_data


def process_validation_data(input_dir, tokenizer, max_seq_length,
                            max_num_choices):

    predict_file = 'work/dev.json'
    dev_example_file = os.path.join(
        input_dir, 'dev_examples_{}.pkl'.format(str(max_seq_length)))
    dev_feature_file = os.path.join(
        input_dir, 'dev_features_{}.pkl'.format(str(max_seq_length)))

    eval_features = generate_input(
        predict_file,
        None,
        dev_example_file,
        dev_feature_file,
        tokenizer,
        max_seq_length=max_seq_length,
        max_num_choices=max_num_choices,
        is_training=False)

    all_example_ids = [f.example_id for f in eval_features]
    all_tags = [f.tag for f in eval_features]
    all_input_ids = paddle.to_tensor(
        [f.input_ids for f in eval_features], dtype="int64")
    all_input_masks = paddle.to_tensor(
        [f.input_masks for f in eval_features], dtype="int64")
    all_segment_ids = paddle.to_tensor(
        [f.segment_ids for f in eval_features], dtype="int64")
    all_choice_masks = paddle.to_tensor(
        [f.choice_masks for f in eval_features], dtype="int64")
    all_example_index = paddle.arange(all_input_ids.shape[0], dtype="int64")

    eval_data = TensorDataset([
        all_input_ids, all_input_masks, all_segment_ids, all_choice_masks,
        all_example_index
    ])

    return eval_data, all_example_ids, all_tags, eval_features


@paddle.no_grad()
def do_evaluate(model, dev_data_loader, all_example_ids, all_tags,
                eval_features):

    all_results = []
    model.eval()
    output_dir = 'data'
    for step, batch in enumerate(tqdm(dev_data_loader)):

        input_ids, input_masks, segment_ids, choice_masks, example_indices = batch
        batch_logits = model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_masks)

        for i, example_index in enumerate(example_indices):
            logits = batch_logits[i].numpy().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    example_id=all_example_ids[unique_id],
                    tag=all_tags[unique_id],
                    logit=logits))

    predict_file = 'dev_predictions.json'
    predict_ans_file = 'work/dev_answer.json'
    print('decoder raw results')
    tmp_predict_file = os.path.join(output_dir, "raw_predictions.pkl")
    output_prediction_file = os.path.join(output_dir, predict_file)
    results = get_final_predictions(all_results, tmp_predict_file, g=True)
    write_predictions(results, output_prediction_file)
    print('predictions saved to {}'.format(output_prediction_file))

    acc = evaluate(predict_ans_file, output_prediction_file)
    print(f'{predict_file} 预测精度：{acc}')
    model.train()
    return acc


def do_train(model, train_data_loader, dev_data_loader, all_example_ids,
             all_tags, eval_features):

    model.train()
    global_step = 0
    tic_train = time.time()
    log_step = 100
    for epoch in range(num_train_epochs):
        metric.reset()
        for step, batch in enumerate(train_data_loader):
            input_ids, input_masks, segment_ids, choice_masks, labels = batch

            logits = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_masks)

            loss = criterion(logits, labels)
            correct = metric.compute(logits, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1

            # 每间隔 log_step 输出训练指标
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

        do_evaluate(model, dev_data_loader, all_example_ids, all_tags,
                    eval_features)
        model.save_pretrained("./checkpoint")


if __name__ == "__main__":
    batch_size = 4
    input_dir = 'output'
    os.makedirs(input_dir, exist_ok=True)
    max_seq_length = 64
    max_num_choices = 10
    num_train_epochs = 3
    max_grad_norm = 1.0
    max_num_choices = 10

    MODEL_NAME = "ernie-1.0"
    tokenizer = ErnieTokenizer.from_pretrained(MODEL_NAME)

    train_data = process_train_data(input_dir, tokenizer, max_seq_length,
                                    max_num_choices)
    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        drop_last=True,
        num_workers=0)

    eval_data, all_example_ids, all_tags, eval_features = process_validation_data(
        input_dir, tokenizer, max_seq_length, max_num_choices)

    # Run prediction for full data
    dev_data_loader = DataLoader(eval_data, batch_size=batch_size)

    model = ErnieForMultipleChoice.from_pretrained(
        MODEL_NAME, num_choices=max_num_choices)

    num_training_steps = len(train_data_loader) * num_train_epochs

    # 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
    lr_scheduler = LinearDecayWithWarmup(2e-5, num_training_steps, 0)
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    grad_clip = paddle.nn.ClipGradByGlobalNorm(max_grad_norm)

    # 定义 Optimizer
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=0.01,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=grad_clip)
    # 交叉熵损失
    criterion = paddle.nn.loss.CrossEntropyLoss()
    # 评估的时候采用准确率指标
    metric = paddle.metric.Accuracy()
    # 模型训练
    do_train(model, train_data_loader, dev_data_loader, all_example_ids,
             all_tags, eval_features)
