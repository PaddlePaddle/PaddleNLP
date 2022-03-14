import json
import numpy as np
from tqdm import tqdm
import os
import pickle
import logging
import time
import random
import pandas as pd
import argparse
import numpy as np

from paddlenlp.transformers import ErnieForMultipleChoice, ErnieTokenizer
from paddlenlp.transformers import RobertaForMultipleChoice, RobertaTokenizer
import paddle
from paddle.io import TensorDataset
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Dict, Pad, Tuple
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import ErnieForMultipleChoice, ErnieTokenizer
from paddlenlp.transformers import RobertaForMultipleChoice, RobertaTokenizer
from C3_preprocess import c3Processor, convert_examples_to_features

MODEL_CLASSES = {
    "ernie": (ErnieForMultipleChoice, ErnieTokenizer),
    "roberta": (RobertaForMultipleChoice, RobertaTokenizer)
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--output_dir",
        default="best_c3_model",
        type=str,
        help="The  path of the checkpoints .", )
    parser.add_argument(
        "--num_train_epochs",
        default=10,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion"
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Linear warmup proportion over total steps.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--learning_rate",
        default=2e-4,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="The max value of grad norm.")

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.")
    args = parser.parse_args()
    return args


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


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


def do_train(args, model, metric, criterion, train_data_loader,
             dev_data_loader):
    model.train()
    global_step = 0
    tic_train = time.time()
    best_acc = 0.0
    for epoch in range(args.num_train_epochs):
        metric.reset()
        for step, batch in enumerate(train_data_loader):
            input_ids, input_mask, segment_ids, label_id = batch
            logits = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask)
            loss = criterion(logits, label_id)

            global_step += 1

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % args.logging_steps == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss,
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
                loss, acc = evaluate(model, dev_data_loader, metric)
                if paddle.distributed.get_rank() == 0 and acc > best_acc:
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)
                    model_to_save.save_pretrained(args.output_dir)
                    best_acc = acc

        print("epoch: %d,  eval loss: %.5f" % (epoch, loss))


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)

    paddle.set_device(args.device)
    set_seed(args)

    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    data_dir = '../data/c3'
    processor = c3Processor(data_dir)

    n_class = 4
    max_num_choices = 4

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(
        args.model_name_or_path, num_choices=max_num_choices)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    train_data = process_train_data(output_dir, processor, tokenizer, n_class,
                                    args.max_seq_length)

    train_data_loader = paddle.io.DataLoader(
        dataset=train_data, batch_size=args.batch_size, num_workers=0)

    dev_data = process_validation_data(output_dir, processor, tokenizer,
                                       n_class, args.max_seq_length)

    dev_data_loader = paddle.io.DataLoader(
        dataset=dev_data, batch_size=args.batch_size, num_workers=0)

    num_training_steps = len(train_data_loader) * args.num_train_epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         0)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    grad_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm)

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=grad_clip)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()
    do_train(args, model, metric, criterion, train_data_loader, dev_data_loader)
