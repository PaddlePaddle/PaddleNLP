# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import argparse

import paddle
from paddle.io import DataLoader, DistributedBatchSampler
from tqdm import tqdm
from trainer import Trainer
import numpy as np
from paddlenlp.transformers import RemBertForSequenceClassification
from data_processor import MrpcProcessor, tokenization, XNLIProcessor, DataGenerator
import paddle.distributed as dist
import random
from paddle.metric import Accuracy

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="RemBert For Sequence Classification")
parser.add_argument("--data_dir", type=str, default=None, help="Data path.")
parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to train the model.")
parser.add_argument("--do_eval",
                    action='store_true',
                    help="Whether to predict.")
parser.add_argument("--num_train_epochs",
                    type=int,
                    default=3,
                    help="Total number of training epochs to perform.")
parser.add_argument("--seed",
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument("--train_batch_size",
                    type=int,
                    default=16,
                    help="per gpu batch size during thr training.")
parser.add_argument("--eval_batch_size",
                    type=int,
                    default=16,
                    help="per gpu batch size during thr evaluating.")
parser.add_argument(
    "--output_dir",
    default='outputs',
    type=str,
    help=
    "The output directory where the model predictions and checkpoints will be written. "
    "Default as `outputs`")
parser.add_argument(
    "--max_seq_length",
    default=512,
    type=int,
    help=
    "The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.")
parser.add_argument('--device',
                    choices=['cpu', 'gpu'],
                    default="gpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=2,
    help=
    "Proportion of training steps to perform linear learning rate warmup for.")
parser.add_argument(
    "--warmup_proportion",
    type=float,
    default=0.02,
    help=
    "Proportion of training steps to perform linear learning rate warmup for.")
parser.add_argument("--learning_rate",
                    type=float,
                    default=8e-6,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.01,
                    help="Weight decay if we apply some.")
parser.add_argument("--task", type=str, required=True, help="Training task")
parser.add_argument("--model_type",
                    default="rembert",
                    type=str,
                    help="Type of pre-trained model.")
parser.add_argument("--eval_step",
                    type=int,
                    default=2000,
                    help="Eavlate the model once after training step X.")
args = parser.parse_args()

nranks = paddle.distributed.ParallelEnv().nranks


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_example(args, mode='train'):
    """Load data to DataLoader"""
    if args.task == 'paws':
        processor = MrpcProcessor()
    if args.task == 'xnli':
        processor = XNLIProcessor()
    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    datagenerator = DataGenerator(examples)

    def collate_fn(batch):

        def create_padded_sequence(key, padding_value):
            """Pad sequence to max length"""
            pad_sequence = []
            max_len = 0
            for example in batch:
                if len(example[key]) > max_len:
                    max_len = len(example[key])
            for example in batch:
                pad_sequence.append(example[key] + [padding_value] *
                                    (max_len - len(example[key])))
            return np.array(pad_sequence, dtype='int64')

        text_a = create_padded_sequence('text_a', 0)  # pad text_a input_ids
        text_b = create_padded_sequence('text_b', 0)  # pad text_b input_ids
        text_a_token_type_ids = create_padded_sequence(
            'text_a_token_type_ids', 0)  # pad text_a_token_type_ids
        text_b_token_type_ids = create_padded_sequence(
            'text_b_token_type_ids', 1)  # pad text_b_token_type_ids
        label = create_padded_sequence(
            'label', 0)  # label will not pad, just convert to numpy array

        input_ids = np.concatenate([text_a, text_b],
                                   axis=-1)[:, :args.max_seq_length]
        token_type_ids = np.concatenate(
            [text_a_token_type_ids, text_b_token_type_ids],
            axis=-1)[:, :args.max_seq_length]

        return input_ids, token_type_ids, label

    if mode in ("dev", "test"):
        dataloader = DataLoader(datagenerator,
                                batch_size=args.eval_batch_size,
                                shuffle=False,
                                collate_fn=collate_fn)
    else:
        sampler = DistributedBatchSampler(datagenerator,
                                          batch_size=args.train_batch_size,
                                          shuffle=True,
                                          drop_last=False)
        dataloader = DataLoader(datagenerator,
                                batch_sampler=sampler,
                                collate_fn=collate_fn)

    return dataloader, processor


def run(args):
    if args.do_train:
        train_dataloader, processor = load_example(args, 'train')
        num_label = len(processor.get_labels())
        model = RemBertForSequenceClassification.from_pretrained(
            args.model_type, num_classes=num_label)
        if nranks > 1:
            dist.init_parallel_env()
            model = paddle.DataParallel(model)

        num_train_steps_per_epoch = len(
            train_dataloader) // args.gradient_accumulation_steps
        num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)
        trainer = Trainer(args,
                          model=model,
                          dataloader=train_dataloader,
                          num_train_steps=num_train_steps,
                          step_callback=evaluate)
        trainer.train()

    if args.do_eval:
        model = RemBertForSequenceClassification.from_pretrained(
            args.output_dir)
        evaluate(model, args)


def evaluate(model, args, mode='test'):
    """evaluate the model"""
    model.eval()
    metric = Accuracy()
    eval_dataloader, processor = load_example(args, mode)
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        logits = model(input_ids=batch[0], token_type_ids=batch[1])
        labels = batch[2].reshape((
            -1,
            1,
        ))
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    print('Accuracy:', res)
    model.train()
    return res


if __name__ == '__main__':
    set_seed(args.seed)
    paddle.set_device(args.device)
    run(args)
