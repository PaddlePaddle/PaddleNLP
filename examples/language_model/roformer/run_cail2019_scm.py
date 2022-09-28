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

import jieba
import argparse
import logging
import os
import random
import time
import distutils.util
from functools import partial

import numpy as np
import paddle

from paddle.io import DataLoader
from paddle.metric import Accuracy
import paddle.nn.functional as F
import paddle.nn as nn
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers.roformer.modeling import RoFormerForSequenceClassification, RoFormerPretrainedModel
from paddlenlp.transformers import RoFormerTokenizer

FORMAT = "%(asctime)s-%(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
jieba.setLogLevel(logging.INFO)


class RoFormerMeanPoolingForSequenceClassification(RoFormerPretrainedModel):

    def __init__(self, roformer, num_classes):
        super(RoFormerMeanPoolingForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.roformer = roformer
        self.classifier = nn.Linear(self.roformer.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        last_hidden_state = self.roformer(input_ids,
                                          token_type_ids=token_type_ids,
                                          attention_mask=attention_mask)[0]

        mask = (input_ids != self.roformer.pad_token_id).astype(
            self.classifier.weight.dtype).unsqueeze(-1)
        mean_pooling = paddle.sum(last_hidden_state * mask,
                                  axis=1) / paddle.sum(mask, axis=1)
        logits = self.classifier(mean_pooling)
        return logits


MODEL_CLASSES = {
    "roformer_cls_pooling":
    (RoFormerForSequenceClassification, RoFormerTokenizer),
    "roformer_mean_pooling":
    (RoFormerMeanPoolingForSequenceClassification, RoFormerTokenizer),
}


class Cail2019_SCM_Accuracy(Accuracy):

    def compute(self, pred, label, *args):
        pred = paddle.cast(pred[::2] > pred[1::2], dtype="int64")
        correct = (pred == 1).unsqueeze(-1)
        return paddle.cast(correct, dtype='float32')


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum(
                [
                    list(classes[-1].pretrained_init_configuration.keys())
                    for classes in MODEL_CLASSES.values()
                ],
                [],
            )),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--logging_steps",
                        type=int,
                        default=100,
                        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-6,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.",
    )
    parser.add_argument(
        "--use_amp",
        type=distutils.util.strtobool,
        default=False,
        help="Enable mixed precision training.",
    )
    parser.add_argument(
        "--scale_loss",
        type=float,
        default=2**15,
        help="The value of scale_loss for fp16.",
    )
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


@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids).squeeze(-1)
        loss = loss_fct(logits, labels)
        correct = metric.compute(F.sigmoid(logits), labels)
        metric.update(correct)
    res = metric.accumulate()
    print("eval loss: %f, acc: %s" % (loss.numpy(), res))


def convert_example(example, tokenizer, max_seq_length=512):
    if example['label'] == 0:
        text1 = example["text_a"]
        text2 = example["text_b"]
        text3 = example["text_c"]
    else:
        text1 = example["text_a"]
        text2 = example["text_c"]
        text3 = example["text_b"]

    data1 = tokenizer(text1, text_pair=text2, max_length=max_seq_length)
    data2 = tokenizer(text1, text_pair=text3, max_length=max_seq_length)

    return [data1["input_ids"], data1["token_type_ids"],
            1], [data2["input_ids"], data2["token_type_ids"], 0]


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)
    args.model_type = args.model_type.lower()
    args.batch_size = args.batch_size // 2
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    train_ds = load_dataset("cail2019_scm", splits="train")
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)

    def batchify_fn(
        samples,
        fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
            Stack(dtype="float32"))):  # label
        new_samples = []
        for sample in samples:
            new_samples.extend(sample)
        return fn(new_samples)

    train_data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True,
    )

    dev_ds = load_dataset("cail2019_scm", splits="dev")
    dev_ds = dev_ds.map(trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds,
                                               batch_size=args.batch_size * 4,
                                               shuffle=False)
    dev_data_loader = DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True,
    )

    test_ds = load_dataset("cail2019_scm", splits="test")
    test_ds = test_ds.map(trans_func, lazy=True)
    test_batch_sampler = paddle.io.BatchSampler(test_ds,
                                                batch_size=args.batch_size * 4,
                                                shuffle=False)
    test_data_loader = DataLoader(
        dataset=test_ds,
        batch_sampler=test_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True,
    )

    model = model_class.from_pretrained(args.model_name_or_path, num_classes=1)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = (args.max_steps if args.max_steps > 0 else
                          (len(train_data_loader) * args.num_train_epochs))

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    loss_fct = paddle.nn.loss.BCEWithLogitsLoss()

    metric = Cail2019_SCM_Accuracy()
    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            model.train()
            global_step += 1

            input_ids, segment_ids, labels = batch

            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"]):
                logits = model(input_ids, segment_ids).squeeze(-1)
                loss = loss_fct(logits, labels)
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.minimize(optimizer, loss)
            else:
                loss.backward()
                optimizer.step()

            optimizer.clear_grad()
            if global_step % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (
                        global_step,
                        num_training_steps,
                        epoch,
                        step,
                        paddle.distributed.get_rank(),
                        loss,
                        optimizer.get_lr(),
                        args.logging_steps / (time.time() - tic_train),
                    ))
                tic_train = time.time()
            if global_step % args.save_steps == 0 or global_step == num_training_steps:
                tic_eval = time.time()
                print("============Dev Dataset============")
                evaluate(model, loss_fct, metric, dev_data_loader)
                print("============Test Dataset============")
                evaluate(model, loss_fct, metric, test_data_loader)
                print("eval done total : %s s" % (time.time() - tic_eval))
                if paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(
                        args.output_dir, "ft_model_%d.pdparams" % (global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Need better way to get inner model of DataParallel
                    model_to_save = (model._layers if isinstance(
                        model, paddle.DataParallel) else model)
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
            if global_step >= num_training_steps:
                return


def print_arguments(args):
    """print arguments"""
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    do_train(args)
