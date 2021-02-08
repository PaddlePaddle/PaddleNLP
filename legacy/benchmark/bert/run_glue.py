# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import logging
import os
import random
import time
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader

from paddlenlp.datasets import GlueQNLI, GlueSST2
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

TASK_CLASSES = {
    "qnli": (GlueQNLI, paddle.metric.Accuracy),  # (dataset, metric)
    "sst-2": (GlueSST2, paddle.metric.Accuracy),
}

MODEL_CLASSES = {"bert": (BertForSequenceClassification, BertTokenizer), }


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(TASK_CLASSES.keys()), )
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
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization")
    args = parser.parse_args()
    return args


def create_data_holder():
    input_ids = paddle.static.data(
        name="input_ids", shape=[-1, -1], dtype="int64")
    segment_ids = paddle.static.data(
        name="segment_ids", shape=[-1, -1], dtype="int64")
    label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")

    return [input_ids, segment_ids, label]


def reset_program_state_dict(model, state_dict, pretrained_state_dict):
    reset_state_dict = {}
    scale = model.initializer_range if hasattr(model, "initializer_range")\
        else model.bert.config["initializer_range"]
    for n, p in state_dict.items():
        if n not in pretrained_state_dict:
            dtype_str = "float32"
            if str(p.dtype) == "VarType.FP64":
                dtype_str = "float64"
            reset_state_dict[p.name] = np.random.normal(
                loc=0.0, scale=scale, size=p.shape).astype(dtype_str)
        else:
            reset_state_dict[p.name] = pretrained_state_dict[n]
    return reset_state_dict


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


def evaluate(exe, metric, loss, correct, dev_program, data_loader):
    metric.reset()
    for batch in data_loader:
        loss_return, correct_return = exe.run(dev_program, feed=batch, \
           fetch_list=[loss, correct])
        metric.update(correct_return)
        accuracy = metric.accumulate()
    print("eval loss: %f, accuracy: %f" % (loss_return, accuracy))


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """convert a glue example into necessary features"""

    def _truncate_seqs(seqs, max_seq_length):
        if len(seqs) == 1:  # single sentence
            # Account for [CLS] and [SEP] with "- 2"
            seqs[0] = seqs[0][0:(max_seq_length - 2)]
        else:  # sentence pair
            # Account for [CLS], [SEP], [SEP] with "- 3"
            tokens_a, tokens_b = seqs
            max_seq_length -= 3
            while True:  # truncate with longest_first strategy
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_seq_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
        return seqs

    def _concat_seqs(seqs, separators, seq_mask=0, separator_mask=1):
        concat = sum((seq + sep for sep, seq in zip(separators, seqs)), [])
        segment_ids = sum(
            ([i] * (len(seq) + len(sep))
             for i, (sep, seq) in enumerate(zip(separators, seqs))), [])
        if isinstance(seq_mask, int):
            seq_mask = [[seq_mask] * len(seq) for seq in seqs]
        if isinstance(separator_mask, int):
            separator_mask = [[separator_mask] * len(sep) for sep in separators]
        p_mask = sum((s_mask + mask
                      for sep, seq, s_mask, mask in zip(
                          separators, seqs, seq_mask, separator_mask)), [])
        return concat, segment_ids, p_mask

    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # get the label
        label = example[-1]
        example = example[:-1]
        #create label maps if classification task
        if label_list:
            label_map = {}
            for (i, l) in enumerate(label_list):
                label_map[l] = i
            label = label_map[label]
        label = [label]
        #label = np.array([label], dtype=label_dtype)
    # tokenize raw text
    tokens_raw = [tokenizer(l) for l in example]
    # truncate to the truncate_length,
    tokens_trun = _truncate_seqs(tokens_raw, max_seq_length)
    # concate the sequences with special tokens
    tokens_trun[0] = [tokenizer.cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = _concat_seqs(tokens_trun, [[tokenizer.sep_token]] *
                                          len(tokens_trun))
    # convert the token to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    # input_mask = [1] * len(input_ids)
    if not is_test:
        return input_ids, segment_ids, label
    else:
        return input_ids, segment_ids


def do_train(args):
    # Set the paddle execute enviroment
    paddle.enable_static()
    place = paddle.CUDAPlace(0)
    set_seed(args)

    # Create the main_program for the training and dev_program for the validation
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()
    dev_program = paddle.static.Program()

    # Get the configuration of tokenizer and model
    args.task_name = args.task_name.lower()
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    dataset_class, metric_class = TASK_CLASSES[args.task_name]

    # Create the tokenizer and dataset
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    train_dataset, dev_dataset = dataset_class.get_datasets(["train", "dev"])

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_dataset.get_labels(),
        max_seq_length=args.max_seq_length)

    train_dataset = train_dataset.apply(trans_func, lazy=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(dtype="int64" if train_dataset.get_labels() else "float32")  # label
    ): [data for i, data in enumerate(fn(samples))]

    train_batch_sampler = paddle.io.BatchSampler(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    dev_dataset = dev_dataset.apply(trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_dataset, batch_size=args.batch_size, shuffle=False)

    feed_list_name = []

    # Define the input data and create the train/dev data_loader
    with paddle.static.program_guard(main_program, startup_program):
        [input_ids, segment_ids, labels] = create_data_holder()

    train_data_loader = DataLoader(
        dataset=train_dataset,
        feed_list=[input_ids, segment_ids, labels],
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=False)

    dev_data_loader = DataLoader(
        dataset=dev_dataset,
        feed_list=[input_ids, segment_ids, labels],
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=False)

    # Create the training-forward program, and clone it for the validation
    with paddle.static.program_guard(main_program, startup_program):
        model, pretrained_state_dict = model_class.from_pretrained(
            args.model_name_or_path,
            num_classes=len(train_dataset.get_labels()))
        loss_fct = paddle.nn.loss.CrossEntropyLoss(
        ) if train_dataset.get_labels() else paddle.nn.loss.MSELoss()
        logits = model(input_ids, segment_ids)
        loss = loss_fct(logits, labels)
        dev_program = main_program.clone(for_test=True)

    # Create the training-backward program, this pass will not be
    # executed in the validation
    with paddle.static.program_guard(main_program, startup_program):
        num_training_steps = args.max_steps if args.max_steps > 0 else len(
            train_data_loader) * args.num_train_epochs

        lr_scheduler = LinearDecayWithWarmup(
            args.learning_rate, num_training_steps, args.warmup_steps)

        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            epsilon=args.adam_epsilon,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in [
                p.name for n, p in model.named_parameters()
               if not any(nd in n for nd in ["bias", "norm"])
        ])
        optimizer.minimize(loss)

    # Create the metric pass for the validation
    with paddle.static.program_guard(dev_program, startup_program):
        metric = metric_class()
        correct = metric.compute(logits, labels)

    # Initialize the fine-tuning parameter, we will load the parameters in
    # pre-training model. And initialize the parameter which not in pre-training model
    # by the normal distribution.
    exe = paddle.static.Executor(place)
    exe.run(startup_program)
    state_dict = model.state_dict()
    reset_state_dict = reset_program_state_dict(model, state_dict,
                                                pretrained_state_dict)
    paddle.static.set_program_state(main_program, reset_state_dict)

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            loss_return = exe.run(main_program, feed=batch, fetch_list=[loss])
            if global_step % args.logging_steps == 0:
                logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss_return[0],
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            lr_scheduler.step()
            if global_step % args.save_steps == 0:
                # Validation pass, record the loss and metric
                evaluate(exe, metric, loss, correct, dev_program,
                         dev_data_loader)
                output_dir = os.path.join(args.output_dir,
                                          "model_%d" % global_step)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                paddle.fluid.io.save_params(exe, output_dir)
                tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
