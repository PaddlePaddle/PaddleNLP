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
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader
from paddlenlp.datasets import load_dataset

from paddle.metric import Accuracy
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import Mcc, PearsonAndSpearman
from paddlenlp.utils.log import logger

METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "sts-b": PearsonAndSpearman,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
}

MODEL_CLASSES = {
    "bert": (BertForSequenceClassification, BertTokenizer),
    "ernie": (ErnieForSequenceClassification, ErnieTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()),
    )
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
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])),
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
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps",
                        type=int,
                        default=500,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for initialization")
    parser.add_argument("--device",
                        type=str,
                        default="gpu",
                        help="Device for selecting for the training.")
    args = parser.parse_args()
    return args


def create_data_holder(task_name):
    """
    Define the input data holder for the glue task.
    """
    input_ids = paddle.static.data(name="input_ids",
                                   shape=[-1, -1],
                                   dtype="int64")
    token_type_ids = paddle.static.data(name="token_type_ids",
                                        shape=[-1, -1],
                                        dtype="int64")
    if task_name == "sts-b":
        label = paddle.static.data(name="label", shape=[-1, 1], dtype="float32")
    else:
        label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")

    return [input_ids, token_type_ids, label]


def reset_program_state_dict(args, model, state_dict, pretrained_state_dict):
    """
    Initialize the parameter from the bert config, and set the parameter by 
    reseting the state dict."
    """
    reset_state_dict = {}
    scale = model.initializer_range if hasattr(model, "initializer_range")\
        else getattr(model, args.model_type).config["initializer_range"]
    reset_parameter_names = []
    for n, p in state_dict.items():
        if n in pretrained_state_dict:
            reset_state_dict[p.name] = np.array(pretrained_state_dict[n])
            reset_parameter_names.append(n)
        elif p.name in pretrained_state_dict and "bert" in n:
            reset_state_dict[p.name] = np.array(pretrained_state_dict[p.name])
            reset_parameter_names.append(n)
        else:
            dtype_str = "float32"
            if str(p.dtype) == "VarType.FP64":
                dtype_str = "float64"
            reset_state_dict[p.name] = np.random.normal(
                loc=0.0, scale=scale, size=p.shape).astype(dtype_str)
    logger.info("the following parameter had reset, please check. {}".format(
        reset_parameter_names))
    return reset_state_dict


def set_seed(args):
    """
    Use the same data seed(for data shuffle) for all procs to guarantee data
    consistency after sharding.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


def evaluate(exe,
             metric,
             loss,
             correct,
             dev_program,
             data_loader,
             phase="eval"):
    """
    The evaluate process, calcluate the eval loss and metric. 
    """
    metric.reset()
    returns = [loss]
    if isinstance(correct, list) or isinstance(correct, tuple):
        returns.extend(list(correct))
    else:
        returns.append(correct)
    for batch in data_loader:
        exe.run(dev_program, feed=batch, \
           fetch_list=returns)
        return_numpys = exe.run(dev_program, feed=batch, \
           fetch_list=returns)
        metric_numpy = return_numpys[1] if len(
            return_numpys[1:]) == 1 else return_numpys[1:]
        metric.update(metric_numpy)
    res = metric.accumulate()
    if isinstance(metric, Mcc):
        print("%s loss: %f, mcc: %s" % (phase, return_numpys[0], res[0]))
    elif isinstance(metric, PearsonAndSpearman):
        print(
            "%s loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s" %
            (phase, return_numpys[0], res[0], res[1], res[2]))
    else:
        print("%s loss: %f, acc: %s, " % (phase, return_numpys[0], res))


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """
    Convert a glue example into necessary features.
    """
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example['labels']
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    if (int(is_test) + len(example)) == 2:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    else:
        example = tokenizer(example['sentence1'],
                            text_pair=example['sentence2'],
                            max_seq_len=max_seq_length)

    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']


def do_train(args):
    # Set the paddle execute enviroment
    paddle.enable_static()
    place = paddle.set_device(args.device)
    fleet.init(is_collective=True)
    set_seed(args)

    # Create the main_program for the training and dev_program for the validation
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()
    dev_program = paddle.static.Program()

    # Get the configuration of tokenizer and model
    args.task_name = args.task_name.lower()
    args.model_type = args.model_type.lower()
    metric_class = METRIC_CLASSES[args.task_name]
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Create the tokenizer and dataset
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    train_ds = load_dataset('glue', args.task_name, splits="train")

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         label_list=train_ds.label_list,
                         max_seq_length=args.max_seq_length)

    train_ds = train_ds.map(trans_func, lazy=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type 
        Stack(dtype="int64" if train_ds.label_list else "float32")  # label
    ): fn(samples)

    train_batch_sampler = paddle.io.BatchSampler(train_ds,
                                                 batch_size=args.batch_size,
                                                 shuffle=True)

    feed_list_name = []

    # Define the input data and create the train/dev data_loader
    with paddle.static.program_guard(main_program, startup_program):
        [input_ids, token_type_ids, labels] = create_data_holder(args.task_name)

    train_data_loader = DataLoader(
        dataset=train_ds,
        feed_list=[input_ids, token_type_ids, labels],
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=False)

    if args.task_name == "mnli":
        dev_ds_matched, dev_ds_mismatched = load_dataset(
            'glue', args.task_name, splits=["dev_matched", "dev_mismatched"])

        dev_ds_matched = dev_ds_matched.map(trans_func, lazy=True)
        dev_ds_mismatched = dev_ds_mismatched.map(trans_func, lazy=True)
        dev_batch_sampler_matched = paddle.io.BatchSampler(
            dev_ds_matched, batch_size=args.batch_size, shuffle=False)
        dev_data_loader_matched = DataLoader(
            dataset=dev_ds_matched,
            batch_sampler=dev_batch_sampler_matched,
            collate_fn=batchify_fn,
            feed_list=[input_ids, token_type_ids, labels],
            num_workers=0,
            return_list=False)
        dev_batch_sampler_mismatched = paddle.io.BatchSampler(
            dev_ds_mismatched, batch_size=args.batch_size, shuffle=False)
        dev_data_loader_mismatched = DataLoader(
            dataset=dev_ds_mismatched,
            batch_sampler=dev_batch_sampler_mismatched,
            collate_fn=batchify_fn,
            num_workers=0,
            feed_list=[input_ids, token_type_ids, labels],
            return_list=False)
    else:
        dev_ds = load_dataset('glue', args.task_name, splits='dev')
        dev_ds = dev_ds.map(trans_func, lazy=True)
        dev_batch_sampler = paddle.io.BatchSampler(dev_ds,
                                                   batch_size=args.batch_size,
                                                   shuffle=False)
        dev_data_loader = DataLoader(
            dataset=dev_ds,
            batch_sampler=dev_batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            feed_list=[input_ids, token_type_ids, labels],
            return_list=False)

    # Create the training-forward program, and clone it for the validation
    with paddle.static.program_guard(main_program, startup_program):
        num_class = 1 if train_ds.label_list is None else len(
            train_ds.label_list)
        model, pretrained_state_dict = model_class.from_pretrained(
            args.model_name_or_path, num_classes=num_class)
        loss_fct = paddle.nn.loss.CrossEntropyLoss(
        ) if train_ds.label_list else paddle.nn.loss.MSELoss()
        logits = model(input_ids, token_type_ids)
        loss = loss_fct(logits, labels)
        dev_program = main_program.clone(for_test=True)

    # Create the training-backward program, this pass will not be
    # executed in the validation
    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.num_train_epochs
    with paddle.static.program_guard(main_program, startup_program):
        lr_scheduler = LinearDecayWithWarmup(args.learning_rate,
                                             num_training_steps,
                                             args.warmup_steps)
        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            epsilon=args.adam_epsilon,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params)
        optimizer = fleet.distributed_optimizer(optimizer)
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
    reset_state_dict = reset_program_state_dict(args, model, state_dict,
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
                if args.task_name == "mnli":
                    evaluate(exe, metric, loss, correct, dev_program,
                             dev_data_loader_matched, "matched eval")
                    evaluate(exe, metric, loss, correct, dev_program,
                             dev_data_loader_mismatched, "mismatched eval")
                else:
                    evaluate(exe, metric, loss, correct, dev_program,
                             dev_data_loader)
                output_dir = os.path.join(args.output_dir,
                                          "model_%d" % global_step)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                paddle.static.save_inference_model(
                    os.path.join(output_dir, "model"),
                    [input_ids, token_type_ids], [logits], exe)
                tokenizer.save_pretrained(output_dir)
            if global_step >= num_training_steps:
                return


if __name__ == "__main__":
    args = parse_args()
    assert args.device in [
        "cpu", "gpu", "xpu"
    ], "Invalid device! Available device should be cpu, gpu, or xpu."

    do_train(args)
