#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import random
import os
import logging

import numpy as np

import paddle
from paddlenlp.transformers import (CosineDecayWithWarmup,
                                    LinearDecayWithWarmup, PolyDecayWithWarmup)

logger = logging.getLogger("__main__")


def get_writer(args):
    if args.writer_type == "visualdl":
        from visualdl import LogWriter

        writer = LogWriter(logdir=args.logging_dir)
    elif args.writer_type == "tensorboard":
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir=args.logging_dir)
    else:
        raise ValueError("writer_type must be in ['visualdl', 'tensorboard']")
    return writer


scheduler_type2cls = {
    "linear": LinearDecayWithWarmup,
    "cosine": CosineDecayWithWarmup,
    "poly": PolyDecayWithWarmup,
}


def get_scheduler(
        learning_rate,
        scheduler_type,
        num_warmup_steps=None,
        num_training_steps=None,
        **scheduler_kwargs, ):

    if scheduler_type not in scheduler_type2cls.keys():
        data = " ".join(scheduler_type2cls.keys())
        raise ValueError(f"scheduler_type must be choson from {data}")

    if num_warmup_steps is None:
        raise ValueError(
            f"requires `num_warmup_steps`, please provide that argument.")

    if num_training_steps is None:
        raise ValueError(
            f"requires `num_training_steps`, please provide that argument.")

    return scheduler_type2cls[scheduler_type](learning_rate=learning_rate,
                                              total_steps=num_training_steps,
                                              warmup=num_warmup_steps,
                                              **scheduler_kwargs)


def set_seed(args):
    random.seed(args.seed + paddle.distributed.get_rank())
    np.random.seed(args.seed + paddle.distributed.get_rank())
    paddle.seed(args.seed + paddle.distributed.get_rank())


def save_checkpoint(tokenizer, model, output_dir):
    logger.info(f"saving checkpoint to {output_dir}")
    if isinstance(model, paddle.DataParallel):
        model = model._layers
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def set_logger(args):
    logger.setLevel(logging.DEBUG if 'DEBUG' in os.environ else logging.INFO)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                filename=f"{args.output_dir}.log",
                mode="w",
                encoding="utf-8", )
        ], )
    # create console handler and set level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)
    # add formatter to console_handler
    console_handler.setFormatter(fmt=logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # add console_handler to logger
    logger.addHandler(console_handler)


def read_json_file(file_name):
    return [json.loads(line) for line in open(file_name)]
