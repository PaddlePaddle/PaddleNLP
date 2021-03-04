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

import argparse
import math
import os
import random
import time

import numpy as np
import paddle
from paddle.io import DataLoader, Dataset

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import GPT2Model, GPT2ForPretraining, GPT2PretrainingCriterion
from paddlenlp.transformers import GPT2Tokenizer, GPT2ChineseTokenizer
from paddlenlp.utils.log import logger
from data import GPT2Dataset
from args import parse_args
import lr

MODEL_CLASSES = {
    "gpt2": (GPT2ForPretraining, GPT2Tokenizer),
    "gpt2-cn": (GPT2ForPretraining, GPT2ChineseTokenizer),
}


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def create_pretrained_dataset(args, input_path, worker_init, worker_index,
                              worker_num, eod_id):
    train_data = GPT2Dataset(
        file_path=input_path,
        worker_index=worker_index,
        num_samples=args.batch_size * args.max_steps * worker_num,
        eod_id=eod_id,
        seed=args.seed + worker_index)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        worker_init_fn=worker_init,
        collate_fn=Tuple(Stack(), Stack(), Stack(), Stack(), Stack()))
    return train_data_loader


def set_seed(args):
    if args.device == "cpu":
        idx = 0
    else:
        idx = paddle.distributed.get_rank()
    random.seed(args.seed + idx)
    np.random.seed(args.seed + idx)
    paddle.seed(args.seed + idx)


def do_train(args):
    assert args.device in [
        "cpu", "gpu", "xpu"
    ], "Invalid device! Available device should be cpu, gpu, or xpu."
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    worker_index = paddle.distributed.get_rank()
    worker_num = paddle.distributed.get_world_size()
    set_seed(args)
    worker_init = WorkerInitObj(args.seed + paddle.distributed.get_rank())
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    eod_id = tokenizer.command_name_map["eod"].Id

    pretrained_models_list = list(
        model_class.pretrained_init_configuration.keys())
    if args.model_name_or_path in pretrained_models_list:
        model = GPT2ForPretraining(
            GPT2Model(**model_class.pretrained_init_configuration[
                args.model_name_or_path]))
    else:
        model = GPT2ForPretraining.from_pretrained(args.model_name_or_path)

    # creat the critrion for the gpt model
    criterion = GPT2PretrainingCriterion()

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if args.decay_steps is None:
        args.decay_steps = args.max_steps
    warmup_step = args.warmup_rate * args.decay_steps
    lr_scheduler = lr.CosineAnnealingWithWarmupDecay(
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_step=warmup_step,
        decay_step=args.decay_steps)

    clip = None
    if args.grad_clip > 0:
        clip = paddle.nn.ClipGradByNorm(clip_norm=args.grad_clip)

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        grad_clip=clip,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])
    if args.model_name_or_path not in pretrained_models_list:
        opt_dict = paddle.load(
            os.path.join(args.model_name_or_path, "model_state.pdopt"))
        optimizer.set_state_dict(opt_dict)

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        files = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if (os.path.isfile(os.path.join(args.input_dir, f)) and "npz_"
                not in str(f))
        ]
        files.sort()
        num_files = len(files)
        for f_id in range(num_files):
            data_file = files[f_id]
            train_data_loader = create_pretrained_dataset(
                args,
                data_file,
                worker_init,
                worker_index,
                worker_num,
                eod_id=eod_id)
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                tokens, loss_mask, attention_mask, position_ids, labels = batch

                loss_mask.stop_gradient = True
                attention_mask.stop_gradient = True

                preds = model(tokens, position_ids, attention_mask)
                loss = criterion(preds, labels, loss_mask)

                if global_step % args.logging_steps == 0:
                    if worker_index == 0:
                        logger.info(
                            "global step %d, epoch: %d, lr: %.10f, batch: %d, loss: %f, speed: %.2f step/s"
                            % (global_step, epoch, optimizer.get_lr(), step,
                               loss,
                               args.logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                if global_step % args.save_steps == 0 or global_step >= args.max_steps:
                    if worker_index == 0:
                        output_dir = os.path.join(args.output_dir,
                                                  "model_%d" % global_step)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # need better way to get inner model of DataParallel
                        model_to_save = model._layers if isinstance(
                            model, paddle.DataParallel) else model
                        logger.info("Save model to %s" % output_dir)
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        paddle.save(
                            optimizer.state_dict(),
                            os.path.join(output_dir, "model_state.pdopt"))
                if global_step >= args.max_steps:
                    logger.info("The training process is complete.")
                    del train_data_loader
                    return

            del train_data_loader


if __name__ == "__main__":
    args = parse_args(MODEL_CLASSES)
    do_train(args)
