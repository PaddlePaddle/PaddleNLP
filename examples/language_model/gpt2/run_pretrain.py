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
from paddlenlp.transformers import GPT2Model, GPT2ForPretraining, GPT2PretrainingCriterion
from paddlenlp.transformers import GPT2Tokenizer, GPT2ChineseTokenizer
from paddlenlp.utils.log import logger
from tensorboardX import SummaryWriter

from data import create_pretrained_dataset
from args import parse_args
import lr

MODEL_CLASSES = {
    "gpt2": (GPT2ForPretraining, GPT2Tokenizer),
    "gpt2-cn": (GPT2ForPretraining, GPT2ChineseTokenizer),
}


class WorkerInitObj:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def set_seed(args):
    if args.device == "cpu":
        idx = 0
    else:
        idx = paddle.distributed.get_rank()
    random.seed(args.seed + idx)
    np.random.seed(args.seed + idx)
    paddle.seed(args.seed + idx)


@paddle.no_grad()
def run_evaluate(data_loader,
                 model,
                 criterion,
                 iter_steps,
                 log_writer,
                 global_step,
                 epoch,
                 task_name="valid"):
    all_loss = []
    local_time = time.time()
    for eval_step, batch in enumerate(data_loader):
        tokens, loss_mask, attention_mask, position_ids, labels = batch
        preds = model(tokens, position_ids, attention_mask)
        loss = criterion(preds, labels, loss_mask)
        all_loss.append(float(loss))
        if eval_step >= iter_steps - 1:
            break

    average_loss = sum(all_loss) / len(all_loss)
    logger.info("%s step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                % (task_name, global_step, epoch, eval_step, average_loss,
                   iter_steps / (time.time() - local_time)))
    log_writer.add_scalar(task_name + "_loss", average_loss, global_step)


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

    # define log writer
    log_writer_path = os.path.join(
        args.output_dir, "gpt2_bs={}_amp={}_recompute={}_card={}".format(
            args.batch_size, False, False, worker_num))
    if os.path.exists(log_writer_path):
        import shutil
        shutil.rmtree(log_writer_path)
    log_writer = SummaryWriter(log_writer_path)

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

    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    opt_param = []
    for pa in model.parameters():
        if "batch_norm" not in pa.name:
            opt_param.append(pa)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=opt_param,
        weight_decay=args.weight_decay,
        grad_clip=clip,
        apply_decay_param_fun=lambda x: x in decay_params)

    # Load checkpoint for path
    if args.model_name_or_path not in pretrained_models_list:
        logger.info("Try to load checkpoint from ", args.model_name_or_path)
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
            train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
                args,
                data_file,
                worker_init,
                worker_index,
                worker_num,
                eod_id=eod_id)
            # bug fix, if not call valid_data_loader, the enumerate will call valid_data_loader
            # many times. and start a new random dataloader.
            valid_data_loader = valid_data_loader()
            test_data_loader = test_data_loader()

            for step, batch in enumerate(train_data_loader()):
                global_step += 1
                tokens, loss_mask, attention_mask, position_ids, labels = batch
                loss_mask.stop_gradient = True
                attention_mask.stop_gradient = True

                preds = model(tokens, position_ids, attention_mask)
                loss = criterion(preds, labels, loss_mask)

                if global_step % args.logging_steps == 0:
                    if worker_index == 0:
                        logger.info(
                            "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s, learning rate: %.9f"
                            % (
                                global_step,
                                epoch,
                                step,
                                loss,
                                args.logging_steps / (time.time() - tic_train),
                                optimizer.get_lr(), ))
                        log_writer.add_scalar("loss", float(loss), global_step)
                        log_writer.add_scalar("learning_rate",
                                              optimizer.get_lr(), global_step)

                    tic_train = time.time()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                if global_step % args.eval_steps == 0:
                    if worker_index == 0:
                        run_evaluate(valid_data_loader, model, criterion,
                                     args.eval_iters, log_writer, global_step,
                                     epoch, "valid")

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
                    run_evaluate(test_data_loader, model, criterion,
                                 args.test_iters, log_writer, global_step,
                                 epoch, "test")
                    logger.info("The training process is complete.")
                    del train_data_loader
                    return

            del train_data_loader


if __name__ == "__main__":
    args = parse_args(MODEL_CLASSES)
    do_train(args)
