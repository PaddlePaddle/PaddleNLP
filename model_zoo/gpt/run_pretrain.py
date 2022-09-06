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
from visualdl import LogWriter
from paddlenlp.transformers import GPTModel, GPTForPretraining, GPTPretrainingCriterion
from paddlenlp.transformers import GPTTokenizer, GPTChineseTokenizer
from paddlenlp.utils.log import logger
from paddlenlp.utils import profiler
from paddlenlp.ops import Topology

from dataset import create_pretrained_dataset
from args import parse_args
import lr
from paddle.distributed import fleet

MODEL_CLASSES = {
    "gpt": (GPTForPretraining, GPTTokenizer),
    "gpt-cn": (GPTForPretraining, GPTChineseTokenizer),
}


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
    model.eval()
    local_time = time.time()
    for eval_step, batch in enumerate(data_loader):
        tokens, loss_mask, attention_mask, position_ids, labels = batch
        preds = model(tokens, position_ids, attention_mask)
        loss = criterion(preds, labels, loss_mask)
        all_loss.append(float(loss))
        if eval_step >= iter_steps - 1:
            break
    model.train()
    average_loss = sum(all_loss) / len(all_loss)
    logger.info(
        "%s step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s" %
        (task_name, global_step, epoch, eval_step, average_loss, iter_steps /
         (time.time() - local_time)))
    log_writer.add_scalar(task_name + "_loss", average_loss, global_step)


def get_train_data_file(args):
    files = [
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if (os.path.isfile(os.path.join(args.input_dir, f))
            and str(f).endswith("_idx.npz"))
    ]
    files = [x.replace("_idx.npz", "") for x in files]
    if len(files) == 0:
        logger.warning(
            "Not found dataset with name of xxx_ids.npy and xxx_idx.npz! Try to found old compatible xxx_ids.npz file."
        )
    else:
        return files

    files = [
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if (os.path.isfile(os.path.join(args.input_dir, f))
            and str(f).endswith("_ids.npz"))
    ]

    files = [x.replace("_ids.npz", "") for x in files]
    return files


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    worker_index = paddle.distributed.get_rank()
    worker_num = paddle.distributed.get_world_size()
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
    set_seed(args)
    # Now, we only support data parallel in dygraph mode for now.
    topo = Topology(device_rank=worker_index,
                    world_size=worker_num,
                    dp_degree=worker_num)

    default_global_batch_size = topo.data_info.size * args.micro_batch_size
    default_global_tokens_num = default_global_batch_size * args.max_seq_len

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    # Define log writer
    log_writer_path = os.path.join(
        args.output_dir, "train_log",
        "{}_globalbsz_{}_amp_{}_recompute_{}_card_{}".format(
            args.model_name_or_path,
            args.micro_batch_size * topo.data_info.size, False, False,
            worker_index).lower())
    if os.path.exists(log_writer_path):
        import shutil
        shutil.rmtree(log_writer_path)
    log_writer = LogWriter(log_writer_path)

    pretrained_models_list = list(
        model_class.pretrained_init_configuration.keys())
    if args.model_name_or_path in pretrained_models_list:
        model_config = model_class.pretrained_init_configuration[
            args.model_name_or_path]
        model_config["hidden_dropout_prob"] = args.hidden_dropout_prob
        model_config[
            "attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
        model = GPTForPretraining(GPTModel(**model_config))
    else:
        model = GPTForPretraining.from_pretrained(
            args.model_name_or_path,
            hidden_dropout_prob=args.hidden_dropout_prob,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob)

    # Create the critrion for the gpt model
    criterion = GPTPretrainingCriterion()

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if args.decay_steps is None:
        args.decay_steps = args.max_steps
    warmup_step = args.warmup_rate * args.decay_steps

    lr_scheduler = None

    if args.lr_decay_style == "none":
        lr_scheduler = None
    elif args.lr_decay_style == "cosine":
        lr_scheduler = lr.CosineAnnealingWithWarmupDecay(
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_step=warmup_step,
            decay_step=args.decay_steps)

    clip = None
    if args.grad_clip > 0:
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.grad_clip)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler if lr_scheduler is not None else args.max_lr,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        grad_clip=clip,
        apply_decay_param_fun=lambda x: x in decay_params)

    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)

    if args.model_name_or_path not in pretrained_models_list:
        logger.info("Try to load checkpoint from %s " % args.model_name_or_path)
        opt_path = os.path.join(args.model_name_or_path, "model_state.pdopt")
        if os.path.exists(opt_path):
            opt_dict = paddle.load(opt_path)
            optimizer.set_state_dict(opt_dict)
        else:
            logger.warning("No optimizer checkpoint file found in %s." %
                           opt_path)

    global_step = 0
    epoch = 0
    tic_train = time.time()
    while True:
        files = get_train_data_file(args)
        files.sort()
        num_files = len(files)
        for f_id in range(num_files):
            data_file = files[f_id]
            train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
                args, [data_file],
                local_rank=local_rank,
                data_world_size=topo.data_info.size,
                data_world_rank=topo.data_info.rank,
                max_seq_len=args.max_seq_len,
                eos_id=tokenizer.eos_token_id)
            # Bug fix, if not call valid_data_loader, the enumerate will call valid_data_loader
            # many times. and start a new random dataloader.
            valid_data_loader = valid_data_loader()
            test_data_loader = test_data_loader()

            # time count
            train_reader_cost = 0.0
            train_run_cost = 0.0
            reader_start = time.time()
            for step, batch in enumerate(train_data_loader()):
                train_reader_cost += time.time() - reader_start
                train_start = time.time()

                global_step += 1
                tokens, loss_mask, attention_mask, position_ids, labels = batch
                loss_mask.stop_gradient = True
                attention_mask.stop_gradient = True
                with paddle.amp.auto_cast(
                        args.use_amp,
                        custom_white_list=["layer_norm", "softmax", "gelu"],
                        custom_black_list=[
                            "reduce_sum", "c_softmax_with_cross_entropy",
                            "c_embedding"
                        ]):

                    preds = model(tokens, position_ids, attention_mask)
                    loss = criterion(preds, labels, loss_mask)

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.minimize(optimizer, loss)
                else:
                    loss.backward()
                    optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.clear_grad()

                loss_numpy = loss.numpy()
                train_run_cost += time.time() - train_start

                # Profile for model benchmark
                profiler.add_profiler_step(args.profiler_options)

                if global_step % args.logging_freq == 0:
                    speed = args.logging_freq / (train_reader_cost +
                                                 train_run_cost)
                    avg_reader_cost = train_reader_cost / args.logging_freq
                    logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %.9f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, speed: %.2f step/s, ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
                        %
                        (global_step, epoch, step, loss_numpy, avg_reader_cost,
                         1. / speed, speed, speed * default_global_tokens_num,
                         speed * default_global_tokens_num / worker_num,
                         optimizer.get_lr()))
                    log_writer.add_scalar("loss", loss_numpy, global_step)
                    log_writer.add_scalar("learning_rate", optimizer.get_lr(),
                                          global_step)

                    tic_train = time.time()
                    train_reader_cost = 0.0
                    train_run_cost = 0.0

                if args.check_accuracy:
                    if global_step >= args.max_steps:
                        return
                    else:
                        continue

                if global_step % args.eval_freq == 0:
                    # Since the valid data broardcast to all devices, we do evaluate on all device.
                    run_evaluate(valid_data_loader, model, criterion,
                                 args.eval_iters, log_writer, global_step,
                                 epoch, "valid")

                if global_step % args.save_steps == 0 or global_step >= args.max_steps:
                    if worker_index == 0:
                        output_dir = os.path.join(args.output_dir,
                                                  "model_%d" % global_step)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Need better way to get inner model of DataParallel
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

                reader_start = time.time()

            del train_data_loader


if __name__ == "__main__":
    args = parse_args(MODEL_CLASSES)
    do_train(args)
