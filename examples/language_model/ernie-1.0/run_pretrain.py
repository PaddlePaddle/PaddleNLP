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
"""
ERNIE-1.0 pretraining scripts.
"""
import argparse
import os
import sys
import random
import time
import yaml
import shutil

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset
from visualdl import LogWriter

from paddlenlp.transformers import ErnieModel, ErnieForPretraining, ErniePretrainingCriterion, ErnieTokenizer
from paddlenlp.transformers import CosineAnnealingWithWarmupDecay, LinearDecayWithWarmup
from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.ops import Topology
from paddlenlp.utils.log import logger

from args import parse_args
sys.path.insert(0, os.path.abspath("../"))
from data_tools.dataset_utils import build_train_valid_test_datasets

MODEL_CLASSES = {
    "ernie": (ErnieModel, ErnieForPretraining, ErniePretrainingCriterion,
              ErnieTokenizer),
}


def create_pretrained_dataset(
        args,
        data_file,
        tokenizer,
        data_world_size,
        data_world_rank,
        max_seq_len,
        places=None,
        data_holders=None,
        current_step=0, ):

    train_valid_test_num_samples = [
        args.global_batch_size * args.max_steps,
        args.micro_batch_size * (args.max_steps // args.eval_freq + 1) *
        args.eval_iters * data_world_size,
        args.micro_batch_size * args.test_iters * data_world_size
    ]
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=data_file,
        args=args,
        tokenizer=tokenizer,
        splits_string=args.split,
        train_valid_test_num_samples=train_valid_test_num_samples,
        max_seq_length=args.max_seq_len,
        masked_lm_prob=args.masked_lm_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=True,
        binary_head=True,
        max_seq_length_dec=None,
        dataset_type='ernie')

    def _collate_data(data, stack_fn=Stack()):
        num_fields = len(data[0])
        out = [None] * num_fields
        # 0. input_ids,
        # 1. segment_ids,
        # 2. input_mask,
        # 3. masked_lm_positions,
        # 4. masked_lm_labels,
        # 5. next_sentence_labels
        for i in (0, 1, 2, 5):
            out[i] = stack_fn([x[i] for x in data])
        out[5] = out[5].reshape([-1, 1])
        batch_size, seq_length = out[0].shape
        size = num_mask = sum(len(x[3]) for x in data)
        # masked_lm_positions
        # Organize as a 1D tensor for gather or use gather_nd
        if size % 8 != 0:
            size += 8 - (size % 8)
        out[3] = np.full(size, 0, dtype=np.int32)
        # masked_lm_labels
        out[4] = np.full([size, 1], -1, dtype=np.int64)
        mask_token_num = 0
        for i, x in enumerate(data):
            for j, pos in enumerate(x[3]):
                out[3][mask_token_num] = i * seq_length + pos
                out[4][mask_token_num] = x[4][j]
                mask_token_num += 1

        return out

    def loader(dataset, consumed_samples=0):
        batch_sampler = DistributedBatchSampler(
            dataset,
            batch_size=args.micro_batch_size,
            num_replicas=data_world_size,
            rank=data_world_rank,
            shuffle=False,
            drop_last=True,
            consumed_samples=consumed_samples)
        data_loader = paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            worker_init_fn=None,
            collate_fn=_collate_data,
            return_list=False)
        return data_loader

    train_dl = loader(train_ds, args.global_batch_size * current_step)
    valid_dl = loader(valid_ds, args.micro_batch_size * (
        (current_step + 1) // args.eval_freq) * args.eval_iters *
                      data_world_size)
    test_dl = loader(test_ds, 0)

    return train_dl, valid_dl, test_dl


def get_train_data_file(args):
    files = [
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if (os.path.isfile(os.path.join(args.input_dir, f)) and "_idx.npz" in
            str(f))
    ]
    files = [x.replace("_idx.npz", "") for x in files]
    return files


@paddle.no_grad()
def run_evaluate(data_loader,
                 model,
                 criterion,
                 iter_steps,
                 log_writer,
                 global_step,
                 args,
                 task_name="valid"):
    model.eval()
    all_loss, all_lm_loss, all_sop_loss = [], [], []
    local_time = time.time()

    for eval_step, batch in enumerate(data_loader):
        input_ids, segment_ids, input_mask, masked_lm_positions, \
        masked_lm_labels, next_sentence_labels = batch

        # Create the model for the gpt pretrain
        prediction_scores, seq_relationship_score = model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            position_ids=None,
            attention_mask=input_mask,
            masked_positions=masked_lm_positions)

        lm_loss, sop_loss = criterion(prediction_scores, seq_relationship_score,
                                      masked_lm_labels, next_sentence_labels)
        loss = lm_loss + sop_loss

        all_loss.append(float(loss.item()))
        all_lm_loss.append(float(lm_loss.item()))
        all_sop_loss.append(float(sop_loss.item()))

        if eval_step >= iter_steps - 1:
            average_loss = sum(all_loss) / len(all_loss)
            average_lm_loss = sum(all_lm_loss) / len(all_lm_loss)
            average_sop_loss = sum(all_sop_loss) / len(all_sop_loss)
            logger.info(
                "%s step %d, batch: %d, loss: %f, lm_loss: %.6f, sop_loss: %.6f, speed: %.0f tokens/s"
                % (task_name, global_step, eval_step, average_loss,
                   average_lm_loss, average_sop_loss,
                   iter_steps * args.micro_batch_size * args.max_seq_len /
                   (time.time() - local_time)))

            log_writer.add_scalar(task_name + "_loss", average_loss,
                                  global_step)
            log_writer.add_scalar(task_name + "_lm_loss", average_lm_loss,
                                  global_step)
            log_writer.add_scalar(task_name + "_sop_loss", average_sop_loss,
                                  global_step)

            break

    model.train()


def set_seed(args):
    if args.device == "cpu":
        idx = 0
    else:
        idx = paddle.distributed.get_rank()
    random.seed(args.seed + idx)
    np.random.seed(args.seed + idx)
    paddle.seed(args.seed + idx)


def args_post_process(args, worker_num):
    default_global_batch_size = worker_num * args.micro_batch_size
    if args.global_batch_size is None:
        args.global_batch_size = default_global_batch_size

    bsz_per_dp = args.global_batch_size // worker_num
    micro_batch_size = args.micro_batch_size
    assert args.global_batch_size % micro_batch_size == 0, \
        "cannot do gradient accumulate, global_batch_size: {} micro_batch_size: {}".format(
        args.global_batch_size, micro_batch_size)
    accumulate_steps = bsz_per_dp // micro_batch_size

    args.eval_iters *= accumulate_steps
    args.test_iters *= accumulate_steps

    args.accumulate_steps = accumulate_steps


def do_train(args):
    paddle.set_device(args.device)

    worker_index = paddle.distributed.get_rank()
    worker_num = paddle.distributed.get_world_size()
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

    if worker_num > 1:
        paddle.distributed.init_parallel_env()

    if args.dp_degree * args.sharding_degree == 1:
        args.dp_degree = worker_num
        args.sharding_degree = 1

    args_post_process(args, worker_num)

    logger.info('{:20}:{}'.format("paddle commit id", paddle.version.commit))
    for arg in vars(args):
        logger.info('{:20}:{}'.format(arg, getattr(args, arg)))

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": args.dp_degree,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1
    }

    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()

    worker_index = paddle.distributed.get_rank()
    worker_num = paddle.distributed.get_world_size()
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

    # Create the random seed for the worker
    set_seed(args)

    assert args.dp_degree * args.sharding_degree == worker_num, \
        "The product of degree num should be equal to worker_num."

    # Create log write, 
    log_writer_path = os.path.join(
        args.output_dir, "train_log",
        "{}_globalbsz_{}_amp_{}_recompute_{}_card_{}".format(
            args.model_name_or_path, args.global_batch_size, args.use_amp,
            args.use_recompute, worker_index).lower())
    log_writer = LogWriter(log_writer_path)

    # Define the input data in the static mode
    base_class, model_class, criterion_class, tokenizer_class = MODEL_CLASSES[
        args.model_type]
    pretrained_models_list = list(
        model_class.pretrained_init_configuration.keys())

    # load config in checkpoint
    global_step = 0
    consumed_samples = 0
    checkpoint_dir = os.path.join(args.output_dir, "model_last")
    if os.path.exists(checkpoint_dir):
        if os.path.isfile(os.path.join(checkpoint_dir, "./config.yml")):
            with open(os.path.join(checkpoint_dir, "./config.yml"), "r") as f:
                step_config = yaml.load(f, Loader=yaml.FullLoader)
                assert step_config[
                    "global_batch_size"] == args.global_batch_size, "Please ensure checkpoint global batch size is the same. Folder: {}".format(
                        checkpoint_dir)
                consumed_samples = step_config["consumed_samples"]
                global_step = step_config["global_step"]

    if args.model_name_or_path in pretrained_models_list:
        model_config = model_class.pretrained_init_configuration[
            args.model_name_or_path]
        model_config["hidden_dropout_prob"] = args.hidden_dropout_prob
        model_config[
            "attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
        model = model_class(base_class(**model_config))
    else:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            hidden_dropout_prob=args.hidden_dropout_prob,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob)

    criterion = criterion_class()

    # Create the learning_rate sheduler and optimizer
    if args.decay_steps is None:
        args.decay_steps = args.max_steps

    lr_scheduler = LinearDecayWithWarmup(
        args.max_lr, args.max_steps, args.warmup_rate, last_epoch=global_step)

    clip = None
    if args.grad_clip > 0:
        clip = paddle.fluid.clip.GradientClipByGlobalNorm(
            clip_norm=args.grad_clip)

    decay_param = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    logger.info("Using paddle.optimizer.AdamW.")
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler if lr_scheduler is not None else args.max_lr,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        grad_clip=clip,
        apply_decay_param_fun=lambda x: x in decay_param,
        multi_precision=args.use_amp)

    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
        scaler = fleet.distributed_scaler(scaler)
        model = paddle.amp.decorate(
            models=model, level='O2', save_dtype='float32')

    if paddle.distributed.get_world_size() > 1:
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    data_file = get_train_data_file(args)

    train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
        args,
        data_file,
        tokenizer,
        data_world_size=worker_num,
        data_world_rank=worker_index,
        max_seq_len=args.max_seq_len,
        current_step=global_step)

    # load checkpoint vars 
    if os.path.exists(checkpoint_dir):
        if os.path.isfile(os.path.join(checkpoint_dir, "./config.yml")):
            logger.info("Try to load checkpoint from %s " % checkpoint_dir)
            opt_path = os.path.join(checkpoint_dir, "model_state.pdopt")
            params_path = os.path.join(checkpoint_dir, "model_state.pdparams")

            if os.path.exists(opt_path):
                opt_dict = paddle.load(opt_path)
                optimizer.set_state_dict(opt_dict)
                model_dict = paddle.load(params_path)
                model.set_state_dict(model_dict)
            else:
                logger.warning("No optimizer checkpoint file found in %s." %
                               opt_path)
            logger.info("Checkpoint loaded from global step: {}".format(
                global_step))

    tic_train = time.time()
    while True:
        # If not call valid_data_loader, the enumerate will call valid_data_loader
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

            # 0. input_ids,
            # 1. segment_ids,
            # 2. input_mask,
            # 3. masked_lm_positions,
            # 4. masked_lm_labels,
            # 5. next_sentence_labels

            input_ids, segment_ids, input_mask, masked_lm_positions, \
            masked_lm_labels, next_sentence_labels = batch

            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_black_list=[
                        "reduce_sum", "c_softmax_with_cross_entropy",
                        "elementwise_div"
                    ],
                    level='O2'):

                # Create the model for the ernie pretrain
                prediction_scores, seq_relationship_score = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    position_ids=None,
                    attention_mask=input_mask,
                    masked_positions=masked_lm_positions)

                lm_loss, sop_loss = criterion(
                    prediction_scores, seq_relationship_score, masked_lm_labels,
                    next_sentence_labels)
                loss = lm_loss + sop_loss

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.minimize(optimizer, loss)
            else:
                loss.backward()
                optimizer.step()

            optimizer.clear_grad()
            train_run_cost += time.time() - train_start

            # Skip for accumulate_steps in global step
            if (step + 1) % args.accumulate_steps != 0:
                continue

            global_step += 1

            if global_step % args.logging_freq == 0:
                speed = args.logging_freq / (time.time() - tic_train)
                common_loginfo = "global step %d, loss: %.9f, lm_loss: %.6f, sop_loss: %.6f, speed: %.2f steps/s, ips: %.2f seqs/s, learning rate: %.5e" % (
                    global_step, loss.item(), lm_loss.item(), sop_loss.item(),
                    speed, speed * args.global_batch_size,
                    lr_scheduler.get_lr())
                addition_info = ""
                if args.use_amp:
                    addition_info = " loss_scaling: %.1f, incr_count: %d, decr_count: %d" % (
                        scaler._scale.numpy(), scaler._incr_count,
                        scaler._decr_count)
                logger.info(common_loginfo + addition_info)
                log_writer.add_scalar("loss", loss.item(), global_step)
                log_writer.add_scalar("lm_loss", lm_loss.item(), global_step)
                log_writer.add_scalar("sop_loss", sop_loss.item(), global_step)

                tic_train = time.time()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if global_step % args.eval_freq == 0:
                # TODO, check the input data of validation

                run_evaluate(
                    valid_data_loader,
                    model,
                    criterion,
                    args.eval_iters,
                    log_writer,
                    global_step,
                    args,
                    task_name="valid")
                tic_train = time.time()

            def save_ckpt(output_dir, model, tokenizer, args, global_step):
                step_config = {
                    "model_name": args.model_name_or_path,
                    "global_step": global_step,
                    "global_batch_size": args.global_batch_size,
                    "consumed_samples": global_step * args.global_batch_size,
                }

                logger.debug("saving models to {}".format(output_dir))
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model

                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                paddle.save(optimizer.state_dict(),
                            os.path.join(output_dir, "model_state.pdopt"))

                with open(os.path.join(output_dir, "config.yml"), "w") as f:
                    yaml.dump(
                        step_config, f, encoding='utf-8', allow_unicode=True)

            if global_step % args.save_steps == 0 or global_step >= args.max_steps:
                output_dir = os.path.join(args.output_dir,
                                          "model_%d" % global_step)
                if worker_index == 0:
                    save_ckpt(output_dir, model, tokenizer, args, global_step)

                if worker_num > 1:
                    paddle.distributed.barrier()
                tic_train = time.time()

            if global_step % args.checkpoint_steps == 0:
                output_dir = os.path.join(args.output_dir, "model_last")
                if worker_index == 0:
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    output_dir_bak = os.path.join(args.output_dir,
                                                  "model_last_bak")
                    if os.path.exists(output_dir):
                        if os.path.exists(output_dir_bak):
                            shutil.rmtree(output_dir_bak)
                        shutil.move(output_dir, output_dir_bak)
                        os.mkdir(output_dir)
                    save_ckpt(output_dir, model, tokenizer, args, global_step)

                if worker_num > 1:
                    paddle.distributed.barrier()

            if global_step >= args.max_steps:
                run_evaluate(
                    test_data_loader,
                    model,
                    criterion,
                    args.test_iters,
                    log_writer,
                    global_step,
                    args,
                    task_name="test")
                del train_data_loader
                return


if __name__ == "__main__":
    config = parse_args(MODEL_CLASSES)
    do_train(config)
