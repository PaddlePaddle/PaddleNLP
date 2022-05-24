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
from modeling import GPTModel, GPTForPretraining, GPTPretrainingCriterion, GPTForPretrainingPipe
from paddlenlp.transformers import GPTTokenizer, GPTChineseTokenizer
from paddlenlp.utils.log import logger

from dataset import create_pretrained_dataset
from args import parse_args
import lr, userdefine_optimizer
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients
import types
from utils import get_timers, set_timers

set_timers()

MODEL_CLASSES = {
    "gpt": (GPTForPretraining, GPTTokenizer),
    "gpt-cn": (GPTForPretraining, GPTChineseTokenizer),
}


def set_hyrbid_parallel_seed(basic_seed, dp_rank, mp_rank, pp_rank):
    assert args.device != "cpu"

    random.seed(basic_seed + dp_rank)
    np.random.seed(basic_seed + dp_rank)
    paddle.seed(basic_seed + dp_rank)

    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = basic_seed + 123 + mp_rank * 10 + pp_rank * 1000
    global_seed = basic_seed + dp_rank
    tracker = get_rng_state_tracker()
    tracker.add('global_seed', global_seed)
    tracker.add('local_seed', local_seed)


@paddle.no_grad()
def run_evaluate(args,
                 data_loader,
                 model,
                 criterion,
                 iter_steps,
                 log_writer,
                 global_step,
                 epoch,
                 task_name="valid"):
    model.eval()
    all_loss = []
    local_time = time.time()
    for eval_step, batch in enumerate(data_loader):
        tokens, loss_mask, labels = batch
        if args.pp_degree < 2:
            preds = model(tokens)
            loss = criterion(preds, labels, loss_mask)
        else:
            data = [tokens, (labels, loss_mask)]
            loss = model.eval_batch(data, compute_loss=True)

        all_loss.append(float(loss))
        if eval_step >= iter_steps - 1:
            break

    average_loss = sum(all_loss) / len(all_loss)
    logger.info("%s step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                % (task_name, global_step, epoch, eval_step, average_loss,
                   iter_steps / (time.time() - local_time)))
    log_writer.add_scalar(task_name + "_loss", average_loss, global_step)
    model.train()

    
def initialize_model_and_expert_group(hcg):
    def get_expert_parallel_world_size(self):
        return self.get_data_parallel_world_size() * self.get_model_parallel_world_size()
    hcg.get_expert_parallel_world_size =  types.MethodType(get_expert_parallel_world_size, hcg)

    # need create mp_dp group for expert parallel group in advance
    _, mp_dp_comm_group = hcg._set_check_group(parallel_method="pipe")
    def get_expert_parallel_group(self):
        return mp_dp_comm_group
    hcg.get_expert_parallel_group = types.MethodType(get_expert_parallel_group, hcg)    


def initialize_mp_dp_parameters(model):
    for param in model.parameters():
        if "expert_" in param.name:
            setattr(param, 'no_sync', True)

def timer_log(log_freq):
    timers=get_timers()
    # Logging
    timers_to_log = []
    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)
    add_to_logging('forward-compute')
    add_to_logging('forward-recv')
    add_to_logging('forward-send')
    add_to_logging('forward-send-backward-recv')
    add_to_logging('backward-compute')
    add_to_logging('backward-recv')
    add_to_logging('backward-send')
    add_to_logging('backward-send-forward-recv')
    add_to_logging('backward-params-all-reduce')
    add_to_logging('backward-embedding-all-reduce')
    add_to_logging('optimizer-copy-to-main-grad')
    add_to_logging('optimizer-unscale-and-check-inf')
    add_to_logging('optimizer-clip-main-grad')
    add_to_logging('optimizer-copy-main-to-model-params')
    add_to_logging('optimizer')
    add_to_logging('batch-generator')
    add_to_logging('Prepare Forward')
    add_to_logging('Gate Computation')
    add_to_logging('Limit_By_Capacity')
    add_to_logging('Prune_Gate_By_Cap')
    add_to_logging('Random Routing')
    add_to_logging('Base Operation')
    add_to_logging('AllGather in Limit')
    add_to_logging('MOEScatter')
    add_to_logging('Expert Computation')
    add_to_logging('MOEGather')
    add_to_logging('Score BMM')
    add_to_logging('AllReduce')
    add_to_logging('AllGather')
    add_to_logging('lec reduce')
    add_to_logging('lec reduce2')
    
    timers.log(timers_to_log, normalizer=log_freq)

def do_train(args):
    paddle.set_device(args.device)
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": args.dp_degree,
        "mp_degree": args.mp_degree,
        "pp_degree": args.pp_degree
    }

    strategy.pipeline_configs = {
        "accumulate_steps": args.local_batch_size // args.micro_batch_size,
        "micro_batch_size": args.micro_batch_size
    }

    fleet.init(is_collective=True, strategy=strategy)

    # obtain rank message of hybrid parallel
    hcg = fleet.get_hybrid_communicate_group()
    global_rank = hcg.get_global_rank()
    mp_rank = hcg.get_model_parallel_rank()
    pp_rank = hcg.get_stage_id()
    dp_rank = hcg.get_data_parallel_rank()
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

    # seed control in hybrid parallel
    set_hyrbid_parallel_seed(args.seed, dp_rank, mp_rank, pp_rank)

    default_global_tokens_num = args.global_batch_size * args.max_seq_len

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    # Define log writer
    log_writer_path = os.path.join(
        args.output_dir, "train_log",
        "{}_globalbsz_{}_amp_{}_recompute_{}_card_{}".format(
            args.model_name_or_path, args.global_batch_size, args.use_amp,
            False, global_rank).lower())

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

        model_config['num_partitions'] = args.mp_degree

        # MOE config
        initialize_model_and_expert_group(hcg)

        model_config['expert_mode'] = args.expert_mode
        model_config['hcg'] = hcg
        model_config['num_experts'] = args.num_experts
        model_config['top_k'] = args.top_k

        if args.pp_degree == 1:
            model = GPTForPretraining(GPTModel(**model_config))
        else:
            model_config['topology'] = hcg.topology()
            model_config["recompute_interval"] = 1 if args.use_recompute else 0
            model = GPTForPretrainingPipe(**model_config)
    else:
        model = GPTForPretraining.from_pretrained(
            args.model_name_or_path,
            hidden_dropout_prob=args.hidden_dropout_prob,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob)

    # Create the critrion for the gpt model
    criterion = GPTPretrainingCriterion()

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

    if paddle.distributed.get_world_size() > 1:
        print(">> initialize....")
        initialize_mp_dp_parameters(model)

    if args.pp_degree > 1:
        # need to no sync parameters
        model = fleet.distributed_model(model)
        optimizer = userdefine_optimizer.MOEOptimizer(optimizer, hcg)
        # optimizer = fleet.distributed_optimizer(optimizer)

    # if args.use_amp:
    #     scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
    #     scaler = fleet.distributed_scaler(scaler)

    p_num = 0
    for param in model.parameters():
        p_num += 1
        print(param.name, param.shape)
    print("all parameters number: ", p_num)

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
    timers = get_timers()
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
                local_rank=local_rank,
                data_world_size=args.dp_degree,
                data_world_rank=dp_rank,
                eos_id=tokenizer.eos_token_id)

            # Bug fix, if not call valid_data_loader, the enumerate will call valid_data_loader
            # many times. and start a new random dataloader.
            valid_data_loader = valid_data_loader()
            test_data_loader = test_data_loader()

            for step, batch in enumerate(train_data_loader()):
                global_step += 1
                tokens, loss_mask, labels = batch

                loss_mask.stop_gradient = True
                labels.stop_gradient = True

                data = [tokens, (labels, loss_mask)]
                loss = model.forward_backward_pipeline(data)
                
                
                optimizer.step()

                learning_rate = optimizer.get_lr()
                lr_scheduler.step()
                optimizer.clear_grad()

                if global_step % args.logging_freq == 0:
                    avg_loss = loss.numpy()
                    speed = args.logging_freq / (time.time() - tic_train)
                    logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %.9f, speed: %.2f step/s, ips: %.0f tokens/s, learning rate: %.5e"
                        % (global_step, epoch, step, avg_loss, speed, speed *
                           default_global_tokens_num, learning_rate))
                    log_writer.add_scalar("loss", float(loss), global_step)
                    log_writer.add_scalar("learning_rate", learning_rate, global_step)

                    tic_train = time.time()
                    timer_log(args.logging_freq)
                if global_step >= args.max_steps:
                    return 

                # if args.check_accuracy:
                #     if global_step >= args.max_steps:
                #         return
                #     else:
                #         continue

                # if global_step % args.eval_freq == 0:
                #     # Since the valid data broardcast to all devices, we do evaluate on all device.
                #     run_evaluate(args, valid_data_loader, model, criterion,
                #                  args.eval_iters, log_writer, global_step,
                #                  epoch, "valid")

                # only dp_rank = 0 save model
                # if (global_step % args.save_steps == 0 or
                #         global_step >= args.max_steps) and dp_rank == 0:

                #     model_to_save = model._layers if paddle.distributed.get_world_size(
                #     ) > 1 else model
                #     output_dir = os.path.join(args.output_dir,
                #                               "step_%d" % global_step)
                #     os.makedirs(output_dir, exist_ok=True)

                #     logger.info("Save model to %s" % output_dir)

                #     if args.pp_degree > 1:
                #         model_to_save.save_state_dict(output_dir)
                #         if mp_rank * pp_rank == 1:
                #             tokenizer.save_pretrained(output_dir)
                #         paddle.save(
                #             optimizer.state_dict(),
                #             os.path.join(
                #                 output_dir,
                #                 "model_state_mp_{:0>2d}_pp_{:0>2d}.pdopt".
                #                 format(mp_rank, pp_rank)))
                #     else:
                #         path = os.path.join(output_dir,
                #                             'model_{:0>2d}'.format(mp_rank))
                #         os.makedirs(path, exist_ok=True)
                #         model_to_save.save_pretrained(path)

                #         paddle.save(optimizer.state_dict(),
                #                     os.path.join(path, "model_state.pdopt"))
                #         tokenizer.save_pretrained(path)

                # if global_step >= args.max_steps:
                #     run_evaluate(args, test_data_loader, model, criterion,
                #                  args.test_iters, log_writer, global_step,
                #                  epoch, "test")
                #     logger.info("The training process is complete.")
                #     del train_data_loader
                #     return

            del train_data_loader


if __name__ == "__main__":
    args = parse_args(MODEL_CLASSES)
    do_train(args)
