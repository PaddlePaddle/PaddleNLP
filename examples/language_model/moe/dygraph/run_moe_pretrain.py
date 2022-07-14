# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import lr
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients
import types
from utils import get_timers, set_timers
from types import MethodType
from paddle import _C_ops
from paddle.framework import core, in_dygraph_mode
import paddle.distributed as dist
from framework import assign_group_by_size, flatten_dense_tensors, obtain_storage, AdamW, group_sharded_parallel
from paddle.incubate.distributed.models import moe
from paddle.distributed.fleet.meta_parallel.sharding.sharding_utils import ShardingScaler
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import GroupShardedScaler

from checkpointing import save_checkpoint, load_checkpoint

MODEL_CLASSES = {
    "gpt": (GPTForPretraining, GPTTokenizer),
    "gpt-cn": (GPTForPretraining, GPTChineseTokenizer),
}

set_timers()


def set_hyrbid_parallel_seed(basic_seed, data_world_rank, mp_rank, pp_rank):
    assert args.device != "cpu"

    random.seed(basic_seed + data_world_rank)
    np.random.seed(basic_seed + data_world_rank)
    paddle.seed(basic_seed + data_world_rank)

    from paddle.distributed.fleet import meta_parallel
    meta_parallel.model_parallel_random_seed(basic_seed + data_world_rank +
                                             1000 * mp_rank)

    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = basic_seed + 123 + mp_rank * 10 + pp_rank * 1000
    global_seed = basic_seed + data_world_rank
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
        with paddle.amp.auto_cast(args.use_pure_fp16,
                                  custom_black_list=[
                                      "reduce_sum",
                                      "c_softmax_with_cross_entropy",
                                      "elementwise_div",
                                  ],
                                  level='O2'):
            preds = model(tokens)
        preds = paddle.cast(preds, dtype="float32")
        loss = criterion(preds, labels, loss_mask)

        all_loss.append(float(loss))
        if eval_step >= iter_steps - 1:
            break

    average_loss = sum(all_loss) / len(all_loss)
    logger.info(
        "%s step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s" %
        (task_name, global_step, epoch, eval_step, average_loss, iter_steps /
         (time.time() - local_time)))
    log_writer.add_scalar(task_name + "_loss", average_loss, global_step)
    model.train()


def initialize_model_and_expert_group(hcg):

    def get_expert_parallel_world_size(self):
        return self.get_data_parallel_world_size(
        ) * self.get_model_parallel_world_size()

    hcg.get_expert_parallel_world_size = types.MethodType(
        get_expert_parallel_world_size, hcg)

    # need create mp_dp group for expert parallel group in advance
    _, mp_dp_comm_group = hcg._set_check_group(parallel_method="pipe")

    def get_expert_parallel_group(self):
        return mp_dp_comm_group

    hcg.get_expert_parallel_group = types.MethodType(get_expert_parallel_group,
                                                     hcg)


def initialize_mp_dp_parameters(model, hcg):
    mp_group = hcg.get_model_parallel_group()
    mp_src_rank = hcg.get_model_parallel_group_src_rank()

    dp_group = hcg.get_data_parallel_group()
    dp_src_rank = hcg.get_data_parallel_group_src_rank()

    for param in model.parameters():
        if "expert_" in param.name:
            continue
        if not param.is_distributed:
            paddle.distributed.broadcast(param.detach(),
                                         src=mp_src_rank,
                                         group=mp_group,
                                         use_calc_stream=True)

        paddle.distributed.broadcast(param.detach(),
                                     src=dp_src_rank,
                                     group=dp_group,
                                     use_calc_stream=True)


def unscale_method(self, optimizer):
    if not self._enable:
        return

    if getattr(optimizer, '_param_groups', None) and isinstance(
            optimizer._param_groups[0], dict):
        param_grads_fp16 = []
        param_grads_fp32 = []
        for group in optimizer._param_groups:
            for param in group['params']:
                if param._grad_ivar() is not None:
                    if param._grad_ivar().dtype == core.VarDesc.VarType.FP16:
                        param_grads_fp16.append(param._grad_ivar())
                    else:
                        param_grads_fp32.append(param._grad_ivar())
    else:
        param_grads_fp16 = [
            param._grad_ivar() for param in optimizer._parameter_list
            if (param._grad_ivar() is not None) and (
                param._grad_ivar().dtype == core.VarDesc.VarType.FP16)
        ]
        param_grads_fp32 = [
            param._grad_ivar() for param in optimizer._parameter_list
            if (param._grad_ivar() is not None) and (
                param._grad_ivar().dtype == core.VarDesc.VarType.FP32)
        ]
    temp_found_inf_fp16 = paddle.to_tensor(np.array([0]).astype(np.bool))
    temp_found_inf_fp32 = paddle.to_tensor(np.array([0]).astype(np.bool))

    if len(param_grads_fp16):
        _C_ops.check_finite_and_unscale(param_grads_fp16, self._scale,
                                        param_grads_fp16, temp_found_inf_fp16)
    if len(param_grads_fp32):
        _C_ops.check_finite_and_unscale(param_grads_fp32, self._scale,
                                        param_grads_fp32, temp_found_inf_fp32)
    self._found_inf = 1 if temp_found_inf_fp16 or temp_found_inf_fp32 else 0

    if dist.get_world_size() > 1:
        is_found_inf = paddle.to_tensor([self._found_inf], dtype="int32")
        paddle.distributed.all_reduce(is_found_inf,
                                      op=paddle.distributed.ReduceOp.MAX,
                                      group=None)
        self._found_inf = is_found_inf.numpy()[0]


def all_reduce_parameters(params, group):
    if group.nranks < 2:
        return

    div_factor = 1.0 / group.nranks
    with paddle.framework.no_grad():
        for p in params:
            grad = p.grad.scale_(div_factor)
            paddle.distributed.all_reduce(grad, use_calc_stream=True)


def parameters_classify(model, use_sharding=False):
    decay_gate_params = []
    decay_expert_params = []
    decay_other_params = []

    gate_params = []
    expert_params = []
    other_params = []

    for param in model.parameters():
        # param_name = param.name
        if "expert_" in param.name:
            if not any(nd in param.name for nd in ["bias", "norm"]):
                decay_expert_params.append(param)
            else:
                expert_params.append(param)
        elif "gate_" in param.name:
            if not any(nd in param.name for nd in ["bias", "norm"]):
                decay_gate_params.append(param)
            else:
                gate_params.append(param)
        else:
            if not any(nd in param.name for nd in ["bias", "norm"]):
                decay_other_params.append(param)
            else:
                other_params.append(param)

    print("all parameters length:", len(model.parameters()))
    print(
        "decay_gate_params len: {}, decay_expert_params len: {}, decay_other_params len: {}"
        .format(len(decay_gate_params), len(decay_expert_params),
                len(decay_other_params)))
    print("gate_params len: {}, expert_params len: {}, other_params len: {}".
          format(len(gate_params), len(expert_params), len(other_params)))

    d_gate = obtain_storage(decay_gate_params)
    gate = obtain_storage(gate_params)

    d_expert = obtain_storage(decay_expert_params)
    expert = obtain_storage(expert_params)

    d_other = decay_other_params if use_sharding else obtain_storage(
        decay_other_params)
    other = other_params if use_sharding else obtain_storage(other_params)

    opt_fused_tensors = []
    decay_fused_tensors = []
    reduce_fused_tensors = []
    gate_fused_tensors = []

    decay_fused_tensors = d_gate + d_other + d_expert
    opt_fused_tensors = decay_fused_tensors + gate + other + expert
    reduce_fused_tensors = d_other + other
    gate_fused_tensors = d_gate + gate

    expert_fusion_names = []
    for i, p in enumerate(d_expert + expert):
        p.name = "fused_expert_tensor_{}".format(i)
        expert_fusion_names.append(p.name)

    for i, p in enumerate(d_gate + gate):
        p.name = "fused_gate_tensor_{}".format(i)

    return opt_fused_tensors, decay_fused_tensors, reduce_fused_tensors, gate_fused_tensors, expert_fusion_names


def timer_log(log_freq):
    timers = get_timers()
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
        "pp_degree": args.pp_degree,
        "sharding_degree": args.sharding_degree
    }

    accumulate_steps = args.local_batch_size // args.micro_batch_size
    strategy.pipeline_configs = {
        "accumulate_steps": accumulate_steps,
        "micro_batch_size": args.micro_batch_size
    }

    fleet.init(is_collective=True, strategy=strategy)

    nranks = paddle.distributed.get_world_size()

    # obtain rank message of hybrid parallel
    hcg = fleet.get_hybrid_communicate_group()
    global_rank = hcg.get_global_rank()
    mp_rank = hcg.get_model_parallel_rank()
    pp_rank = hcg.get_stage_id()
    dp_rank = hcg.get_data_parallel_rank()
    sharding_rank = hcg.get_sharding_parallel_rank()
    sharding_group = hcg.get_sharding_parallel_group()

    if args.sharding_degree > 1:
        assert args.dp_degree == args.mp_degree == args.pp_degree == 1, "sharding stage2 will support hybrid parallel later"

    sharding_size = hcg.get_sharding_parallel_world_size()
    data_world_rank = dp_rank * sharding_size + sharding_rank
    data_world_size = args.dp_degree * args.sharding_degree
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

    # seed control in hybrid parallel
    set_hyrbid_parallel_seed(args.seed, data_world_rank, mp_rank, pp_rank)

    default_global_tokens_num = args.global_batch_size * args.max_seq_len

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    # Define log writer
    log_writer_path = os.path.join(
        args.output_dir, "train_log",
        "{}_globalbsz_{}_pure_fp16_{}_recompute_{}_card_{}".format(
            args.model_name_or_path, args.global_batch_size, args.use_pure_fp16,
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
        if args.expert_mode:
            model_config['gate'] = args.gate

        if args.pp_degree == 1:
            model_config["recompute_interval"] = 1 if args.use_recompute else 0
            model_config["recompute_partition"] = args.recompute_partition
            model_config["recompute_offload"] = args.recompute_offload
            if args.use_recompute and args.recompute_partition:
                raise Exception(
                    "when use_recompute is True, recompute_partition must be False in MoE."
                )

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

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    if args.use_pure_fp16:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
        if args.sharding_degree == 1:
            scaler = fleet.distributed_scaler(scaler)
            scaler._unscale = MethodType(unscale_method, scaler)
        else:
            wrap_scale_func = GroupShardedScaler if in_dygraph_mode(
            ) else ShardingScaler
            scaler = wrap_scale_func(scaler)

        model = paddle.amp.decorate(models=model,
                                    optimizers=None,
                                    level='O2',
                                    save_dtype='float32')

    opt_fused_tensors, decay_fused_tensors, reduce_fused_tensors, gate_fused_tensors, \
        expert_fusion_names = parameters_classify(model, use_sharding=(args.sharding_degree > 1))
    decay_params = [p.name for p in decay_fused_tensors]

    clip = None
    if args.grad_clip > 0:
        is_expert_param_fun = lambda param: param.name in expert_fusion_names
        clip = moe.ClipGradByGlobalNorm(clip_norm=args.grad_clip, \
                                        is_expert_param_func = is_expert_param_fun, \
                                        moe_group = hcg.get_expert_parallel_group())

    optimizer = AdamW(
        learning_rate=lr_scheduler if lr_scheduler is not None else args.max_lr,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        epsilon=args.adam_epsilon,
        parameters=opt_fused_tensors,
        weight_decay=args.weight_decay,
        grad_clip=clip,
        apply_decay_param_fun=lambda x: x in decay_params,  #decay_params,
        multi_precision=args.use_pure_fp16)

    #in order to restore reader.
    pass_num = 0
    file_id = 0
    start_epoch = 0
    args.resume_dir = None if len(args.resume_dir) <= 0 else args.resume_dir

    if paddle.distributed.get_world_size() > 1 and args.resume_dir is None:
        print(">> initialize....")
        if args.sharding_degree > 1:
            model, optimizer = group_sharded_parallel(model, optimizer,
                                                      sharding_group,
                                                      args.sharding_offload)
            for p in gate_fused_tensors:
                dist.broadcast(p,
                               src=sharding_group.ranks[0],
                               group=sharding_group,
                               use_calc_stream=True)
            # Multi stream operation will be supported later
            dist.wait(tensor=p, group=sharding_group, use_calc_stream=True)
        else:
            initialize_mp_dp_parameters(model, hcg)

    if args.resume_dir is not None:
        global_step, loss_scale, data_meta = load_checkpoint(
            args, model, optimizer, lr_scheduler, tokenizer, dp_rank, mp_rank,
            pp_rank)
        pass_num = data_meta["pass_num"]
        file_id = data_meta["file_id"]
        start_epoch = data_meta["start_epoch"]

    if args.model_name_or_path not in pretrained_models_list:
        logger.info("Try to load checkpoint from %s " % args.model_name_or_path)
        opt_path = os.path.join(args.model_name_or_path, "model_state.pdopt")
        if os.path.exists(opt_path):
            opt_dict = paddle.load(opt_path)
            optimizer.set_state_dict(opt_dict)
        else:
            logger.warning("No optimizer checkpoint file found in %s." %
                           opt_path)

    global_step = 0 if args.resume_dir is None else global_step
    timers = get_timers()
    tic_train = time.time()
    for epoch in range(start_epoch, args.num_train_epochs):
        files = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if (os.path.isfile(os.path.join(args.input_dir, f))
                and "npz_" not in str(f))
        ]
        files.sort()
        num_files = len(files)
        for f_id in range(file_id, num_files):
            data_file = files[f_id]
            train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
                args,
                data_file,
                local_rank=local_rank,
                data_world_size=data_world_size,
                data_world_rank=data_world_rank,
                eos_id=tokenizer.eos_token_id)

            # Bug fix, if not call valid_data_loader, the enumerate will call valid_data_loader
            # many times. and start a new random dataloader.
            valid_data_loader = valid_data_loader()
            test_data_loader = test_data_loader()

            for step, batch in enumerate(train_data_loader()):
                # to remove the train data that has been studyed.
                if step < global_step - pass_num: continue

                global_step += 1
                tokens, loss_mask, labels = batch

                loss_mask.stop_gradient = True
                labels.stop_gradient = True

                loss = 0.0
                for i in range(accumulate_steps):
                    start_index = i * args.micro_batch_size
                    end_index = start_index + args.micro_batch_size
                    timers('forward-compute').start()
                    with paddle.amp.auto_cast(
                            args.use_pure_fp16,
                            custom_black_list=[
                                "reduce_sum",
                                "c_softmax_with_cross_entropy",
                                "elementwise_div",
                            ],
                            level='O2'):
                        preds = model(tokens[start_index:end_index, :])
                        loss_mbs = criterion(
                            preds, labels[start_index:end_index, :],
                            loss_mask[start_index:end_index, :])
                    timers('forward-compute').stop()

                    if args.gate != "naive" and args.balance_loss_weight:
                        aux_loss_list = [
                            l.moe_mlp.gate.get_loss(clear=False)
                            for l in model.gpt.decoder.layers
                            if hasattr(l.moe_mlp, "gate")
                        ]
                        bal_loss = paddle.concat(aux_loss_list)
                        if bal_loss.dtype == paddle.float16:
                            bal_loss = paddle.cast(bal_loss,
                                                   dtype=paddle.float32)
                        bal_loss = bal_loss.mean()
                        loss_mbs += bal_loss * args.balance_loss_weight
                    loss_mbs = loss_mbs / accumulate_steps

                    timers('backward-compute').start()
                    if args.use_pure_fp16:
                        scaler.scale(loss_mbs).backward()
                    else:
                        loss_mbs.backward()
                    timers('backward-compute').stop()
                    loss = loss + loss_mbs

                timers('backward-params-all-reduce').start()
                all_reduce_parameters(gate_fused_tensors,
                                      hcg.get_expert_parallel_group())
                if args.sharding_degree == 1:
                    all_reduce_parameters(reduce_fused_tensors,
                                          hcg.get_data_parallel_group())
                timers('backward-params-all-reduce').stop()

                if args.use_pure_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                learning_rate = optimizer.get_lr()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.clear_grad()

                if global_step % args.logging_freq == 0:
                    avg_loss = loss.numpy()
                    speed = args.logging_freq / (time.time() - tic_train)
                    if args.gate != "naive" and args.balance_loss_weight:
                        bal_loss = bal_loss.numpy()
                        avg_loss -= bal_loss
                    else:
                        bal_loss = -1
                    logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %.9f, bal_loss: %.9f, speed: %.2f step/s, ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
                        % (global_step, epoch, step, avg_loss, bal_loss, speed,
                           speed * default_global_tokens_num, speed *
                           default_global_tokens_num / nranks, learning_rate))
                    log_writer.add_scalar("loss", float(loss), global_step)
                    log_writer.add_scalar("learning_rate", learning_rate,
                                          global_step)

                    tic_train = time.time()
                    timer_log(args.logging_freq)

                if (global_step % args.save_steps == 0
                        or global_step >= args.max_steps):
                    loss_scale = scaler._scale if args.use_pure_fp16 else None
                    save_checkpoint(args, global_step, model, optimizer,
                                    lr_scheduler, tokenizer, loss_scale,
                                    dp_rank, mp_rank, pp_rank, pass_num,
                                    file_id, epoch)
                    print(
                        "save checkpoint for step_{} successfully...loss_scale = {}"
                        .format(global_step, loss_scale))

                if global_step % args.eval_freq == 0:
                    # Since the valid data broardcast to all devices, we do evaluate on all device.
                    run_evaluate(args, valid_data_loader, model, criterion,
                                 args.eval_iters, log_writer, global_step,
                                 epoch, "valid")

                if global_step >= args.max_steps:
                    run_evaluate(args, test_data_loader, model, criterion,
                                 args.test_iters, log_writer, global_step,
                                 epoch, "test")
                    logger.info("The training process is complete.")
                    del train_data_loader
                    return

            # to record sum of the length of train_data_loader that has been read.
            pass_num += len(train_data_loader())
            del train_data_loader


if __name__ == "__main__":
    args = parse_args(MODEL_CLASSES)
    do_train(args)
