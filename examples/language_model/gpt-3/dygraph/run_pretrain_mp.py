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

import inspect
import os
import random
import sys
import time

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
    DygraphShardingOptimizer,
)
from paddle.distributed.fleet.meta_parallel import TensorParallel, get_rng_state_tracker
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)
from paddle.distributed.sharding import group_sharded_parallel
from visualdl import LogWriter

from paddlenlp.trainer import get_last_checkpoint
from paddlenlp.trainer.trainer import paddlenlp_load
from paddlenlp.trainer.training_args import default_logdir
from paddlenlp.transformers import GPTChineseTokenizer, GPTTokenizer, PretrainedModel
from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.utils.log import logger

try:
    from paddle.fluid.dygraph.parallel import sync_params_buffers
except ImportError:
    from paddle.distributed.parallel import sync_params_buffers


# to import data_tools
filepath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(filepath, "../"))
import lr  # noqa e402
from args import parse_args  # noqa e402
from dataset import create_pretrained_dataset  # noqa e402
from modeling import GPTForPretraining, GPTModel, GPTPretrainingCriterion  # noqa e402
from run_glue_mp import _rotate_checkpoints, all_gather  # noqa e402
from run_pretrain import get_train_data_file  # noqa e402

MODEL_CLASSES = {
    "gpt": (GPTForPretraining, GPTTokenizer),
    "gpt-cn": (GPTForPretraining, GPTChineseTokenizer),
}


def is_dp_group_support_in_group_sharded_parallel():
    return "dp_group" in set(inspect.signature(paddle.distributed.sharding.group_sharded_parallel).parameters.keys())


def set_hyrbid_parallel_seed(basic_seed, data_world_rank, mp_rank, pp_rank=0):
    assert args.device != "cpu"

    random.seed(basic_seed + data_world_rank)
    np.random.seed(basic_seed + data_world_rank)
    paddle.seed(basic_seed + data_world_rank)

    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = basic_seed + 123 + mp_rank * 10 + pp_rank * 1000
    global_seed = basic_seed + data_world_rank
    tracker = get_rng_state_tracker()
    tracker.add("global_seed", global_seed)
    tracker.add("local_seed", local_seed)


@paddle.no_grad()
def run_evaluate(args, data_loader, model, criterion, iter_steps, log_writer, global_step, task_name="valid"):
    model.eval()
    all_loss = []
    local_time = time.time()
    iter_step = 0
    for eval_step, batch in enumerate(data_loader):
        with paddle.amp.auto_cast(
            args.use_pure_fp16,
            custom_black_list=["c_softmax_with_cross_entropy", "elementwise_div"],
            custom_white_list=["fused_attention", "fused_feedforward"],
            level="O2",
        ):
            tokens, loss_mask, position_ids, labels = batch
            preds = model(tokens, position_ids)
            loss = criterion(preds, labels, loss_mask)

        all_loss.append(float(loss))

        if (eval_step + 1) % args.accumulate_steps == 0:
            iter_step += 1
        else:
            continue

        if iter_step >= iter_steps:
            break

    average_loss = sum(all_loss) / len(all_loss)
    v = paddle.to_tensor(average_loss).detach()
    average_loss = all_gather(v)

    if log_writer is not None:
        logger.info("--" * 30)
        logger.info(
            "%s step %d, batch: %d, loss: %f, speed: %.2f step/s"
            % (task_name, global_step, iter_steps, average_loss, iter_steps / (time.time() - local_time))
        )
        logger.info("--" * 30)
        log_writer.add_scalar(task_name + "_loss", average_loss, global_step)

    model.train()


def do_train(args):
    paddle.set_device(args.device)
    nranks = paddle.distributed.get_world_size()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": args.dp_degree,
        "mp_degree": args.mp_degree,
        "pp_degree": 1,
        "sharding_degree": args.sharding_degree,
    }

    # set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": args.seed}

    fleet.init(is_collective=True, strategy=strategy)

    # obtain rank message of hybrid parallel
    hcg = fleet.get_hybrid_communicate_group()
    # global_rank = hcg.get_global_rank()
    mp_rank = hcg.get_model_parallel_rank()
    dp_rank = hcg.get_data_parallel_rank()
    sharding_rank = hcg.get_sharding_parallel_rank()

    sharding_size = hcg.get_sharding_parallel_world_size()
    data_world_rank = dp_rank * sharding_size + sharding_rank
    data_world_size = args.dp_degree * args.sharding_degree
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

    # seed control in hybrid parallel
    set_hyrbid_parallel_seed(args.seed, data_world_rank, mp_rank)

    default_global_tokens_num = args.global_batch_size * args.max_seq_len

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name_or_path)

    # Detecting last checkpoint.
    last_checkpoint = None
    training_args = args
    training_args.overwrite_output_dir = False
    training_args.resume_from_checkpoint = True
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 1:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    global_step = 0
    if training_args.resume_from_checkpoint and last_checkpoint is not None:
        global_step = int(str(last_checkpoint).split("-")[-1])

    log_writer = None
    if dp_rank == 0 and mp_rank == 0 and sharding_rank == 0:
        log_writer_path = os.path.join(args.output_dir, default_logdir())
        log_writer = LogWriter(logdir=log_writer_path)

    WEIGHTS_NAME = "model_state.pdparams"
    OPTIMIZER_NAME = "model_state_mp_{:0>2d}_sharding_{:0>2d}.pdopt".format(mp_rank, sharding_rank)
    if args.mp_degree > 1:
        WEIGHTS_NAME = "model_state_mp_{:0>2d}.pdparams".format(mp_rank)
        GPTForPretraining.resource_files_names = {"model_state": WEIGHTS_NAME}

    model_config = model_class.pretrained_init_configuration[args.model_name_or_path]
    model_config["hidden_dropout_prob"] = args.hidden_dropout_prob
    model_config["attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
    model_config["num_partitions"] = args.mp_degree
    model_config["use_recompute"] = args.use_recompute
    model_config["fuse"] = False
    model = GPTForPretraining(GPTModel(**model_config))
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
            decay_step=args.decay_steps,
            last_epoch=global_step,
        )

    clip = None
    if args.grad_clip > 0:
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.grad_clip)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]

    if args.sharding_stage == 1 and args.sharding_degree > 1:
        optimizer = DygraphShardingOptimizer(
            hcg=fleet.get_hybrid_communicate_group(),
            user_defined_strategy=strategy,
            params=model.parameters(),
            inner_optimizer_class=paddle.optimizer.AdamW,
            learning_rate=lr_scheduler if lr_scheduler is not None else args.max_lr,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_epsilon,
            weight_decay=args.weight_decay,
            grad_clip=clip,
            apply_decay_param_fun=lambda x: x in decay_params,
        )
    else:
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler if lr_scheduler is not None else args.max_lr,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_epsilon,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            grad_clip=clip,
            apply_decay_param_fun=lambda x: x in decay_params,
            # TODO: remove 'multi_precision' in definition of optimizer
            # and add it to 'paddle.amp.decorate'
            multi_precision=args.use_pure_fp16,
        )

    if args.use_pure_fp16:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
        # level O2 means converting the network to FP16
        if args.sharding_stage not in [2, 3]:
            scaler = fleet.distributed_scaler(scaler)
        model = paddle.amp.decorate(models=model, level="O2")

    if training_args.resume_from_checkpoint and last_checkpoint is not None:
        model.set_state_dict(
            paddle.load(os.path.join(last_checkpoint, model.resource_files_names["model_state"]), return_numpy=True)
        )
    # wrap sharding stage2/3 and add collective group
    # TODO(Baibaifan): combine ShardingStage1/2/3 and fleet.distributed_model in feature
    if args.sharding_stage in [2, 3] and args.sharding_degree > 1:
        scaler = scaler if args.use_pure_fp16 else None
        model, optimizer, scaler = wrap_sharding_2_3(model, optimizer, scaler, args)

    elif paddle.distributed.get_world_size() > 1:
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

    files = get_train_data_file(args)
    train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
        args,
        files,
        local_rank=local_rank,
        data_world_size=data_world_size,
        data_world_rank=data_world_rank,
        max_seq_len=args.max_seq_len,
        eos_id=tokenizer.eos_token_id,
        current_step=global_step,
    )
    # Bug fix, if not call valid_data_loader, the enumerate will call valid_data_loader
    # many times. and start a new random dataloader.
    valid_data_loader = valid_data_loader()
    test_data_loader = test_data_loader()

    global_step = 0
    # time count
    train_reader_cost = 0.0
    train_run_cost = 0.0
    reader_start = time.time()

    if training_args.resume_from_checkpoint and last_checkpoint is not None:
        optimizer.set_state_dict(
            paddlenlp_load(
                os.path.join(last_checkpoint, OPTIMIZER_NAME),
                return_numpy=True,
            )
        )
        global_step = int(str(last_checkpoint).split("-")[-1])

    _globalstep_last_logged = global_step
    if isinstance(train_data_loader.batch_sampler, DistributedBatchSampler):
        _globalstep_last_logged = 0

    tr_loss = paddle.to_tensor(0.0)
    loss_global = paddle.to_tensor(0.0)

    for step, batch in enumerate(train_data_loader()):
        train_reader_cost += time.time() - reader_start
        train_start = time.time()

        if _globalstep_last_logged > 0:
            _globalstep_last_logged -= 1
            continue

        tokens, loss_mask, position_ids, labels = batch

        # In ParallelMode of DataParallel, 'no_sync' can be used for improving
        # performance of model by gradient accumulation.

        with paddle.amp.auto_cast(
            args.use_pure_fp16,
            custom_black_list=["c_softmax_with_cross_entropy", "elementwise_div"],
            custom_white_list=["fused_attention", "fused_feedforward"],
            level="O2",
        ):
            preds = model(tokens, position_ids)
            loss = criterion(preds, labels, loss_mask)

        if args.accumulate_steps > 1:
            tr_loss_step = loss / args.accumulate_steps
        else:
            tr_loss_step = loss

        if args.use_pure_fp16:
            scaler.scale(tr_loss_step).backward()
        else:
            tr_loss_step.backward()

        tr_loss_step = tr_loss_step.detach()

        tr_loss += tr_loss_step
        loss_global += loss.detach()

        # Skip for accumulate_steps in global step
        if (step + 1) % args.accumulate_steps != 0:
            continue

        if args.sharding_degree > 1 and args.sharding_stage in [2, 3]:
            if args.dp_degree > 1 and not is_dp_group_support_in_group_sharded_parallel():
                fused_allreduce_gradients(model.parameters(), fleet.get_hybrid_communicate_group())

        if args.use_pure_fp16:
            # scaler.minimize(optimizer, tr_loss)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.clear_grad()
        tr_loss.subtract_(tr_loss)

        global_step += 1

        # Sync for profile time, delete it may be a little faster
        # paddle.device.cuda.synchronize()
        train_run_cost += time.time() - train_start

        if global_step % args.logging_freq == 0:
            avg_loss = all_gather(loss_global) / args.logging_freq / args.accumulate_steps
            loss_global.subtract_(loss_global)
            speed = args.logging_freq / (train_reader_cost + train_run_cost)
            avg_reader_cost = train_reader_cost / args.logging_freq

            logger.info(
                "global step %d,  loss: %.9f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, speed: %.2f step/s, ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
                % (
                    global_step,
                    avg_loss,
                    avg_reader_cost,
                    1.0 / speed,
                    speed,
                    speed * default_global_tokens_num,
                    speed * default_global_tokens_num / nranks,
                    optimizer.get_lr(),
                )
            )
            if log_writer is not None:
                log_writer.add_scalar("loss", float(loss), global_step)
                log_writer.add_scalar("learning_rate", optimizer.get_lr(), global_step)

            # tic_train = time.time()
            train_reader_cost = 0.0
            train_run_cost = 0.0

        if lr_scheduler is not None:
            lr_scheduler.step()

        if global_step % args.eval_freq == 0:
            # Since the valid data broardcast to all devices, we do evaluate on all device.
            run_evaluate(args, valid_data_loader, model, criterion, args.eval_iters, log_writer, global_step, "valid")

        # TODO: 1. merge paramters while saving model. 2. ensure that the model is saved and loaded correctly
        # only dp_rank = 0 save model
        if (global_step % args.save_steps == 0 or global_step >= args.max_steps) and dp_rank == 0:

            model_to_save = (
                model._layers
                if paddle.distributed.get_world_size() > 1 and args.sharding_stage not in [2, 3]
                else model
            )

            if args.sharding_stage == 3:
                # If parameter need to convert to cpu, please add convert2cpu=True
                model_to_save.get_all_parameters(convert2cpu=True)

            while hasattr(model_to_save, "_layers") or hasattr(model_to_save, "_layer"):
                if hasattr(model_to_save, "_layers"):
                    model_to_save = model_to_save._layers
                else:
                    model_to_save = model_to_save._layer

            output_dir = os.path.join(args.output_dir, "checkpoint-%d" % global_step)
            os.makedirs(output_dir, exist_ok=True)
            logger.info("Save model to %s" % output_dir)

            # tokenizer only need to save on one node
            if mp_rank == 0 and sharding_rank == 0 and dp_rank == 0:
                tokenizer.save_pretrained(output_dir)

            # paramerters is the same in sharding group
            if sharding_rank == 0 and dp_rank == 0:
                if isinstance(model_to_save, PretrainedModel):
                    model_to_save.save_pretrained(output_dir)
                else:
                    logger.info("Trainer.model is not a `PretrainedModel`, only saving its state dict.")
                    state_dict = model_to_save.state_dict()
                    paddle.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # ckpt optimizer weight should save on echo sharding rank
            if dp_rank == 0:
                paddle.save(
                    optimizer.state_dict(),
                    os.path.join(
                        output_dir,
                        OPTIMIZER_NAME,
                    ),
                )

            if mp_rank == 0 and sharding_rank == 0 and dp_rank == 0:
                _rotate_checkpoints(args.save_total_limit, output_dir=args.output_dir)

        if global_step >= args.max_steps:
            return

        reader_start = time.time()


def wrap_sharding_2_3(model, optimizer, scaler, dist_config):
    """_summary_

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        scaler (_type_): _description_
        dist_config (_type_): _description_

    Returns:
        _type_: _description_
    """
    # group = fleet.get_hybrid_communicate_group().get_sharding_parallel_group()
    # level = "p_g_os" if dist_config.sharding_stage == 3 else "os_g"
    # return group_sharded_parallel(
    #     model=model, optimizer=optimizer, level=level, scaler=scaler, group=group, offload=dist_config.sharding_offload,

    # )

    hcg = fleet.get_hybrid_communicate_group()
    dp_group = hcg.get_data_parallel_group()
    sharding_group = hcg.get_sharding_parallel_group()

    # sync params (broadcast) buffers in dp group
    if (
        not is_dp_group_support_in_group_sharded_parallel()
        and dist_config.dp_degree > 1
        and dist_config.sharding_stage == 2
    ):
        sync_params_buffers(model, comm_group=dp_group, src_rank=dp_group.ranks[0])

    if dist_config.dp_degree > 1 and dist_config.sharding_stage == 3:
        sync_params_buffers(model, comm_group=dp_group, src_rank=dp_group.ranks[0])

    if dist_config.mp_degree > 1:
        assert dist_config.sharding_stage == 2, "only support mp + sharding stage2 hybrid parallel now."
        model = TensorParallel(model, hcg, strategy=None)

    level = "p_g_os" if dist_config.sharding_stage == 3 else "os_g"
    # origin_model = model

    extra_kwargs = {}
    if is_dp_group_support_in_group_sharded_parallel():
        extra_kwargs["dp_group"] = dp_group if dp_group.nranks > 1 else None
        extra_kwargs["exclude_layer"] = ["GroupNorm"]

    model, optimizer, scaler = group_sharded_parallel(
        model=model,
        optimizer=optimizer,
        level=level,
        scaler=scaler,
        group=sharding_group,
        offload=dist_config.sharding_offload,
        # dp_group=dp_group if dp_group.nranks > 1 else None,
        **extra_kwargs,
    )

    # if dist_config.sharding.reduce_overlap:
    #     model._set_reduce_overlap(dist_config.sharding.reduce_overlap)

    # if dist_config.sharding.broadcast_overlap:
    #     optimizer._set_broadcast_overlap(
    #         dist_config.sharding.broadcast_overlap,
    #         layers=origin_model,
    #         num_groups=2)

    return model, optimizer, scaler


if __name__ == "__main__":
    args = parse_args(MODEL_CLASSES)
    do_train(args)
