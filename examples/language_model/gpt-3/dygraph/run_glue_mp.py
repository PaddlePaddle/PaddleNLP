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

import os
import random
import sys
import time
from functools import partial

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
    DygraphShardingOptimizer,
)
from paddle.distributed.fleet.meta_parallel import TensorParallel, get_rng_state_tracker
from paddle.distributed.sharding import group_sharded_parallel

try:
    from paddle.fluid.dygraph.parallel import sync_params_buffers
except ImportError:
    from paddle.distributed.parallel import sync_params_buffers

from paddle.metric import Accuracy
from visualdl import LogWriter

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.transformers import GPTChineseTokenizer, GPTTokenizer
from paddlenlp.utils.log import logger

# to import data_tools
filepath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(filepath, "../"))
import lr  # noqa e402
from args import parse_args  # noqa e402
from dataset import create_pretrained_dataset  # noqa e402
from modeling import (  # noqa e402
    GPTForSequenceClassification,
    GPTModel,
    GPTPretrainingCriterion,
)

# from run_pretrain import get_train_data_file  # noqa e402


METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "mrpc": AccuracyAndF1,
    "sts-b": PearsonAndSpearman,
    "qqp": AccuracyAndF1,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
}

MODEL_CLASSES = {
    "gpt": (GPTForSequenceClassification, GPTTokenizer),
    "gpt-cn": (GPTForSequenceClassification, GPTChineseTokenizer),
}


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


def convert_example(example, tokenizer, label_list, max_seq_length=512, is_test=False):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example["labels"]
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    if (int(is_test) + len(example)) == 2:
        example = tokenizer(
            example["sentence"], padding="max_length", max_length=max_seq_length, return_token_type_ids=False
        )
    else:
        example = tokenizer(
            example["sentence1"],
            text_pair=example["sentence2"],
            padding=True,
            max_length=max_seq_length,
            return_token_type_ids=False,
        )

    if not is_test:
        example["labels"] = label

    return example


@paddle.no_grad()
def run_evaluate(args, data_loader, model, log_writer, global_step, metric, task_name="valid"):
    model.eval()
    metric.reset()
    local_time = time.time()
    iter_steps = sys.maxsize
    all_loss = []
    for eval_step, batch in enumerate(data_loader):
        with paddle.amp.auto_cast(
            args.use_pure_fp16,
            custom_black_list=["c_softmax_with_cross_entropy", "elementwise_div"],
            custom_white_list=["fused_attention", "fused_feedforward"],
            level="O2",
        ):
            loss = model(**batch, return_dict=True)
            if isinstance(loss, dict):
                logits = loss["logits"]
                loss = loss["loss"]
                correct = metric.compute(logits.detach(), batch["labels"].detach())
                metric.update(correct)

            all_loss.append(float(loss))

        if eval_step >= iter_steps - 1:
            break

    res = metric.accumulate()

    average_loss = sum(all_loss) / len(all_loss)

    logger.info("--" * 30)
    if isinstance(metric, AccuracyAndF1):
        logger.info(
            "eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, "
            % (average_loss, res[0], res[1], res[2], res[3], res[4]),
        )
    elif isinstance(metric, Mcc):
        logger.info(
            "eval loss: %f, mcc: %s, " % (average_loss, res[0]),
        )
    elif isinstance(metric, PearsonAndSpearman):
        logger.info(
            "eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s, "
            % (average_loss, res[0], res[1], res[2]),
        )
    else:
        logger.info("eval loss: %f, acc: %s, " % (average_loss, res))

    logger.info("--" * 30)
    logger.info(
        "%s step %d, batch: %d, loss: %f, speed: %.2f step/s"
        % (task_name, global_step, eval_step + 1, average_loss, (eval_step + 1) / (time.time() - local_time))
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

    accumulate_steps = args.local_batch_size // args.micro_batch_size
    strategy.pipeline_configs = {"accumulate_steps": accumulate_steps, "micro_batch_size": args.micro_batch_size}

    # set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": args.seed}

    fleet.init(is_collective=True, strategy=strategy)

    # obtain rank message of hybrid parallel
    hcg = fleet.get_hybrid_communicate_group()
    global_rank = hcg.get_global_rank()
    mp_rank = hcg.get_model_parallel_rank()
    dp_rank = hcg.get_data_parallel_rank()
    sharding_rank = hcg.get_sharding_parallel_rank()

    sharding_size = hcg.get_sharding_parallel_world_size()
    data_world_rank = dp_rank * sharding_size + sharding_rank
    data_world_size = args.dp_degree * args.sharding_degree
    # local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

    # seed control in hybrid parallel
    set_hyrbid_parallel_seed(args.seed, data_world_rank, mp_rank)

    default_global_tokens_num = args.global_batch_size * args.max_seq_len

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    # Define log writer
    log_writer_path = os.path.join(
        args.output_dir,
        "train_log",
        "{}_globalbsz_{}_pure_fp16_{}_card_{}".format(
            args.model_name_or_path, args.global_batch_size, args.use_pure_fp16, global_rank
        ).lower(),
    )

    if os.path.exists(log_writer_path):
        import shutil

        shutil.rmtree(log_writer_path)

    log_writer = LogWriter(log_writer_path)

    args.task_name = args.task_name.lower()
    metric_class = METRIC_CLASSES[args.task_name]

    train_ds = load_dataset("glue", args.task_name, splits="train")
    tokenizer = GPTTokenizer.from_pretrained(args.model_name_or_path)

    trans_func = partial(
        convert_example, tokenizer=tokenizer, label_list=train_ds.label_list, max_seq_length=args.max_seq_length
    )
    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_replicas=data_world_size,
        rank=data_world_rank,
    )

    if args.task_name == "mnli":
        dev_ds = load_dataset("glue", args.task_name, splits=["dev_matched"])
    else:
        dev_ds = load_dataset("glue", args.task_name, splits="dev")

    dev_ds = dev_ds.map(trans_func, lazy=True)
    valid_batch_sampler = paddle.io.BatchSampler(
        dev_ds,
        batch_size=args.micro_batch_size,
        shuffle=False,
    )

    print("ds", len(train_ds))
    train_data_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        return_list=True,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=args.max_seq_length),
    )

    valid_data_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_sampler=valid_batch_sampler,
        num_workers=0,
        return_list=True,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=args.max_seq_length),
    )

    num_classes = 1 if train_ds.label_list is None else len(train_ds.label_list)

    # model = GPTForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_classes)
    pretrained_models_list = list(model_class.pretrained_init_configuration.keys())

    model = GPTForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        num_partitions=args.mp_degree,
        use_recompute=args.use_recompute,
        fuse=True,
        num_labels=num_classes,
    )

    metric = metric_class()

    if args.decay_steps is None:
        args.decay_steps = args.max_steps
    warmup_step = args.warmup_rate * args.decay_steps

    lr_scheduler = None
    if args.lr_decay_style == "none":
        lr_scheduler = None
    elif args.lr_decay_style == "cosine":
        lr_scheduler = lr.CosineAnnealingWithWarmupDecay(
            max_lr=args.max_lr, min_lr=args.min_lr, warmup_step=warmup_step, decay_step=args.decay_steps
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

    # wrap sharding stage2/3 and add collective group
    # TODO(Baibaifan): combine ShardingStage1/2/3 and fleet.distributed_model in feature
    if args.sharding_stage in [2, 3] and args.sharding_degree > 1:
        scaler = scaler if args.use_pure_fp16 else None
        model, optimizer, scaler = wrap_sharding_2_3(model, optimizer, scaler, args)

    elif paddle.distributed.get_world_size() > 1:
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

    if args.model_name_or_path not in pretrained_models_list:
        logger.info("Try to load checkpoint from %s " % args.model_name_or_path)
        opt_path = os.path.join(args.model_name_or_path, "model_state.pdopt")
        if os.path.exists(opt_path):
            opt_dict = paddle.load(opt_path)
            optimizer.set_state_dict(opt_dict)
        else:
            logger.warning("No optimizer checkpoint file found in %s." % opt_path)

    global_step = 0

    # time count
    train_reader_cost = 0.0
    train_run_cost = 0.0
    reader_start = time.time()

    for epoch in range(sys.maxsize):
        train_data_loader.batch_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_data_loader):
            train_reader_cost += time.time() - reader_start
            train_start = time.time()
            global_step += 1

            # In ParallelMode of DataParallel, 'no_sync' can be used for improving
            # performance of model by gradient accumulation.
            loss = 0.0
            for i in range(accumulate_steps):
                with paddle.amp.auto_cast(
                    args.use_pure_fp16,
                    custom_black_list=["c_softmax_with_cross_entropy", "elementwise_div"],
                    custom_white_list=["fused_attention", "fused_feedforward"],
                    level="O2",
                ):
                    loss_mbs = model(**batch)
                    if isinstance(loss_mbs, tuple):
                        loss_mbs = loss_mbs[0]

                loss_mbs = loss_mbs / accumulate_steps
                if args.use_pure_fp16:
                    scaler.scale(loss_mbs).backward()
                else:
                    loss_mbs.backward()
                loss = loss + loss_mbs

            if args.use_pure_fp16:
                if args.sharding_stage in [2, 3]:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    scaler.minimize(optimizer, loss)
            else:
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            optimizer.clear_grad()

            # Sync for profile time, delete it may be a little faster
            paddle.device.cuda.synchronize()
            train_run_cost += time.time() - train_start

            if global_step % args.logging_freq == 0:
                avg_loss = loss.numpy()
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
                log_writer.add_scalar("loss", float(loss), global_step)
                log_writer.add_scalar("learning_rate", optimizer.get_lr(), global_step)

                # tic_train = time.time()
                train_reader_cost = 0.0
                train_run_cost = 0.0

            if global_step % args.eval_freq == 0:
                # Since the valid data broardcast to all devices, we do evaluate on all device.
                run_evaluate(args, valid_data_loader, model, log_writer, global_step, metric, "valid")

            # TODO: 1. merge paramters while saving model. 2. ensure that the model is saved and loaded correctly
            # only dp_rank = 0 save model
            if (global_step % args.save_steps == 0 or global_step >= args.max_steps) and dp_rank == 0:

                model_to_save = (
                    model._layers
                    if paddle.distributed.get_world_size() > 1 and args.sharding_stage not in [2, 3]
                    else model
                )
                output_dir = os.path.join(args.output_dir, "step_%d" % global_step)
                os.makedirs(output_dir, exist_ok=True)

                logger.info("Save model to %s" % output_dir)

                if args.sharding_stage == 3:
                    # If parameter need to convert to cpu, please add convert2cpu=True
                    model_to_save.get_all_parameters(convert2cpu=True)

                if mp_rank == 0 and sharding_rank == 0:
                    tokenizer.save_pretrained(output_dir)
                model_to_save.save_pretrained(output_dir)
                paddle.save(
                    optimizer.state_dict(),
                    os.path.join(
                        output_dir,
                        "model_state_mp_{:0>2d}_sharding_{:0>2d}.pdopt".format(mp_rank, sharding_rank),
                    ),
                )

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

    if dist_config.dp_degree > 1 and dist_config.sharding_stage == 3:
        sync_params_buffers(model, comm_group=dp_group, src_rank=dp_group.ranks[0])

    if dist_config.mp_degree > 1:
        assert dist_config.sharding_stage == 2, "only support mp + sharding stage2 hybrid parallel now."
        model = TensorParallel(model, hcg, strategy=None)

    level = "p_g_os" if dist_config.sharding_stage == 3 else "os_g"
    # origin_model = model
    model, optimizer, scaler = group_sharded_parallel(
        model=model,
        optimizer=optimizer,
        level=level,
        scaler=scaler,
        group=sharding_group,
        offload=dist_config.sharding_offload,
        dp_group=dp_group if dp_group.nranks > 1 else None,
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
