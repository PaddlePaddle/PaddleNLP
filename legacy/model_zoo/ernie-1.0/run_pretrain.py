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
import contextlib
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import yaml
from args import parse_args
from data_tools.dataset_utils import build_train_valid_test_datasets
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)
from visualdl import LogWriter

from paddlenlp.data import Stack
from paddlenlp.transformers import (
    ErnieConfig,
    ErnieForMaskedLM,
    ErnieForPretraining,
    ErniePretrainingCriterion,
    ErnieTokenizer,
    LinearAnnealingWithWarmupDecay,
)
from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "ernie": (ErnieConfig, ErnieForPretraining, ErniePretrainingCriterion, ErnieTokenizer),
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
    binary_head=True,
    current_step=0,
):

    train_valid_test_num_samples = [
        args.global_batch_size * args.max_steps,
        args.micro_batch_size * (args.max_steps // args.eval_freq + 1) * args.eval_iters * data_world_size,
        args.micro_batch_size * args.test_iters * data_world_size,
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
        binary_head=binary_head,
        max_seq_length_dec=None,
        dataset_type="ernie",
    )

    def print_dataset(data, mode="train"):
        logger.info(f"Sample data for {mode} mode")
        input_ids, segment_ids, input_mask, masked_lm_positions, masked_lm_labels, next_sentence_labels = data
        if tokenizer.pad_token_id in input_ids:
            input_ids = input_ids[0 : list(input_ids).index(tokenizer.pad_token_id)]
        logger.info(tokenizer._decode(input_ids))
        for pos, label in zip(masked_lm_positions, masked_lm_labels):
            input_ids[pos] = label
        logger.info(tokenizer._decode(input_ids))
        logger.info(tokenizer.convert_ids_to_tokens(masked_lm_labels))

    print_dataset(train_ds[0], "train")
    print_dataset(valid_ds[0], "valid")
    print_dataset(test_ds[0], "test")

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
        _, seq_length = out[0].shape
        size = sum(len(x[3]) for x in data)
        # masked_lm_positions
        # Organize as a 1D tensor for gather or use gather_nd
        if args.device == "npu":
            # For NPU device, fixed input sentence length, in
            # order to reduce the number of op compile.
            if size % 80 != 0:
                size += 80 - (size % 80)
        else:
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
            consumed_samples=consumed_samples,
        )
        data_loader = paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            worker_init_fn=None,
            collate_fn=_collate_data,
            return_list=False,
        )
        return data_loader

    train_dl = loader(train_ds, args.global_batch_size * current_step)
    valid_dl = loader(
        valid_ds, args.micro_batch_size * ((current_step + 1) // args.eval_freq) * args.eval_iters * data_world_size
    )
    test_dl = loader(test_ds, 0)

    return train_dl, valid_dl, test_dl


def get_train_data_file(args):
    if len(args.input_dir.split()) > 1:
        # weight-1 data-prefix-1 weight-2 data-prefix-2 ...
        return args.input_dir.split()
    else:
        files = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if (os.path.isfile(os.path.join(args.input_dir, f)) and ("_idx.npz" in str(f) or ".idx" in str(f)))
        ]
        files = [x.replace("_idx.npz", "") for x in files]
        files = [x.replace(".idx", "") for x in files]

        if len(files) > 1:
            ret = []
            logger.info("You are using multi-dataset:")
            for x in files:
                ret.append(1.0)
                ret.append(x)
                logger.info("    > set weight of %s dataset to 1.0" % x)
            return ret

    return files


def all_gather(v):
    if dist.get_world_size() <= 1:
        return v.item()
    ret = []
    dist.all_gather(ret, v)
    output_tensors = [t if len(t.shape) > 0 else t.reshape_([-1]) for t in ret]
    concat = paddle.concat(output_tensors, axis=0)
    return concat.mean().item()


@paddle.no_grad()
def run_evaluate(data_loader, model, criterion, iter_steps, log_writer, global_step, args, task_name="valid"):
    model.eval()

    if args.binary_head:
        loss_global = {
            "loss": paddle.to_tensor(0.0),
            "lm_loss": paddle.to_tensor(0.0),
            "sop_loss": paddle.to_tensor(0.0),
        }
    else:
        loss_global = {
            "loss": paddle.to_tensor(0.0),
        }

    local_time = time.time()

    for eval_step, batch in enumerate(data_loader):
        input_ids, segment_ids, input_mask, masked_lm_positions, masked_lm_labels, next_sentence_labels = batch
        with paddle.amp.auto_cast(
            args.use_amp,
            custom_white_list=[
                "softmax",
                "layer_norm",
                "gelu",
            ],
            custom_black_list=[
                "c_softmax_with_cross_entropy",
            ],
            level=args.fp16_opt_level,
        ):

            if args.binary_head:
                prediction_scores, seq_relationship_score = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    position_ids=None,
                    attention_mask=input_mask,
                    masked_positions=masked_lm_positions,
                )

                lm_loss, sop_loss = criterion(
                    prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels
                )
                loss = lm_loss + sop_loss
            else:
                prediction_scores = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    position_ids=None,
                    attention_mask=input_mask,
                    masked_positions=masked_lm_positions,
                )

                loss = criterion(prediction_scores, None, masked_lm_labels)

            loss_global["loss"] += loss.detach()
            if args.binary_head:
                loss_global["lm_loss"] += lm_loss.detach()
                loss_global["sop_loss"] += sop_loss.detach()

        if eval_step >= iter_steps - 1:
            log_info_dict = dict()
            for k, v in loss_global.items():
                log_info_dict[k] = all_gather(v) / iter_steps
                v.subtract_(v)
            if dist.get_rank() == 0:
                log_info_dict["samples_per_second"] = (
                    iter_steps * args.micro_batch_size * dist.get_world_size() / (time.time() - local_time)
                )
                loss_info = ", ".join(
                    ["{}: {:.6f}".format(k, log_info_dict[k]) for k in log_info_dict.keys() if k.endswith("loss")]
                )

                logger.info(
                    "%s step %d, batch: %d, %s, ips: %.0f seqs/s"
                    % (task_name, global_step, iter_steps, loss_info, log_info_dict["samples_per_second"])
                )

                for k, v in log_info_dict.items():
                    log_writer.add_scalar("%s/%s" % (task_name, k), v, global_step)

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
    assert (
        args.global_batch_size % micro_batch_size == 0
    ), "cannot do gradient accumulate, global_batch_size: {} micro_batch_size: {}".format(
        args.global_batch_size, micro_batch_size
    )
    accumulate_steps = bsz_per_dp // micro_batch_size
    assert (
        accumulate_steps >= 1
    ), f"Larger global_batch_size: {args.global_batch_size} is expect, micro_batch_size is {micro_batch_size}, but only {bsz_per_dp} on each card!"

    args.eval_iters *= accumulate_steps
    args.test_iters *= accumulate_steps

    args.accumulate_steps = accumulate_steps


def default_logdir() -> str:
    """
    Same default
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


def do_train(args):
    paddle.set_device(args.device)

    worker_index = paddle.distributed.get_rank()
    worker_num = paddle.distributed.get_world_size()

    if worker_num > 1:
        paddle.distributed.init_parallel_env()

    if args.dp_degree * args.sharding_degree == 1:
        args.dp_degree = worker_num
        args.sharding_degree = 1

    args_post_process(args, worker_num)

    logger.info("{:20}:{}".format("paddle commit id", paddle.version.commit))
    for arg in vars(args):
        logger.info("{:20}:{}".format(arg, getattr(args, arg)))

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {"dp_degree": args.dp_degree, "mp_degree": 1, "pp_degree": 1, "sharding_degree": 1}

    fleet.init(is_collective=True, strategy=strategy)

    # Create the random seed for the worker
    set_seed(args)

    assert (
        args.dp_degree * args.sharding_degree == worker_num
    ), "The product of degree num should be equal to worker_num."

    # Create log write,
    log_writer = None
    if worker_index == 0:
        log_writer = LogWriter(os.path.join(args.output_dir, default_logdir()))

    # Define the input data in the static mode
    config_class, model_class, criterion_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.binary_head is False:
        model_class = ErnieForMaskedLM

    pretrained_models_list = list(model_class.pretrained_init_configuration.keys())

    # load config in checkpoint
    global_step = 0
    checkpoint_dir = os.path.join(args.output_dir, "model_last")
    if os.path.exists(checkpoint_dir):
        if os.path.isfile(os.path.join(checkpoint_dir, "./config.yml")):
            with open(os.path.join(checkpoint_dir, "./config.yml"), "r") as f:
                step_config = yaml.load(f, Loader=yaml.FullLoader)
                assert (
                    step_config["global_batch_size"] == args.global_batch_size
                ), "Please ensure checkpoint global batch size is the same. Folder: {}".format(checkpoint_dir)
                global_step = step_config["global_step"]

    if args.model_name_or_path in pretrained_models_list and not args.continue_training:
        logger.warning(f"Your model {args.model_name_or_path} is training from scratch !!!")
        model_config = model_class.pretrained_init_configuration[args.model_name_or_path]
        model_config["hidden_dropout_prob"] = args.hidden_dropout_prob
        model_config["attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
        model_config["enable_recompute"] = args.use_recompute
        model = model_class(config_class(**model_config))
    else:
        logger.warning(f"Your model is continue training from {args.model_name_or_path}")
        model = model_class.from_pretrained(
            args.model_name_or_path,
            hidden_dropout_prob=args.hidden_dropout_prob,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob,
            enable_recompute=args.use_recompute,
        )

    criterion = criterion_class(with_nsp_loss=args.binary_head)

    if worker_index == 0:
        # log the model config and args
        model_config_json = json.dumps(model.config.to_dict(), ensure_ascii=False, indent=2)
        log_writer.add_text("model_config", model_config_json)
        args_dict = {"paddle commit id": str(paddle.version.commit)}
        for arg in vars(args):
            args_dict[arg] = str(getattr(args, arg))
        log_writer.add_text("args", json.dumps(args_dict, indent=2))

    # Create the learning_rate sheduler and optimizer
    if args.decay_steps is None:
        args.decay_steps = args.max_steps
    assert args.warmup_rate <= 1.0 and args.warmup_rate >= 0.0, "warmup_rate should be in [0, 1]"
    args.warmup_steps = args.warmup_rate * args.max_steps

    lr_scheduler = LinearAnnealingWithWarmupDecay(
        args.max_lr, args.min_lr, warmup_step=args.warmup_steps, decay_step=args.decay_steps, last_epoch=global_step
    )

    clip = None
    if args.grad_clip > 0:
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.grad_clip)

    decay_param = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
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
        multi_precision=args.use_amp,
    )

    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
        scaler = fleet.distributed_scaler(scaler)
        model = paddle.amp.decorate(models=model, level=args.fp16_opt_level)
    else:
        scaler = None

    if paddle.distributed.get_world_size() > 1:
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name_or_path)
    # Must extend chinese char for ErnieTokenizer
    tokenizer.extend_chinese_char()

    data_file = get_train_data_file(args)
    train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
        args,
        data_file,
        tokenizer,
        data_world_size=worker_num,
        data_world_rank=worker_index,
        max_seq_len=args.max_seq_len,
        binary_head=args.binary_head,
        current_step=global_step,
    )

    # load checkpoint vars
    if os.path.exists(checkpoint_dir):
        if os.path.isfile(os.path.join(checkpoint_dir, "./config.yml")):
            logger.info("Try to load checkpoint from %s " % checkpoint_dir)
            opt_path = os.path.join(checkpoint_dir, "model_state.pdopt")
            params_path = os.path.join(checkpoint_dir, "model_state.pdparams")

            if os.path.exists(opt_path):
                load_dict = paddle.load(params_path)
                model_dict = model.state_dict()
                if args.use_amp and args.fp16_opt_level == "O2":
                    for k, v in load_dict.items():
                        if k not in model_dict:
                            logger.warning(f"Checkpoint have too much keys: {k}")
                            continue
                        if "layer_norm" not in model_dict[k].name:
                            load_dict[k] = v.astype("float16")
                model.set_state_dict(load_dict)
                opt_dict = paddle.load(opt_path)
                optimizer.set_state_dict(opt_dict)
            else:
                logger.warning("No optimizer checkpoint file found in %s." % opt_path)
            if scaler is not None and os.path.isfile(os.path.join(checkpoint_dir, "scaler.pdparams")):
                scaler.load_state_dict(paddle.load(os.path.join(checkpoint_dir, "scaler.pdparams"), return_numpy=True))
            logger.info("Checkpoint loaded from global step: {}".format(global_step))

    if args.binary_head:
        loss_global = {
            "loss": paddle.to_tensor(0.0),
            "lm_loss": paddle.to_tensor(0.0),
            "sop_loss": paddle.to_tensor(0.0),
        }
    else:
        loss_global = {
            "loss": paddle.to_tensor(0.0),
        }

    tic_train = time.time()
    while True:
        # If not call valid_data_loader, the enumerate will call valid_data_loader
        # many times. and start a new random dataloader.
        valid_data_loader = valid_data_loader()
        test_data_loader = test_data_loader()

        # time count
        train_reader_cost = 0.0
        train_run_cost = 0.0
        tr_loss = paddle.to_tensor(0.0)
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

            input_ids, segment_ids, input_mask, masked_lm_positions, masked_lm_labels, next_sentence_labels = batch

            ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()

            if worker_num > 1 and (args.use_recompute or ((step + 1) % args.accumulate_steps != 0)):
                # grad acc, no_sync when (step + 1) % args.accumulate_steps != 0:
                # recompute, no_sync every where
                # recompute + grad_acc, no_sync every where
                ctx_manager = model.no_sync()
            else:
                ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()

            with ctx_manager:
                # For NPU device, using fp16 data type to execute `dropout` NPU op
                # can improve performance, which can change `Cast` CANN OP from
                # AICPU operator to AICore operator.
                with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=[
                        "softmax",
                        "layer_norm",
                        "gelu",
                        "dropout",
                    ],
                    custom_black_list=[
                        "c_softmax_with_cross_entropy",
                    ],
                    level=args.fp16_opt_level,
                ):

                    # Create the model for the ernie pretrain
                    if args.binary_head:
                        prediction_scores, seq_relationship_score = model(
                            input_ids=input_ids,
                            token_type_ids=segment_ids,
                            position_ids=None,
                            attention_mask=input_mask,
                            masked_positions=masked_lm_positions,
                        )
                        lm_loss, sop_loss = criterion(
                            prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels
                        )
                        loss = lm_loss + sop_loss
                    else:
                        prediction_scores = model(
                            input_ids=input_ids,
                            token_type_ids=segment_ids,
                            position_ids=None,
                            attention_mask=input_mask,
                            masked_positions=masked_lm_positions,
                        )

                        loss = criterion(prediction_scores, None, masked_lm_labels)

                if args.accumulate_steps >= 1:
                    tr_loss_step = loss / args.accumulate_steps
                else:
                    tr_loss_step = loss

                if args.use_amp:
                    scaler.scale(tr_loss_step).backward()
                else:
                    tr_loss_step.backward()

            tr_loss += tr_loss_step.detach()

            loss_global["loss"] += loss.detach()
            if args.binary_head:
                loss_global["lm_loss"] += lm_loss.detach()
                loss_global["sop_loss"] += sop_loss.detach()

            # Skip for accumulate_steps in global step
            if (step + 1) % args.accumulate_steps != 0:
                continue

            if worker_num > 1 and args.use_recompute:
                fused_allreduce_gradients(list(model.parameters()), None)

            if args.use_amp:
                scaler.minimize(optimizer, tr_loss)
            else:
                optimizer.step()

            if args.device == "npu":
                # For NPU device, set set_to_zero to False can improve
                # performance.
                optimizer.clear_grad(set_to_zero=False)
            else:
                optimizer.clear_grad()
            train_run_cost += time.time() - train_start
            tr_loss.subtract_(tr_loss)

            global_step += 1

            if global_step % args.logging_freq == 0:
                log_info_dict = dict()
                log_info_dict["global_step"] = global_step
                for k, v in loss_global.items():
                    log_info_dict[k] = all_gather(v) / args.logging_freq / args.accumulate_steps
                    v.subtract_(v)
                if worker_index == 0:
                    speed = args.logging_freq / (time.time() - tic_train)
                    log_info_dict["learning_rate"] = lr_scheduler.get_lr()
                    log_info_dict["steps_per_second"] = speed
                    log_info_dict["samples_per_second"] = speed * args.global_batch_size

                    for k, v in log_info_dict.items():
                        log_writer.add_scalar("train/%s" % k, v, global_step)

                    loss_info = ", ".join(
                        ["{}: {:.6f}".format(k, log_info_dict[k]) for k in log_info_dict.keys() if k.endswith("loss")]
                    )

                    common_loginfo = (
                        "global step %d, %s, speed: %.2f steps/s, ips: %.2f seqs/s, learning rate: %.5e"
                        % (
                            global_step,
                            loss_info,
                            speed,
                            log_info_dict["samples_per_second"],
                            log_info_dict["learning_rate"],
                        )
                    )

                    addition_info = ""
                    if args.use_amp:
                        amp_info = {
                            "loss_scaling": scaler._scale.item(),
                            "incr_count": scaler._incr_count,
                            "decr_count": scaler._decr_count,
                        }
                        addition_info = ", ".join("%s: %.2f" % (k, v) for k, v in amp_info.items())
                        for k, v in amp_info.items():
                            log_writer.add_scalar("amp/%s" % k, v, global_step)

                    logger.info(", ".join([common_loginfo, addition_info]))

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
                    task_name="valid",
                )
                tic_train = time.time()

            def save_ckpt(output_dir, model, tokenizer, optimizer, scaler, args, global_step):
                step_config = {
                    "model_name": args.model_name_or_path,
                    "global_step": global_step,
                    "global_batch_size": args.global_batch_size,
                    "consumed_samples": global_step * args.global_batch_size,
                }

                logger.debug("saving models to {}".format(output_dir))
                model_to_save = model._layers if isinstance(model, paddle.DataParallel) else model

                tokenizer.save_pretrained(output_dir)
                # added token is not need for downstream finetune tasks.
                added_token_path = os.path.join(output_dir, "added_tokens.json")
                if os.path.exists(added_token_path):
                    os.remove(added_token_path)

                model_to_save.save_model_config(output_dir)
                model_dict = model_to_save.state_dict()
                if scaler is not None:
                    paddle.save(scaler.state_dict(), os.path.join(output_dir, "scaler.pdparams"))
                    for k, v in model_dict.items():
                        if v.dtype is paddle.float16:
                            model_dict[k] = v.astype("float32")
                paddle.save(model_dict, os.path.join(output_dir, "model_state.pdparams"))
                paddle.save(optimizer.state_dict(), os.path.join(output_dir, "model_state.pdopt"))

                with open(os.path.join(output_dir, "config.yml"), "w") as f:
                    yaml.dump(step_config, f, encoding="utf-8", allow_unicode=True)

            if global_step % args.save_steps == 0 or global_step >= args.max_steps:
                output_dir = os.path.join(args.output_dir, "model_%d" % global_step)
                if worker_index == 0:
                    save_ckpt(output_dir, model, tokenizer, optimizer, scaler, args, global_step)

                if worker_num > 1:
                    paddle.distributed.barrier()
                tic_train = time.time()

            if global_step % args.checkpoint_steps == 0:
                output_dir = os.path.join(args.output_dir, "model_last")
                if worker_index == 0:
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    output_dir_bak = os.path.join(args.output_dir, "model_last_bak")
                    if os.path.exists(output_dir):
                        if os.path.exists(output_dir_bak):
                            shutil.rmtree(output_dir_bak)
                        shutil.move(output_dir, output_dir_bak)
                        os.mkdir(output_dir)
                    save_ckpt(output_dir, model, tokenizer, optimizer, scaler, args, global_step)

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
                    task_name="test",
                )
                del train_data_loader
                del valid_data_loader
                del test_data_loader
                return


if __name__ == "__main__":
    config = parse_args(MODEL_CLASSES)
    do_train(config)
