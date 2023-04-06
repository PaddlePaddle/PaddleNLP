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
from args import parse_args
from modeling import GPTLMHeadModel, GPTModel
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
    DygraphShardingOptimizer,
)
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)
from utils import (
    _rotate_checkpoints,
    all_gather,
    convert_example,
    is_dp_group_support_in_group_sharded_parallel,
    left_padding,
    optimizer_name_suffix,
    weight_name_suffix,
    wrap_sharding_2_3,
)
from visualdl import LogWriter

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import get_last_checkpoint
from paddlenlp.trainer.trainer import paddlenlp_load
from paddlenlp.trainer.training_args import default_logdir
from paddlenlp.transformers import (
    CosineAnnealingWithWarmupDecay,
    GPTChineseTokenizer,
    GPTTokenizer,
    LinearAnnealingWithWarmupDecay,
    PretrainedModel,
)
from paddlenlp.transformers.model_utils import _add_variant
from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "gpt": (GPTLMHeadModel, GPTTokenizer),
    "gpt-cn": (GPTLMHeadModel, GPTChineseTokenizer),
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


@paddle.no_grad()
def run_evaluate(args, data_loader, model, iter_steps, log_writer, global_step, task_name="valid"):
    model.eval()
    all_loss = []
    local_time = time.time()
    iter_step = 0
    iter_steps = sys.maxsize
    for eval_step, batch in enumerate(data_loader):
        with paddle.amp.auto_cast(
            args.use_pure_fp16,
            custom_black_list=["c_softmax_with_cross_entropy", "elementwise_div"],
            custom_white_list=["fused_attention", "fused_feedforward"],
            level="O2",
        ):
            loss = model(**batch)

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
            % (task_name, global_step, iter_step, average_loss, iter_step / (time.time() - local_time))
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
    # local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

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
    OPTIMIZER_NAME = "optimizer.pdopt"

    if args.mp_degree > 1 or args.sharding_degree > 1:
        WEIGHTS_NAME = _add_variant(WEIGHTS_NAME, weight_name_suffix())
        OPTIMIZER_NAME = _add_variant(OPTIMIZER_NAME, optimizer_name_suffix())
        # GPTLMHeadModel using old style save_pretrained
        # remove if CLASS using save_pretrained_v2
        logger.info(f"{WEIGHTS_NAME}, {OPTIMIZER_NAME}, {optimizer_name_suffix()}")
        if not GPTLMHeadModel.constructed_from_pretrained_config():
            GPTLMHeadModel.resource_files_names = {"model_state": WEIGHTS_NAME}

    model_config = model_class.pretrained_init_configuration[args.model_name_or_path]
    model_config["hidden_dropout_prob"] = args.hidden_dropout_prob
    model_config["attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
    model_config["num_partitions"] = args.mp_degree
    model_config["use_recompute"] = args.use_recompute
    model_config["enable_fuse_transformer"] = False
    model = GPTLMHeadModel(GPTModel(**model_config), pad_token_id=tokenizer.pad_token_id)
    # Create the critrion for the gpt model

    # Create the learning_rate sheduler and optimizer
    if args.decay_steps is None:
        args.decay_steps = args.max_steps
    assert args.warmup_rate <= 1.0 and args.warmup_rate >= 0.0, "warmup_rate should be in [0, 1]"
    args.warmup_steps = args.warmup_rate * args.max_steps

    lr_scheduler = None

    if args.lr_decay_style == "none":
        lr_scheduler = None
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_step=args.warmup_steps,
            decay_step=args.decay_steps,
            last_epoch=0,
        )
    elif args.lr_decay_style == "linear":
        lr_scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_step=args.warmup_steps,
            decay_step=args.decay_steps,
            last_epoch=0,
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

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        # decoder_start_token_id=model.config.bos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
    )

    logger.info("Loading train and dev dataset: %s" % args.dataset_name)
    train_set, dev_set = load_dataset(args.dataset_name, splits=["train_v1", "dev_v1"])
    logger.info("Loaded train and dev dataset: %s" % args.dataset_name)
    train_set = train_set.map(trans_func, lazy=True)

    # print(train_set[0])
    # exit()

    train_batch_sampler = DistributedBatchSampler(
        train_set,
        batch_size=args.micro_batch_size,
        shuffle=True,
        drop_last=True,
        num_replicas=data_world_size,
        rank=data_world_rank,
    )

    train_data_loader = paddle.io.DataLoader(
        dataset=train_set,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            max_length=args.max_seq_length,
            label_pad_token_id=tokenizer.pad_token_id,
        ),
        return_list=True,
    )
    dev_set = dev_set.map(trans_func, lazy=True)
    valid_batch_sampler = paddle.io.BatchSampler(dev_set, batch_size=args.micro_batch_size, shuffle=False)
    valid_data_loader = paddle.io.DataLoader(
        dataset=dev_set,
        batch_sampler=valid_batch_sampler,
        num_workers=0,
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            max_length=args.max_seq_length,
            label_pad_token_id=tokenizer.pad_token_id,
        ),
        return_list=True,
    )

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

    for epoch in range(sys.maxsize):
        for step, batch in enumerate(train_data_loader()):
            train_reader_cost += time.time() - reader_start
            train_start = time.time()

            if global_step >= args.max_steps:
                return

            if _globalstep_last_logged > 0:
                _globalstep_last_logged -= 1
                continue

            # In ParallelMode of DataParallel, 'no_sync' can be used for improving
            # performance of model by gradient accumulation.

            with paddle.amp.auto_cast(
                args.use_pure_fp16,
                custom_black_list=["c_softmax_with_cross_entropy", "elementwise_div"],
                custom_white_list=["fused_attention", "fused_feedforward"],
                level="O2",
            ):
                loss = model(**batch)
                # loss = criterion(preds, labels, loss_mask)

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
                    "global step %d, epoch: %d, loss: %.9f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, speed: %.2f step/s, ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
                    % (
                        global_step,
                        epoch,
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
                run_evaluate(args, valid_data_loader, model, args.eval_iters, log_writer, global_step, "valid")

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


def do_export(args):

    if args.do_export:
        from utils import merge_model_parallel

        last_checkpoint = get_last_checkpoint(args.output_dir)
        from modeling import GPTForGeneration

        from paddlenlp.transformers import GPTConfig

        _, tokenizer_class = MODEL_CLASSES[args.model_type]
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

        config = GPTConfig.from_pretrained(last_checkpoint)
        config.fuse_attention_qkv = True
        # config.max_predict_len = 8
        config.max_dec_len = 20
        config.eos_token_id = tokenizer.eos_token_id
        config.eol_token_id = tokenizer.eol_token_id
        config.pad_token_id = tokenizer.eos_token_id
        config.use_cache = True
        config.top_k = 1

        model = GPTForGeneration(config)
        missing_keys, unexpected_keys = model.set_state_dict(merge_model_parallel(last_checkpoint, config))
        print("missing_keys", missing_keys)
        print("unexpected_keys", unexpected_keys)

        # Switch to eval model
        model.eval()
        # Convert to static graph with specific input description
        input_text = ["Nice to meet", "Hello "]
        inputs = tokenizer(input_text)

        # input_ids = tokenizer.encode(input_text)['input_ids']
        inputs = tokenizer(input_text)
        inputs = left_padding(inputs, tokenizer.bos_token_id)
        input_ids = inputs["input_ids"]

        input_ids = paddle.to_tensor(input_ids, dtype="int64")
        ret = model(input_ids=input_ids)

        # ret =  model.generate(input_ids = data["input_ids"])
        for out_ids, in_txt in zip(ret[0].tolist(), input_text):
            print("==" * 30)
            print(in_txt + tokenizer.convert_ids_to_string(out_ids))

        model = paddle.jit.to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            ],
        )
        infer_path = os.path.join(args.output_dir, "infer", f"{args.dataset_name}")

        # Save converted static graph model
        paddle.jit.save(model, infer_path)
        # Also save tokenizer for inference usage
        tokenizer.save_pretrained(os.path.dirname(infer_path))


if __name__ == "__main__":
    args = parse_args(MODEL_CLASSES)
    args.do_export = True
    os.environ["softmax_mask_fuse_upper_triangle"] = "False"
    do_train(args)
    do_export(args)
