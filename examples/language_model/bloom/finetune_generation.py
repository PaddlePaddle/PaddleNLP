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
import sys
import time
from functools import partial

import paddle
from args import parse_args
from model_split_merge import split_model_parallel
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
    DygraphShardingOptimizer,
)
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)
from utils import (
    _rotate_checkpoints,
    all_gather,
    is_dp_group_support_in_group_sharded_parallel,
    set_hyrbid_parallel_seed,
    wrap_sharding_2_3,
)
from visualdl import LogWriter

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.datasets import load_dataset
from paddlenlp.layers import LoRAConfig, LoRAModel
from paddlenlp.trainer import get_last_checkpoint
from paddlenlp.trainer.trainer import paddlenlp_load
from paddlenlp.trainer.training_args import default_logdir
from paddlenlp.transformers import (  # AutoTokenizer,
    AutoTokenizer,
    BloomConfig,
    BloomForCausalLM,
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
    PretrainedModel,
)
from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.utils.log import logger


def convert_example(
    example,
    tokenizer,
    decoder_start_token_id,
    max_source_length,
    max_target_length,
    is_train=True,
):
    """Convert all examples into necessary features."""
    source = None
    title = None
    target = None
    if "source" in example and "title" in example:
        source = example["source"]
        if "title" in example.keys():
            title = example["title"]
    elif "context" in example and "answer" in example:
        source = example["context"]
        if "answer" in example.keys():
            title = example["answer"]
    else:
        assert False, "Source and title are not in the input dictionary, nor are context and answer."
    if "target" in example.keys():
        target = example["target"]
    elif "question" in example.keys():
        target = example["question"]

    source = "答案：" + title + tokenizer.eos_token + "上下文：" + source + "。" + tokenizer.eos_token + "在已知答案的前提下，问题："

    if target:
        target += tokenizer.eos_token

    # 1. tokenize input-tokens
    input_tokens = tokenizer.tokenize(source)[:max_source_length]

    # 2. tokenize output tokens
    output_tokens = tokenizer.tokenize(target)[:max_target_length]

    # 3. concat the inputs
    tokens = input_tokens + output_tokens

    labels = [tokenizer.pad_token_id] * len(input_tokens) + tokenizer.convert_tokens_to_ids(output_tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


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
            loss, _ = model(**batch)

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


def do_train():
    args = parse_args()
    paddle.set_device(args.device)

    nranks = paddle.distributed.get_world_size()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": args.dp_degree,
        "mp_degree": args.mp_degree,
        "pp_degree": 1,
        "sharding_degree": args.sharding_degree,
    }

    # Set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": args.seed}

    fleet.init(is_collective=True, strategy=strategy)

    # Obtain rank message of hybrid parallel
    hcg = fleet.get_hybrid_communicate_group()
    mp_rank = hcg.get_model_parallel_rank()
    dp_rank = hcg.get_data_parallel_rank()
    sharding_rank = hcg.get_sharding_parallel_rank()

    sharding_size = hcg.get_sharding_parallel_world_size()
    data_world_rank = dp_rank * sharding_size + sharding_rank
    data_world_size = args.dp_degree * args.sharding_degree

    # Seed control in hybrid parallel
    set_hyrbid_parallel_seed(args.seed, data_world_rank, mp_rank)

    default_global_tokens_num = args.global_batch_size * args.max_seq_length

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

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

    # Load the model
    config = BloomConfig.from_pretrained(args.model_name_or_path)
    WEIGHTS_NAME = "model_state.pdparams"
    OPTIMIZER_NAME = "model_state_mp_{:0>2d}_sharding_{:0>2d}.pdopt".format(mp_rank, sharding_rank)
    if args.mp_degree > 1:
        WEIGHTS_NAME = "model_state_mp_{:0>2d}.pdparams".format(mp_rank)
        BloomForCausalLM.resource_files_names = {"model_state": WEIGHTS_NAME}
        args.model_name_or_path = split_model_parallel(
            args.model_name_or_path, config, args.mp_degree, args.sharding_degree
        )
    config.mp_rank = mp_rank
    config.mp_degree = args.mp_degree

    config["hidden_dropout_prob"] = args.hidden_dropout_prob
    config["attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
    config["use_recompute"] = args.use_recompute
    config["use_pure_fp16"] = args.use_pure_fp16
    config["enable_fuse_transformer"] = False
    model = BloomForCausalLM.from_pretrained(args.model_name_or_path, config=config, low_cpu_mem_usage=True)

    if args.lora:
        # hardcode parameters for now
        lora_config = LoRAConfig(
            target_modules=[".*query_key_value.*"],
            r=4,
            lora_alpha=8,
            merge_weights=True,
        )
        model = LoRAModel(model, lora_config)
        model.mark_only_lora_as_trainable()
        model.print_trainable_parameters()

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
    )

    train_ds, dev_ds = load_dataset("dureader_qg", splits=("train", "dev"))
    # train_ds = train_ds.map(trans_func, lazy=True)
    train_ds = train_ds.map(trans_func)

    train_batch_sampler = DistributedBatchSampler(
        train_ds,
        batch_size=args.micro_batch_size,
        shuffle=True,
        drop_last=True,
        num_replicas=data_world_size,
        rank=data_world_rank,
    )

    train_data_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            max_length=args.max_seq_length,
            label_pad_token_id=tokenizer.pad_token_id,
            return_tensors="np",
        ),
        return_list=True,
    )
    dev_ds = dev_ds.map(trans_func, lazy=True)
    valid_batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=args.micro_batch_size, shuffle=False)
    valid_data_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_sampler=valid_batch_sampler,
        num_workers=0,
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            max_length=args.max_seq_length,
            label_pad_token_id=tokenizer.pad_token_id,
            return_tensors="np",
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

    model_path = "splits_mp_{:0>2d}_sharding_{:0>2d}".format(args.mp_degree, args.sharding_degree)
    for epoch in range(sys.maxsize):
        for step, batch in enumerate(train_data_loader):
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
                loss, _ = model(**batch)

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

                if config.mp_degree == 1 and config.pp_degree == 1:
                    output_dir = os.path.join(args.output_dir, str(global_step))
                else:
                    output_dir = os.path.join(args.output_dir, str(global_step), model_path)
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


if __name__ == "__main__":
    do_train()
