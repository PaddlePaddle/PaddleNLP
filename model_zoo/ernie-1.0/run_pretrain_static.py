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
ERNIE pretraining scripts for paddlepaddle static graph mode.
"""
import argparse
import math
import os
import random
import time
import yaml
import shutil
import collections

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_optimizers.sharding.utils import save_persistables
from paddle.io import DataLoader, Dataset
from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.utils.log import logger

from paddlenlp.transformers import ErnieModel, ErnieForPretraining, ErniePretrainingCriterion, ErnieTokenizer

from paddlenlp.transformers import CosineAnnealingWithWarmupDecay, LinearAnnealingWithWarmupDecay

from paddlenlp.ops import guard, Topology, get_rng_state_tracker
from paddlenlp.utils.log import logger
import paddlenlp.ops as ops
from visualdl import LogWriter

from args import parse_args
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
    places,
    data_holders,
    binary_head=True,
    current_step=0,
):

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
        binary_head=binary_head,
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
        data_loader = paddle.io.DataLoader(dataset=dataset,
                                           places=places,
                                           feed_list=data_holders,
                                           batch_sampler=batch_sampler,
                                           num_workers=args.num_workers,
                                           worker_init_fn=None,
                                           collate_fn=_collate_data,
                                           return_list=False)
        return data_loader

    train_dl = loader(train_ds, args.global_batch_size * current_step)
    valid_dl = loader(
        valid_ds,
        args.micro_batch_size * ((current_step + 1) // args.eval_freq) *
        args.eval_iters * data_world_size)
    test_dl = loader(test_ds, 0)

    return train_dl, valid_dl, test_dl


def create_data_holder(args=None):
    input_ids = paddle.static.data(name="input_ids",
                                   shape=[-1, -1],
                                   dtype="int64")
    segment_ids = paddle.static.data(name="segment_ids",
                                     shape=[-1, -1],
                                     dtype="int64")
    input_mask = paddle.static.data(name="input_mask",
                                    shape=[-1, 1, 1, -1],
                                    dtype="float32")
    masked_lm_positions = paddle.static.data(name="masked_lm_positions",
                                             shape=[-1],
                                             dtype="int32")
    masked_lm_labels = paddle.static.data(name="masked_lm_labels",
                                          shape=[-1, 1],
                                          dtype="int64")

    next_sentence_labels = paddle.static.data(name="next_sentence_labels",
                                              shape=[-1, 1],
                                              dtype="int64")

    return [
        input_ids, segment_ids, input_mask, masked_lm_positions,
        masked_lm_labels, next_sentence_labels
    ]


def dist_optimizer(args, topo):
    default_global_batch_size = topo.data_info.size * args.micro_batch_size
    if args.global_batch_size is None:
        args.global_batch_size = default_global_batch_size

    bsz_per_dp = args.global_batch_size // topo.data_info.size
    micro_batch_size = args.micro_batch_size
    assert args.global_batch_size % micro_batch_size == 0, \
        "cannot do gradient accumulate, global_batch_size: {} micro_batch_size: {}".format(
        args.global_batch_size, micro_batch_size)
    accumulate_steps = bsz_per_dp // micro_batch_size

    exec_strategy = paddle.static.ExecutionStrategy()
    exec_strategy.num_threads = 1
    exec_strategy.num_iteration_per_drop_scope = 10000

    build_strategy = paddle.static.BuildStrategy()
    #build_strategy.enable_sequential_execution = True # for profile
    build_strategy.fuse_broadcast_ops = True
    build_strategy.enable_inplace = True
    build_strategy.enable_addto = args.enable_addto

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.build_strategy = build_strategy
    dist_strategy.nccl_comm_num = 3
    dist_strategy.fuse_grad_size_in_MB = 16

    dist_strategy.recompute = args.use_recompute
    dist_strategy.pipeline = args.pp_degree > 1

    if args.pp_degree <= 1 and args.sharding_degree <= 1 and accumulate_steps > 1:
        dist_strategy.gradient_merge = True
        dist_strategy.gradient_merge_configs = {'k_steps': accumulate_steps}
    args.eval_iters *= accumulate_steps
    args.test_iters *= accumulate_steps

    if args.use_amp:
        dist_strategy.amp = True
        dist_strategy.amp_configs = {
            "custom_white_list": [
                'softmax',
                'layer_norm',
                'gelu',
            ],
            "custom_black_list": ['c_softmax_with_cross_entropy'],
            "init_loss_scaling": 32768,
            "use_dynamic_loss_scaling": True,
        }
    if args.use_sharding:
        dist_strategy.sharding = True
        dist_strategy.sharding_configs = {
            "segment_broadcast_MB":
            32,
            "sharding_degree":
            args.sharding_degree,
            "mp_degree":
            args.mp_degree,
            "pp_degree":
            args.pp_degree,
            "dp_degree":
            args.dp_degree,
            "gradient_merge_acc_step":
            accumulate_steps if args.sharding_degree > 1 else 1,
            "optimize_offload":
            False,
        }
    if args.pp_degree > 1:
        dist_strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_micro_batch_size": micro_batch_size,
            "accumulate_steps": accumulate_steps,
        }

    args.accumulate_steps = accumulate_steps
    return dist_strategy


def get_train_data_file(args):
    if len(args.input_dir.split()) > 1:
        # weight-1 data-prefix-1 weight-2 data-prefix-2 ...
        return args.input_dir.split()
    else:
        files = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if (os.path.isfile(os.path.join(args.input_dir, f))
                and "_idx.npz" in str(f))
        ]
        files = [x.replace("_idx.npz", "") for x in files]

        if len(files) > 1:
            ret = []
            logger.info("You are using multi-dataset:")
            for x in files:
                ret.append(1.0)
                ret.append(x)
                logger.info("    > set weight of %s dataset to 1.0" % x)
            return ret

    return files


def run_evaluate(data_loader,
                 exe,
                 program,
                 iter_steps,
                 log_writer,
                 global_step,
                 args,
                 is_last,
                 eval_fetch,
                 task_name="valid"):
    all_ret = collections.defaultdict(list)
    average_ret = collections.defaultdict(float)

    local_time = time.time()

    for eval_step, batch in enumerate(data_loader):
        ret = exe.run(program, feed=batch, fetch_list=list(eval_fetch.values()))
        if is_last:
            for k, v in zip(list(eval_fetch.keys()), ret):
                all_ret[k].append(float(v[0]))

        if eval_step >= iter_steps - 1:
            if not is_last:
                break

            for k in list(eval_fetch.keys()):
                average_ret[k] = sum(all_ret[k]) / len(all_ret[k])

            speed = iter_steps / (time.time() - local_time)
            speed_tokens = speed * args.micro_batch_size * args.max_seq_len
            ips = speed * args.micro_batch_size

            loss_info = ", ".join([
                "{}: {:.6f}".format(k, average_ret[k])
                for k in eval_fetch.keys()
            ])

            logger.info(
                "%s step %d, batch: %d, %s, speed: %.0f tokens/s, ips: %.2f seqs/s"
                % (task_name, global_step, eval_step + 1, loss_info,
                   speed_tokens, ips))

            for k in list(eval_fetch.keys()):
                log_writer.add_scalar("%s/%s" % (task_name, k), average_ret[k],
                                      global_step)

            break


def do_train(args):
    # Initialize the paddle and paddle fleet execute environment
    paddle.enable_static()
    fleet.init(is_collective=True)

    # Create the random seed for the worker
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    get_rng_state_tracker().add('global_seed', args.seed)
    get_rng_state_tracker().add('local_seed',
                                args.seed + fleet.worker_index() + 2021)

    assert args.device in [
        "cpu", "gpu", "xpu"
    ], "Invalid device! Available device should be cpu, gpu, or xpu."
    place = paddle.set_device(args.device)

    worker_num = fleet.worker_num()
    worker_index = fleet.worker_index()
    assert args.dp_degree * args.sharding_degree * args.mp_degree * args.pp_degree == worker_num, \
        "The product of degree num should be equal to worker_num."

    topo = Topology(device_rank=worker_index,
                    world_size=worker_num,
                    dp_degree=args.dp_degree,
                    pp_degree=args.pp_degree,
                    sharding_degree=args.sharding_degree,
                    mp_degree=args.mp_degree)

    logger.info("The topo of hybrid parallelism:\n{}".format(topo))

    dist_strategy = dist_optimizer(args, topo)

    # Create log write, train results show on last card of pipeline.
    if topo.is_last:
        log_writer_path = os.path.join(
            args.output_dir, "train_log",
            "{}_globalbsz_{}_amp_{}_recompute_{}_card_{}".format(
                args.model_name_or_path, args.global_batch_size, args.use_amp,
                args.use_recompute, worker_index).lower())
        # if os.path.exists(log_writer_path):
        #     shutil.rmtree(log_writer_path)
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

    data_file = get_train_data_file(args)
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()
    with paddle.static.program_guard(main_program, startup_program):
        data_holders = create_data_holder(args)
        # 0. input_ids,
        # 1. segment_ids,
        # 2. input_mask,
        # 3. masked_lm_positions,
        # 4. masked_lm_labels,
        # 5. next_sentence_labels

        [
            input_ids, segment_ids, input_mask, masked_lm_positions,
            masked_lm_labels, next_sentence_labels
        ] = data_holders

        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

        train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
            args,
            data_file,
            tokenizer,
            data_world_size=topo.data_info.size,
            data_world_rank=topo.data_info.rank,
            max_seq_len=args.max_seq_len,
            places=paddle.static.cuda_places(),
            data_holders=data_holders,
            current_step=global_step)
        fleet.init(is_collective=True)

        if args.model_name_or_path in pretrained_models_list:
            model_config = model_class.pretrained_init_configuration[
                args.model_name_or_path]
            if model_config["vocab_size"] % 8 != 0:
                model_config["vocab_size"] += 8 - (model_config["vocab_size"] %
                                                   8)
            model_config["hidden_dropout_prob"] = args.hidden_dropout_prob
            model_config[
                "attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
            model = model_class(base_class(**model_config))
        else:
            model, _ = model_class.from_pretrained(
                args.model_name_or_path,
                hidden_dropout_prob=args.hidden_dropout_prob,
                attention_probs_dropout_prob=args.attention_probs_dropout_prob,
            )

        # Create the model for the gpt pretrain
        prediction_scores, seq_relationship_score = model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            position_ids=None,
            attention_mask=input_mask,
            masked_positions=masked_lm_positions)

        criterion = criterion_class(with_nsp_loss=args.binary_head)
        if args.binary_head:
            lm_loss, sop_loss = criterion(prediction_scores,
                                          seq_relationship_score,
                                          masked_lm_labels,
                                          next_sentence_labels)
            loss = lm_loss + sop_loss
        else:
            loss = criterion(prediction_scores, seq_relationship_score,
                             masked_lm_labels)

        # Create the learning_rate sheduler and optimizer
        if args.decay_steps is None:
            args.decay_steps = args.max_steps

        lr_scheduler = LinearAnnealingWithWarmupDecay(
            args.max_lr,
            args.min_lr,
            warmup_step=args.warmup_rate * args.max_steps,
            decay_step=args.decay_steps,
            last_epoch=global_step)

        clip = None
        if args.grad_clip > 0:
            clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.grad_clip)

        decay_param = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        logger.info("Using paddle.optimizer.AdamW.")
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_epsilon,
            grad_clip=clip,
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_param)
        # alias
        optimizer.apply_optimize = optimizer._apply_optimize

        # if args.use_recompute:
        #     dist_strategy.recompute = True
        #     dist_strategy.recompute_configs = {
        #         "checkpoints": model.ernie.checkpoints
        #     }

        # Use the fleet api to compile the distributed optimizer
        optimizer = fleet.distributed_optimizer(optimizer,
                                                strategy=dist_strategy)

        optimizer.minimize(loss)
        logger.info(f'final strategy: {fleet._final_strategy()}')
        logger.info("The training meta optimizer is/are %s" %
                    fleet._get_applied_meta_list())

    program_desc_dir = os.path.join(args.output_dir, "program_desc")
    if not os.path.isdir(program_desc_dir):
        os.mkdir(program_desc_dir)

    with open(program_desc_dir + "/main_program.txt.%d" % worker_index,
              'w') as f:
        f.write(str(main_program))

    with open(program_desc_dir + "/startup_program.txt.%d" % worker_index,
              'w') as f:
        f.write(str(startup_program))

    # Define the Executor for running the static model
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    test_program = main_program.clone(for_test=True)

    if args.model_name_or_path not in pretrained_models_list:
        logger.info("Try to load checkpoint from %s " % args.model_name_or_path)
        dygrah_path = os.path.join(args.model_name_or_path,
                                   "model_state.pdparams")
        static_path = os.path.join(args.model_name_or_path, "static_vars")

        flag_loaded = False
        if os.path.exists(static_path):
            if args.mp_degree > 1:
                logger.warning("MP should init with dygraph params")
            else:
                logger.info("Loading parameters from %s" % static_path)
                paddle.static.load(main_program, static_path, exe)
                flag_loaded = True

        if not flag_loaded and os.path.exists(dygrah_path):
            if args.sharding_degree > 1:
                logger.warning("Sharding should init with static vars")
            else:
                logger.info("Loading parameters from %s" % dygrah_path)
                init_static_with_params(
                    model, paddle.load(dygrah_path, return_numpy=True), topo,
                    main_program)
                flag_loaded = True

        if not flag_loaded:
            logger.error("No checkpoint load.")

    # load checkpoint vars
    if os.path.exists(checkpoint_dir):
        if os.path.isfile(os.path.join(checkpoint_dir, "./config.yml")):
            paddle.static.load(main_program,
                               os.path.join(checkpoint_dir, "static_vars"), exe)

    fetch_loss_vars = collections.OrderedDict()
    fetch_other_vars = collections.OrderedDict()
    fetch_loss_vars["loss"] = loss
    if args.binary_head:
        fetch_loss_vars["lm_loss"] = lm_loss
        fetch_loss_vars["sop_loss"] = sop_loss

    fetch_other_vars["learning_rate"] = main_program.global_block(
    ).vars["learning_rate_0"]

    additional_vars = collections.OrderedDict()
    if args.use_amp:
        for key in ["loss_scaling", "num_good_steps", "num_bad_steps"]:
            additional_vars[key] = main_program.global_block().vars[key + "_0"]

    tic_train = time.time()
    while True:
        fetchs = []
        fetchs_keys = []
        if topo.is_last:
            fetchs = list(fetch_loss_vars.values()) + list(
                fetch_other_vars.values()) + list(additional_vars.values())
            fetchs_keys = list(fetch_loss_vars.keys()) + list(
                fetch_other_vars.keys()) + list(additional_vars.keys())

        # Bug fix, if not call valid_data_loader, the enumerate will call valid_data_loader
        # many times. and start a new random dataloader.
        valid_data_loader = valid_data_loader()
        test_data_loader = test_data_loader()

        for step, batch in enumerate(train_data_loader()):
            ret = exe.run(main_program,
                          feed=batch,
                          fetch_list=fetchs,
                          use_program_cache=True)
            # Skip for accumulate_steps in global step
            if (step + 1) % args.accumulate_steps != 0:
                continue
            global_step += 1
            # In the new 2.0 api, must call this function to change the learning_rate
            lr_scheduler.step()

            if global_step % args.logging_freq == 0:
                if topo.is_last:
                    res = collections.defaultdict(float)
                    for k, v in zip(fetchs_keys, ret):
                        res[k] = v[0]

                    speed = args.logging_freq / (time.time() - tic_train)

                    loss_info = "loss: %.6f, lm_loss: %.6f, sop_loss: %.6f"

                    loss_info = ", ".join([
                        "{}: {:.6f}".format(k, res[k])
                        for k in fetch_loss_vars.keys()
                    ])

                    common_loginfo = "global step %d, %s, speed: %.2f steps/s, ips: %.2f seqs/s, learning rate: %.5e" % (
                        global_step, loss_info, speed,
                        speed * args.global_batch_size, res["learning_rate"])
                    additional_loginfo = ", ".join([
                        "{}: {}".format(k, res[k])
                        for k in additional_vars.keys()
                    ])
                    if additional_loginfo:
                        common_loginfo += ", " + additional_loginfo
                    logger.info(common_loginfo)
                    for k, v in res.items():
                        log_writer.add_scalar("train/" + k, v, global_step)

                tic_train = time.time()

            #if args.check_accuracy:
            #    if global_step >= args.max_steps:
            #        return
            #    else:
            #        continue

            if global_step % args.eval_freq == 0:
                # TODO, check the input data of validation
                eval_fetch = collections.OrderedDict()
                if topo.is_last:
                    eval_fetch["loss"] = loss
                    if args.binary_head:
                        eval_fetch["lm_loss"] = lm_loss
                        eval_fetch["sop_loss"] = sop_loss

                run_evaluate(valid_data_loader, exe, test_program,
                             args.eval_iters, log_writer, global_step, args,
                             topo.is_last, eval_fetch, "valid")
                tic_train = time.time()

            if global_step % args.save_steps == 0 or global_step >= args.max_steps:
                output_dir = os.path.join(args.output_dir,
                                          "model_%d" % global_step)
                logger.debug("saving models to {}".format(output_dir))
                save_persistables(exe, os.path.join(output_dir, "static_vars"),
                                  main_program)
                if global_step == args.save_steps:
                    model.init_config["init_args"][0].init_config.pop(
                        "topo", None)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
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

                    step_config = {
                        "model_name": args.model_name_or_path,
                        "global_step": global_step,
                        "global_batch_size": args.global_batch_size,
                        "consumed_samples":
                        global_step * args.global_batch_size,
                    }

                    with open(os.path.join(output_dir, "config.yml"), "w") as f:
                        yaml.dump(step_config,
                                  f,
                                  encoding='utf-8',
                                  allow_unicode=True)

                fleet.barrier_worker()

                logger.debug("saving models to {}".format(output_dir))
                if args.sharding_degree <= 1:
                    # Save on the first worker by default.
                    if worker_index == 0:
                        paddle.static.save(
                            main_program, os.path.join(output_dir,
                                                       "static_vars"))
                else:
                    # Use save_persistables in sharding, but more slower
                    save_persistables(exe,
                                      os.path.join(output_dir, "static_vars"),
                                      main_program)

            if global_step >= args.max_steps:
                eval_fetch = collections.OrderedDict()
                if topo.is_last:
                    eval_fetch["loss"] = loss
                    if args.binary_head:
                        eval_fetch["lm_loss"] = lm_loss
                        eval_fetch["sop_loss"] = sop_loss

                run_evaluate(test_data_loader, exe, test_program,
                             args.test_iters, log_writer, global_step, args,
                             topo.is_last, eval_fetch, "test")
                del train_data_loader
                return


if __name__ == "__main__":
    args = parse_args(MODEL_CLASSES)
    logger.info('{:20}:{}'.format("paddle commit id", paddle.version.commit))
    for arg in vars(args):
        logger.info('{:20}:{}'.format(arg, getattr(args, arg)))

    do_train(args)
