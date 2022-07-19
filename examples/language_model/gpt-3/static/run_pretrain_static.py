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
Pretrain  GPT in static graph mode.
"""
import argparse
import math
import os
import random
import time
import sys

os.path.expandvars('$HOME')
os.path.expanduser('~')

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_optimizers.sharding.utils import save_persistables
from modeling import GPTModel, GPTForPretraining, GPTPretrainingCriterion
from paddlenlp.transformers import GPTTokenizer, GPTChineseTokenizer
from paddlenlp.ops import guard, Topology, get_rng_state_tracker
from paddlenlp.utils.log import logger
from paddlenlp.utils import profiler
import paddlenlp.ops as ops
from visualdl import LogWriter

# Used to load the data_tools path, should import before dataset
filepath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(filepath, "../"))
from dataset import create_pretrained_dataset
from args import parse_args
import lr

MODEL_CLASSES = {
    "gpt": (GPTForPretraining, GPTTokenizer),
    "gpt-cn": (GPTForPretraining, GPTChineseTokenizer),
}


def create_data_holder(args):
    """creat data holder"""
    tokens = paddle.static.data(name="tokens",
                                shape=[-1, args.max_seq_len],
                                dtype="int64")
    loss_mask = paddle.static.data(name="loss_mask",
                                   shape=[-1, args.max_seq_len],
                                   dtype="float32")
    position_ids = paddle.static.data(name="position_ids",
                                      shape=[-1, args.max_seq_len],
                                      dtype="int64")
    labels = paddle.static.data(name="labels",
                                shape=[-1, args.max_seq_len],
                                dtype="int64")
    return [tokens, loss_mask, position_ids, labels]


def dist_optimizer(args, topo):
    default_global_batch_size = topo.data_info.size * args.micro_batch_size
    if args.global_batch_size is None:
        args.global_batch_size = default_global_batch_size

    bsz_per_dp = args.global_batch_size // topo.data_info.size
    micro_batch_size = args.micro_batch_size
    assert args.global_batch_size % micro_batch_size == 0, "cannot do gradient accumulate, global_batch_size: {} micro_batch_size: {}".format(
        args.global_batch_size, micro_batch_size)
    acc_steps = bsz_per_dp // micro_batch_size

    exec_strategy = paddle.static.ExecutionStrategy()
    exec_strategy.num_threads = 2
    exec_strategy.num_iteration_per_drop_scope = 1

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.nccl_comm_num = 3

    dist_strategy.recompute = args.use_recompute
    dist_strategy.pipeline = args.pp_degree > 1

    if args.use_amp:
        dist_strategy.amp = True
        dist_strategy.amp_configs = {
            "custom_white_list": [
                'softmax', 'layer_norm', 'gelu',
                "fused_softmax_mask_upper_triangle", "elementwise_add"
            ],
            "custom_black_list":
            ["reduce_sum", "c_softmax_with_cross_entropy", "elementwise_div"],
            "init_loss_scaling":
            32768,
            "use_dynamic_loss_scaling":
            True,
            "use_pure_fp16":
            args.amp_level == "O2",
            "use_fp16_guard":
            False
        }
    if args.use_sharding:
        dist_strategy.sharding = True
        dist_strategy.sharding_configs = {
            "segment_broadcast_MB": 32,
            "sharding_degree": args.sharding_degree,
            "mp_degree": args.mp_degree,
            "pp_degree": args.pp_degree,
            "dp_degree": args.dp_degree,
            "optimize_offload": False,
        }
    elif args.mp_degree > 1 and args.pp_degree == 1:
        # For MP or MP + DP, use executor instead of parallel_executor
        dist_strategy.without_graph_optimization = True
    if args.pp_degree > 1:
        dist_strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_batch_size": micro_batch_size,
            "accumulate_steps": acc_steps,
        }
    else:
        assert acc_steps == 1, "Only support accumulate steps in piplinemode. Please set you global_batch_size={}".format(
            default_global_batch_size)

    return dist_strategy


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


def init_static_with_params(model, dygraph_params, topo, prog=None):
    from paddlenlp.utils.tools import dygraph_params_to_static
    static_params = dygraph_params_to_static(model, dygraph_params, topo)
    if prog is None:
        prog = paddle.static.default_main_program()
    paddle.static.set_program_state(prog, static_params)


def run_evaluate(data_loader,
                 exe,
                 program,
                 iter_steps,
                 log_writer,
                 global_step,
                 args,
                 epoch,
                 is_last,
                 eval_fetch,
                 task_name="valid"):
    all_loss = []
    local_time = time.time()

    for eval_step, batch in enumerate(data_loader):
        loss_return = exe.run(program, feed=batch, fetch_list=eval_fetch)
        if is_last:
            all_loss.append(float(loss_return[0]))
        if eval_step >= iter_steps - 1:
            if not is_last:
                break
            average_loss = sum(all_loss) / len(all_loss)
            logger.info(
                "%s step %d, epoch: %d, batch: %d, loss: %f, speed: %.0f tokens/s"
                % (task_name, global_step, epoch, eval_step, average_loss,
                   iter_steps * args.micro_batch_size * args.max_seq_len /
                   (time.time() - local_time)))
            log_writer.add_scalar(task_name + "_loss", average_loss,
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

    if args.use_amp and args.amp_level == "O2":
        assert (args.mp_degree == 1 and args.pp_degree == 1
                ), "When amp level is O2, mp_degree and pp_degree should be 1."
        assert (args.use_sharding == False
                ), "When amp level is O2, use_sharding should be False."

    assert args.device in [
        "cpu", "gpu", "xpu"
    ], "Invalid device! Available device should be cpu, gpu, or xpu."
    place = paddle.set_device(args.device)

    worker_num = fleet.worker_num()
    worker_index = fleet.worker_index()
    local_rank = 0 if fleet.local_rank() is None else int(fleet.local_rank())

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
        if os.path.exists(log_writer_path):
            import shutil
            shutil.rmtree(log_writer_path)
        log_writer = LogWriter(log_writer_path)

    # Define the input data in the static mode

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    pretrained_models_list = list(
        model_class.pretrained_init_configuration.keys())

    data_file = get_train_data_file(args)
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()
    with paddle.static.program_guard(main_program, startup_program):
        with paddle.utils.unique_name.guard():
            with paddle.static.device_guard('gpu:0'):
                data_holders = create_data_holder(args)
                [tokens, loss_mask, position_ids, labels] = data_holders

                tokenizer = tokenizer_class.from_pretrained(
                    args.model_name_or_path)
                eos_id = tokenizer.eos_token_id

                train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
                    args,
                    data_file,
                    local_rank=local_rank,
                    data_world_size=topo.data_info.size,
                    data_world_rank=topo.data_info.rank,
                    eos_id=eos_id,
                    max_seq_len=args.max_seq_len,
                    places=paddle.static.cuda_places(),
                    data_holders=data_holders,
                    pipeline_mode=True if args.pp_degree > 1 else False,
                )

                if args.model_name_or_path in pretrained_models_list:
                    model_config = model_class.pretrained_init_configuration[
                        args.model_name_or_path]

                    model_config[
                        "hidden_dropout_prob"] = args.hidden_dropout_prob
                    model_config[
                        "attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
                    model_config["topo"] = topo
                    model_config["fuse"] = args.fuse_transformer

                    model = guard(f'gpu:{args.pp_degree -1}')(
                        GPTForPretraining)(
                            guard(f'gpu:0')(GPTModel)(**model_config))
                else:
                    model, _ = GPTForPretraining.from_pretrained(
                        args.model_name_or_path,
                        hidden_dropout_prob=args.hidden_dropout_prob,
                        attention_probs_dropout_prob=args.
                        attention_probs_dropout_prob,
                        topo=topo)
                # Create the model for the gpt pretrain
                preds = model(tokens, position_ids)

                criterion = guard(f'gpu:{args.pp_degree -1}')(
                    GPTPretrainingCriterion)(topo)
                loss = criterion(preds, labels, loss_mask)

            # Create the learning_rate sheduler and optimizer
            if args.decay_steps is None:
                args.decay_steps = args.max_steps
            warmup_step = args.warmup_rate * args.decay_steps

            # TODO @ZHUI Use paddle network to support lr scheduler
            lr_scheduler = lr.CosineAnnealingWithWarmupDecay(
                max_lr=args.max_lr,
                min_lr=args.min_lr,
                warmup_step=warmup_step,
                decay_step=args.decay_steps)

            clip = None
            if args.grad_clip > 0:
                clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.grad_clip)

            decay_param = [
                p.name for n, p in model.named_parameters()
                if not any(nd in n for nd in ["bias", "norm"])
            ]
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

            if args.use_recompute:
                dist_strategy.recompute = True
                dist_strategy.recompute_configs = {
                    "checkpoints": model.gpt.checkpoints
                }

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
        if args.pp_degree > 1:
            f.write(str(main_program._pipeline_opt['section_program']))
        else:
            f.write(str(main_program))

    with open(program_desc_dir + "/startup_program.txt.%d" % worker_index,
              'w') as f:
        f.write(str(startup_program))

    # Define the Executor for running the static model
    exe = paddle.static.Executor(place)
    exe.run(startup_program)
    test_program = main_program.clone(for_test=True)

    if args.use_amp and args.amp_level == "O2":
        optimizer.amp_init(place)

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

    global_step = 0
    tic_train = time.time()
    epoch = 0
    learning_rate = main_program.global_block().vars["learning_rate_0"]
    step = 0
    while True:
        fetchs = []
        if topo.is_last:
            fetchs = [loss, learning_rate]

        # Bug fix, if not call valid_data_loader, the enumerate will call valid_data_loader
        # many times. and start a new random dataloader.
        # Note: for pipeline mode, validation and test are not supported, so
        # both valid_data_loader and test_data_loader are None.
        valid_data_loader = None if args.pp_degree > 1 else valid_data_loader()
        test_data_loader = None if args.pp_degree > 1 else test_data_loader()

        train_reader_cost = 0.0
        train_run_cost = 0.0
        reader_start = time.time()
        if args.pp_degree == 1:
            for step, batch in enumerate(train_data_loader()):
                train_reader_cost += time.time() - reader_start
                train_start = time.time()

                global_step += 1

                ret = exe.run(main_program,
                              feed=batch,
                              fetch_list=fetchs,
                              use_program_cache=True)
                # In the new 2.0 api, must call this function to change the learning_rate
                lr_scheduler.step()
                train_run_cost += time.time() - train_start

                # Profile for model benchmark
                profiler.add_profiler_step(args.profiler_options)

                if global_step % args.logging_freq == 0:
                    if topo.is_last:
                        loss_return, lr_return = ret
                        #speed = args.logging_freq / (time.time() - tic_train)
                        speed = args.logging_freq / (train_reader_cost +
                                                     train_run_cost)
                        avg_reader_cost = train_reader_cost / args.logging_freq
                        logger.info(
                            "global step %d, epoch: %d, batch: %d, loss: %.9f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, speed: %.2f steps/s, ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
                            % (global_step, epoch, step, loss_return[0],
                               avg_reader_cost, 1. / speed, speed, speed *
                               args.global_batch_size * args.max_seq_len,
                               speed * args.global_batch_size *
                               args.max_seq_len / worker_num, lr_return[0]))
                        log_writer.add_scalar("loss", loss_return[0],
                                              global_step)
                        log_writer.add_scalar("learning_rate", lr_return[0],
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
                    # TODO, check the input data of validation
                    eval_fetch = []
                    if topo.is_last:
                        eval_fetch = [loss]

                    run_evaluate(valid_data_loader, exe, test_program,
                                 args.eval_iters, log_writer, global_step, args,
                                 epoch, topo.is_last, eval_fetch, "valid")
                    tic_train = time.time()

                if global_step % args.save_steps == 0 or global_step >= args.max_steps:
                    output_dir = os.path.join(args.output_dir,
                                              "model_%d" % global_step)
                    logger.debug("saving models to {}".format(output_dir))
                    save_persistables(exe,
                                      os.path.join(output_dir, "static_vars"),
                                      main_program)
                    if global_step <= args.save_steps:
                        model.init_config["init_args"][0].init_config.pop(
                            "topo", None)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    tic_train = time.time()

                if global_step >= args.max_steps:
                    eval_fetch = []
                    if topo.is_last:
                        eval_fetch = [loss]

                    run_evaluate(test_data_loader, exe, test_program,
                                 args.test_iters, log_writer, global_step, args,
                                 epoch, topo.is_last, eval_fetch, "test")
                    del train_data_loader
                    return
                reader_start = time.time()
            epoch += 1
        else:  # for pipeline, use noniterable dataloader
            train_data_loader.start()
            try:
                while True:
                    train_reader_cost += time.time() - reader_start
                    train_start = time.time()

                    global_step += 1

                    ret = exe.run(main_program,
                                  fetch_list=fetchs,
                                  use_program_cache=True)
                    # In the new 2.0 api, must call this function to change the learning_rate
                    lr_scheduler.step()
                    train_run_cost += time.time() - train_start

                    # Profile for model benchmark
                    profiler.add_profiler_step(args.profiler_options)

                    if global_step % args.logging_freq == 0:
                        if topo.is_last:
                            loss_return, lr_return = ret
                            #speed = args.logging_freq / (time.time() - tic_train)
                            speed = args.logging_freq / (train_reader_cost +
                                                         train_run_cost)
                            avg_reader_cost = train_reader_cost / args.logging_freq
                            logger.info(
                                "global step %d, epoch: %d, batch: %d, loss: %.9f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, speed: %.2f steps/s, ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
                                % (global_step, epoch, step, loss_return[0],
                                   avg_reader_cost, 1. / speed, speed, speed *
                                   args.global_batch_size * args.max_seq_len,
                                   speed * args.global_batch_size *
                                   args.max_seq_len / worker_num, lr_return[0]))
                            log_writer.add_scalar("loss", loss_return[0],
                                                  global_step)
                            log_writer.add_scalar("learning_rate", lr_return[0],
                                                  global_step)
                        tic_train = time.time()
                        train_reader_cost = 0.0
                        train_run_cost = 0.0
                    step += 1

                    if args.check_accuracy:
                        if global_step >= args.max_steps:
                            return
                        else:
                            continue

                    if global_step % args.save_steps == 0 or global_step >= args.max_steps:
                        output_dir = os.path.join(args.output_dir,
                                                  "model_%d" % global_step)
                        logger.debug("saving models to {}".format(output_dir))
                        save_persistables(
                            exe, os.path.join(output_dir, "static_vars"),
                            main_program._pipeline_opt['section_program'])
                        if global_step <= args.save_steps:
                            model.init_config["init_args"][0].init_config.pop(
                                "topo", None)
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        tic_train = time.time()

                    if global_step >= args.max_steps:
                        train_data_loader.reset()
                        del train_data_loader
                        return

                    reader_start = time.time()
            except paddle.framework.core.EOFException:
                train_data_loader.reset()
                epoch += 1
                step = 0
                global_step = 0


if __name__ == "__main__":
    config = parse_args(MODEL_CLASSES)
    do_train(config)
