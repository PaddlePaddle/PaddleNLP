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
Pretrain  GPT-2 in static graph mode.
"""
import argparse
import math
import os
import random
import time

os.environ['FLAGS_enable_parallel_graph'] = "0"
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.1"
os.environ['FLAGS_sync_nccl_allreduce'] = "1"
os.environ['FLAGS_eager_delete_tensor_gb'] = "0"
os.environ['FLAGS_fuse_parameter_memory_size'] = "32"
os.environ['FLAGS_fuse_parameter_groups_size'] = "50"
os.environ['FLAGS_check_nan_inf'] = "1"
os.environ['FLAGS_enable_sequential_execution'] = "1"
os.path.expandvars('$HOME')
os.path.expanduser('~')

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_optimizers.sharding.utils import save_persistables
from paddlenlp.transformers import GPTModel, GPTForPretraining, GPTPretrainingCriterion
from paddlenlp.transformers import GPTTokenizer, GPTChineseTokenizer
from paddlenlp.ops import guard, Topology, get_rng_state_tracker
from paddlenlp.utils.log import logger
import paddlenlp.ops as ops
from visualdl import LogWriter

from dataset import create_pretrained_dataset
from args import parse_args
import lr

MODEL_CLASSES = {
    "gpt": (GPTForPretraining, GPTTokenizer),
    "gpt-cn": (GPTForPretraining, GPTChineseTokenizer),
}


def create_data_holder(args):
    """creat data holder"""
    tokens = paddle.static.data(
        name="tokens", shape=[-1, args.max_seq_len], dtype="int64")
    loss_mask = paddle.static.data(
        name="loss_mask", shape=[-1, args.max_seq_len], dtype="float32")
    attention_mask = paddle.static.data(
        name="attention_mask",
        shape=[-1, 1, args.max_seq_len, args.max_seq_len],
        dtype="float32")
    position_ids = paddle.static.data(
        name="position_ids", shape=[-1, args.max_seq_len], dtype="int64")
    labels = paddle.static.data(
        name="labels", shape=[-1, args.max_seq_len], dtype="int64")
    return [tokens, loss_mask, attention_mask, position_ids, labels]


def dist_optimizer(args, topo):
    default_global_batch_size = topo.data_info.size * args.micro_batch_size
    if args.global_batch_size is None:
        args.global_batch_size = default_global_batch_size

    bsz_per_dp = args.global_batch_size // topo.data_info.size
    micro_batch_size = args.micro_batch_size
    assert args.global_batch_size % micro_batch_size == 0, f"cannot do gradient accumulate, global_batch_size: {args.global_batch_size} micro_batch_size: {micro_batch_size}"
    acc_steps = bsz_per_dp // micro_batch_size

    exec_strategy = paddle.fluid.ExecutionStrategy()
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
            "custom_white_list": ['softmax', 'layer_norm', 'gelu'],
            "init_loss_scaling": 32768,
            "use_dynamic_loss_scaling": True,
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
    if args.pp_degree > 1:
        dist_strategy.pipeline_configs = {
            "schedule_mode": "1F1B",
            "micro_micro_batch_size": micro_batch_size,
            "accumulate_steps": acc_steps,
        }
    else:
        assert acc_steps == 1, f"Only support accumulate steps in piplinemode. Please set you global_batch_size={default_global_batch_size}"

    return dist_strategy


def get_train_data_file(args):
    files = [
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if (os.path.isfile(os.path.join(args.input_dir, f)) and "npz_" not in
            str(f))
    ]

    data_file = files[0]
    return data_file


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
    args.test_iters = args.eval_iters * 5
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

    topo = Topology(
        device_rank=worker_index,
        world_size=worker_num,
        dp_degree=args.dp_degree,
        pp_degree=args.pp_degree,
        sharding_degree=args.sharding_degree,
        mp_degree=args.mp_degree)

    logger.info(f"The topo of hybrid parallelism:\n{topo}")

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

    data_file = get_train_data_file(args)
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()
    with paddle.static.program_guard(main_program, startup_program):
        with paddle.utils.unique_name.guard():
            with paddle.static.device_guard('gpu:0'):
                data_holders = create_data_holder(args)
                [tokens, loss_mask, attention_mask, position_ids,
                 labels] = data_holders

                model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
                tokenizer = tokenizer_class.from_pretrained(
                    args.model_name_or_path)
                eod_id = tokenizer.eod_token_id
                model_config = model_class.pretrained_init_configuration[
                    args.model_name_or_path]
                if model_config["vocab_size"] % 8 != 0:
                    model_config["vocab_size"] += 8 - (
                        model_config["vocab_size"] % 8)
                model_config["topo"] = topo

                train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
                    args,
                    data_file,
                    topo,
                    eod_id=eod_id,
                    max_seq_len=args.max_seq_len,
                    places=paddle.static.cuda_places(),
                    data_holders=data_holders,
                    pipeline_mode=False, )

                # Create the model for the gpt pretrain
                model = guard(f'gpu:{args.pp_degree -1}')(GPTForPretraining)(
                    guard(f'gpu:0')(GPTModel)(**model_config))
                preds = model(tokens, position_ids, attention_mask)

                criterion = guard(f'gpu:{args.pp_degree -1}')(
                    GPTPretrainingCriterion)()
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
                # TODO @ZHUI Use nn.ClipGradByNorm
                clip = paddle.fluid.clip.GradientClipByNorm(
                    clip_norm=args.grad_clip)

            decay_param = [
                p.name for n, p in model.named_parameters()
                if not any(nd in n for nd in ["bias", "norm"])
            ]
            # TODO @ZHUI Use paddle.optimizer.AdamW
            if ops.optimizer._jit_compile():
                optimizer = ops.optimizer.AdamwOptimizer(
                    learning_rate=lr_scheduler,
                    beta1=args.adam_beta1,
                    beta2=args.adam_beta2,
                    epsilon=args.adam_epsilon,
                    grad_clip=clip,
                    weight_decay=args.weight_decay,
                    apply_decay_param_fun=lambda x: x in decay_param)
            else:
                optimizer = paddle.fluid.optimizer.AdamOptimizer(
                    learning_rate=lr_scheduler,
                    beta1=args.adam_beta1,
                    beta2=args.adam_beta2,
                    epsilon=args.adam_epsilon,
                    grad_clip=clip)
                logger.error(
                    "Compile custom adamw failed, use fluid.Adam and on WEIGHT DECAY!!!!"
                )

            if args.use_recompute:
                dist_strategy.recompute = True
                dist_strategy.recompute_configs = {
                    "checkpoints": model.gpt.checkpoints
                }

            # Use the fleet api to compile the distributed optimizer
            optimizer = fleet.distributed_optimizer(
                optimizer, strategy=dist_strategy)

            optimizer.minimize(loss)
            logger.info(f'final strategy: {fleet._final_strategy()}')
            logger.info("The training meta optimizer is/are %s" %
                        fleet._get_applied_meta_list())

    program_desc_dir = os.path.join(args.output_dir, "program_desc")
    if not os.path.isdir(program_desc_dir):
        os.mkdir(program_desc_dir)

    with open(program_desc_dir + "/main_program.txt.%d" %
              (int(os.environ.get('FLAGS_selected_gpus', 0))), 'w') as f:
        f.write(str(main_program))

    with open(program_desc_dir + "/startup_program.txt.%d" %
              (int(os.environ.get('FLAGS_selected_gpus', 0))), 'w') as f:
        f.write(str(startup_program))

    # Define the Executor for running the static model
    exe = paddle.static.Executor(place)
    exe.run(startup_program)
    test_program = main_program.clone(for_test=True)

    global_step = 0
    tic_train = time.time()
    epoch = 0
    learning_rate = main_program.global_block().vars["learning_rate_0"]
    while True:
        fetchs = []
        if topo.is_last:
            fetchs = [loss, learning_rate]

        # Bug fix, if not call valid_data_loader, the enumerate will call valid_data_loader
        # many times. and start a new random dataloader.
        valid_data_loader = valid_data_loader()
        test_data_loader = test_data_loader()

        for step, batch in enumerate(train_data_loader()):
            global_step += 1
            ret = exe.run(main_program, feed=batch, fetch_list=fetchs)
            # In the new 2.0 api, must call this function to change the learning_rate
            lr_scheduler.step()

            if global_step % args.logging_freq == 0:
                if topo.is_last:
                    loss_return, lr_return = ret
                    speed = args.logging_freq / (time.time() - tic_train)
                    logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f steps/s, %.0f token/s, learning rate: %.9f"
                        % (global_step, epoch, step, loss_return[0], speed,
                           speed * args.global_batch_size * args.max_seq_len,
                           lr_return[0]))
                    log_writer.add_scalar("loss", loss_return[0], global_step)
                    log_writer.add_scalar("learning_rate", lr_return[0],
                                          global_step)
                tic_train = time.time()

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
                save_persistables(exe, output_dir, main_program)
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
        epoch += 1


if __name__ == "__main__":
    config = parse_args(MODEL_CLASSES)
    do_train(config)
