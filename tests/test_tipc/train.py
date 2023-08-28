# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import time
from pprint import pprint

import numpy as np
import paddle
import paddle.distributed as dist
from benchmark import options
from benchmark.modules.benchmark_utils import clone_inputs
from benchmark.options import LR_SCHEDULER_REGISTRY, MODEL_REGISTRY, OPTIMIZER_REGISTRY
from benchmark.utils.record import AverageStatistical

from paddlenlp.utils import profiler
from paddlenlp.utils.log import logger


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def do_generated_inputs(args):
    if args.device == "cpu":
        rank = 0
        trainer_count = 1
        paddle.set_device("cpu")
    else:
        rank = dist.get_rank()
        trainer_count = dist.get_world_size()

    if trainer_count > 1:
        dist.init_parallel_env()

    # Set seed for CE
    if args.seed is not None:
        set_seed(args.seed)

    benchmark_model = MODEL_REGISTRY[args.model]()
    benchmark_optimizer = OPTIMIZER_REGISTRY[args.optimizer]()

    if args.max_steps is None or (args.max_steps is not None and args.max_steps < 0):
        args.max_steps = 10000

    # Define model
    model = benchmark_model.build_model(args)

    if args.to_static:
        input_spec = benchmark_model.create_input_specs()
        model = paddle.jit.to_static(model, input_spec=input_spec)
        logger.info("Successfully to apply @to_static with specs: {}".format(input_spec))

    # Define data loader
    example_inputs = benchmark_model.generate_inputs_for_model(args, model)

    if args.lr_scheduler is not None:
        benchmark_lr_scheduler = LR_SCHEDULER_REGISTRY[args.lr_scheduler]()
        lr = benchmark_lr_scheduler.build_scheculer(args)
    else:
        lr = args.learning_rate

    optimizer = benchmark_optimizer.build_optimizer(args, lr, model)

    # for amp training
    if args.use_amp:
        scaler = paddle.amp.GradScaler(enable=True, init_loss_scaling=args.scale_loss)
        model = paddle.amp.decorate(models=model, level=args.amp_level, save_dtype="float32")

    # for distributed training
    if trainer_count > 1:
        model = paddle.DataParallel(model)

    step_id = 1

    # For benchmark
    reader_cost_avg = AverageStatistical()
    batch_cost_avg = AverageStatistical()
    batch_ips_avg = AverageStatistical()

    # Train loop
    for pass_id in range(args.epoch):
        epoch_start = time.time()

        batch_start = time.time()
        for batch_id in range(args.max_steps):
            train_reader_cost = time.time() - batch_start
            cloned_inputs = clone_inputs(example_inputs)

            if args.use_amp:
                # paddle version >= 2.5.0 or develop
                paddle_version = float(paddle.__version__[:3])
                if (paddle_version == 0.0) or (paddle_version >= 2.5):
                    with paddle.amp.auto_cast(
                        custom_black_list=args.custom_black_list if args.amp_level == "O2" else {},
                        level=args.amp_level,
                        use_promote=args.amp_use_promote,
                    ):
                        loss, sample_per_cards = benchmark_model.forward(model, args, cloned_inputs)
                else:
                    with paddle.amp.auto_cast(
                        custom_black_list=args.custom_black_list if args.amp_level == "O2" else {},
                        level=args.amp_level,
                    ):
                        loss, sample_per_cards = benchmark_model.forward(model, args, cloned_inputs)

                scaled = scaler.scale(loss)
                scaled.backward()

                scaler.minimize(optimizer, scaled)
                if "set_to_zero" in inspect.getfullargspec(optimizer.clear_grad).args:
                    optimizer.clear_grad(set_to_zero=False)
                else:
                    optimizer.clear_grad()
            else:
                loss, sample_per_cards = benchmark_model.forward(model, args, cloned_inputs)

                loss.backward()

                optimizer.step()
                optimizer.clear_grad()

            if args.profiler_options is not None:
                profiler.add_profiler_step(args.profiler_options)

            if args.max_steps and step_id == args.max_steps:
                if args.save_model and rank == 0:
                    model_dir = args.save_model
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    paddle.save(model.state_dict(), os.path.join(model_dir, "model.pdparams"))
                    paddle.save(optimizer.state_dict(), os.path.join(model_dir, "model.pdopt"))
                return

            if args.lr_scheduler is not None and not args.scheduler_update_by_epoch:
                lr.step()

            if step_id % args.logging_steps == 0:
                total_avg_loss = loss.numpy()

                train_batch_cost = time.time() - batch_start
                reader_cost_avg.record(train_reader_cost)
                batch_cost_avg.record(train_batch_cost)
                batch_ips_avg.record(train_batch_cost, sample_per_cards)

                benchmark_model.logger(
                    args,
                    step_id=step_id,
                    pass_id=pass_id,
                    batch_id=batch_id,
                    loss=total_avg_loss,
                    batch_cost=batch_cost_avg.get_average(),
                    reader_cost=reader_cost_avg.get_average(),
                    num_samples=sample_per_cards,
                    ips=batch_ips_avg.get_average_per_sec(),
                )

                reader_cost_avg.reset()
                batch_cost_avg.reset()
                batch_ips_avg.reset()
            else:
                train_batch_cost = time.time() - batch_start
                reader_cost_avg.record(train_reader_cost)
                batch_cost_avg.record(train_batch_cost)
                batch_ips_avg.record(train_batch_cost, sample_per_cards)

            batch_start = time.time()

            batch_id += 1
            step_id += 1

        if args.lr_scheduler is not None and args.scheduler_update_by_epoch:
            lr.step()

        train_epoch_cost = time.time() - epoch_start
        logger.info("train epoch: %d, epoch_cost: %.5f s" % (pass_id, train_epoch_cost))


def do_train(args):
    if args.device == "gpu":
        rank = dist.get_rank()
        trainer_count = dist.get_world_size()
    else:
        rank = 0
        trainer_count = 1
        paddle.set_device("cpu")

    if trainer_count > 1:
        dist.init_parallel_env()

    # Set seed for CE
    if args.seed is not None:
        set_seed(args.seed)

    benchmark_model = MODEL_REGISTRY[args.model]()
    benchmark_optimizer = OPTIMIZER_REGISTRY[args.optimizer]()

    # Define data loader
    train_loader, eval_loader = benchmark_model.create_data_loader(args)

    if args.max_steps is None or (args.max_steps is not None and args.max_steps < 0):
        args.max_steps = len(train_loader) * args.epoch

    # Define model
    model = benchmark_model.build_model(args)

    if args.to_static:
        input_spec = benchmark_model.create_input_specs()
        model = paddle.jit.to_static(model, input_spec=input_spec)
        logger.info("Successfully to apply @to_static with specs: {}".format(input_spec))

    if args.lr_scheduler is not None:
        benchmark_lr_scheduler = LR_SCHEDULER_REGISTRY[args.lr_scheduler]()
        lr = benchmark_lr_scheduler.build_scheculer(args)
    else:
        lr = args.learning_rate

    optimizer = benchmark_optimizer.build_optimizer(args, lr, model)

    # for amp training
    if args.use_amp:
        scaler = paddle.amp.GradScaler(enable=True, init_loss_scaling=args.scale_loss)
        model = paddle.amp.decorate(models=model, level=args.amp_level, save_dtype="float32")

    # for distributed training
    if trainer_count > 1:
        model = paddle.DataParallel(model)

    step_id = 1

    # For benchmark
    reader_cost_avg = AverageStatistical()
    batch_cost_avg = AverageStatistical()
    batch_ips_avg = AverageStatistical()

    # Train loop
    for pass_id in range(args.epoch):
        epoch_start = time.time()

        batch_id = 0
        batch_start = time.time()
        for input_data in train_loader:
            train_reader_cost = time.time() - batch_start

            if args.use_amp:
                # paddle version >= 2.5.0 or develop
                paddle_version = float(paddle.__version__[:3])
                if (paddle_version == 0.0) or (paddle_version >= 2.5):
                    with paddle.amp.auto_cast(
                        custom_black_list=args.custom_black_list if args.amp_level == "O2" else {},
                        level=args.amp_level,
                        use_promote=args.amp_use_promote,
                    ):
                        loss, sample_per_cards = benchmark_model.forward(model, args, input_data)
                else:
                    with paddle.amp.auto_cast(
                        custom_black_list=args.custom_black_list if args.amp_level == "O2" else {},
                        level=args.amp_level,
                    ):
                        loss, sample_per_cards = benchmark_model.forward(model, args, input_data)

                scaled = scaler.scale(loss)
                scaled.backward()

                scaler.minimize(optimizer, scaled)
                if "set_to_zero" in inspect.getfullargspec(optimizer.clear_grad).args:
                    optimizer.clear_grad(set_to_zero=False)
                else:
                    optimizer.clear_grad()
            else:
                loss, sample_per_cards = benchmark_model.forward(model, args, input_data)

                loss.backward()

                optimizer.step()
                optimizer.clear_grad()

            if args.profiler_options is not None:
                profiler.add_profiler_step(args.profiler_options)

            if args.max_steps and step_id == args.max_steps:
                if args.save_model and rank == 0:
                    model_dir = args.save_model
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    paddle.save(model.state_dict(), os.path.join(model_dir, "model.pdparams"))
                    paddle.save(optimizer.state_dict(), os.path.join(model_dir, "model.pdopt"))
                return

            if args.lr_scheduler is not None and not args.scheduler_update_by_epoch:
                lr.step()

            if step_id % args.logging_steps == 0:
                total_avg_loss = loss.numpy()

                train_batch_cost = time.time() - batch_start
                reader_cost_avg.record(train_reader_cost)
                batch_cost_avg.record(train_batch_cost)
                batch_ips_avg.record(train_batch_cost, sample_per_cards)

                benchmark_model.logger(
                    args,
                    step_id=step_id,
                    pass_id=pass_id,
                    batch_id=batch_id,
                    loss=total_avg_loss,
                    batch_cost=batch_cost_avg.get_average(),
                    reader_cost=reader_cost_avg.get_average(),
                    num_samples=sample_per_cards,
                    ips=batch_ips_avg.get_average_per_sec(),
                )

                reader_cost_avg.reset()
                batch_cost_avg.reset()
                batch_ips_avg.reset()
            else:
                train_batch_cost = time.time() - batch_start
                reader_cost_avg.record(train_reader_cost)
                batch_cost_avg.record(train_batch_cost)
                batch_ips_avg.record(train_batch_cost, sample_per_cards)

            batch_start = time.time()

            batch_id += 1
            step_id += 1

        if args.lr_scheduler is not None and args.scheduler_update_by_epoch:
            lr.step()

        train_epoch_cost = time.time() - epoch_start
        logger.info("train epoch: %d, epoch_cost: %.5f s" % (pass_id, train_epoch_cost))


def do_hapi(args):
    paddle.set_device(args.device)

    # Set seed for CE
    if args.seed is not None:
        set_seed(args.seed)

    benchmark_model = MODEL_REGISTRY[args.model]()
    benchmark_optimizer = OPTIMIZER_REGISTRY[args.optimizer]()

    # Define data loader
    train_loader, eval_loader = benchmark_model.create_data_loader(args)

    if args.lr_scheduler is not None:
        benchmark_lr_scheduler = LR_SCHEDULER_REGISTRY[args.lr_scheduler]()
        lr = benchmark_lr_scheduler.build_scheculer(args)
    else:
        lr = args.learning_rate

    model = benchmark_model.build_model(args)

    if args.to_static:
        input_spec = benchmark_model.create_input_specs()
        model = paddle.jit.to_static(model, input_spec=input_spec)
        logger.info("Successfully to apply @to_static with specs: {}".format(input_spec))

    optimizer = benchmark_optimizer.build_optimizer(args, lr, model)

    benchmark_model.forward(model, args, optimizer=optimizer, train_loader=train_loader, eval_loader=eval_loader)


if __name__ == "__main__":
    parser = options.get_training_parser()
    args = options.parse_args_and_model(parser)
    pprint(args)

    if args.generated_inputs:
        do_generated_inputs(args)
    elif getattr(args, "use_hapi", False):
        do_hapi(args)
    else:
        do_train(args)
