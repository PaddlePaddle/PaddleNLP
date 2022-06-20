import os
import sys
import time
import random
import inspect
import numpy as np
from pprint import pprint

from paddlenlp.utils import profiler

import paddle
import paddle.distributed as dist

import benchmark
from benchmark import options
from benchmark.options import MODEL_REGISTRY, OPTIMIZER_REGISTRY, LR_SCHEDULER_REGISTRY
from benchmark.utils.record import AverageStatistical

from paddlenlp.utils.log import logger


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


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

    if args.max_steps is None or (args.max_steps is not None
                                  and args.max_steps < 0):
        args.max_steps = len(train_loader) * args.epoch

    # Define model
    model = benchmark_model.build_model(args)

    if args.lr_scheduler is not None:
        benchmark_lr_scheduler = LR_SCHEDULER_REGISTRY[args.lr_scheduler]()
        lr = benchmark_lr_scheduler.build_scheculer(args)
    else:
        lr = args.learning_rate

    optimizer = benchmark_optimizer.build_optimizer(args, lr, model)

    # for amp training
    if args.use_amp:
        scaler = paddle.amp.GradScaler(enable=True,
                                       init_loss_scaling=args.scale_loss)
        model = paddle.amp.decorate(models=model,
                                    level=args.amp_level,
                                    save_dtype='float32')

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
                with paddle.amp.auto_cast(
                        custom_black_list=args.custom_black_list
                        if amp_level == 'O2' else {},
                        level=amp_level):
                    loss, sample_per_cards = benchmark_model.forward(
                        model, args, input_data)

                scaled = scaler.scale(loss)
                scaled.backward()

                scaler.minimize(optimizer, scaled)
                if 'set_to_zero' in inspect.getfullargspec(
                        optimizer.clear_grad).args:
                    optimizer.clear_grad(set_to_zero=False)
                else:
                    optimizer.clear_grad()
            else:
                loss, sample_per_cards = benchmark_model.forward(
                    model, args, input_data)

                loss.backward()

                optimizer.step()
                optimizer.clear_grad()

            train_batch_cost = time.time() - batch_start
            reader_cost_avg.record(train_reader_cost)
            batch_cost_avg.record(train_batch_cost)
            batch_ips_avg.record(train_batch_cost, sample_per_cards)

            if args.profiler_options is not None:
                profiler.add_profiler_step(args.profiler_options)

            if step_id % args.logging_steps == 0:
                total_avg_loss = loss.numpy()

                benchmark_model.logger(
                    args,
                    step_id=step_id,
                    pass_id=pass_id,
                    batch_id=batch_id,
                    loss=total_avg_loss,
                    batch_cost=batch_cost_avg.get_average(),
                    reader_cost=reader_cost_avg.get_average(),
                    num_samples=sample_per_cards,
                    ips=batch_ips_avg.get_average_per_sec())

                reader_cost_avg.reset()
                batch_cost_avg.reset()
                batch_ips_avg.reset()

            if args.max_steps and step_id == args.max_steps:
                if args.save_model and rank == 0:
                    model_dir = args.save_model
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    paddle.save(model.state_dict(),
                                os.path.join(model_dir, "model.pdparams"))
                    paddle.save(optimizer.state_dict(),
                                os.path.join(model_dir, "model.pdopt"))
                return
            batch_id += 1
            step_id += 1
            if args.lr_scheduler is not None and not args.scheduler_update_by_epoch:
                lr.step()
            batch_start = time.time()

        if args.lr_scheduler is not None and args.scheduler_update_by_epoch:
            lr.step()

        train_epoch_cost = time.time() - epoch_start
        logger.info("train epoch: %d, epoch_cost: %.5f s" %
                    (pass_id, train_epoch_cost))


def do_hapi(args):
    device = paddle.set_device(args.device)

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

    optimizer = benchmark_optimizer.build_optimizer(args, lr, model)

    benchmark_model.forward(model,
                            args,
                            optimizer=optimizer,
                            train_loader=train_loader,
                            eval_loader=eval_loader)


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_model(parser)

    if getattr(args, 'use_hapi', False):
        do_hapi(args)
    else:
        do_train(args)
