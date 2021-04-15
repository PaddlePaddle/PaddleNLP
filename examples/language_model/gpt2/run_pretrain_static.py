# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
os.environ['FLAGS_check_nan_inf'] = "0"
os.environ['FLAGS_enable_sequential_execution'] = "1"
os.path.expandvars('$HOME')
os.path.expanduser('~')

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
from paddlenlp.transformers import GPT2Model, GPT2ForPretraining, GPT2PretrainingCriterion
from paddlenlp.transformers import GPT2Tokenizer, GPT2ChineseTokenizer
from paddlenlp.utils.log import logger
from tensorboardX import SummaryWriter

from data import create_pretrained_dataset
import lr
from topo import Topology

MODEL_CLASSES = {
    "gpt2": (GPT2ForPretraining, GPT2Tokenizer),
    "gpt2-cn": (GPT2ForPretraining, GPT2ChineseTokenizer),
}


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input directory where the data will be read from.", )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--grad_clip",
        default=0.0,
        type=float,
        help="Grad clip for the parameter.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_steps",
        default=3600000,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--decay_steps",
        default=360000,
        type=int,
        help="The steps use to control the learing rate. If the step > decay_steps, will use the min_lr.",
    )
    parser.add_argument(
        "--max_lr",
        default=1e-5,
        type=float,
        help="The initial max learning rate for Adam.")
    parser.add_argument(
        "--min_lr",
        default=5e-5,
        type=float,
        help="The initial min learning rate for Adam.")
    parser.add_argument(
        "--warmup_rate",
        default=0.01,
        type=float,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--use_amp",
        type=str2bool,
        nargs='?',
        const=True,
        help="Enable mixed precision training.")
    parser.add_argument(
        "--use_sharding",
        type=str2bool,
        nargs='?',
        const=True,
        help="Spliting the parameters to many cards.")
    parser.add_argument(
        "--use_recompute",
        type=str2bool,
        nargs='?',
        const=True,
        help="Using the recompute to save the memory.")
    parser.add_argument(
        "--enable_addto",
        type=str2bool,
        nargs='?',
        const=True,
        help="Whether to enable the addto strategy for gradient accumulation or not. This is only used for AMP training."
    )
    parser.add_argument(
        "--scale_loss",
        type=float,
        default=128,
        help="The value of scale_loss for fp16.")
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every X updates steps.")
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate for every X updates steps.")
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=10,
        help="Evaluate for every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        help="select cpu, gpu, xpu devices.")
    parser.add_argument(
        "--init_params_path",
        type=str,
        default="./data/gpt2.pdparams",
        help="select cpu, gpu, xpu devices.")
    config = parser.parse_args()
    return config


class WorkerInitObj:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def create_data_holder():
    """creat data holdr"""
    tokens = paddle.static.data(name="tokens", shape=[-1, -1], dtype="int64")
    loss_mask = paddle.static.data(
        name="loss_mask", shape=[-1, -1], dtype="float32")
    attention_mask = paddle.static.data(
        name="attention_mask", shape=[-1, 1, -1, -1], dtype="float32")
    position_ids = paddle.static.data(
        name="position_ids", shape=[-1, -1], dtype="int64")
    labels = paddle.static.data(name="labels", shape=[-1, -1], dtype="int64")
    return [tokens, loss_mask, attention_mask, position_ids, labels]


def create_strategy(args):
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    build_strategy.enable_addto = args.enable_addto
    build_strategy.enable_sequential_execution = args.use_recompute

    exec_strategy.num_threads = 1
    exec_strategy.num_iteration_per_drop_scope = 100
    return build_strategy, exec_strategy


def build_compiled_program(args, main_program, loss):
    build_strategy, exec_strategy = create_strategy(args)
    main_program = paddle.static.CompiledProgram(
        main_program).with_data_parallel(
            loss_name=loss.name,
            exec_strategy=exec_strategy,
            build_strategy=build_strategy)
    return main_program


def dist_optimizer(args, optimizer, model, worker_num):
    build_strategy, exec_strategy = create_strategy(args)

    exec_strategy.num_threads = 2
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.build_strategy = build_strategy
    dist_strategy.nccl_comm_num = 3

    dist_strategy.fuse_grad_size_in_MB = 16
    if args.use_amp:
        dist_strategy.amp = True
        dist_strategy.amp_configs = {
            "custom_white_list": ['softmax', 'gelu'],
            "init_loss_scaling": 32768,
            "use_dynamic_loss_scaling": True,
        }
        # dist_strategy.amp_configs = {
        #     "custom_white_list": ['softmax', 'layer_norm', 'gelu'],
        #     "init_loss_scaling": args.scale_loss,
        #     "incr_every_n_steps": 10,
        #     "decr_every_n_nan_or_inf": 1,
        #     "incr_ratio": 2.0,
        #     "decr_ratio": 10,
        #     "use_dynamic_loss_scaling": True,
        # }
    if args.use_sharding:
        dist_strategy.sharding = True
        dist_strategy.sharding_configs = {
            "segment_broadcast_MB": 32,
            "sharding_degree": args.sharding_degree,
            "mp_degree": args.mp_degree,
            "pp_degree": args.pp_degree,
            "dp_degree": args.dp_degree,
        }

    if args.use_recompute:
        dist_strategy.recompute = True
        dist_strategy.recompute_configs = {
            "checkpoints": model.gpt2.checkpoints
        }

    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
    return optimizer


def set_seed(args, worker_index):
    random.seed(args.seed + worker_index)
    np.random.seed(args.seed + worker_index)
    paddle.seed(args.seed + worker_index)


def reset_program_state_dict(model, state_dict):
    scale = model.initializer_range if hasattr(model, "initializer_range")\
        else model.gpt2.config["initializer_range"]
    print("the init scale is :{}".format(scale))
    new_state_dict = dict()
    for n, p in state_dict.items():
        if "layer_norm" not in p.name:
            dtype_str = "float32"
            if str(p.dtype) == "VarType.FP64":
                dtype_str = "float64"
            # print(p.name)
            new_state_dict[p.name] = np.random.normal(
                loc=0.0, scale=scale, size=p.shape).astype(dtype_str)
    return new_state_dict


def trans_param_from_static_to_dygrah(model):
    from paddlenlp.utils.tools import static_params_to_dygraph
    base_path = "/ssd1/zhonghui03/GPT2ModelCheckpoints"
    load_path = "model_10000_recompute"
    abs_path = os.path.join(base_path, load_path)
    static_tensor_dict = paddle.static.load_program_state(abs_path)
    new_dict = static_params_to_dygraph(model, static_tensor_dict)
    paddle.save(new_dict,
                os.path.join(base_path, "gpt2_%s.pdparams" % load_path))
    exit(0)


def init_static_with_params(model, dygraph_params):
    from paddlenlp.utils.tools import dygraph_params_to_static
    static_params = dygraph_params_to_static(model, dygraph_params)
    prog = paddle.static.default_main_program()
    paddle.static.set_program_state(prog, static_params)


def do_train(args):
    args.test_iters = args.eval_iters * 5
    args.sharding_degree = 2
    args.mp_degree = 2
    args.pp_degree = 1
    args.dp_degree = 1
    # Initialize the paddle and paddle fleet execute environment
    paddle.enable_static()
    assert args.device in [
        "cpu", "gpu", "xpu"
    ], "Invalid device! Available device should be cpu, gpu, or xpu."
    place = paddle.set_device(args.device)
    fleet.init(is_collective=True)

    worker_num = fleet.worker_num()
    worker_index = fleet.worker_index()

    topo = Topology(
        rank=worker_index,
        world_size=worker_num,
        dp=args.dp_degree,
        pp=args.pp_degree,
        sharding=args.sharding_degree,
        mp=args.mp_degree)

    # Create the random seed for the worker
    set_seed(args, worker_index)
    worker_init = WorkerInitObj(args.seed + worker_index)

    # create log write
    if worker_index == 0:
        log_writer_path = os.path.join(
            args.output_dir, "gpt2_bs_{}_amp_{}_recompute_{}_card_{}".format(
                args.batch_size, args.use_amp, args.use_recompute, worker_num))
        if os.path.exists(log_writer_path):
            import shutil
            shutil.rmtree(log_writer_path)
        log_writer = SummaryWriter(log_writer_path)

    # Define the input data in the static mode
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()
    data_holders = create_data_holder()
    [tokens, loss_mask, attention_mask, position_ids, labels] = data_holders

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    eod_id = tokenizer.command_name_map["eod"].Id
    model_config = model_class.pretrained_init_configuration[
        args.model_name_or_path]
    if model_config["vocab_size"] % 8 != 0:
        model_config["vocab_size"] += 8 - (model_config["vocab_size"] % 8)
    if args.mp_degree != 1:
        model_config["topo"] = topo

    # create the model for the gpt model
    model = GPT2ForPretraining(GPT2Model(**model_config))
    criterion = GPT2PretrainingCriterion()
    preds = model(tokens, position_ids, attention_mask)
    loss = criterion(preds, labels, loss_mask)

    # Create the learning_rate sheduler and optimizer
    if args.decay_steps is None:
        args.decay_steps = args.max_steps
    warmup_step = args.warmup_rate * args.decay_steps
    lr_scheduler = lr.CosineAnnealingWithWarmupDecay(
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_step=warmup_step,
        decay_step=args.decay_steps)

    clip = None
    if args.grad_clip > 0:
        clip = paddle.nn.ClipGradByNorm(clip_norm=args.grad_clip)

    opt_param = []
    for pa in model.parameters():
        if "batch_norm" not in pa.name:
            opt_param.append(pa)
        else:
            print(pa.name)
    decay_param = lambda x: x in [
             p.name for n, p in model.named_parameters()
             if not any(nd in n for nd in ["bias", "norm"])]

    #optimizer = paddle.optimizer.AdamW(
    optimizer = paddle.fluid.optimizer.Adam(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameter_list=opt_param,
        #weight_decay=args.weight_decay,
        grad_clip=clip,
        #apply_decay_param_fun=decay_param
    )

    # Use the fleet api to compile the distributed optimizer
    #optimizer.apply_optimize = optimizer._apply_optimize

    optimizer = dist_optimizer(args, optimizer, model, worker_num)
    optimizer.minimize(loss)
    logger.info("The training meta optimizer is/are %s" %
                fleet._get_applied_meta_list())

    # Define the Executor for running the static model
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    # trans_param_from_static_to_dygrah(model)

    # Use the state dict to update the parameter
    # state_dict = model.state_dict()
    # reset_state_dict = reset_program_state_dict(model, state_dict)
    # paddle.static.set_program_state(reset_state_dict)

    # init with Megatron params
    init_dir = None
    if args.batch_size == 4:
        print("Loading bs4 init params.")
        init_dir = os.path.join(os.environ['HOME'],
                                './init_checkponits/gpt2-init-bs4.pdparams')
    elif args.batch_size == 32:
        print("Loading bs32 init params.")
        init_dir = os.path.join(os.environ['HOME'],
                                './init_checkponits/gpt2-init-bs32.pdparams')
    if "small" in args.model_name_or_path:
        init_dir = os.path.join(
            os.environ['HOME'],
            './init_checkponits/gpt2-init-small-bs32.pdparams')
    if init_dir is not None:
        print(init_dir)
        # Sharding is incompatible with init params.
        #init_static_with_params(model, paddle.load(init_dir))

    test_program = main_program.clone(for_test=True)
    if worker_num == 1:
        # Construct the compiled program
        main_program = build_compiled_program(args, main_program, loss)

    program_desc_dir = os.path.join(args.output_dir, "program_desc")
    if not os.path.isdir(program_desc_dir):
        os.mkdir(program_desc_dir)

    with open(program_desc_dir + "/main_program.txt.%d" %
              (int(os.environ.get('FLAGS_selected_gpus', 0))), 'w') as f:
        f.write(str(main_program))

    with open(program_desc_dir + "/startup_program.txt.%d" %
              (int(os.environ.get('FLAGS_selected_gpus', 0))), 'w') as f:
        f.write(str(startup_program))

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        files = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if (os.path.isfile(os.path.join(args.input_dir, f)) and "npz_"
                not in str(f))
        ]
        files.sort()
        num_files = len(files)
        random.Random(args.seed + epoch).shuffle(files)
        for f_id in range(math.ceil(len(files) / worker_num)):
            data_file = files[(f_id * worker_num + worker_index) % num_files]
            train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
                args,
                data_file,
                worker_init,
                worker_index,
                worker_num,
                eod_id=eod_id,
                places=paddle.static.cuda_places(),
                data_holders=data_holders)
            # bug fix, if not call valid_data_loader, the enumerate will call valid_data_loader
            # many times. and start a new random dataloader.
            valid_data_loader = valid_data_loader()
            test_data_loader = test_data_loader()

            for step, batch in enumerate(train_data_loader()):
                global_step += 1
                loss_return, lr_return = exe.run(
                    main_program,
                    feed=batch,
                    fetch_list=[loss.name, 'learning_rate_0'])
                # In the new 2.0 api, must call this function to change the learning_rate
                lr_scheduler.step()

                if global_step % args.logging_steps == 0:
                    if worker_index == 0:
                        logger.info(
                            "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s, learning rate: %.9f"
                            % (global_step, epoch, step, loss_return[0],
                               args.logging_steps / (time.time() - tic_train),
                               lr_return[0]))
                        log_writer.add_scalar("loss", loss_return[0],
                                              global_step)
                        log_writer.add_scalar("learning_rate", lr_return[0],
                                              global_step)
                    tic_train = time.time()

                def run_evaluate(data_loader,
                                 program,
                                 iter_steps,
                                 task_name="valid"):
                    all_loss = []
                    local_time = time.time()
                    for eval_step, batch in enumerate(data_loader):
                        loss_return = exe.run(program,
                                              feed=batch,
                                              fetch_list=[loss.name])
                        all_loss.append(float(loss_return[0]))
                        if eval_step >= iter_steps - 1:
                            average_loss = sum(all_loss) / len(all_loss)
                            logger.info(
                                "%s step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                                % (task_name, global_step, epoch, eval_step,
                                   average_loss,
                                   iter_steps / (time.time() - local_time)))
                            log_writer.add_scalar(task_name + "_loss",
                                                  average_loss, global_step)
                            break

                if global_step % args.eval_steps == 0:
                    if worker_index == 0:
                        run_evaluate(valid_data_loader, test_program,
                                     args.eval_iters, "valid")
                        tic_train = time.time()

                if global_step % args.save_steps == 0 or global_step >= args.max_steps:
                    if worker_index == 0:
                        output_dir = os.path.join(args.output_dir,
                                                  "model_%d" % global_step)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        paddle.fluid.io.save_inference_model(
                            output_dir,
                            feeded_var_names=[
                                tokens.name,
                                loss_mask.name,
                                attention_mask.name,
                                position_ids.name,
                                labels.name,
                            ],
                            target_vars=[loss],
                            executor=exe)
                        tic_train = time.time()

                if global_step >= args.max_steps:
                    run_evaluate(test_data_loader, test_program,
                                 args.test_iters, "test")
                    del train_data_loader
                    return
            del train_data_loader
        epoch += 1


if __name__ == "__main__":
    config = parse_args()
    do_train(config)
