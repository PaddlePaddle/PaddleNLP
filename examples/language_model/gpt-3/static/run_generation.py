#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import random
import math
import time
import numpy as np

os.path.expandvars('$HOME')
os.path.expanduser('~')

import paddle
import paddle.distributed.fleet as fleet

from paddlenlp.transformers import GPTTokenizer, GPTChineseTokenizer
from paddlenlp.ops import guard, Topology, get_rng_state_tracker
from paddlenlp.utils.log import logger
import paddlenlp.ops as ops

from paddle.distributed import init_parallel_env
from paddle.static.amp import AutoMixedPrecisionLists

from modeling import GPTModel, GPTForPretraining, GPTForGeneration

# Used to load the data_tools path, should import before dataset
filepath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(filepath, "../../"))
from dataset import create_pretrained_dataset
from args import parse_args
import lr

MODEL_CLASSES = {
    "gpt": (GPTForGeneration, GPTTokenizer),
    "gpt-cn": (GPTForGeneration, GPTChineseTokenizer),
}

USE_LOCAL_HPI = True

device = "gpu"
ascend = False
int_type = "int64"
device_id = int(os.environ.get('FLAGS_selected_gpus', 0))

# yapf: enable.


def create_data_holder(args):
    shapes = [[-1, -1], [-1, -1], [-1, -1]]
    dtypes = [int_type, 'float32', int_type]
    names = ['src_ids', 'input_mask', 'pos_ids']  # three inputs
    #names = ['src_ids']  # one input

    inputs = [
        paddle.static.data(name=names[i], shape=shapes[i], dtype=dtypes[i])
        for i in range(len(names))
    ]
    return inputs


def debug_program(name, program):
    with open("{}.txt.{}".format(name, device_id), 'w') as f:
        f.write(str(program))


def get_data_file(args):
    files = [
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if (os.path.isfile(os.path.join(args.input_dir, f))
            and str(f).endswith("_idx.npz"))
    ]
    files = [x.replace("_idx.npz", "") for x in files]
    if len(files) == 0:
        logger.warning(
            "Not found dataset with name of xxx_ids.npy and xxx_idx.npz! \
            Try to found old compatible xxx_ids.npz file.")
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


def do_generation(args):
    # Initialize the paddle and paddle fleet execute environment
    paddle.enable_static()

    assert args.dp_degree == 1, "Data parallel is not supported in inference"
    assert args.sharding_degree == 1, "Sharding parallel is temporarily not supported in inference"
    assert args.pp_degree == 1, "Pipeline parallel will be supported later"

    if args.mp_degree == 1:
        args.mp_degree = paddle.distributed.get_world_size()
    else:
        assert args.mp_degree == paddle.distributed.get_world_size(), \
            "If mp_degree is specified, the size must be the same as world_size"

    strategy = fleet.DistributedStrategy()
    strategy.tensor_parallel = True
    strategy.tensor_parallel_configs = {
        "tensor_parallel_degree": args.mp_degree
    }

    white_list = [
        'softmax', 'layer_norm', 'gelu', 'fused_softmax_mask_upper_triangle',
        'elementwise_add'
    ]
    black_list = [
        'reduce_sum', 'c_softmax_with_cross_entropy', 'elementwise_div'
    ]

    if args.use_amp:
        strategy.amp = True
        strategy.amp_configs = {
            "custom_white_list": white_list,
            "custom_black_list": black_list,
            "init_loss_scaling": 32768,
            "use_dynamic_loss_scaling": True,
            "use_pure_fp16": args.amp_level == "O2",
            "use_fp16_guard": False
        }

    fleet.init(is_collective=True, strategy=strategy)

    # temp use dynamic init, use HybridParallelInferenceHelper in future?
    paddle.distributed.init_parallel_env()

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

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    pretrained_models_list = list(
        model_class.pretrained_init_configuration.keys())

    data_file = get_data_file(args)
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()
    with paddle.static.program_guard(main_program, startup_program):
        with paddle.utils.unique_name.guard():
            with paddle.static.device_guard(
                    'gpu:0' if topo.pp_info.size > 1 else None):
                feeds = create_data_holder(args)
                tokenizer = tokenizer_class.from_pretrained(
                    args.model_name_or_path)
                eos_id = tokenizer.eos_token_id

                _, _, test_data_loader = create_pretrained_dataset(
                    args,
                    data_file,
                    local_rank=local_rank,
                    data_world_size=topo.data_info.size,
                    data_world_rank=topo.data_info.rank,
                    eos_id=eos_id,
                    max_seq_len=args.max_seq_len,
                    places=paddle.static.cuda_places(),
                    data_holders=feeds,
                    pipeline_mode=False)

                if args.model_name_or_path in pretrained_models_list:
                    model_config = model_class.pretrained_init_configuration[
                        args.model_name_or_path]
                    model_config[
                        "hidden_dropout_prob"] = args.hidden_dropout_prob
                    model_config[
                        "attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
                    model_config["topo"] = topo
                    model_config["fuse"] = args.fuse
                    model_config["fuse_mt"] = args.fuse_mt
                    model = GPTForGeneration(
                        GPTModel(**model_config),
                        max_length=args.max_dec_len,
                        decoding_strategy=args.decoding_strategy,
                        temperature=args.temperature,
                        top_k=args.topk,
                        top_p=args.topp,
                        eos_id=eos_id,
                        fuse=args.fuse,
                        fuse_mt=args.fuse_mt)
                else:
                    logger.error("No checkpoint load.")
                model.eval()
                ins = {v.name: v for v in feeds}
                preds = model(ins)

            if args.use_amp:
                amp_lists = AutoMixedPrecisionLists(
                    custom_white_list=set(white_list),
                    custom_black_list=set(black_list))
                # Cast model with while loop to fp16
                from utils.fp16_util import cast_model_to_fp16_block
                fp16_var_names = cast_model_to_fp16_block(
                    main_program, amp_lists, False)

                # Change parameters' and ops' dtype to fp16, or will cause OOM while running
                # startup program
                block = startup_program.block(0)
                for var_name in fp16_var_names:
                    var = block.var(var_name)
                    block._remove_var(var_name, sync=False)
                    cast_var = block.create_var(name=var_name,
                                                dtype='float16',
                                                shape=var.shape,
                                                persistable=True)
                    for op in block.ops:
                        for out_name in op.output_arg_names:
                            if out_name == var_name:
                                op._set_attr("dtype", paddle.float16)

    # Define the Executor for running the static model
    exe = paddle.static.Executor(place)
    exe.run(startup_program)
    main_program = main_program.clone(for_test=True)
    #debug_program('main_program', main_program)

    model_urls = model.pretrained_resource_files_map['model_state']
    model_path = args.model_name_or_path
    if model_path in pretrained_models_list and model_path in model_urls:
        flag_loaded = False
        from paddle.utils.download import get_weights_path_from_url
        dygraph_path = get_weights_path_from_url(model_urls[model_path])
        if os.path.exists(dygraph_path):
            if args.sharding_degree > 1:
                logger.warning("Sharding should init with static vars")
            else:
                logger.info("Loading parameters from %s" % dygraph_path)
                init_static_with_params(
                    model, paddle.load(dygraph_path, return_numpy=True), topo,
                    main_program)
                flag_loaded = True
        if not flag_loaded:
            logger.error("No checkpoint load.")

    global_step = 0
    epoch = 0
    fetchs = [preds]

    ### check resutls
    text = [
        "Question: Where is the capital of China? Answer:",
        "Question:Who is the CEO of Apple? Answer:"
    ]
    inputs = tokenizer(text,
                       padding=True,
                       return_attention_mask=True,
                       return_position_ids=True)
    ids = np.array(inputs["input_ids"]).reshape(len(text), -1).astype('int64')
    position_ids = np.array(inputs["position_ids"]).reshape(len(text),
                                                            -1).astype('int64')
    attention_mask = np.array(inputs["attention_mask"]).reshape(
        len(text), -1).astype('float32')

    t_ids = paddle.framework.core.Tensor()
    t_ids.set(ids, place)
    t_mask = paddle.framework.core.Tensor()
    t_mask.set(attention_mask, place)
    t_pos = paddle.framework.core.Tensor()
    t_pos.set(position_ids, place)
    feed_data = {'src_ids': t_ids, 'pos_ids': t_pos, 'input_mask': t_mask}
    ret = exe.run(main_program, feed=feed_data, fetch_list=fetchs)
    ret = np.array(ret[0])
    for i in range(ret.shape[0]):
        o = [int(x) for x in ret[i]]
        ret_str = tokenizer.convert_ids_to_string(o)
        ret_str = text[i] + ret_str
        logger.info(ret_str)
    ##################

    for step, batch in enumerate(test_data_loader()):
        ret = exe.run(main_program, feed=batch, fetch_list=fetchs)
        if step == 5:
            break

    if args.save_inference_model_then_exist:
        save_inference_model_dir = 'inference_model_pp{pp_degree}mp{mp_degree}'.format(
            pp_degree=args.pp_degree, mp_degree=args.mp_degree)
        inference_save_path = os.path.join(save_inference_model_dir,
                                           'rank_' + str(fleet.worker_index()),
                                           'step_' + str(0))
        print("saving inference models to {}".format(inference_save_path))
        feed_names = [v.name for v in feeds]
        fetchs_names = [v.name for v in fetchs]
        print('feeds: ', feed_names, 'fetches: ', fetchs_names)
        paddle.static.save_inference_model(inference_save_path,
                                           feeds,
                                           fetchs,
                                           exe,
                                           program=main_program)


if __name__ == '__main__':
    args = parse_args(MODEL_CLASSES)
    do_generation(args)
