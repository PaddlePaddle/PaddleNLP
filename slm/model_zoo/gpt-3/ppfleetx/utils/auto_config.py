# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import argparse
import os
import sys

import paddle
import paddle.distributed as dist
import paddle.distributed.auto_parallel as auto

from .config import (
    AttrDict,
    check_config,
    create_attr_dict,
    override_config,
    parse_config,
    print_config,
)
from .log import logger


def process_dist_configs(config):
    """
    process distributed strategy for auto parallel
    """
    nranks = dist.get_world_size()

    configs = config["Distributed"]

    mp_degree = configs.setdefault("mp_degree", 1)
    pp_degree = configs.setdefault("pp_degree", 1)

    # disenable sequence parallel is mp_degree < 2.
    sequence_parallel = config["Model"]["sequence_parallel"]
    if mp_degree < 2 and sequence_parallel:
        config["Model"]["sequence_parallel"] = False
        logger.warning(
            "sequence_parallel is turn off since mp_degree < 2."
        )


    # sharding default
    sharding_config = configs["sharding"]
    sharding_degree = sharding_config.setdefault("sharding_degree", 1)
    sharding_config.setdefault("sharding_stage", 2)
    sharding_config.setdefault("reduce_overlap", False)
    sharding_config.setdefault("broadcast_overlap", False)

    other_degree = mp_degree * pp_degree

    assert nranks % other_degree == 0, "Requires nranks should be divided by mp_degree*pp_degree."
    dp_degree = configs.setdefault("dp_degree", nranks // other_degree)
    assert nranks % dp_degree == 0, "unreasonable config of dist_strategy."
    assert nranks == dp_degree * other_degree, (
        "Mismatched config using {} cards with dp_degree[{}],"
        "mp_degree[{}], pp_degree[{}] and sharding_degree[{}]".format(
            nranks, dp_degree, mp_degree, pp_degree, sharding_degree
        )
    )


def process_global_configs(config):
    """
    process global configs for auto parallel
    """
    dp_degree = config["Distributed"]["dp_degree"]
    # pp_degree = config["Distributed"]["pp_degree"]
    # sharding_degree = config["Distributed"]["sharding"]["sharding_degree"]

    # TODO: support partial_send_recv
    # config["Global"]["enable_partial_send_recv"] = True
    # if config.get("Model", None) is not None and "sequence_parallel" in config["Model"] and pp_degree > 1:
    #     if config["Model"]["sequence_parallel"]:
    #         config["Global"]["enable_partial_send_recv"] = False
    #         logger.warning(
    #             "if config.Distributed.pp_degree > 1 and config.Model.sequence_parallel is True, "
    #             "config.Global.enable_partial_send_recv will be set False."
    #         )

    global_cfg = config["Global"]

    # Set environment variable
    flags = global_cfg.get("flags", {})
    paddle.set_flags(flags)
    for k, v in flags.items():
        logger.info("Environment variable {} is set {}.".format(k, v))

    if global_cfg["global_batch_size"] is None and global_cfg["local_batch_size"] is None:
        raise ValueError("global_batch_size or local_batch_size should be set.")
    elif global_cfg["global_batch_size"] is not None and global_cfg["local_batch_size"] is not None:
        assert (
            global_cfg["global_batch_size"] // global_cfg["local_batch_size"] == dp_degree
        ), "global_batch_size[{}] should be divided by local_batch_size[{}] when dp_degree is [{}]".format(
            global_cfg["global_batch_size"], global_cfg["local_batch_size"], dp_degree
        )
    elif global_cfg["global_batch_size"] is not None and global_cfg["local_batch_size"] is None:
        assert (
            global_cfg["global_batch_size"] % dp_degree == 0
        ), "global_batch_size[{}] should be divided by dp_degree[{}]".format(
            global_cfg["global_batch_size"], dp_degree
        )
        global_cfg["local_batch_size"] = global_cfg["global_batch_size"] // dp_degree
    else:
        global_cfg["global_batch_size"] = global_cfg["local_batch_size"] * dp_degree
    assert global_cfg["local_batch_size"] % global_cfg["micro_batch_size"] == 0


def process_engine_configs(config):
    """
    process engine configs for auto parallel
    """
    if config.Engine.get("verbose", None) is None:
        config.Engine["verbose"] = 2
    if config.Engine.get("logging_freq", None) is None:
        config.Engine["logging_freq"] = 10
    config.Engine["save_load"] = config.Engine.get("save_load", {})
    save_load_cfg = config.Engine.save_load
    save_steps = save_load_cfg.get("save_steps", None)
    save_epoch = save_load_cfg.get("save_epoch", None)
    if save_steps is None or save_steps == -1:
        save_load_cfg["save_steps"] = sys.maxsize if sys.version > "3" else sys.maxint

    if save_epoch is None or save_epoch == -1:
        save_load_cfg["save_epoch"] = 1

    save_load_cfg["output_dir"] = save_load_cfg.get("output_dir", "./output")
    save_load_cfg["ckpt_dir"] = save_load_cfg.get("ckpt_dir", None)

    config.Engine["max_steps"] = config.Engine.get("max_steps", 500000)
    config.Engine["eval_freq"] = config.Engine.get("eval_freq", -1)
    config.Engine["eval_iters"] = config.Engine.get("eval_iters", 0)
    config.Engine["logging_freq"] = config.Engine.get("logging_freq", 1)
    config.Engine["num_train_epochs"] = config.Engine.get("num_train_epochs", 1)
    config.Engine["test_iters"] = (
        config.Engine["eval_iters"] * 10
        if config.Engine.get("test_iters", None) is None
        else config.Engine["test_iters"]
    )
    config.Engine["accumulate_steps"] = config.Global.local_batch_size // config.Global.micro_batch_size

def use_pir():
    is_pir_mode = os.environ.get("FLAGS_enable_pir_in_executor", None)
    return str(is_pir_mode).lower() not in ('false', 'off', '0', 'none')

def process_strategy(config):
    """
    process auto strategy for auto parallel
    """
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    # strategy.seed = config["Global"]["seed"]

    if config.get("FusedPasses", None) is not None:
        # fused passes config
        fused_passes_list = []
        fused_linear = config.FusedPasses.pop("fused_linear", False)
        fused_adamw = config.FusedPasses.pop("fused_adamw", False)
        if fused_linear:
            if use_pir():
                fused_passes_list.append("fused_gemm_epilogue_pass")
            else:
                fused_passes_list.append("fuse_gemm_epilogue")
        if fused_adamw:
            fused_passes_list.append("fuse_adamw")
        fused_passes = strategy.fused_passes
        fused_passes.enable = len(fused_passes_list) > 0
        fused_passes.fused_passes_list = fused_passes_list

    if config.get("Model", None) is not None:
        # recompute config
        if not config.Model.get("no_recompute_layers", None):
            config.Model["no_recompute_layers"] = []
        else:
            assert isinstance(config.Model["no_recompute_layers"], list), "no_recompute_layers should be a list"
            for i in config.Model["no_recompute_layers"]:
                assert isinstance(i, int), "all values in no_recompute_layers should be an integer"
            assert min(config.Model["no_recompute_layers"]) >= 0, "the min value in no_recompute_layers should >= 0"
            assert (
                max(config.Model["no_recompute_layers"]) < config.Model["num_layers"]
            ), "the max value in no_recompute_layers should < num_layers"
            config.Model["no_recompute_layers"] = sorted(list(set(config.Model["no_recompute_layers"])))
        recompute = strategy.recompute
        recompute.enable = config.Model.get("use_recompute", False)
        recompute.sr = config.Model.pop("sr", 0)
        recompute.refined_ops_patterns = config.Model.pop("refined_ops_patterns", []) # gpt.GPTModelAuto don't need this parameter
        recompute.no_recompute_segments = config.Model.pop("no_recompute_layers", [])
        recompute.enable_tuning = config.get("Tuning", False) and config.Tuning.get("tuning_recompute", False)

    # amp config
    amp_cfg = config.Engine.get("mix_precision", {})
    amp = strategy.amp
    amp.enable = amp_cfg.get("enable", False)
    amp.dtype = amp_cfg.get("dtype", "float16")
    amp.level = amp_cfg.get("level", "o2")
    amp.init_loss_scaling = amp_cfg.get("scale_loss", 32768)
    amp.custom_black_list = amp_cfg.get("custom_black_list", [])
    amp.custom_white_list = amp_cfg.get("custom_white_list", [])
    amp.use_fp16_guard = amp_cfg.get("use_fp16_guard", False)
    amp.use_bf16_guard = amp_cfg.get("use_bf16_guard", False)

    # mp_optimization config
    mp_degree = config.Distributed.get("mp_degree", 1)
    if mp_degree > 1:
        mp_cfg = config.Distributed.get("mp_optimization", {})
        strategy.mp_optimization.allreduce_matmul_grad_overlapping = mp_cfg.get("allreduce_matmul_grad_overlapping", False)

    # sharding config
    sharding_cfg = config.Distributed.get("sharding", {})
    sharding = strategy.sharding
    sharding.enable = sharding_cfg.get("sharding_degree", 1) > 1
    sharding.degree = sharding_cfg.get("sharding_degree", 1)
    sharding.stage = sharding_cfg.get("sharding_stage", 1)
    sharding.enable_overlap = sharding_cfg.get("reduce_overlap", False) and sharding_cfg.get("broadcast_overlap", False)
    sharding.param_comm_stream_num = sharding_cfg.get("param_comm_stream_num", 1)
    sharding.grad_comm_stream_num = sharding_cfg.get("grad_comm_stream_num", 1)
    sharding.param_bucket_size_numel = sharding_cfg.get("param_bucket_size_numel", 1)
    sharding.grad_bucket_size_numel = sharding_cfg.get("grad_bucket_size_numel", 1)
    sharding.enable_hierarchical_comm = sharding_cfg.get("enable_hierarchical_comm", False)

    pp_degree = config["Distributed"]["pp_degree"]
    accumulate_steps = config.Engine.get("accumulate_steps", 1)
    if pp_degree > 1 and accumulate_steps > 1:
        # pipeline config
        pipeline_cfg = config.Distributed.get("pipeline", {})
        pipeline = strategy.pipeline
        pipeline.enable = True
        pipeline.enable_send_recv_overlap = pipeline_cfg.get("enable_send_recv_overlap", False)
        pipeline.schedule_mode = pipeline_cfg.get("schedule_mode", "1F1B")
        pipeline.micro_batch_size = config.Global.micro_batch_size
        pipeline.accumulate_steps = accumulate_steps
        pipeline.job_schedule_profiler_start = pipeline_cfg.get("job_schedule_profiler_start", -1)
        pipeline.job_schedule_profiler_stop = pipeline_cfg.get("job_schedule_profiler_stop", -1)
        
    elif accumulate_steps > 1:
        # gradient merge config
        gradient_merge = strategy.gradient_merge
        gradient_merge.enable = True
        gradient_merge.k_steps = accumulate_steps

    # quantization config
    qat_cfg = config.get("Quantization", {})
    qat = strategy.qat
    qat.enable = qat_cfg.get("enable", False)
    qat.channel_wise_abs_max = qat_cfg.get("channel_wise_abs_max", True)
    qat.weight_bits = qat_cfg.get("weight_bits", 8)
    qat.activation_bits = qat_cfg.get("activation_bits", 8)
    qat.onnx_format = qat_cfg.get("onnx_format", True)

    # tuning config
    tuning_cfg = config.get("Tuning", {})
    tuning = strategy.tuning
    tuning.enable = tuning_cfg.get("enable", False)
    tuning.profile_start_step = tuning_cfg.get("profile_start_step", 1)
    tuning.profile_end_step = tuning_cfg.get("profile_end_step", 1)
    tuning.run_after_tuning = tuning_cfg.get("run_after_tuning", True)
    tuning.debug = tuning_cfg.get("debug", True)

    # sequence parallel config
    if config.Model.get("sequence_parallel", False):
        sp_optimization = strategy.sp_optimization
        sp_optimization.enable = True

    engine_cfg = config["Engine"]
    engine_cfg["strategy"] = strategy


def process_ckpt_dir(config):
    configs = config["Engine"]["save_load"]
    ckpt_dir = configs.get("ckpt_dir", None)
    if ckpt_dir is None:
        return

    assert (
        os.path.isdir(ckpt_dir) is False
    ), "Wrong setting of ckpt_dir!ckpt_dir can't be a folder, but {} is a folder. Your `ckpt_dir` should be `dirname/prefix` like `output/auto` if your model path is `output/auto_dist0.pdparams`".format(
        ckpt_dir
    )

    assert os.path.exists(ckpt_dir) is False, (
        "Wrong setting of ckpt_dir,"
        "if you want to load weight,you should set ckpt_dir like this!"
        "for example:\ngpt_auto_model_save\n\t--auto_dist0.pdparams\n\t--auto_dist0.pdparams\n"
        '\t--auto_dist0.pdattr\nyou should set ckpt_dir="gpt_auto_model_save/auto"'
    )

    parent_path = os.path.split(ckpt_dir)[0]

    if os.path.exists(parent_path) is False:
        logger.warning("{} path is not existed!we will set ckpt_dir None.".format(parent_path))
        configs["ckpt_dir"] is None


def get_config(fname, overrides=None, show=False):
    """
    Read config from file for auto parallel
    """
    assert os.path.exists(fname), "config file({}) is not exist".format(fname)
    config = parse_config(fname)
    override_config(config, overrides)

    process_dist_configs(config)
    process_global_configs(config)
    process_engine_configs(config)
    process_strategy(config)
    process_ckpt_dir(config)
    create_attr_dict(AttrDict(config))

    if show:
        print_config(config)
    check_config(config)
    return config


def parse_args():
    parser = argparse.ArgumentParser("train script")
    parser.add_argument("-c", "--config", type=str, default="configs/config.yaml", help="config file path")
    parser.add_argument("-o", "--override", action="append", default=[], help="config options to be overridden")
    args = parser.parse_args()
    return args
