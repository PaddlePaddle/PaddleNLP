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

# import copy
import json
import os
import shutil

# import numpy as np
from llama.tests.parallel_launch import TestMultipleGpus

# export NVIDIA_TF32_OVERRIDE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=xgbe0
# export NCCL_IB_TIMEOUT=22
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_IB_GDR_LEVEL=4
# export NCCL_SOCKET_IFNAME=eth2


environment_variables = {
    # "NCCL_ALGO": "Tree",
    # "NVIDIA_TF32_OVERRIDE": "0",
    "NCCL_IB_TIMEOUT": "22",
    # "NCCL_DEBUG": "INFO",
    # "FLAGS_embedding_deterministic": "1",
    # "FLAGS_cudnn_deterministic": "1",
    # "Flags_mp_aysnc_allreduce": "1",
    # "Flags_skip_mp_c_identity": "1",
    # "FLAGS_shard_norm_align_dp": "0",
    # "FLAGS_shard_use_reduce": "1",
    "test_ci_no_save_model": "1",
}

pretrain_arguments = {
    "learning_rate": 1e-04,
    "min_learning_rate": 1e-05,
    "warmup_steps": 100,
    "logging_steps": 1,
    "max_steps": 10,
    "save_steps": 2000,
    "eval_steps": 1000,
    "continue_training": 0,
    "skip_memory_metrics": 0,
    "do_train": "true",
    "do_eval": "false",
    "do_predict": "false",
    "disable_tqdm": "true",
    "save_total_limit": 2,
}

best_pretrain_config_for_a100_80g = {
    # "qwen/qwen-7b": "./qwen/pretrain_argument_stage2.json",
    # "baichuan-inc/Baichuan2-13B-Base" "./llama/pretrain-baichuan2_13b-tp4sd2_stage2.json",
    # "baichuan-inc/Baichuan2-13B-Base": "./llama/pretrain-baichuan2_13b-tp2sd4_stage2.json",
    "facebook/llama-7b": "./llama/pretrain-llama_7b-tp2sd4_stage2.json",
    "facebook/llama-13b": "./llama/pretrain-llama_13b-tp2sd4_stage2.json",
    "meta-llama/Llama-2-7b": "./llama/pretrain-llama2_7b-tp2sd4_stage2.json",
    "meta-llama/Llama-2-13b": "./llama/pretrain-llama2_13b-tp2sd4_stage2.json",
    "qwen/qwen-7b": "./qwen/pretrain-qwen_7b-tp2sd4_stage2.json",
    "baichuan-inc/Baichuan2-13B-Base": "./baichuan/pretrain-baichuan2_13b-sd8_stage2.json",
    "baichuan-inc/Baichuan2-7B-Base": "./baichuan/pretrain-baichuan2_7b-tp2sd4_stage2.json",
    "FlagAlpha/Llama2-Chinese-13b-Chat": "./llama/pretrain-flagalpha_llama2_13b-tp2sd4_stage2.json",
    "FlagAlpha/Llama2-Chinese-7b-Chat": "./llama/pretrain-flagalpha_llama2_7b-tp2sd4_stage2.json",
    "linly-ai/chinese-llama-2-7b": "./llama/pretrain-linly_llama2_7b-tp2sd4_stage2.json",
    "idea-ccnl/ziya-llama-13b-v1": "./llama/pretrain-ziya_llama_13b-tp2sd4_stage2.json",
}


def log_test_result(model_name_or_path, config_name, config, log_dir="log"):
    model_name_or_path = model_name_or_path
    max_seq_len = config["max_seq_length"]
    distribued_info = config_name.split("b-")[-1].split(".json")[0]
    speed = "NA"
    memory = "NA"
    config_name = config_name
    time = "NA"

    file_path = os.path.join(log_dir, "workerlog.n0.c0")

    get_memory_cmd = (
        "grep -aE 'gpu_mem_max_memory_reserved ' " + file_path + " | awk '{print $8}' |  awk -F '\x1b'  '{print $1}'"
    )
    get_time_cmd = (
        "grep -aE 'gpu_mem_max_memory_reserved ' "
        + file_path
        + " | awk -F '['  '{print $3}' | awk -F ','  '{print $1}'"
    )
    get_ips_cmd = "grep -aE 'global_step: ' " + file_path + "  | awk -F ',' '{print $6}' | awk   '{print $2}'  "

    import subprocess

    res = subprocess.check_output(get_memory_cmd, shell=True, text=True)
    if "MB" in res:
        memory = res.strip()

    res = subprocess.check_output(get_time_cmd, shell=True, text=True)
    if len(res) > 0:
        time = res.strip()

    res = subprocess.check_output(get_ips_cmd, shell=True, text=True)
    ips = [float(x) for x in res.strip().split()]
    if len(ips) > 4:
        ips = sum(ips[2:-2]) / (len(ips) - 4)
        speed = round(ips * max_seq_len / 8, 2)

    write_result(
        [
            f"`{model_name_or_path}`",
            max_seq_len,
            f"`{distribued_info}`",
            speed,
            memory,
            f"`{config_name}`",
            time,
        ]
    )

    return res


result_title = r"""| 模型 | 序列长度 | 分布式策略 | 速度(`tokens/card/sec`) | 显存占用(`MB^1`) | 配置文件| 测试时间 |"""
result_file_name = "results_of_best_pretrain_config_for_a100_80g.md"


def write_result(res):
    fileds_name = [x.strip() for x in result_title.split("|")[1:-1]]
    assert len(fileds_name) == len(res)

    def format_list_to_str(lst):
        content = "|".join([""] + ["{:10}".format(x) for x in lst] + [""])
        return content

    if not os.path.exists(result_file_name):
        with open(result_file_name, "w") as f:
            f.write(format_list_to_str(fileds_name) + "\n")
            f.write(format_list_to_str([" :-: "] * len(fileds_name)) + "\n")

    with open(result_file_name, "a+") as f:
        f.write(format_list_to_str(res) + "\n")


def remove_logs(log_dir="log"):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)


def remove_ckpt(ckpt_dir):
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)


class TestModelOnN1C8(TestMultipleGpus):
    def setUp(self):
        os.environ.update(environment_variables)

    def test_facebook_llama_7b(self):
        name = "facebook/llama-7b"
        arguments = json.load(open(best_pretrain_config_for_a100_80g[name], "r"))
        arguments.update(pretrain_arguments)
        remove_logs()
        remove_ckpt(arguments["output_dir"])
        self.run_n1c8("run_pretrain.py", **arguments)
        log_test_result(name, best_pretrain_config_for_a100_80g[name], arguments, log_dir="log")

    def test_facebook_llama_13b(self):
        name = "facebook/llama-13b"
        arguments = json.load(open(best_pretrain_config_for_a100_80g[name], "r"))
        arguments.update(pretrain_arguments)
        remove_logs()
        remove_ckpt(arguments["output_dir"])
        self.run_n1c8("run_pretrain.py", **arguments)
        log_test_result(name, best_pretrain_config_for_a100_80g[name], arguments, log_dir="log")

    def test_metallama_Llama2_7b(self):
        name = "meta-llama/Llama-2-7b"
        arguments = json.load(open(best_pretrain_config_for_a100_80g[name], "r"))
        arguments.update(pretrain_arguments)
        remove_logs()
        remove_ckpt(arguments["output_dir"])
        self.run_n1c8("run_pretrain.py", **arguments)
        log_test_result(name, best_pretrain_config_for_a100_80g[name], arguments, log_dir="log")

    def test_metallama_Llama2_13b(self):
        name = "meta-llama/Llama-2-13b"
        arguments = json.load(open(best_pretrain_config_for_a100_80g[name], "r"))
        arguments.update(pretrain_arguments)
        remove_logs()
        remove_ckpt(arguments["output_dir"])
        self.run_n1c8("run_pretrain.py", **arguments)
        log_test_result(name, best_pretrain_config_for_a100_80g[name], arguments, log_dir="log")

    def test_qwen_qwen_7b(self):
        name = "qwen/qwen-7b"
        arguments = json.load(open(best_pretrain_config_for_a100_80g[name], "r"))
        arguments.update(pretrain_arguments)
        remove_logs()
        remove_ckpt(arguments["output_dir"])
        self.run_n1c8("run_pretrain.py", **arguments)
        log_test_result(name, best_pretrain_config_for_a100_80g[name], arguments, log_dir="log")

    def test_baichuan_Baichuan2_13B_Base(self):
        name = "baichuan-inc/Baichuan2-13B-Base"
        arguments = json.load(open(best_pretrain_config_for_a100_80g[name], "r"))
        arguments.update(pretrain_arguments)
        remove_logs()
        remove_ckpt(arguments["output_dir"])
        self.run_n1c8("run_pretrain.py", **arguments)
        log_test_result(name, best_pretrain_config_for_a100_80g[name], arguments, log_dir="log")

    def test_baichuan_Baichuan2_7B_Base(self):
        name = "baichuan-inc/Baichuan2-7B-Base"
        arguments = json.load(open(best_pretrain_config_for_a100_80g[name], "r"))
        arguments.update(pretrain_arguments)
        remove_logs()
        remove_ckpt(arguments["output_dir"])
        self.run_n1c8("run_pretrain.py", **arguments)
        log_test_result(name, best_pretrain_config_for_a100_80g[name], arguments, log_dir="log")

    def test_FlagAlpha_Llama2Chinese_13b_Chat(self):
        name = "FlagAlpha/Llama2-Chinese-13b-Chat"
        arguments = json.load(open(best_pretrain_config_for_a100_80g[name], "r"))
        arguments.update(pretrain_arguments)
        remove_logs()
        remove_ckpt(arguments["output_dir"])
        self.run_n1c8("run_pretrain.py", **arguments)
        log_test_result(name, best_pretrain_config_for_a100_80g[name], arguments, log_dir="log")

    def test_FlagAlpha_Llama2Chinese_7b_Chat(self):
        name = "FlagAlpha/Llama2-Chinese-7b-Chat"
        arguments = json.load(open(best_pretrain_config_for_a100_80g[name], "r"))
        arguments.update(pretrain_arguments)
        remove_logs()
        remove_ckpt(arguments["output_dir"])
        self.run_n1c8("run_pretrain.py", **arguments)
        log_test_result(name, best_pretrain_config_for_a100_80g[name], arguments, log_dir="log")

    def test_linlyai_chinesellama2_7b(self):
        name = "linly-ai/chinese-llama-2-7b"
        arguments = json.load(open(best_pretrain_config_for_a100_80g[name], "r"))
        arguments.update(pretrain_arguments)
        remove_logs()
        remove_ckpt(arguments["output_dir"])
        self.run_n1c8("run_pretrain.py", **arguments)
        log_test_result(name, best_pretrain_config_for_a100_80g[name], arguments, log_dir="log")

    def test_ideaccnl_ziyallama_13b(self):
        name = "idea-ccnl/ziya-llama-13b-v1"
        arguments = json.load(open(best_pretrain_config_for_a100_80g[name], "r"))
        arguments.update(pretrain_arguments)
        remove_logs()
        remove_ckpt(arguments["output_dir"])
        self.run_n1c8("run_pretrain.py", **arguments)
        log_test_result(name, best_pretrain_config_for_a100_80g[name], arguments, log_dir="log")
