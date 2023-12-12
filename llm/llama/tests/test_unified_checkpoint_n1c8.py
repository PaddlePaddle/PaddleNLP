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

import copy
import os
import shutil

import numpy as np
from parallel_launch import TestMultipleGpus
from utils import get_pretrain_arguments

# export NVIDIA_TF32_OVERRIDE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=xgbe0
# export NCCL_IB_TIMEOUT=22
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_IB_GDR_LEVEL=4
# export NCCL_SOCKET_IFNAME=eth2


environment_variables = {
    "NCCL_ALGO": "Tree",
    "NVIDIA_TF32_OVERRIDE": "0",
    "NCCL_IB_TIMEOUT": "22",
    "NCCL_DEBUG": "INFO",
    "FLAGS_embedding_deterministic": "1",
    "FLAGS_cudnn_deterministic": "1",
    "Flags_mp_aysnc_allreduce": "1",
    "Flags_skip_mp_c_identity": "1",
    "FLAGS_shard_norm_align_dp": "0",
    "FLAGS_shard_use_reduce": "1",
    "test_ci_no_save_model": "1",
}

pretrain_arguments = {
    "model_name_or_path": "./tests/unified-ckpt-llama-500m",
    "tokenizer_name_or_path": "facebook/llama-7b",
    "input_dir": "./data",
    "output_dir": "./checkpoints/llama_pretrain_ckpts",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "per_device_eval_batch_size": 16,
    "tensor_parallel_degree": 2,
    "pipeline_parallel_degree": 4,
    "sharding": "",
    "virtual_pp_degree": 1,
    "sequence_parallel": 0,
    "use_flash_attention": "false",
    "use_fused_rms_norm": "false",
    "max_seq_length": 1024,
    "learning_rate": 1e-04,
    "min_learning_rate": 1e-05,
    "warmup_steps": 100,
    "logging_steps": 1,
    "max_steps": 30,
    "save_steps": 20,
    "eval_steps": 1000,
    "weight_decay": 0.01,
    "fp16": "true",
    "fp16_opt_level": "O2",
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 0,
    "continue_training": 0,
    "do_train": "true",
    "do_eval": "false",
    "do_predict": "false",
    "disable_tqdm": "true",
    "recompute": 1,
    "unified_checkpoint": 1,
    "distributed_dataloader": 0,
    "recompute_granularity": "full",
    "save_total_limit": 2,
}

# GBS: 16 MAX_steps: 30


def check_acc(log_dir="log"):
    file_path = os.path.join(log_dir, "workerlog.n0.c0")
    cmd = "grep -a 'global_step: 30' " + file_path + " | awk -F ','  '{print $2}' | awk  '{print $6}'"
    import subprocess

    res = subprocess.check_output(cmd, shell=True, text=True)
    res = [float(x) for x in res.split()]

    return res


def remove_logs(log_dir="log"):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)


def remove_ckpt(ckpt_dir):
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)


class TestModelOnN1C8(TestMultipleGpus):
    def setUp(self):
        os.environ.update(environment_variables)

    def testTP8(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 8
        train_args["pipeline_parallel_degree"] = 1

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testTP4PP2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 4
        train_args["pipeline_parallel_degree"] = 2

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testTP4DP2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 4
        train_args["pipeline_parallel_degree"] = 1
        train_args["sharding"] = ""
        train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testTP4Sharding2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 4
        train_args["pipeline_parallel_degree"] = 1
        train_args["sharding"] = "stage1"
        train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testTP2PP4(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 2
        train_args["pipeline_parallel_degree"] = 4

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testTP2Sharding4(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 2
        train_args["pipeline_parallel_degree"] = 1
        train_args["sharding"] = "stage1"
        train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 4

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testPP8(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 1
        train_args["pipeline_parallel_degree"] = 8

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testPP4DP2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 1
        train_args["pipeline_parallel_degree"] = 4
        train_args["sharding"] = ""
        train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testPP4Sharding2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 1
        train_args["pipeline_parallel_degree"] = 4
        train_args["sharding"] = "stage1"
        train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testSharding8S1(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 1
        train_args["pipeline_parallel_degree"] = 1
        train_args["sharding"] = "stage1"
        train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testSharding8S2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 1
        train_args["pipeline_parallel_degree"] = 1
        train_args["sharding"] = "stage2"
        train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testSharding4S1DP2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 1
        train_args["pipeline_parallel_degree"] = 1
        train_args["sharding_parallel_degree"] = 4
        train_args["sharding"] = "stage1"
        train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testSharding4S2DP2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 1
        train_args["pipeline_parallel_degree"] = 1
        train_args["sharding_parallel_degree"] = 4
        train_args["sharding"] = "stage2"
        train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testSharding2S1DP4(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 1
        train_args["pipeline_parallel_degree"] = 1
        train_args["sharding_parallel_degree"] = 2
        train_args["sharding"] = "stage1"
        train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testSharding2S2DP4(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 1
        train_args["pipeline_parallel_degree"] = 1
        train_args["sharding_parallel_degree"] = 2
        train_args["sharding"] = "stage2"
        train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])

    def testDP8(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])
        train_args = copy.deepcopy(pretrain_arguments)
        train_args["tensor_parallel_degree"] = 1
        train_args["pipeline_parallel_degree"] = 1
        train_args["sharding_parallel_degree"] = 1
        train_args["sharding"] = "stage2"
        train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8

        self.run_n1c8("run_pretrain.py", **train_args)
        self.run_n1c8("run_pretrain.py", **train_args)
        res = check_acc()
        assert len(res) == 2
        np.testing.assert_allclose(res[0], res[1])


class TestModelOnN1C8Dynamic(TestMultipleGpus):
    def setUp(self):
        os.environ.update(environment_variables)
        self.configs = get_pretrain_arguments(pretrain_arguments)

    def testTP8(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["TP8"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "TP8":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testTP4PP2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["TP4PP2"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "TP4PP2":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testTP4DP2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["TP4DP2"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "TP4DP2":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testTP4Sharding2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["TP4Sharding2"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "TP4Sharding2":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testTP2PP4(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["TP2PP4"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "TP2PP4":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testTP2Sharding4(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["TP2Sharding4"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "TP2Sharding4":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testPP8(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["PP8"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "PP8":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testPP4DP2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["PP4DP2"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "PP4DP2":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testPP4Sharding2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["PP4Sharding2"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "PP4Sharding2":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testSharding8S1(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["Sharding8S1"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "Sharding8S1":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testSharding8S2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["Sharding8S2"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "Sharding8S2":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testSharding4S1DP2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["Sharding4S1DP2"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "Sharding4S1DP2":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testSharding4S2DP2(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["Sharding4S2DP2"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "Sharding4S2DP2":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testSharding2S1DP4(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["Sharding2S1DP4"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "Sharding2S1DP4":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testSharding2S2DP4(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["Sharding2S2DP4"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "Sharding2S2DP4":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)

    def testDP8(self):
        remove_logs()
        remove_ckpt(pretrain_arguments["output_dir"])

        train_args = self.configs["DP8"]
        self.run_n1c8("run_pretrain.py", **train_args)

        for config_name, config in self.configs.items():
            if config_name == "DP8":
                continue
            self.run_n1c8("run_pretrain.py", **config)
            res = check_acc()
            np.testing.assert_allclose(res[0], res[-1], rtol=1e-4)
