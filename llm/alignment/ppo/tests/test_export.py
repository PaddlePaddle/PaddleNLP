# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os

from tests.parallel_launch import TestMultipleGpus

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


class TestExportEvalModel(TestMultipleGpus):
    def setUp(self):
        os.environ.update(environment_variables)
        super().setUp()

    def test_pptp_to_tp(self):
        config = {
            "output_dir": "./tmp",
            "model_name_or_path": "__internal_testing__/tiny-random-llama",
            "tensor_parallel_degree": 2,
            "pipeline_parallel_degree": 2,
        }
        scripts = "tests/run_model.py"
        self.run_4gpu(scripts, **config)

    def test_tp_to_single(self):
        config = {
            "output_dir": "./tmp",
            "model_name_or_path": "__internal_testing__/tiny-random-llama",
            "tensor_parallel_degree": 2,
            "pipeline_parallel_degree": 1,
        }
        scripts = "tests/run_model.py"
        self.run_2gpu(scripts, **config)

    def test_group_rank_guard(self):
        config = {
            "output_dir": "./tmp",
            "model_name_or_path": "__internal_testing__/tiny-random-llama",
            "tensor_parallel_degree": 2,
            "pipeline_parallel_degree": 1,
            "test_mode": "rank_guard",
        }
        scripts = "tests/run_model.py"
        self.run_2gpu(scripts, **config)
