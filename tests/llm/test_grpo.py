# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import unittest

from parameterized import parameterized_class

from .testing_utils import LLMTest


@parameterized_class(
    ["model_dir"],
    [["qwen"]],
)
class GRPOTest(LLMTest, unittest.TestCase):
    config_path: str = None
    model_dir: str = None

    def setUp(self) -> None:
        LLMTest.setUp(self)
        sys.path.insert(0, "./llm/alignment/rl")
        sys.path.insert(0, self.model_dir)

    def tearDown(self) -> None:
        LLMTest.tearDown(self)

    def test_grpo(self):
        # 设置必要的环境变量
        env_vars = {
            "PYTHONPATH": f"{os.path.abspath('./')}:{os.path.abspath('./llm')}:" + os.environ.get("PYTHONPATH", ""),
            "FLAGS_set_to_1d": "False",
            "NVIDIA_TF32_OVERRIDE": "0",
            "FLAGS_dataloader_use_file_descriptor": "False",
            "HF_DATASETS_DOWNLOAD_TIMEOUT": "1",
            "FLAGS_gemm_use_half_precision_compute_type": "False",
            "FLAGS_force_cublaslt_no_reduced_precision_reduction": "True",
            "FLAGS_mla_use_tensorcore": "0",
            "FLAGS_cascade_attention_max_partition_size": "2048",
        }
        case_env = os.environ.copy()
        case_env.update(env_vars)

        # 修改执行路径
        repo_path = os.getcwd()
        rl_dir = os.path.join(os.getcwd(), "./llm/alignment/rl")
        os.chdir(rl_dir)

        # 下载并解压数据
        if not os.path.exists("ppo-kk.tgz"):
            subprocess.run(
                "wget -q https://paddlenlp.bj.bcebos.com/datasets/examples/ppo-kk.tgz && tar zxf ppo-kk.tgz",
                shell=True,
                check=True,
            )

        # 启动 reward server
        reward_dir = os.path.join(os.getcwd(), "./reward")
        reward_log = os.path.join(reward_dir, "reward_server.log")
        reward_server_script = os.path.join(reward_dir, "reward_server.py")

        with open(reward_log, "w") as log_file:
            reward_proc = subprocess.Popen(
                [sys.executable, reward_server_script],
                cwd=reward_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # 便于后续 kill 整个进程组
            )

        try:
            # 等待 reward server 启动
            time.sleep(30)

            # 运行主逻辑
            cmd = 'python -u -m paddle.distributed.launch \
                    --devices "$CUDA_VISIBLE_DEVICES" run_rl.py \
                    ../../config/qwen/grpo_argument.yaml \
                    --actor_model_name_or_path "Qwen/Qwen2-1.5B" \
                    --max_dec_len 128 \
                    --max_steps 3 \
                    --kl_coeff 0.000 \
                    --kl_loss_coeff 0.000 \
                    --use_fused_rms_norm true '
            pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = pro.communicate()
            print(out)
            pro.wait()
            pro.returncode == 0
            assert str(out).find("Error") == -1
            assert str(err).find("Error") == -1
            os.chdir(repo_path)

        finally:
            # main 执行完毕，关闭 reward server
            if reward_proc.poll() is None:  # 确保进程还在
                os.killpg(os.getpgid(reward_proc.pid), signal.SIGTERM)  # kill 整个进程组
