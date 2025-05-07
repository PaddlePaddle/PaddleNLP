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

import sys
import os
import subprocess
import time
import signal
import unittest
from unittest import skip

from parameterized import parameterized_class

from tests.testing_utils import argv_context_guard, load_test_config

from .testing_utils import LLMTest


@parameterized_class(
    ["model_dir"],
    [["qwen"]],
)
class FinetuneTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/reinforce_plus_plus.yaml"
    model_dir: str = None

    def setUp(self) -> None:
        LLMTest.setUp(self)
        sys.path.insert(0, "./llm/alignment/rl")
        sys.path.insert(0, self.model_dir)

    def tearDown(self) -> None:
        LLMTest.tearDown(self)

    def test_finetune(self):
        # 启动 reward server
        reward_dir = os.path.join(os.getcwd(), "reward")
        reward_log = os.path.join(reward_dir, "reward_server.log")
        reward_server_script = os.path.join(reward_dir, "reward_server.py")

        with open(reward_log, "w") as log_file:
            reward_proc = subprocess.Popen(
                [sys.executable, reward_server_script],
                cwd=reward_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # 便于后续 kill 整个进程组
            )

        try:
            # 等待 reward server 启动
            time.sleep(3)

            # 运行主逻辑
            grpo_config = load_test_config(self.config_path, "grpo", self.model_dir)
            grpo_config["output_dir"] = self.output_dir

            with argv_context_guard(grpo_config):
                from alignment.rl.run_rl import main
                main()
        finally:
            # main 执行完毕，关闭 reward server
            if reward_proc.poll() is None:  # 确保进程还在
                os.killpg(os.getpgid(reward_proc.pid), signal.SIGTERM)  # kill 整个进程组