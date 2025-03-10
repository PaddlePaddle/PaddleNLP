#!/usr/bin/env python
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import json
import os
import socket
import subprocess
import sys
import time
import unittest

import pytest
import requests

from paddlenlp.transformers import LlamaTokenizer


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.01)
        try:
            s.bind(("localhost", port))
            return False
        except socket.error:
            return True


class UITest(unittest.TestCase):
    def setUp(self):
        # start web ui
        self.flask_port = self.avaliable_free_port()
        self.port = self.avaliable_free_port([self.flask_port])
        self.model_path = "__internal_testing__/micro-random-llama"
        command = (
            "cd ./llm && PYTHONPATH=../:$PYTHONPATH"
            + " {python} predict/flask_server.py --model_name_or_path {model_path} "
            + '--port {port} --flask_port {flask_port} --src_length 1024 --dtype "float16"'
        ).format(flask_port=self.flask_port, port=self.port, model_path=self.model_path, python=sys.executable)
        current_env = copy.copy(os.environ.copy())
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""

        self.ui_process = subprocess.Popen(command, shell=True, stdout=sys.stdout, stderr=sys.stderr, env=current_env)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        return super().setUp()

    def tearDown(self):
        self.ui_process.terminate()

    def avaliable_free_port(self, exclude=None):
        exclude = exclude or []
        for port in range(8000, 10000):
            if port in exclude:
                continue
            if is_port_in_use(port):
                continue
            return port

        raise ValueError("can not get valiable port in [8000, 8200]")

    def wait_until_server_is_ready(self):
        while True:
            if is_port_in_use(self.flask_port) and is_port_in_use(self.port):
                break
            print("waiting for server ...")
            time.sleep(1)

    def get_gradio_ui_result(self, *args, **kwargs):
        _, _, file = self.client.predict(*args, **kwargs)
        with open(file, "r", encoding="utf-8") as f:
            content = json.load(f)
        return content[-1]["utterance"]

    @pytest.mark.timeout(4 * 60)
    def test_argument(self):
        self.wait_until_server_is_ready()

        def get_response(data):
            res = requests.post(f"http://localhost:{self.flask_port}/v1/chat/completions", json=data, stream=True)
            result_ = ""
            for line in res.iter_lines():
                if not line:
                    continue
                decoded_line = line.decode("utf-8").strip()
                # 如果返回行以 "data:" 开头，则去除该前缀
                if decoded_line.startswith("data:"):
                    data_str = decoded_line[len("data:") :].strip()
                else:
                    data_str = decoded_line
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                # 根据 OpenAI 的流式返回，每个 chunk 在 choices[0]["delta"] 中包含回复增量
                delta = chunk["choices"][0]["delta"].get("content", "")
                result_ += delta
            return result_

        # 测试用例1：greedy search 模式（top_p 为1.0）
        data = {
            "messages": [{"role": "user", "content": "你好"}],
            "temperature": 1.0,
            "max_tokens": 20,
            "top_p": 1.0,
            "stream": True,
        }
        result_1 = get_response(data)

        # 测试用例2：采样模式（top_p 为 0.7）
        data = {
            "messages": [{"role": "user", "content": "你好"}],
            "temperature": 1.0,
            "max_tokens": 20,
            "top_p": 0.7,
            "stream": True,
        }
        result_2 = get_response(data)

        # 对生成文本的长度进行简单检测
        assert 10 <= len(self.tokenizer.tokenize(result_1)) <= 50
        assert 10 <= len(self.tokenizer.tokenize(result_2)) <= 50

        # 测试用例3：更长的 max_tokens 参数
        data = {
            "messages": [{"role": "user", "content": "你好"}],
            "temperature": 1.0,
            "max_tokens": 100,
            "top_p": 0.7,
            "stream": True,
        }
        result_3 = get_response(data)
        assert result_3 != result_2
        assert 70 <= len(self.tokenizer.tokenize(result_3)) <= 150
