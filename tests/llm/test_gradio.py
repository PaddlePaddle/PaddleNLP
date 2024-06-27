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
            + ' {python} predict/flask_server.py --model_name_or_path {model_path} --port {port} --flask_port {flask_port} --src_length 1024 --dtype "float16"'.format(
                flask_port=self.flask_port, port=self.port, model_path=self.model_path, python=sys.executable
            )
        )
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
            res = requests.post(f"http://localhost:{self.flask_port}/api/chat", json=data, stream=True)
            result_ = ""
            for line in res.iter_lines():
                print(line)
                result = json.loads(line)
                bot_response = result["result"]["response"]

                if bot_response["utterance"].endswith("[END]"):
                    bot_response["utterance"] = bot_response["utterance"][:-5]

                result_ += bot_response["utterance"]

            return result_

        data = {
            "context": "你好",
            "top_k": 1,
            "top_p": 1.0,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
            "max_length": 20,
            "min_length": 1,
        }
        # Case 1: greedy search
        # result_0 = get_response(data)
        result_1 = get_response(data)

        # TODO(wj-Mcat): enable logit-comparision later
        # assert result_0 == result_1

        data = {
            "context": "你好",
            "top_k": 0,
            "top_p": 0.7,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
            "max_length": 20,
            "min_length": 1,
        }

        # Case 2: sampling
        result_2 = get_response(data)
        # assert result_1 != result_2

        # 测试长度应该保持一致
        assert 10 <= len(self.tokenizer.tokenize(result_1)) <= 50
        assert 10 <= len(self.tokenizer.tokenize(result_2)) <= 50

        data = {
            "context": "你好",
            "top_k": 1,
            "top_p": 0.7,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
            "max_length": 100,
            "min_length": 1,
        }
        # Case 3: max_length
        result_3 = get_response(data)
        assert result_3 != result_2
        assert 70 <= len(self.tokenizer.tokenize(result_3)) <= 150
