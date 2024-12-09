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

import logging
import os
import sys

from fastdeploy_client.chatbot import ChatBot


def _get_service_configuration():
    """
    get service url from environment

    Returns:
        tuple: (hostname, port)
    """
    url = os.getenv("FASTDEPLOY_MODEL_URL")

    if url is None:
        raise ValueError("Please set service url by `export FASTDEPLOY_MODEL_URL`."
                         "For example: `export FASTDEPLOY_MODEL_URL=127.0.0.1:8500`")
    hostname, port = url.strip().split(':')
    port = int(port)
    if port <= 0 or port > 65535:
        raise ValueError("Invalid port number")

    return hostname, port


def stream_generate(prompt):
    """
    Streaming interface
    """
    hostname, port = _get_service_configuration()
    chatbot = ChatBot(hostname=hostname, port=port)
    stream_result = chatbot.stream_generate(prompt)
    for res in stream_result:
        print(res)


def generate(prompt):
    """
    entire sentence interface
    """
    hostname, port = _get_service_configuration()
    chatbot = ChatBot(hostname=hostname, port=port)
    result = chatbot.generate(prompt)
    print(result)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ["generate", "stream_generate"]:
        logging.error("Usage 1: fdclient generate \"Hello, How are you?\"")
        return
    prompt = sys.argv[2]
    if sys.argv[1] == "generate":
        return generate(prompt)
    else:
        return stream_generate(prompt)
