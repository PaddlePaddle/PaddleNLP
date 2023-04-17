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

import json
import os

import requests

from pipelines.nodes.base import BaseComponent


class ErnieBot(BaseComponent):
    outgoing_edges = 1
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    def __init__(self, ernie_bot_access_token=None):
        access_token = ernie_bot_access_token or os.environ.get("ernie_bot_access_token", None)
        if access_token is None:
            raise ValueError(
                "Did not find `ernie_bot_access_token`, please add an environment variable `ernie_bot_access_token` which contains it, or pass"
                "  `ernie_bot_access_token` as a named parameter."
            )
        self.url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token={access_token}"

    def run(self, query, **kwargs):
        payload = {"messages": [{"role": "user", "content": f"{query}"}]}

        if kwargs.get("stream", False):
            payload["stream"] = True
        response = requests.request("POST", self.url, headers=self.headers, data=json.dumps(payload))
        return json.loads(response.text), "eb_output"
