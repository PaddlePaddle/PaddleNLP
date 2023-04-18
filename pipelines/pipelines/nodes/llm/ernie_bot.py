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
import json
import os

import requests

from pipelines.nodes.base import BaseComponent


class ErnieBot(BaseComponent):
    """
    The ErnieBot class is a subclass of the BaseComponent class, which is designed to interface with
    the Ernie Bot API for generating AI chatbot responses. It handles the interaction with the API using
    the provided access token. It allows you to make a request with a given query and optional conversation
    history, receiving a response from the chatbot and extending the conversation history accordingly.
    """

    outgoing_edges = 1
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    def __init__(self, ernie_bot_access_token=None):
        """
        Initialize the ErnieBot instance with the provided access token or retrieve it from an
        environment variable.

        :param ernie_bot_access_token: The access token to authenticate with the Ernie Bot API. If not provided,
            the method will attempt to retrieve it from the `ernie_bot_access_token` environment variable.
            Defaults to None.
        """

        access_token = ernie_bot_access_token or os.environ.get("ernie_bot_access_token", None)
        if access_token is None:
            raise ValueError(
                "Did not find `ernie_bot_access_token`, please add an environment variable `ernie_bot_access_token` which contains it, or pass"
                "  `ernie_bot_access_token` as a named parameter."
            )
        self.url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token={access_token}"

    def run(self, query, history=None, stream=False):
        """
        Send a request to the Ernie Bot API with the given query and optional conversation history.
        Returns the chatbot response and updates the conversation history accordingly.

        :param query: The user's input/query to be sent to the Ernie Bot API.
        :param history: A list of dictionaries representing the conversation history,
        :param stream: Whether to use streaming mode when making the request. Currently not in use. Defaults to False.
        """

        payload = {"messages": []}
        if history is not None:
            if len(history) % 2 == 0:
                for past_msg in history:
                    if past_msg["role"] not in ["user", "assistant"]:
                        raise ValueError(
                            "Invalid history: The `role` in each message in history must be `user` or `assistant`."
                        )
                payload["messages"].extend(history)
            else:
                raise ValueError("Invalid history: an even number of `messages` is expected!")
        payload["messages"].append({"role": "user", "content": f"{query}"})
        # Do not use stream for now
        if stream:
            payload["stream"] = True
        response = requests.request("POST", self.url, headers=self.headers, data=json.dumps(payload))
        response_json = json.loads(response.text)
        if history is None:
            return_history = []
        else:
            return_history = copy.deepcopy(history)
        return_history.extend(
            [{"role": "user", "content": query}, {"role": "assistant", "content": response_json["result"]}]
        )
        response_json["history"] = return_history
        return response_json, "eb_output"
