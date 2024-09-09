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
import logging
import os

import requests
from pipelines.nodes.base import BaseComponent

logger = logging.getLogger(__name__)

ernie_dict = {
    "ERNIE-Bot-turbo": " https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={}",
    "ERNIE-Bot": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token={}",
}


class ErnieBot(BaseComponent):
    """
    The ErnieBot class is a subclass of the BaseComponent class, which is designed to interface with
    the Ernie Bot API for generating AI chatbot responses. It handles the interaction with the API using
    the provided api_key, secret_key . It allows you to make a request with a given query and optional conversation
    history, receiving a response from the chatbot and extending the conversation history accordingly.
    """

    outgoing_edges = 1
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    def __init__(self, api_key=None, secret_key=None, model_name="ERNIE-Bot-turbo"):
        """
        Initialize the ErnieBot instance with the provided api_key and secret_key.

        :param api_key: api_key for applying token to request wenxin api.
        :param secret_key: secret_key for applying token to request wenxin api.
        """
        api_key = api_key or os.environ.get("ERNIE_BOT_API_KEY", None)
        secret_key = secret_key or os.environ.get("ERNIE_BOT_SECRET_KEY", None)
        self.model_name = model_name
        if api_key is None or secret_key is None:
            raise Exception(
                "Please apply api_key and secret_key from https://cloud.baidu.com/doc/WENXINWORKSHOP/s/flfmc9do2"
            )
        self.api_key = api_key
        self.secret_key = secret_key
        self.token = self._apply_token(self.api_key, self.secret_key)

    def _apply_token(self, api_key, secret_key):
        payload = ""
        self.token_host = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
        response = requests.request("POST", self.token_host, headers=self.headers, data=payload)
        if response:
            res = response.json()
        else:
            raise RuntimeError("Request access token error.")

        return res["access_token"]

    def predict(self, query, history=None, stream=False, api_key=None, secret_key=None):
        if api_key is not None and secret_key is not None:
            self.token = self._apply_token(api_key, secret_key)
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
        chat_url = ernie_dict[self.model_name].format(self.token)
        response = requests.request("POST", chat_url, headers=self.headers, data=json.dumps(payload))
        response_json = json.loads(response.text)
        if history is None:
            return_history = []
        else:
            return_history = copy.deepcopy(history)
        try:
            return_history.extend(
                [{"role": "user", "content": query}, {"role": "assistant", "content": response_json["result"]}]
            )
            response_json["history"] = return_history
        except Exception as e:
            logger.error(e)
            logger.error(response_json)
        return response_json

    def run(self, query, history=None, stream=False, api_key=None, secret_key=None, **kwargs):
        """
        Send a request to the Ernie Bot API with the given query and optional conversation history.
        Returns the chatbot response and updates the conversation history accordingly.

        :param query: The user's input/query to be sent to the Ernie Bot API.
        :param history: A list of dictionaries representing the conversation history,
        :param stream: Whether to use streaming mode when making the request. Currently not in use. Defaults to False.
        """
        debug = kwargs.get("debug", False)
        if debug:
            logger.debug(f"Query: {query}")
        response_json = self.predict(query, history, stream)
        return response_json, "output_1"
