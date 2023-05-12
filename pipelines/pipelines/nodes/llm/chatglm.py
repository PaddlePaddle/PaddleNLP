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

import logging

from paddlenlp import Taskflow
from pipelines.nodes.base import BaseComponent

logger = logging.getLogger(__name__)


class ChatGLMBot(BaseComponent):
    def __init__(self):
        self.chatglm = Taskflow("text2text_generation", batch_size=2, max_seq_length=2048, tgt_length=256)

    def run(self, query, history=None, stream=False):

        logger.info(query)
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
        result = self.chatglm(query)

        return result, "output_1"
