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

from pipelines.nodes.base import BaseComponent

from paddlenlp import Taskflow

logger = logging.getLogger(__name__)


class ChatGLMBot(BaseComponent):
    def __init__(
        self,
        model_name_or_path="THUDM/chatglm-6b-v1.1",
        batch_size: int = 2,
        max_seq_length: int = 2048,
        tgt_length: int = 2048,
        **kwargs
    ):
        """
        Initialize the ChatGLMBot instance.

        :param batch_size: batch_size for chatglm prediction.
        :param max_seq_length: max_seq_length for the processing input.
        :param tgt_length: tgt_length for models output
        """
        self.kwargs = kwargs
        self.chatglm = Taskflow(
            "text2text_generation",
            model=model_name_or_path,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            tgt_length=tgt_length,
            **self.kwargs,
        )

    def predict(self, query, stream=False):
        result = self.chatglm(query)
        return result

    def run(self, query, stream=False, **kwargs):
        """
        Using the chatbot to generate the answers
        :param query: The user's input/query to be sent to the chatGLM.
        :param stream: Whether to use streaming mode when making the request. Currently not in use. Defaults to False.
        """
        debug = kwargs.get("debug", False)
        if debug:
            logger.debug(f"Query: {query}")
        result = self.predict(query=query, stream=stream)
        return result, "output_1"
