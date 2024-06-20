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

from pipelines.nodes.base import BaseComponent


class TruncatedConversationHistory(BaseComponent):
    """This class represents a component that truncates conversation history to a specified maximum length."""

    outgoing_edges = 1

    def __init__(self, max_length):
        """Initializes the TruncatedConversationHistory class with the specified maximum length."""
        self.max_length = max_length

    def run(self, query, history=None):
        """Truncates the conversation history to the maximum allowed length, then returns the modified history along with the query."""
        if history is None:
            return {"query": query}, "output_1"
        for past_msg in history:
            if len(past_msg["content"]) > self.max_length:
                past_msg["content"] = past_msg["content"][: self.max_length]
        return {"query": query, "history": history}, "output_1"
