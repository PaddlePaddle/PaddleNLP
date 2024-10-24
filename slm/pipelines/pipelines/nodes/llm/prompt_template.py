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


class LLMPromptTemplate(BaseComponent):
    outgoing_edges = 1

    def __init__(self, template):
        self.template = template

    def run(self, query=None, documents=None, history=None):
        if documents is not None:
            documents = [i.content for i in documents]
            context = "".join(documents)
            result = {"documents": context, "query": query}
        elif history is not None:
            chat_history = "\n".join(history)
            question = query
            result = {"chat_history": chat_history, "question": question}
        else:
            raise NotImplementedError("This prompt template is not implemented!")

        return {"query": self.template.format(**result)}, "output_1"
