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


from typing import Dict, List, Optional, Union

from pipelines.document_stores import BaseDocumentStore
from pipelines.nodes import BaseComponent
from pipelines.nodes.prompt import PromptNode, PromptTemplate
from pipelines.nodes.retriever import BaseRetriever
from pipelines.schema import Document, FilterType


class MockNode(BaseComponent):
    outgoing_edges = 1

    def run(self, *a, **k):
        pass

    def run_batch(self, *a, **k):
        pass


class MockDocumentStore(BaseDocumentStore):
    outgoing_edges = 1

    def _create_document_field_map(self, *a, **k):
        pass

    def delete_documents(self, *a, **k):
        pass

    def delete_labels(self, *a, **k):
        pass

    def get_all_documents(self, *a, **k):
        pass

    def get_all_documents_generator(self, *a, **k):
        pass

    def get_all_labels(self, *a, **k):
        pass

    def get_document_by_id(self, *a, **k):
        pass

    def get_document_count(self, *a, **k):
        pass

    def get_documents_by_id(self, *a, **k):
        pass

    def get_label_count(self, *a, **k):
        pass

    def query_by_embedding(self, *a, **k):
        pass

    def write_documents(self, *a, **k):
        pass

    def write_labels(self, *a, **k):
        pass

    def delete_index(self, *a, **k):
        pass

    def update_document_meta(self, *a, **kw):
        pass


class MockRetriever(BaseRetriever):
    outgoing_edges = 1

    def retrieve(
        self,
        query: str,
        filters: Optional[FilterType] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
        **kwargs,
    ) -> List[Document]:
        return []

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        return [[]]


class MockPromptNode(PromptNode):
    def __init__(self):
        self.default_prompt_template = None
        self.model_name_or_path = ""

    def prompt(self, prompt_template: Optional[Union[str, PromptTemplate]], *args, **kwargs) -> List[str]:
        return [""]

    def get_prompt_template(self, prompt_template: Union[str, PromptTemplate, None]) -> Optional[PromptTemplate]:
        if prompt_template == "think-step-by-step":
            return PromptTemplate(
                name="think-step-by-step",
                prompt_text="You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions "
                "correctly, you have access to the following tools:\n\n"
                "{tool_names_with_descriptions}\n\n"
                "To answer questions, you'll need to go through multiple steps involving step-by-step thinking and "
                "selecting appropriate tools and their inputs; tools will respond with observations. When you are ready "
                "for a final answer, respond with the `Final Answer:`\n\n"
                "Use the following format:\n\n"
                "Question: the question to be answered\n"
                "Thought: Reason if you have the final answer. If yes, answer the question. If not, find out the missing information needed to answer it.\n"
                "Tool: [{tool_names}]\n"
                "Tool Input: the input for the tool\n"
                "Observation: the tool will respond with the result\n"
                "...\n"
                "Final Answer: the final answer to the question, make it short (1-5 words)\n\n"
                "Thought, Tool, Tool Input, and Observation steps can be repeated multiple times, but sometimes we can find an answer in the first pass\n"
                "---\n\n"
                "Question: {query}\n"
                "Thought: Let's think step-by-step, I first need to {generated_text}",
            )
        else:
            return PromptTemplate(name="", prompt_text="")
