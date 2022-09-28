# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 deepset GmbH. All Rights Reserved.
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

from typing import List

from pipelines.schema import Document, Answer, Span
from pipelines.nodes.base import BaseComponent


class Docs2Answers(BaseComponent):
    """
    This Node is used to convert retrieved documents into predicted answers format.
    It is useful for situations where you are calling a Retriever only pipeline via REST API.
    This ensures that your output is in a compatible format.
    """

    outgoing_edges = 1

    def __init__(self):
        self.set_config()

    def run(self, query: str, documents: List[Document]):  # type: ignore
        # conversion from Document -> Answer
        answers: List[Answer] = []
        for doc in documents:
            # For FAQ style QA use cases
            if "answer" in doc.meta:
                doc.meta[
                    "query"] = doc.content  # question from the existing FAQ
                cur_answer = Answer(
                    answer=doc.meta["answer"],
                    type="other",
                    score=doc.score,
                    context=doc.meta["answer"],
                    offsets_in_context=[
                        Span(start=0, end=len(doc.meta["answer"]))
                    ],
                    document_id=doc.id,
                    meta=doc.meta,
                )
            else:
                # Regular docs
                cur_answer = Answer(
                    answer="",
                    type="other",
                    score=doc.score,
                    context=doc.content,
                    document_id=doc.id,
                    meta=doc.meta,
                )
            answers.append(cur_answer)

        output = {"query": query, "answers": answers}

        return output, "output_1"
