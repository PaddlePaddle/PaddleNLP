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

import unittest
from unittest.mock import MagicMock, patch

from pipelines.nodes.combine_documents import (
    MapReduceDocuments,
    ReduceDocuments,
    StuffDocuments,
)


class TestCombineDocuments(unittest.TestCase):
    def setUp(self):
        self.api_key = "your api_key"
        self.secret_key = "your secret_key"
        self.document_list = [{"content": "今天天气很好"}, {"content": "今天天气不错"}]
        self.document_prompt = "文件{index}: 文件内容{content}"
        self.llm_prompt = "根据下列多个的文件内容给出一个摘要：\n{}"

    @patch("requests.request")
    def test_StuffDocuments(self, mock_request):
        mock_response = MagicMock()
        mock_response.text = '{"result": "Hello, how can I help you?"}'
        mock_request.return_value = mock_response
        stuff_documents = StuffDocuments(
            api_key=self.api_key,
            secret_key=self.secret_key,
            document_prompt=self.document_prompt,
            llm_prompt=self.llm_prompt,
        )
        stuff_documents.run(self.document_list)

    @patch("requests.request")
    def test_ReduceDocuments(self, mock_request):
        mock_response = MagicMock()
        mock_response.text = '{"result": "Hello, how can I help you?"}'
        mock_request.return_value = mock_response
        combine_documents = StuffDocuments(
            api_key=self.api_key,
            secret_key=self.secret_key,
            document_prompt=self.document_prompt,
            llm_prompt=self.llm_prompt,
        )
        reducedocuments = ReduceDocuments(combine_documents=combine_documents)
        reducedocuments.run(self.document_list)

    @patch("requests.request")
    def test_MapReduceDocuments(self, mock_request):
        mock_response = MagicMock()
        mock_response.text = '{"result": "Hello, how can I help you?"}'
        mock_request.return_value = mock_response
        combine_documents = StuffDocuments(
            api_key=self.api_key,
            secret_key=self.secret_key,
            document_prompt=self.document_prompt,
            llm_prompt=self.llm_prompt,
        )
        reduce_documents = ReduceDocuments(combine_documents=combine_documents)
        mapReduceDocuments = MapReduceDocuments(
            api_key=self.api_key,
            secret_key=self.secret_key,
            llm_prompt=self.llm_prompt,
            reduce_documents=reduce_documents,
        )
        mapReduceDocuments.run(self.document_list)
