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

from pipelines.nodes.llm import LLMPromptTemplate as PromptTemplate
from pipelines.schema import Document


class TestPromptTemplate(unittest.TestCase):
    def test_prompt_templates(self):
        template = PromptTemplate("Here is some fake template with query {query}, documents {documents}")
        query = "this is a test"
        documents = [Document(content="document {} ".format(i), content_type="text") for i in range(2)]
        results, ouput_edge = template.run(query=query, documents=documents)
        format_query = "Here is some fake template with query this is a test, documents document 0 document 1 "
        self.assertEqual(format_query, results["query"])
        self.assertEqual(ouput_edge, "output_1")
