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

from pipelines.nodes.llm import ChatGLMBot


class TestChatGLMBot(unittest.TestCase):
    def setUp(self):
        self.chatbot = ChatGLMBot(
            model_name_or_path="__internal_testing__/tiny-random-chatglm", dtype="float32", tgt_length=8
        )

    def test_run(self):
        prompt_text = "很高兴认识你"
        result = self.chatbot.run(query=prompt_text)
        expected_output = ({"result": ["strained睡到睡到睡到睡到睡到睡到睡到"]}, "output_1")
        self.assertEqual(
            result,
            expected_output,
        )
