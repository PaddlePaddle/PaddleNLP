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

from pipelines.nodes.prompt.invocation_layer.handlers import DefaultPromptHandler


class TestHandlers(unittest.TestCase):
    def test_prompt_handler_basics(self):
        handler = DefaultPromptHandler(model_name_or_path="gpt2", model_max_length=20, max_length=10)
        assert callable(handler)

        handler = DefaultPromptHandler(model_name_or_path="gpt2", model_max_length=20)
        assert handler.max_length == 100

    def test_gpt2_prompt_handler(self):
        # test gpt2 BPE based tokenizer
        handler = DefaultPromptHandler(model_name_or_path="gpt2", model_max_length=20, max_length=10)

        # test no resize
        assert handler("This is a test") == {
            "prompt_length": 4,
            "resized_prompt": "This is a test",
            "max_length": 10,
            "model_max_length": 20,
            "new_prompt_length": 4,
        }

        # test resize
        assert handler("This is a prompt that will be resized because it is longer than allowed") == {
            "prompt_length": 15,
            "resized_prompt": "This is a prompt that will be resized because",
            "max_length": 10,
            "model_max_length": 20,
            "new_prompt_length": 10,
        }
