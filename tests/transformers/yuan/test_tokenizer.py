# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers import YuanTokenizer


class YuanTokenizationTest(unittest.TestCase):
    def test_extract_non_learnable_parts(self):
        models_with_templates = [
            "IEITYuan/Yuan2-2B",
            "IEITYuan/Yuan2-51B",
            "IEITYuan/Yuan2-102B",
        ]
        dummy_conversastions = [
            ["Q.", "A."],
            ["Q.A.", "A."],
            ["Q?", "A!"],
        ]
        decode_outputs = [
            ["Q.<n>", "A.<n>"],
            ["Q.A.<n>", "A.<n>"],
            ["Q?<n>", " A!<sep>"],  # notify there is an extra space
        ]
        context_data = {}
        context_data["is_training"] = True
        for model_id in models_with_templates:
            tokenizer = YuanTokenizer.from_pretrained(model_id)
            if tokenizer.chat_template is None:
                continue
            conversation_result: list[tuple[list[int], list[int]]] = tokenizer.encode_chat_inputs(
                dummy_conversastions,
                context_data=context_data,
            )
            for idx, round in enumerate(conversation_result["conversations"]):
                self.assertEquals(tokenizer.decode(round[0]), decode_outputs[idx][0])
                self.assertEquals(tokenizer.decode(round[1]), decode_outputs[idx][1])
