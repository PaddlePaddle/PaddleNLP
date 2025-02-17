# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import gc
import unittest

import paddle

from paddlenlp.transformers import AutoTokenizer, BiEncoderModel

from ...testing_utils import require_gpu


class BiEncoderModelIntegrationTest(unittest.TestCase):
    @require_gpu(1)
    def test_model_tiny_logits(self):
        input_texts = [
            "This is a test",
            "This is another test",
        ]

        model_name_or_path = "BAAI/bge-large-en-v1.5"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = BiEncoderModel(model_name_or_path=model_name_or_path, dtype="float16", tokenizer=tokenizer)
        with paddle.no_grad():
            out = model.encode_sentences(sentences=input_texts)

        print(out)
        """
        [[ 0.00674057  0.03396606  0.00722122 ...  0.01176453  0.00311279 -0.02825928]
         [ 0.00708771  0.03982544 -0.00155735 ...  0.00658417  0.01318359 -0.03259277]]
        """

        del model
        paddle.device.cuda.empty_cache()
        gc.collect()
