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

from paddlenlp.transformers import NVEncodeModel, PretrainedConfig

from ...testing_utils import require_gpu


class NVEncodeModelIntegrationTest(unittest.TestCase):
    @require_gpu(1)
    def test_model_tiny_logits(self):
        input_texts = [
            "This is a test",
            "This is another test",
        ]

        config = PretrainedConfig(
            attention_dropout=0.0,
            bos_token_id=1,
            dtype="float16",
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=4096,
            initializer_range=0.02,
            intermediate_size=14336,
            max_position_embeddings=32768,
            num_attention_heads=32,
            num_hidden_layers=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-05,
            rope_theta=10000.0,
            sliding_window=4096,
            tie_word_embeddings=False,
            vocab_size=32000,
        )
        model = NVEncodeModel(
            config=config,
            tokenizer_path="BAAI/bge-large-en-v1.5",
            query_instruction="",
            document_instruction="",
        )
        with paddle.no_grad():
            out = model.encode_sentences(input_texts, instruction_len=0)

        print(out)
        """
        [[-0.00473404  0.00711441  0.01237488 ... -0.00228691 -0.01416779 -0.00429535]
         [-0.00343323  0.00911713  0.00894928 ... -0.00637054 -0.0165863 -0.00852966]]
        """

        del model
        paddle.device.cuda.empty_cache()
        gc.collect()
