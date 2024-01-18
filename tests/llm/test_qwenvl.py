# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import unittest

import paddle

from paddlenlp.experimental.transformers import QWenForQWenVLInferenceModel
from paddlenlp.transformers import AutoConfig, AutoTokenizer, QWenForCausalLM

from .testing_utils import LLMTest, argv_context_guard, load_test_config

paddle.seed(1234)


class QWenVLTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    model_name_or_path: str = "__internal_testing__/tiny-fused-qwen"
    model_class = QWenForCausalLM

    def setUp(self) -> None:
        super().setUp()
        paddle.set_default_dtype("float32")
        self.model_class.from_pretrained(self.model_name_or_path, dtype="float16").save_pretrained(self.output_dir)
        AutoTokenizer.from_pretrained(self.model_name_or_path).save_pretrained(self.output_dir)

    def test_forward(self):
        self.disable_static()
        config = AutoConfig.from_pretrained(self.output_dir)
        config["quant_type"] = None
        config["weight_only_quant_bits"] = None

        paddle.set_default_dtype("float16")
        model = QWenForQWenVLInferenceModel.from_pretrained(self.output_dir, config=config, dtype="float16")

        batch = 1
        seq = 271
        max_len = 1024
        dtype = "float16"
        input_ids = paddle.randint(0, 100, [batch, seq], dtype="int64")
        image_features = paddle.randn([batch, 256, 4096], dtype="float16")
        tgt_generation_mask = paddle.full([batch, 1, 1, max_len], 1, dtype=dtype)
        img_pos = paddle.to_tensor([0, 4, 261], dtype="int64")
        attention_mask = paddle.full([batch, 1, max_len, max_len], 0, dtype=dtype)
        attention_mask[:, 0, :seq, :seq] = paddle.tril(paddle.ones(shape=(seq, seq), dtype=dtype))
        position_ids = paddle.full([batch, seq], 0, dtype="int64")
        for i in range(batch):
            position_ids[i, :] = paddle.to_tensor([i for i in range(seq)], dtype="int64")

        inputs = [
            input_ids,  # input_ids
            image_features,  # image_features
            img_pos,  # img_pos
            attention_mask,  # attention_mask
            position_ids,  # position_ids
            paddle.full([batch, 1], 1.0, dtype="float32"),  # penalty_score
            paddle.full([batch, 1], 0.0, dtype="float32"),  # frequency_score,
            paddle.full([batch, 1], 0.0, dtype="float32"),  # presence_score,
            paddle.full([batch, 1], 1, dtype="int64"),  # min_length,
            paddle.full([batch, 1], max_len - seq, dtype="int64"),  # max_length,
            paddle.full([batch, 1], 1.0, dtype="float32"),  # temperature,
            paddle.full([batch, 1], 0.0, dtype="float32"),  # top_p,
            paddle.full([1], 151643, dtype="int64"),  # eos_token_id,
            paddle.full([batch, 1], seq, dtype="int32"),  # seq_len_encoder,
            paddle.full([batch, 1], seq, dtype="int32"),  # seq_len_decoder,
            paddle.full([batch, 1], 0, dtype="int64"),  # step_idx,
            paddle.full([batch, 1], False, dtype="bool"),  # stop_flags,
            paddle.full([batch, 1], -123, dtype="int64"),  # tgt_ids can be be initialized arbitrarily
            paddle.full([batch, 1], seq - 1, dtype="int64"),  # tgt_pos,
            tgt_generation_mask,  # tgt_generation_mask,
            paddle.full([batch, max_len], -100, dtype="int64"),  # pre_ids, can be initialized arbitrarily
            paddle.full([1], batch, dtype="int64"),  # stop_nums, be batch
        ]
        for i in range(config.num_hidden_layers):
            tmp = paddle.rand(shape=[2, batch, 32, max_len, 128], dtype=dtype)
            inputs.append(tmp)

        model.eval()
        model.generate_text_with_image_features(
            input_ids=inputs[0],
            image_features=inputs[1],
            img_pos=inputs[2],
            attention_mask=inputs[3],
            position_ids=inputs[4],
            penalty_score=inputs[5],
            frequency_score=inputs[6],
            presence_score=inputs[7],
            min_length=inputs[8],
            max_length=inputs[9],
            temperature=inputs[10],
            top_p=inputs[11],
            eos_token_id=inputs[12],
            seq_len_encoder=inputs[13],
            seq_len_decoder=inputs[14],
            step_idx=inputs[15],
            stop_flags=inputs[16],
            tgt_ids=inputs[17],
            tgt_pos=inputs[18],
            tgt_generation_mask=inputs[19],
            pre_ids=inputs[20],
            stop_nums=inputs[21],
            cache_kvs=inputs[22:],
        )

    def test_export(self):
        self.disable_static()
        config = load_test_config(self.config_path, "inference-to-static")
        config["model_name_or_path"] = self.model_name_or_path
        config["output_path"] = self.output_dir
        config["dtype"] = "float16"
        config["inference_model"] = True
        config["model_prefix"] = "qwen"
        config["model_type"] = "qwen-img2txt"

        with argv_context_guard(config):
            from export_model import main

            main()
