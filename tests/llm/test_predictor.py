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

import os
import unittest

import paddle
from parameterized import parameterized_class

from paddlenlp.experimental.transformers import QWenForQWenVLInferenceModel
from paddlenlp.transformers import (
    AutoConfig,
    AutoTokenizer,
    BloomForCausalLM,
    ChatGLMForCausalLM,
    ChatGLMv2ForCausalLM,
    LlamaForCausalLM,
    QWenForCausalLM,
)
from paddlenlp.utils.downloader import (
    COMMUNITY_MODEL_PREFIX,
    get_path_from_url_with_filelock,
    url_file_exists,
)

from .testing_utils import LLMTest, argv_context_guard, load_test_config


@parameterized_class(
    ["model_name_or_path", "model_class"],
    [
        ["__internal_testing__/tiny-random-llama", LlamaForCausalLM],
        ["__internal_testing__/tiny-fused-bloom", BloomForCausalLM],
        ["__internal_testing__/tiny-fused-chatglm", ChatGLMForCausalLM],
        ["__internal_testing__/tiny-fused-chatglm2", ChatGLMv2ForCausalLM],
        ["__internal_testing__/tiny-fused-qwen-inference5.2", QWenForCausalLM],
    ],
)
class PredictorTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    model_name_or_path: str = None
    model_class = None

    def setUp(self) -> None:
        super().setUp()
        paddle.set_default_dtype("float32")
        self.model_class.from_pretrained(self.model_name_or_path, dtype="float16").save_pretrained(self.output_dir)
        AutoTokenizer.from_pretrained(self.model_name_or_path).save_pretrained(self.output_dir)

    def test_predictor(self):
        self.run_predictor({"inference_model": True, "src_length": 512, "max_length": 48})
        result_0 = self._read_result(os.path.join(self.output_dir, "predict.json"))
        self.run_predictor({"inference_model": False, "src_length": 512, "max_length": 48})
        result_1 = self._read_result(os.path.join(self.output_dir, "predict.json"))

        # compare the generation result of inference & dygraph model
        assert len(result_0) == len(result_1)

        count, full_match = 0, 0
        for inference_item, no_inference_item in zip(result_0, result_1):
            min_length = min(len(inference_item), len(no_inference_item))
            count += int(inference_item[: min_length // 2] == no_inference_item[: min_length // 2])
            full_match += int(inference_item[:min_length] == no_inference_item[:min_length])

        self.assertGreaterEqual(full_match / len(result_0), 0.25)

        if self.model_name_or_path == "__internal_testing__/tiny-fused-chatglm":
            self.assertGreaterEqual(count / len(result_0), 0.3)
        else:
            self.assertGreaterEqual(count / len(result_0), 0.4)

    def test_flash_attention(self):
        self.run_predictor(
            {"inference_model": False, "use_flash_attention": False, "src_length": 512, "max_length": 48}
        )
        result_0 = self._read_result(os.path.join(self.output_dir, "predict.json"))

        self.run_predictor(
            {"inference_model": False, "use_flash_attention": True, "src_length": 512, "max_length": 48}
        )
        result_1 = self._read_result(os.path.join(self.output_dir, "predict.json"))

        # compare the generation result of dygraph & flash attention model
        assert len(result_0) == len(result_1)

        count, full_match = 0, 0
        for inference_item, no_inference_item in zip(result_0, result_1):
            if self.model_name_or_path == "__internal_testing__/tiny-random-llama":
                min_length = 5
            else:
                min_length = min(len(inference_item), len(no_inference_item))
            count += int(inference_item[: min_length // 2] == no_inference_item[: min_length // 2])
            full_match += int(inference_item[:min_length] == no_inference_item[:min_length])

        if self.model_name_or_path == "__internal_testing__/tiny-random-llama":
            self.assertGreaterEqual(count / len(result_0), 0.2)
        else:
            self.assertEqual(full_match / len(result_0), 1.0)

    def test_wint8(self):
        self.run_predictor(
            {"inference_model": True, "quant_type": "weight_only_int8", "src_length": 512, "max_length": 48}
        )
        result_0 = self._read_result(os.path.join(self.output_dir, "predict.json"))
        self.run_predictor({"inference_model": False, "src_length": 512, "max_length": 48})
        result_1 = self._read_result(os.path.join(self.output_dir, "predict.json"))

        assert len(result_0) == len(result_1)
        count, full_match = 0, 0

        for inference_item, no_inference_item in zip(result_0, result_1):
            min_length = min(len(inference_item), len(no_inference_item))
            count += int(inference_item[: min_length // 2] == no_inference_item[: min_length // 2])
            full_match += int(inference_item[:min_length] == no_inference_item[:min_length])

        self.assertGreaterEqual(full_match / len(result_0), 0.1)

        if self.model_name_or_path == "__internal_testing__/tiny-fused-chatglm":
            self.assertGreaterEqual(count / len(result_0), 0.3)
        else:
            self.assertGreaterEqual(count / len(result_0), 0.4)


@parameterized_class(
    ["model_name_or_path", "model_class"],
    [["__internal_testing__/tiny-random-llama", LlamaForCausalLM]],
)
class PredictorPrecacheTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    model_name_or_path: str = None
    model_class = None

    def setUp(self) -> None:
        super().setUp()

        AutoTokenizer.from_pretrained(self.model_name_or_path).save_pretrained(self.output_dir)
        self.download_precache_files()

    def download_precache_files(self):
        files = [
            "prefix_config.json",
            "config.json",
            "model_state.pdparams",
            "pre_caches.npy",
            "prefix_model_state.pdparams",
        ]
        for file in files:
            file_url = os.path.join(COMMUNITY_MODEL_PREFIX, self.model_name_or_path, file)
            if not url_file_exists(file_url):
                continue
            get_path_from_url_with_filelock(file_url, root_dir=self.output_dir)

    def test_predictor(self):
        self.run_predictor(
            {
                "inference_model": True,
                "export_precache": True,
                "prefix_path": self.output_dir,
                "src_length": 512,
                "max_length": 48,
            }
        )
        result_0 = self._read_result(os.path.join(self.output_dir, "predict.json"))
        self.run_predictor(
            {
                "inference_model": False,
                "export_precache": True,
                "prefix_path": self.output_dir,
                "src_length": 512,
                "max_length": 48,
            }
        )
        result_1 = self._read_result(os.path.join(self.output_dir, "predict.json"))

        # compare the generation result of inference & dygraph model
        assert len(result_0) == len(result_1)
        count, full_match = 0, 0
        for inference_item, no_inference_item in zip(result_0, result_1):

            min_length = min(len(inference_item), len(no_inference_item))
            count += int(inference_item[: min_length // 2] == no_inference_item[: min_length // 2])
            full_match += int(inference_item[:min_length] == no_inference_item[:min_length])

        self.assertGreaterEqual(full_match / len(result_0), 0.6)
        self.assertGreaterEqual(count / len(result_0), 0.8)


@parameterized_class(
    ["model_name_or_path", "model_class"],
    [
        ["__internal_testing__/tiny-fused-llama-inference5.2", LlamaForCausalLM],
        # ["__internal_testing__/tiny-fused-bloom", BloomForCausalLM],
    ],
)
class BlockAttnPredictorTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    model_name_or_path: str = None
    model_class = None

    def setUp(self) -> None:
        super().setUp()
        paddle.set_default_dtype("float32")
        self.model_class.from_pretrained(self.model_name_or_path, dtype="float16").save_pretrained(self.output_dir)
        AutoTokenizer.from_pretrained(self.model_name_or_path).save_pretrained(self.output_dir)

    def test_blha(self):
        self.run_predictor({"inference_model": True, "block_attn": True, "src_length": 512, "max_length": 48})
        result_0 = self._read_result(os.path.join(self.output_dir, "predict.json"))
        self.run_predictor({"inference_model": False, "src_length": 512, "max_length": 48})
        result_1 = self._read_result(os.path.join(self.output_dir, "predict.json"))

        # compare the generation result of inference & dygraph model
        assert len(result_0) == len(result_1)

        count, full_match = 0, 0
        for inference_item, no_inference_item in zip(result_0, result_1):
            min_length = min(len(inference_item), len(no_inference_item))
            count += int(inference_item[: min_length // 2] == no_inference_item[: min_length // 2])
            full_match += int(inference_item[:min_length] == no_inference_item[:min_length])

        self.assertGreaterEqual(full_match / len(result_0), 0.3)

        if self.model_name_or_path == "__internal_testing__/tiny-fused-chatglm":
            self.assertGreaterEqual(count / len(result_0), 0.3)
        else:
            self.assertGreaterEqual(count / len(result_0), 0.4)

    def test_wint8(self):
        self.run_predictor(
            {
                "inference_model": True,
                "quant_type": "weight_only_int8",
                "block_attn": True,
                "src_length": 512,
                "max_length": 48,
            }
        )
        result_0 = self._read_result(os.path.join(self.output_dir, "predict.json"))
        self.run_predictor(
            {"inference_model": True, "quant_type": "weight_only_int8", "src_length": 512, "max_length": 48}
        )
        result_1 = self._read_result(os.path.join(self.output_dir, "predict.json"))

        assert len(result_0) == len(result_1)
        count, full_match = 0, 0

        for inference_item, no_inference_item in zip(result_0, result_1):
            min_length = min(len(inference_item), len(no_inference_item))
            count += int(inference_item[: min_length // 2] == no_inference_item[: min_length // 2])
            full_match += int(inference_item[:min_length] == no_inference_item[:min_length])

        self.assertGreaterEqual(full_match / len(result_0), 0.4)

        if self.model_name_or_path == "__internal_testing__/tiny-fused-chatglm":
            self.assertGreaterEqual(count / len(result_0), 0.3)
        else:
            self.assertGreaterEqual(count / len(result_0), 0.4)

    def test_cachekv_int8(self):
        self.run_predictor(
            {
                "inference_model": True,
                "block_attn": True,
                "cachekv_int8_type": "dynamic",
                "src_length": 512,
                "max_length": 48,
            }
        )
        result_0 = self._read_result(os.path.join(self.output_dir, "predict.json"))
        self.run_predictor({"inference_model": True, "block_attn": True, "src_length": 512, "max_length": 48})
        result_1 = self._read_result(os.path.join(self.output_dir, "predict.json"))
        print(f"result_0 {result_0}, result_1 {result_1}")

        assert len(result_0) == len(result_1)
        count, full_match = 0, 0

        for inference_item, no_inference_item in zip(result_0, result_1):
            min_length = min(len(inference_item), len(no_inference_item))
            count += int(inference_item[: min_length // 2] == no_inference_item[: min_length // 2])
            full_match += int(inference_item[:min_length] == no_inference_item[:min_length])

        self.assertGreaterEqual(count / len(result_0), 0.1)


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
        config.quant_type = ""

        paddle.set_default_dtype("float16")
        # need to use dtype guard
        model = QWenForQWenVLInferenceModel.from_pretrained(self.output_dir, config=config, dtype="float16")

        batch = 1
        seq = 31
        max_len = 50
        dtype = "float16"
        input_ids = paddle.randint(0, 100, [batch, seq], dtype="int64")
        image_features = paddle.randn([batch, 16, config.hidden_size], dtype="float16")
        tgt_generation_mask = paddle.full([batch, 1, 1, max_len], 1, dtype=dtype)
        img_pos = paddle.to_tensor([[0, 4, 21]], dtype="int64")
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
            tmp = paddle.rand(shape=[2, batch, 1, max_len, 64], dtype=dtype)
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
            from predict.export_model import main

            main()
