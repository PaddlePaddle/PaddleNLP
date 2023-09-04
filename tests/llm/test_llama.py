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
import sys
from unittest import TestCase

from paddlenlp.utils.downloader import get_path_from_url
from tests.testing_utils import argv_context_guard, load_test_config, update_params


class LLaMATest(TestCase):
    def setUp(self) -> None:
        self.path = "./llm/"
        self.config_path = "./tests/fixtures/llm/llama.yaml"
        sys.path.insert(0, self.path)

    def tearDown(self) -> None:
        sys.path.remove(self.path)

    def test_pretrain(self):

        # 1. run pretrain
        if not os.path.exists("./llm/llama/data"):
            URL = "https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy"
            URL2 = "https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz"
            get_path_from_url(URL, root_dir="./llm/llama/data")
            get_path_from_url(URL2, root_dir="./llm/llama/data")

        # fix fused_layers import error
        sys.path.insert(0, self.path + "/llama")

        pretrain_config = load_test_config(self.config_path, "pretrain")
        with argv_context_guard(pretrain_config):
            from run_pretrain import main

            main()

    def test_finetune(self):

        run_finetune_format = "python -u  -m paddle.distributed.launch --gpus 0,1 llm/finetune_generation.py "

        # 1. run sft
        sft_json_file = os.path.join(self.path + "llama/sft_argument.json")
        sft_params = {
            "dataset_name_or_path": "./fixtures/llm/data",
            "save_steps": 5,
            "max_steps": 5,
            "tensor_parallel_degree": 2,
        }
        update_params(sft_json_file, sft_params)
        os.system(f"{run_finetune_format + sft_json_file}")

        # 2. run lora
        lora_json_file = os.path.join(self.path + "llama/lora_argument.json")
        lora_params = {
            "dataset_name_or_path": "./fixtures/llm/data",
            "save_steps": 5,
            "max_steps": 5,
            "tensor_parallel_degree": 2,
        }
        update_params(lora_json_file, lora_params)
        os.system(f"{run_finetune_format + lora_json_file}")

        # 3. run prefix tuning
        pt_json_file = os.path.join(self.path + "llama/pt_argument.json")
        pt_params = {
            "dataset_name_or_path": "./fixtures/llm/data",
            "save_steps": 5,
            "max_steps": 5,
            "tensor_parallel_degree": 2,
        }
        update_params(pt_json_file, pt_params)
        os.system(f"{run_finetune_format + pt_json_file}")

        # 4. run  ptq quantization
        ptq_json_file = os.path.join(self.path + "llama/ptq_argument.json")
        os.system(f"python finetune_generation.py {ptq_json_file}")

        # 5. run gptq quantization
        gptq_json_file = os.path.join(self.path + "llama/gptq_argument.json")
        os.system(f"python finetune_generation.py {gptq_json_file}")

    def test_merge_params(self):

        # 1. Merge Tensor Parallelism
        merge_config = load_test_config(self.config_path, "merge")
        with argv_context_guard(merge_config):
            from merge_tp_params import main

            main()

        # 2. Merge Lora
        lora_config = {
            "model_name_or_path": merge_config["meta-llama/Llama-2-7b-chat"],
            "lora_path": merge_config["./checkpoints/llama_lora_ckpts/checkpoint-5"],
        }
        with argv_context_guard(lora_config):
            from merge_tp_params import merge

            merge()

    def test_predict(self):
        pretrain_config = load_test_config(self.config_path, "pretrain")
        # 1. dynamic predict
        dy_config = {
            "model_name_or_path": pretrain_config["facebook/llama-7b"],
            "batch_size": pretrain_config["1"],
            "data_file": pretrain_config["./data/dev.json"],
            "dtype": pretrain_config["float16"],
            "mode": pretrain_config["dynamic"],
        }
        with argv_context_guard(dy_config):
            from predictor import predict

            predict()

        # 2. export model
        export_config = {
            "model_name_or_path": pretrain_config["meta-llama/Llama-2-7b-chat"],
            "output_path": pretrain_config["1"],
            "dtype": pretrain_config["./inference"],
        }
        with argv_context_guard(export_config):
            from export_model import main

            main()

        # 3. static predict
        st_config = {
            "model_name_or_path": pretrain_config["./inference "],
            "batch_size": pretrain_config["1"],
            "data_file": pretrain_config["./data/dev.json"],
            "dtype": pretrain_config["float16"],
            "mode": pretrain_config["static"],
        }
        with argv_context_guard(st_config):
            from predictor import predict

            predict()
