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
import subprocess
import sys
from unittest import TestCase

from paddlenlp.utils.downloader import get_path_from_url
from tests.testing_utils import (
    argv_context_guard,
    load_test_config,
    slow,
    update_params,
)


class LLaMATest(TestCase):
    def setUp(self) -> None:
        self.path = "./llm"
        self.config_path = "./tests/fixtures/llm/llama.yaml"
        sys.path.insert(0, self.path)

    def tearDown(self) -> None:
        sys.path.remove(self.path)

    @slow
    def test_pretrain(self):

        # 1. run pretrain
        if not os.path.exists("./llm/llama/data"):
            URL = "https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy"
            URL2 = "https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz"
            get_path_from_url(URL, root_dir="./llm/llama/data")
            get_path_from_url(URL2, root_dir="./llm/llama/data")

        sys.path.insert(0, self.path + "/llama")
        print(self.path + "/llama")
        pretrain_config = load_test_config(self.config_path, "pretrain")
        with argv_context_guard(pretrain_config):
            from run_pretrain import main

            main()

    @slow
    def test_finetune(self):

        finetune_params = {
            "model_name_or_path": "__internal_testing__/micro-random-llama",
            "dataset_name_or_path": "./tests/fixtures/llm/data",
            "output_dir": "./llm/checkpoints/llama_sft_ckpts",
            "save_steps": 2,
            "max_steps": 2,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "tensor_parallel_degree": 1,
            "pipeline_parallel_degree": 1,
        }
        quant_params = {
            "dataset_name_or_path": "./tests/fixtures/llm/data",
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "model_name_or_path": "./llm/checkpoints/llama_sft_ckpts/checkpoint-2",
            "output_dir": "./llm/checkpoints/llama_ptq_ckpts",
        }

        # run sft
        run_fintune = "llm/finetune_generation.py"
        sft_json_file = "./llm/llama/sft_argument.json"
        update_params(sft_json_file, finetune_params)
        subprocess.check_output("python %s %s " % (run_fintune, sft_json_file), shell=True)

        # run sft_pp
        sft_pp_json_file = "./llm/llama/sft_pp_argument.json"
        finetune_params.update({"output_dir": "./llm/checkpoints/llama_sft_pp_ckpts"})
        update_params(sft_pp_json_file, finetune_params)
        subprocess.check_output("python %s %s " % (run_fintune, sft_pp_json_file), shell=True)

        # run lora
        lora_json_file = "./llm/llama/lora_argument.json"
        finetune_params.update({"output_dir": "./llm/checkpoints/llama_lora_ckpts"})
        update_params(lora_json_file, finetune_params)
        subprocess.check_output("python %s %s " % (run_fintune, lora_json_file), shell=True)

        # run prefix tuning
        pt_json_file = "./llm/llama/pt_argument.json"
        finetune_params.update({"output_dir": "./llm/checkpoints/llama_pt_ckpts"})
        update_params(pt_json_file, finetune_params)
        subprocess.check_output("python %s %s " % (run_fintune, pt_json_file), shell=True)

        # run  ptq quant
        ptq_json_file = "./llm/llama/ptq_argument.json"
        update_params(ptq_json_file, quant_params)
        subprocess.check_output("python %s %s " % (run_fintune, ptq_json_file), shell=True)

        # run gptq quant
        gptq_json_file = "./llm/llama/gptq_argument.json"
        quant_params.update({"output_dir": "./llm/checkpoints/llama_gptq_ckpts"})
        update_params(gptq_json_file, quant_params)
        subprocess.check_output("python %s %s " % (run_fintune, gptq_json_file), shell=True)

    @slow
    def test_merge_params(self):

        # 1. Merge Tensor Parallelism Params
        merge_tp_config = load_test_config(self.config_path, "merge_tp_params")
        with argv_context_guard(merge_tp_config):
            from merge_tp_params import main

            main()

        # 2. Merge Lora Params
        merge_lora_config = load_test_config(self.config_path, "merge_lora_params")
        with argv_context_guard(merge_lora_config):
            from merge_lora_params import merge

            merge()

    @slow
    def test_predict(self):
        # SFT dynamic predict
        predict_config = load_test_config(self.config_path, "predict")
        with argv_context_guard(predict_config):
            from predictor import predict

            predict()

        # LoRA dynamic predict
        lora_predict_config = {
            "model_name_or_path": predict_config["model_name_or_path"],
            "batch_size": predict_config["batch_size"],
            "data_file": predict_config["data_file"],
            "dtype": predict_config["dtype"],
            "mode": predict_config["mode"],
            "lora_path": "./llm/checkpoints/llama_lora_ckpts/checkpoint-2/",
        }
        with argv_context_guard(lora_predict_config):
            from predictor import predict

            predict()

        # Prefix Tuning dynamic predict
        pt_predict_config = {
            "model_name_or_path": predict_config["model_name_or_path"],
            "batch_size": predict_config["batch_size"],
            "data_file": predict_config["data_file"],
            "dtype": predict_config["dtype"],
            "mode": predict_config["mode"],
            "prefix_path": "./llm/checkpoints/llama_pt_ckpts/checkpoint-2/",
        }
        with argv_context_guard(pt_predict_config):
            from predictor import predict

            predict()

        # export model
        export_config = {
            "model_name_or_path": predict_config["model_name_or_path"],
            "output_path": "./llm/inference",
            "dtype": predict_config["dtype"],
        }
        with argv_context_guard(export_config):
            from export_model import main

            main()

        # static predict
        st_predict_config = {
            "model_name_or_path": "./llm/inference",
            "batch_size": predict_config["batch_size"],
            "data_file": predict_config["data_file"],
            "dtype": predict_config["dtype"],
            "mode": "static",
        }
        with argv_context_guard(st_predict_config):
            from predictor import predict

            predict()
