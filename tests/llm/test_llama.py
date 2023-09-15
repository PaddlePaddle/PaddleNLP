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
import shutil
import sys
import tempfile
from unittest import TestCase

from paddlenlp.utils.downloader import get_path_from_url
from tests.testing_utils import (
    argv_context_guard,
    load_test_config,
    run_command,
    update_params,
)


class LLaMATest(TestCase):
    def setUp(self) -> None:
        self.path = "./llm"
        self.config_path = "./tests/fixtures/llm/llama.yaml"
        self.output_dir = tempfile.mkdtemp()
        sys.path.insert(0, self.path)

    def tearDown(self) -> None:
        sys.path.remove(self.path)
        shutil.rmtree(self.output_dir)

    def test_pretrain(self):

        # Run pretrain
        if not os.path.exists("./llm/llama/data"):
            URL = "https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy"
            URL2 = "https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz"
            get_path_from_url(URL, root_dir="./llm/llama/data")
            get_path_from_url(URL2, root_dir="./llm/llama/data")

        sys.path.insert(0, self.path + "/llama")
        pretrain_config = load_test_config(self.config_path, "pretrain")
        pretrain_config.update({"output": os.path.join(self.output_dir, "pretrain")})
        with argv_context_guard(pretrain_config):
            from run_pretrain import main

            main()

    def llm_finetune(self, model_name: str, method: str):
        """
        Tests a `complete` llm fintune examples

        Args:
            model_name (`str`):
                The LLM name, eg: llama, chatglm
            method (`str`):
                intruction  finetune method, eg: sft or lora
                quanzitation method aslo can be used by qtp or gqtp

        """
        run_fintune = "llm/finetune_generation.py"
        finetune_params = load_test_config(self.config_path, "finetune")
        quant_params = load_test_config(self.config_path, "quant")
        predict_config = load_test_config(self.config_path, "predict")
        merge_lora_config = load_test_config(self.config_path, "merge_lora_params")

        # Copy json file to tmp dir
        model_path = os.path.join("./llm/", model_name)
        for json_file in os.listdir(model_path):
            if json_file.endswith("json"):
                shutil.copyfile(model_path + "/" + json_file, os.path.join(self.output_dir, json_file))

        # 1.Run finetune
        params_file = method + "_argument.json"
        checkpoint_output = model_name + "_" + method + "_ckpts"
        json_file = os.path.join(self.output_dir, params_file)
        finetune_params.update({"output_dir": os.path.join(self.output_dir, checkpoint_output)})
        update_params(json_file, finetune_params)
        run_command("python %s %s " % (run_fintune, json_file))

        # 2. Run quant
        if method == "ptq" or method == "gptq":

            quant_params.update(
                {
                    "model_name_or_path": os.path.join(self.output_dir, "llama_sft_ckpts"),
                    "output_dir": os.path.join(self.output_dir, checkpoint_output),
                }
            )
            update_params(json_file, quant_params)
            run_command("python %s %s " % (run_fintune, json_file))

        # 3. Run predict
        if method == "lora":
            # LoRA dynamic predict
            lora_predict_config = {
                "model_name_or_path": predict_config["model_name_or_path"],
                "batch_size": predict_config["batch_size"],
                "data_file": predict_config["data_file"],
                "dtype": predict_config["dtype"],
                "mode": predict_config["mode"],
                "lora_path": os.path.join(self.output_dir, "llama_lora_ckpts/"),
            }

            with argv_context_guard(lora_predict_config):
                from predictor import predict

                predict()

            # Merge Lora Params
            merge_lora_config = {
                "model_name_or_path": merge_lora_config["model_name_or_path"],
                "lora_path": os.path.join(self.output_dir, "llama_lora_ckpts/"),
            }
            with argv_context_guard(merge_lora_config):
                from merge_lora_params import merge

                merge()

        if method == "pt":
            # Prefix Tuning dynamic predict
            pt_predict_config = {
                "model_name_or_path": predict_config["model_name_or_path"],
                "batch_size": predict_config["batch_size"],
                "data_file": predict_config["data_file"],
                "dtype": predict_config["dtype"],
                "mode": predict_config["mode"],
                "prefix_path": os.path.join(self.output_dir, "llama_pt_ckpts/"),
            }
            with argv_context_guard(pt_predict_config):
                from predictor import predict

                predict()

            # Merge Tensor Parallelism Params
            merge_tp_config = {"model_name_or_path": os.path.join(self.output_dir, "llama_pt_ckpts/")}
            with argv_context_guard(merge_tp_config):
                from merge_tp_params import main

                main()

    def test_llm_finetune(self):
        self.llm_finetune("llama", "sft")
        self.llm_finetune("llama", "sft_pp")

        self.llm_finetune("llama", "lora")
        self.llm_finetune("llama", "pt")

        self.llm_finetune("llama", "ptq")
        self.llm_finetune("llama", "gptq")

    def test_predict(self):
        # dynamic predict
        predict_config = load_test_config(self.config_path, "predict")
        with argv_context_guard(predict_config):
            from predictor import predict

            predict()

        # Export model
        export_config = {
            "model_name_or_path": predict_config["model_name_or_path"],
            "output_path": "./llm/inference",
            "dtype": predict_config["dtype"],
        }
        with argv_context_guard(export_config):
            from export_model import main

            main()

        # Static predict
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
