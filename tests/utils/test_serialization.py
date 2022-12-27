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

import os
import tempfile
from unittest import TestCase

import numpy as np
from huggingface_hub import hf_hub_download
from parameterized import parameterized

from paddlenlp.utils import load_torch
from tests.testing_utils import require_package


class SerializationTest(TestCase):
    @parameterized.expand(
        [
            "float32",
            "float16",
        ]
    )
    @require_package("torch")
    def test_simple_load(self, dtype: str):
        import torch

        # torch "normal_kernel_cpu" not implemented for 'Char', 'Int', 'Long', so only support float
        dtype_mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
        }
        dtype = dtype_mapping[dtype]

        with tempfile.TemporaryDirectory() as tempdir:
            weight_file_path = os.path.join(tempdir, "pytorch_model.bin")
            torch.save({"a": torch.randn(2, 3, dtype=dtype), "b": torch.randn(3, 4, dtype=dtype)}, weight_file_path)
            numpy_data = load_torch(weight_file_path)
            torch_data = torch.load(weight_file_path)

            for key, arr in numpy_data.items():
                assert np.allclose(
                    arr,
                    torch_data[key].numpy(),
                )

    @parameterized.expand(
        [
            "hf-internal-testing/tiny-random-codegen",
            "hf-internal-testing/tiny-random-Data2VecTextModel",
            "hf-internal-testing/tiny-random-SwinModel",
        ]
    )
    @require_package("torch")
    def test_load_bert_model(self, repo_id):
        import torch

        with tempfile.TemporaryDirectory() as tempdir:
            weight_file = hf_hub_download(
                repo_id=repo_id,
                filename="pytorch_model.bin",
                cache_dir=tempdir,
                library_name="PaddleNLP",
            )
            torch_weight = torch.load(weight_file)
            torch_weight = {key: value for key, value in torch_weight.items()}
            paddle_weight = load_torch(weight_file)

            for key, arr in paddle_weight.items():
                assert np.allclose(
                    arr,
                    torch_weight[key].numpy(),
                )
