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

import tempfile
import time

import numpy as np
import paddle
import pytest
from tokenizers import Tokenizer as HFTokenizer

from paddlenlp.transformers import AutoTokenizer

temp_path = tempfile.mkdtemp()
MODEL_NAME = "ernie-m-base"


def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    execution_time = end_time - start_time

    return result, execution_time


@pytest.fixture
def setup_inputs():
    single_s = "今天天气很好"
    return single_s


@pytest.fixture
def tokenizer_fast():
    fast_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, from_slow=True)
    return fast_tokenizer


def test_tokenizer_type(tokenizer_fast):
    assert isinstance(tokenizer_fast._tokenizer, HFTokenizer)
    tokenizer_fast.save_pretrained(temp_path)


def test_tokenizer_cost(tokenizer_fast, setup_inputs):
    costs = {}
    results = []
    for tokenizer in ["tokenizer_fast"]:
        (
            _result,
            _time,
        ) = measure_time(eval(tokenizer), setup_inputs)
        costs[tokenizer] = _time
        results.append(_result["input_ids"])

    print(costs)
    assert tokenizer_fast.decode(results[0]) == f"[CLS] {setup_inputs}[SEP]"


def test_output_type(setup_inputs, tokenizer_fast):
    isinstance(tokenizer_fast.encode(setup_inputs[0], return_tensors="pd")["input_ids"], paddle.Tensor)
    isinstance(tokenizer_fast.encode(setup_inputs[0], return_tensors="np")["input_ids"], np.ndarray)
