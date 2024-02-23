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

import time

import numpy as np
import paddle
import pytest
from datasets import load_dataset
from tokenizers import Tokenizer as HFTokenizer

from paddlenlp.transformers import AutoTokenizer

MODEL_NAME = "facebook/llama-7b"


def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    execution_time = end_time - start_time

    return result, execution_time


@pytest.fixture
def setup_inputs():
    dataset = load_dataset("tatsu-lab/alpaca")
    return dataset["train"]["text"]


@pytest.fixture
def tokenizer_hf():
    from transformers import AutoTokenizer as AutoTokenizer_HF

    fast_tokenizer_hf = AutoTokenizer_HF.from_pretrained(
        "hf-internal-testing/llama-tokenizer", use_fast=False, trust_remote_code=True
    )
    return fast_tokenizer_hf


@pytest.fixture
def tokenizer_fast():
    fast_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, from_slow=True)
    return fast_tokenizer


@pytest.fixture
def tokenizer_base():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


def test_tokenizer_type(tokenizer_hf, tokenizer_fast, tokenizer_base):
    assert isinstance(tokenizer_fast._tokenizer, HFTokenizer)
    assert not hasattr(tokenizer_base, "_tokenizer")


def test_tokenizer_cost(tokenizer_hf, tokenizer_fast, tokenizer_base, setup_inputs):
    costs = {}
    results = []
    for tokenizer in ["tokenizer_hf", "tokenizer_fast", "tokenizer_base"]:
        (
            _result,
            _time,
        ) = measure_time(eval(tokenizer), setup_inputs[:20000])
        costs[tokenizer] = _time
        results.append(_result["input_ids"])
    record = []
    for i in range(1000):
        if results[0][i] != results[1][i]:
            record.append(i)
    print(costs, record)
    assert results[0] == results[1] == results[2]


def test_output_type(tokenizer_fast, setup_inputs):
    isinstance(
        tokenizer_fast.batch_encode(setup_inputs[:20], return_tensors="pd", padding=True)["input_ids"], paddle.Tensor
    )
    isinstance(
        tokenizer_fast.batch_encode(setup_inputs[:20], return_tensors="np", padding=True)["input_ids"], np.ndarray
    )
