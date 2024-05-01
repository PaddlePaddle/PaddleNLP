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
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time

import numpy as np
import paddle
import pytest
from datasets import load_dataset
from tokenizers import Tokenizer as HFTokenizer

from paddlenlp.transformers import AutoTokenizer

MODEL_NAME = "google/gemma-7b"


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
def fast_tokenizer_hf():
    # Need authoried to use the tokenizer
    return None


@pytest.fixture
def tokenizer_fast():
    fast_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, from_slow=True)
    return fast_tokenizer


@pytest.fixture
def tokenizer_base():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


def test_tokenizer_type(tokenizer_fast, tokenizer_base):
    assert isinstance(tokenizer_fast._tokenizer, HFTokenizer)
    assert not hasattr(tokenizer_base, "_tokenizer")


def test_tokenizer_cost(tokenizer_fast, tokenizer_base, setup_inputs):
    costs = {}
    results = []
    for tokenizer in ["tokenizer_fast", "tokenizer_base"]:
        (
            _result,
            _time,
        ) = measure_time(eval(tokenizer), setup_inputs[:2000])
        costs[tokenizer] = _time
        results.append(_result["input_ids"])
        # slow is also different from fast in HF


def test_output_type(tokenizer_fast, setup_inputs):
    isinstance(tokenizer_fast.encode(setup_inputs[0], return_tensors="pd")["input_ids"], paddle.Tensor)
    isinstance(tokenizer_fast.encode(setup_inputs[0], return_tensors="np")["input_ids"], np.ndarray)
