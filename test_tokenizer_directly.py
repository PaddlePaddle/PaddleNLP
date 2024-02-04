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

import psutil
import pytest
from tokenizers import Tokenizer as HFTokenizer

from fast_tokenizer import Tokenizer as FastTokenizer
from paddlenlp.transformers import AutoTokenizer

MODEL_HF = r"/Users/tanzhehao/.paddlenlp/models/models--bert-base-uncased/snapshots/1dbc166cf8765166998eff31ade2eb64c8a40076/tokenizer.json"
MODEL_FAST = r"/Users/tanzhehao/.paddlenlp/models/bert-base-uncased/tokenizer.json"
MODEL_NAME = "bert-base-uncased"


def pytest_addoption(parser):
    parser.addoption(
        "--test-times",
        action="store",
        default=10000,
    )


def measure_time_and_memory(func: callable, *args, **kwargs):
    start_time = time.time()
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024

    result = func(*args, **kwargs)

    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024

    execution_time = end_time - start_time
    memory_usage = end_memory - start_memory

    return result, execution_time, memory_usage


@pytest.fixture
def test_times(request):
    return request.config.getoption("--test-times")


@pytest.fixture
def setup_inputs():
    single_s = "natual language processing"
    return single_s


@pytest.fixture
def tokenizer_fast_hf():
    fast_tokenizer_hf = HFTokenizer.from_file(MODEL_HF)
    return fast_tokenizer_hf


@pytest.fixture
def tokenizer_fast():
    fast_tokenizer = FastTokenizer.from_file(MODEL_FAST)
    return fast_tokenizer


@pytest.fixture
def tokenizer_base():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


def test_tokenizer_type(tokenizer_fast_hf, tokenizer_fast, tokenizer_base):
    assert isinstance(tokenizer_fast_hf, HFTokenizer)
    assert isinstance(tokenizer_fast, FastTokenizer)
    assert not hasattr(tokenizer_base, "_tokenizer")


def test_tokenizer_cost(tokenizer_fast_hf, tokenizer_fast, tokenizer_base, setup_inputs, test_times):
    # breakpoint()
    print(test_times)
    costs = []
    for tokenizer in ["tokenizer_fast_hf", "tokenizer_fast", "tokenizer_base"]:
        if tokenizer != "tokenizer_base":
            (
                _,
                _time,
                _memory,
            ) = measure_time_and_memory(eval(f"{tokenizer}.encode_batch"), [setup_inputs] * test_times)
        else:
            (
                _,
                _time,
                _memory,
            ) = measure_time_and_memory(eval(tokenizer), [setup_inputs] * test_times)
        costs.append({tokenizer: (_memory, _time)})
    print(costs)


def test_tokenizer_decode(tokenizer_fast_hf, tokenizer_fast, tokenizer_base, setup_inputs):
    token_hf = tokenizer_fast_hf.encode(setup_inputs)
    token_fast = tokenizer_fast.encode(setup_inputs)
    token_base = tokenizer_base(setup_inputs)
    assert token_hf.ids == token_fast.ids == token_base["input_ids"]
