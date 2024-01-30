import pytest
import time
import psutil
from paddlenlp.transformers import AutoTokenizer
from tokenizers import Tokenizer as HFTokenizer
from fast_tokenizer import Tokenizer as FastTokenizer

MODEL_NAME = 'bert-base-uncased'

def measure_time_and_memory(func, *args, **kwargs):
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
def setup_inputs():
    single_s = "natual language processing"
    return single_s

@pytest.fixture
def tokenizer_fast_hf():
    fast_tokenizer_hf = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, from_hf_hub=True)
    return fast_tokenizer_hf

@pytest.fixture
def tokenizer_fast():
    fast_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    return fast_tokenizer

@pytest.fixture
def tokenizer_base():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer

def test_tokenizer_type(tokenizer_fast_hf, tokenizer_fast, tokenizer_base):
    assert isinstance(tokenizer_fast_hf._tokenizer, HFTokenizer)
    assert isinstance(tokenizer_fast._tokenizer, FastTokenizer)
    assert not hasattr(tokenizer_base, '_tokenizer')

def test_tokenizer_cost(tokenizer_fast_hf, tokenizer_fast, tokenizer_base, setup_inputs):
    costs = []
    for tokenizer in ['tokenizer_fast_hf', 'tokenizer_fast', 'tokenizer_base']:
        _,_time,_memory, = measure_time_and_memory(eval(tokenizer), [setup_inputs]*100)
        costs.append({tokenizer: (_memory, _time)})

def test_tokenizer_decode(tokenizer_fast_hf, tokenizer_fast, tokenizer_base, setup_inputs):
    token_hf = tokenizer_fast_hf(setup_inputs)
    token_fast = tokenizer_fast(setup_inputs)
    token_base = tokenizer_base(setup_inputs)

    assert token_hf['input_ids'] == token_fast['input_ids'] == token_base['input_ids']
    