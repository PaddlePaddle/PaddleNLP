import copy
import random
import unittest

import numpy as np
from parameterized import parameterized
import paddle

from paddlenlp.generation import GenerationConfig
from paddlenlp.transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DynamicCache,
        LlamaConfig,
        RobertaForCausalLM,

)
from paddlenlp.transformers.cache_utils import StaticCache
from tests.testing_utils import slow
def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
class CacheTest(unittest.TestCase):
    def test_dynamic_cache_retrocompatibility(self):
        """Tests that we can convert back and forth between the legacy cache format and DynamicCache"""
        legacy_cache = ()
        new_cache = DynamicCache()

        # Creates a new cache with 10 layers in both formats
        for layer_idx in range(10):
            new_key = paddle.rand((2, 4, 8, 16))
            new_value = paddle.rand((2, 4, 8, 16))
            new_cache.update(new_key, new_value, layer_idx)
            legacy_cache += ((new_key, new_value),)

        # Sanity check 1: they must have the same shapes
        self.assertTrue(len(legacy_cache), len(new_cache))
        for layer_idx in range(10):
            self.assertTrue(len(legacy_cache[layer_idx]), len(legacy_cache[layer_idx]))
            for key_value_idx in range(2):
                self.assertTrue(
                    legacy_cache[layer_idx][key_value_idx].shape == new_cache[layer_idx][key_value_idx].shape
                )

        # Sanity check 2: we can get the sequence length in multiple ways with DynamicCache, and they return the
        # expected value
        self.assertTrue(legacy_cache[0][0].shape[-3] == new_cache[0][0].shape[-3] == new_cache.get_seq_length() == 4)

        # Sanity check 3: they must be equal, and both support indexing
        for layer_idx in range(10):
            for key_value_idx in range(2):
                self.assertTrue(
                    paddle.allclose(new_cache[layer_idx][key_value_idx], legacy_cache[layer_idx][key_value_idx])
                )

        # Test 1: We can convert from legacy to new with no changes
        from_legacy = DynamicCache.from_legacy_cache(legacy_cache)
        for layer_idx in range(10):
            for key_value_idx in range(2):
                self.assertTrue(
                    paddle.allclose(from_legacy[layer_idx][key_value_idx], legacy_cache[layer_idx][key_value_idx])
                )

        # Test 2: We can convert from new to legacy with no changes
        to_legacy = new_cache.to_legacy_cache()
        for layer_idx in range(10):
            for key_value_idx in range(2):
                self.assertTrue(
                    paddle.allclose(to_legacy[layer_idx][key_value_idx], new_cache[layer_idx][key_value_idx])
                )


    def test_static_cache_mha_mqa_gqa(self):
        """
        Tests that static cache works with multi-head attention (MHA), grouped query attention (GQA), and multi-query
        attention (MQA)
        """

        def _random_kvs(config):
            # shape for key and values: (batch_size, num_heads, seq_len, head_dim)
            random_keys = paddle.rand(
                (1, config.num_key_value_heads, 1, config.hidden_size // config.num_attention_heads)
            )
            random_values = paddle.rand(
                (1, config.num_key_value_heads, 1, config.hidden_size // config.num_attention_heads)
            )
            return random_keys, random_values

        mha_config = LlamaConfig(num_attention_heads=32)
        mha_static_cache = StaticCache(config=mha_config, batch_size=1, max_cache_len=10)
        cached_keys, cached_values = mha_static_cache.update(
            *_random_kvs(mha_config), 0, cache_kwargs={"cache_position": paddle.arange(1)}
        )

        self.assertTrue(cached_keys.shape == [1, 32, 10, 128])
        self.assertTrue(cached_values.shape == [1, 32, 10, 128])

        gqa_config = LlamaConfig(num_attention_heads=32, num_key_value_heads=4)
        gqa_static_cache = StaticCache(config=gqa_config, batch_size=1, max_cache_len=10)
        cached_keys, cached_values = gqa_static_cache.update(
            *_random_kvs(gqa_config), 0, cache_kwargs={"cache_position": paddle.arange(1)}
        )
        self.assertTrue(cached_keys.shape == [1, 4, 10, 128])
        self.assertTrue(cached_values.shape == [1, 4, 10, 128])

        mqa_config = LlamaConfig(num_attention_heads=32, num_key_value_heads=1)
        mqa_static_cache = StaticCache(config=mqa_config, batch_size=1, max_cache_len=10)
        cached_keys, cached_values = mqa_static_cache.update(
            *_random_kvs(mqa_config), 0, cache_kwargs={"cache_position": paddle.arange(1)}
        )
        self.assertTrue(cached_keys.shape == [1, 1, 10, 128])
        self.assertTrue(cached_values.shape == [1, 1, 10, 128])
        
class CacheIntegrationTest(unittest.TestCase):
    def test_offloaded_cache_equivalent_to_dynamic_cache(self):
        """Tests that OffloadedCache produces the same result as the default DynamicCache"""
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", paddle_dtype=paddle.float16)
        device = model.device
        input_text = "Fun fact:"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        common = {
            "num_beams": 4,
            "num_beam_groups": 2,
            "num_return_sequences": 4,
            "diversity_penalty": 1.0,
            "max_new_tokens": 20,
            "early_stopping": True,
        }
        original = GenerationConfig(**common)
        offloaded = GenerationConfig(cache_implementation="offloaded", **common)
        original_outputs = model.generate(generation_config=original, **inputs)
        offloaded_outputs = model.generate(generation_config=offloaded, **inputs)
        for original_output, offloaded_output in zip(original_outputs, offloaded_outputs):
            assert paddle.all(original_output == offloaded_output).item()

    def test_offloaded_cache_uses_less_memory_than_dynamic_cache(self):
        """Tests that OffloadedCache uses less memory than the default DynamicCache"""
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", paddle_dtype=paddle.float16)
        device = model.device
        input_text = "Fun fact:"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        common = {
            "num_beams": 4,
            "num_beam_groups": 2,
            "num_return_sequences": 4,
            "diversity_penalty": 1.0,
            "max_new_tokens": 20,
            "early_stopping": True,
        }
        original = GenerationConfig(**common)
        offloaded = GenerationConfig(cache_implementation="offloaded", **common)
        paddle.cuda.reset_peak_memory_stats(device)
        model.generate(generation_config=original, **inputs)
        original_peak_memory = paddle.cuda.max_memory_allocated
        (device)
        paddle.cuda.reset_peak_memory_stats(device)
        model.generate(generation_config=offloaded, **inputs)
        offloaded_peak_memory = paddle.cuda.max_memory_allocated(device)
        assert offloaded_peak_memory < original_peak_memory
