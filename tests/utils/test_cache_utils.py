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

import random
import unittest

import numpy as np
import paddle

from paddlenlp.transformers import DynamicCache, LlamaConfig, LlamaForCausalLM
from paddlenlp.transformers.cache_utils import StaticCache


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

    def test_cache_equivalence(self):
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

    def test_reorder_cache_retrocompatibility(self):
        """Tests that Cache.reorder_cache is retrocompatible with the legacy code path"""
        legacy_reorder_fn = LlamaForCausalLM.reorder_cache  # An example of a legacy `reorder_cache` function

        legacy_cache = ()
        new_cache = DynamicCache()

        # Creates a new cache with 10 layers in both formats
        for layer_idx in range(10):
            new_key = paddle.rand((4, 4, 8, 16))
            new_value = paddle.rand((4, 4, 8, 16))
            new_cache.update(new_key, new_value, layer_idx)
            legacy_cache += ((new_key, new_value),)

        # Let's create some dummy beam indices. From the shape above, it is equivalent to the case where num_beams=4
        # and batch_size=1
        # beam_idx = paddle.randint(low=0, high=4, size=(4,))
        beam_idx = paddle.randint(low=0, high=4, shape=(4,))
        beam_idx = paddle.Tensor(beam_idx, dtype=paddle.int64)

        legacy_cache_reordered = legacy_reorder_fn(None, legacy_cache, beam_idx)
        new_cache.reorder_cache(beam_idx)

        # Let's check that the results are the same
        for layer_idx in range(10):
            for key_value_idx in range(2):
                self.assertTrue(
                    paddle.allclose(
                        new_cache[layer_idx][key_value_idx], legacy_cache_reordered[layer_idx][key_value_idx]
                    )
                )


# @slow
# class CacheIntegrationTest(unittest.TestCase):
#     def test_dynamic_cache_hard(self):
#         tokenizer = AutoTokenizer.from_pretrained(
#             "meta-llama/Llama-2-7b-hf", padding_side="left", from_hf_hub=True, use_fast=True
#         )
#         model = AutoModelForCausalLM.from_pretrained(
#             "meta-llama/Llama-2-7b-hf",
#             dtype=paddle.float16,
#             from_hf_hub=True,
#         )
#         inputs = tokenizer(["Here's everything I know about cats. Cats"], return_tensors="np")
#         for key in inputs:
#             inputs[key] = paddle.to_tensor(inputs[key])

#         # DynamicCache and the legacy cache format should be equivalent
#         set_seed(0)
#         gen_out_legacy = model.generate(**inputs, do_sample=True, max_new_tokens=256)
#         set_seed(0)
#         gen_out = model.generate(**inputs, do_sample=True, max_new_tokens=256, past_key_values=DynamicCache())
#         self.assertListEqual(gen_out_legacy[0].tolist(), gen_out[0].tolist())
#         self.assertListEqual(gen_out_legacy[1].tolist(), gen_out[1].tolist())

#         decoded = tokenizer.batch_decode(gen_out[0], skip_special_tokens=True)

#         expected_text = (
#             "Here's everything I know about cats. Cats are mysterious creatures. They can't talk, and they don't like "
#             "to be held. They don't play fetch, and they don't like to be hugged. But they do like to be petted.\n"
#             "Cats are also very independent. They don't like to be told what to do, and they don't like to be told "
#             "what to eat. They are also very territorial. They don't like to share their food or their toys.\nCats "
#             "are also very curious. They like to explore, and they like to play. They are also very fast. They can "
#             "run very fast, and they can jump very high.\nCats are also very smart. They can learn tricks, and they "
#             "can solve problems. They are also very playful. They like to play with toys, and they like to play with "
#             "other cats.\nCats are also very affectionate. They like to be petted, and they like to be held. They "
#             "also like to be scratched.\nCats are also very clean. They like to groom themselves, and they like to "
#             "clean their litter box.\nCats are also very independent. They don't"
#         )
#         self.assertEqual(decoded[0], expected_text)

#     def test_dynamic_cache_batched(self):
#         tokenizer = AutoTokenizer.from_pretrained(
#             "meta-llama/Llama-2-7b-hf", padding_side="left", from_hf_hub=True, use_fast=True
#         )
#         tokenizer.pad_token = tokenizer.eos_token
#         model = AutoModelForCausalLM.from_pretrained(
#             "meta-llama/Llama-2-7b-hf",
#             dtype=paddle.float16,
#             from_hf_hub=True,
#         )
#         inputs = tokenizer(["A sequence: 1, 2, 3, 4, 5", "A sequence: A, B, C"], padding=True, return_tensors="np").to(
#             model.device
#         )
#         for key in inputs:
#             inputs[key] = paddle.to_tensor(inputs[key])

#         gen_out = model.generate(**inputs, do_sample=False, max_new_tokens=10, past_key_values=DynamicCache())
#         decoded = tokenizer.batch_decode(gen_out[0], skip_special_tokens=True)
#         expected_text = ["A sequence: 1, 2, 3, 4, 5, 6, 7, 8,", "A sequence: A, B, C, D, E, F, G, H"]
#         self.assertListEqual(decoded, expected_text)

#     def test_dynamic_cache_beam_search(self):
#         tokenizer = AutoTokenizer.from_pretrained(
#             "meta-llama/Llama-2-7b-hf", padding_side="left", from_hf_hub=True, use_fast=True
#         )
#         model = AutoModelForCausalLM.from_pretrained(
#             "meta-llama/Llama-2-7b-hf",
#             dtype=paddle.float16,
#             from_hf_hub=True,
#         )

#         inputs = tokenizer(["The best color is"], return_tensors="np")
#         for key in inputs:
#             inputs[key] = paddle.to_tensor(inputs[key])
#         gen_out = model.generate(
#             **inputs,
#             do_sample=False,
#             max_new_tokens=20,
#             num_beams=2,
#             num_return_sequences=2,
#         )
#         decoded = tokenizer.batch_decode(gen_out[0], skip_special_tokens=True)
#         expected_text = [
#             "The best color is the one that makes you feel good.\nThe best color is the one that makes you feel good",
#             "The best color is the one that suits you.\nThe best color is the one that suits you. The",
#         ]
#         self.assertListEqual(decoded, expected_text)
