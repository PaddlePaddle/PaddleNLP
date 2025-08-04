#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Run all DISCO tests independently"""

import sys
import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

# Direct import to avoid circular dependency
exec(open("paddlenlp/transformers/llama/disco_cache.py").read())


class DISCOCacheTest(unittest.TestCase):
    """Test cases for DISCO cache implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_layers = 4
        self.cache_size = 128
        self.window_size = 16
        self.gamma = 0.1
        self.batch_size = 2
        self.num_heads = 8
        self.head_dim = 64
        self.seq_length = 32

    def test_disco_cache_initialization(self):
        """Test DISCO cache initialization with different configurations."""
        # Test default initialization
        cache = DISCOCache(
            num_layers=self.num_layers,
            cache_size=self.cache_size,
            window_size=self.window_size,
            gamma=self.gamma,
        )

        self.assertEqual(cache.num_layers, self.num_layers)
        self.assertEqual(cache.cache_size, self.cache_size)
        self.assertEqual(cache.window_size, self.window_size)
        self.assertEqual(cache.gamma, self.gamma)

        # Check default layer budget
        expected_budget = [self.cache_size // self.num_layers] * self.num_layers
        self.assertEqual(cache.layer_budget, expected_budget)

        # Test with custom layer budget
        custom_budget = [40, 30, 30, 28]
        cache_custom = DISCOCache(
            num_layers=self.num_layers,
            cache_size=self.cache_size,
            window_size=self.window_size,
            gamma=self.gamma,
            layer_budget=custom_budget,
        )
        self.assertEqual(cache_custom.layer_budget, custom_budget)

    def test_cache_update_and_eviction(self):
        """Test cache update and eviction mechanism."""
        cache = DISCOCache(
            num_layers=self.num_layers,
            cache_size=64,  # Small cache for testing eviction
            window_size=8,
            gamma=self.gamma,
        )

        # Create dummy key and value states
        key_states = paddle.randn([self.batch_size, self.num_heads, self.seq_length, self.head_dim])
        value_states = paddle.randn([self.batch_size, self.num_heads, self.seq_length, self.head_dim])

        # Test update for layer 0
        updated_keys, updated_values = cache.update(key_states, value_states, layer_idx=0)

        # Check shapes - first update should not evict
        self.assertEqual(list(updated_keys.shape), list(key_states.shape))
        self.assertEqual(list(updated_values.shape), list(value_states.shape))

        # Test multiple updates to trigger eviction
        for i in range(3):
            new_keys = paddle.randn([self.batch_size, self.num_heads, 16, self.head_dim])
            new_values = paddle.randn([self.batch_size, self.num_heads, 16, self.head_dim])
            updated_keys, updated_values = cache.update(new_keys, new_values, layer_idx=0)

        # Check that cache size is within budget
        actual_size = updated_keys.shape[2]
        expected_max = cache.layer_budget[0]
        self.assertLessEqual(actual_size, expected_max)

    def test_layerwise_eviction_manager(self):
        """Test LayerwiseEvictionManager scoring functions."""
        manager = LayerwiseEvictionManager(cache_size=32, window_size=8, gamma=0.1)

        # Create dummy inputs
        key_states = paddle.randn([self.batch_size, self.num_heads, self.seq_length, self.head_dim])
        value_states = paddle.randn([self.batch_size, self.num_heads, self.seq_length, self.head_dim])
        attention_weights = F.softmax(
            paddle.randn([self.batch_size, self.num_heads, self.window_size, self.seq_length]), axis=-1
        )

        # Test attention-based scoring
        scores = manager.compute_scores(key_states, value_states, attention_weights)
        self.assertEqual(scores.shape[0], self.batch_size)
        self.assertEqual(scores.shape[1], self.num_heads)

        # Test distance-based scoring
        distance_scores = manager.compute_distance_scores(key_states, value_states)
        self.assertEqual(distance_scores.shape[0], self.batch_size)
        self.assertEqual(distance_scores.shape[-1], self.seq_length)

    def test_kv_fusion_scoring(self):
        """Test key-value fusion scoring function."""
        key_states = paddle.randn([self.batch_size, self.num_heads, self.seq_length, self.head_dim])
        value_states = paddle.randn([self.batch_size, self.num_heads, self.seq_length, self.head_dim])

        scores = get_scores_with_kv_fusion(key_states, value_states, window_size=16)

        # Check output shape
        self.assertEqual(scores.shape[0], self.batch_size)
        self.assertEqual(scores.shape[1], self.num_heads)

    def test_cache_get_operations(self):
        """Test cache retrieval operations."""
        cache = DISCOCache(
            num_layers=self.num_layers,
            cache_size=self.cache_size,
            window_size=self.window_size,
            gamma=self.gamma,
        )

        # Initially empty
        self.assertEqual(cache.get_seq_length(), 0)
        self.assertEqual(cache.get_seq_length(layer_idx=0), 0)

        # Add some data
        key_states = paddle.randn([self.batch_size, self.num_heads, self.seq_length, self.head_dim])
        value_states = paddle.randn([self.batch_size, self.num_heads, self.seq_length, self.head_dim])
        cache.update(key_states, value_states, layer_idx=0)

        # Check sequence length
        self.assertEqual(cache.get_seq_length(layer_idx=0), self.seq_length)
        self.assertEqual(cache.get_seq_length(), self.seq_length)

        # Get cache for specific layer
        retrieved_keys, retrieved_values = cache.get_cache(layer_idx=0)
        self.assertIsNotNone(retrieved_keys)
        self.assertIsNotNone(retrieved_values)
        self.assertEqual(list(retrieved_keys.shape), list(key_states.shape))

        # Get cache for non-existent layer
        empty_keys, empty_values = cache.get_cache(layer_idx=999)
        self.assertIsNone(empty_keys)
        self.assertIsNone(empty_values)


def run_tests():
    """Run all tests and report results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(DISCOCacheTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running DISCO Cache Tests")
    print("=" * 60)
    success = run_tests()
    print("=" * 60)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
