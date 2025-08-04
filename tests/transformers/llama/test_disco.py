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

"""Unit tests for DISCO (Dynamic Score-based Cache Optimization) algorithm."""

import tempfile
import unittest

import numpy as np
import paddle

from paddlenlp.transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from paddlenlp.transformers.llama.disco_cache import DISCOCache, LayerwiseEvictionManager, get_scores_with_kv_fusion
import paddle.nn.functional as F

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
        
        self.assertEqual(len(cache.layer_budget), self.num_layers)
        self.assertEqual(sum(cache.layer_budget), self.cache_size)
        self.assertEqual(cache.window_size, self.window_size)
        self.assertEqual(cache.gamma, self.gamma)
        
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
        
        # Check shapes
        self.assertEqual(updated_keys.shape, key_states.shape)
        self.assertEqual(updated_values.shape, value_states.shape)
        
        # Test multiple updates to trigger eviction
        for i in range(3):
            new_keys = paddle.randn([self.batch_size, self.num_heads, 16, self.head_dim])
            new_values = paddle.randn([self.batch_size, self.num_heads, 16, self.head_dim])
            updated_keys, updated_values = cache.update(new_keys, new_values, layer_idx=0)
            
        # Check that cache size is within budget
        actual_size = updated_keys.shape[2]
        budget = cache.layer_budget[0]
        self.assertLessEqual(actual_size, budget)

    def test_layerwise_eviction_manager(self):
        """Test LayerwiseEvictionManager scoring functions."""
        manager = LayerwiseEvictionManager(
            cache_size=32,
            window_size=8,
            gamma=0.1
        )
        
        # Create dummy inputs
        key_states = paddle.randn([self.batch_size, self.num_heads, self.seq_length, self.head_dim])
        value_states = paddle.randn([self.batch_size, self.num_heads, self.seq_length, self.head_dim])
        attention_weights = F.softmax(
            paddle.randn([self.batch_size, self.num_heads, self.window_size, self.seq_length]),
            axis=-1
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
        
        self.assertEqual(scores.shape[0], self.batch_size)
        self.assertEqual(scores.shape[1], self.num_heads)

    def test_llama_with_disco_config(self):
        """Test LLaMA model initialization with DISCO configuration."""
        config = LlamaConfig(
            vocab_size=1000,  # Small vocab for testing
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            # DISCO parameters
            use_disco=True,
            disco_cache_size=64,
            disco_window_size=8,
            disco_gamma=0.15,
        )
        
        # Test model creation doesn't fail
        model = LlamaModel(config)
        
        # Verify DISCO cache is initialized
        self.assertIsNotNone(model.disco_cache)
        self.assertEqual(model.disco_cache.num_layers, config.num_hidden_layers)
        self.assertEqual(model.disco_cache.cache_size, config.disco_cache_size)
        
        # Verify attention layers have DISCO attributes
        for idx, layer in enumerate(model.layers):
            self.assertTrue(layer.self_attn.use_disco)
            self.assertEqual(layer.self_attn.layer_idx, idx)
            self.assertEqual(layer.self_attn.disco_cache, model.disco_cache)

    def test_llama_forward_with_disco(self):
        """Test LLaMA forward pass with DISCO enabled."""
        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            use_disco=True,
            disco_cache_size=32,
            disco_window_size=8,
        )
        
        model = LlamaModel(config)
        model.eval()
        
        # Create input
        input_ids = paddle.randint(0, config.vocab_size, shape=[self.batch_size, 16])
        
        # Forward pass with DISCO
        with paddle.no_grad():
            outputs = model(input_ids, use_cache=True)
        
        # Check outputs
        self.assertEqual(outputs[0].shape[0], self.batch_size)
        self.assertEqual(outputs[0].shape[1], 16)
        self.assertEqual(outputs[0].shape[2], config.hidden_size)

    def test_disco_disabled_compatibility(self):
        """Test that DISCO can be disabled without affecting normal operation."""
        # Config with DISCO disabled
        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            use_disco=False,  # DISCO disabled
        )
        
        model = LlamaModel(config)
        model.eval()
        
        # Verify DISCO cache is not initialized
        self.assertIsNone(model.disco_cache)
        
        # Verify model still works normally
        input_ids = paddle.randint(0, config.vocab_size, shape=[self.batch_size, 16])
        with paddle.no_grad():
            outputs = model(input_ids, use_cache=True)
        
        self.assertEqual(outputs[0].shape[0], self.batch_size)
        self.assertEqual(outputs[0].shape[1], 16)
        self.assertEqual(outputs[0].shape[2], config.hidden_size)

    def test_score_function_loading(self):
        """Test loading score function from file."""
        # Create a temporary score function file
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            coeffs = np.array([0.1, 0.2, 0.3, 0.4])
            np.save(f.name, coeffs)
            
            cache = DISCOCache(
                num_layers=self.num_layers,
                cache_size=self.cache_size,
                score_func_path=f.name,
            )
            
            self.assertIsNotNone(cache.score_func)
            # Test score function evaluation
            score = cache.score_func(1)
            self.assertIsInstance(score, (float, np.floating))

    def test_causal_lm_with_disco(self):
        """Test LlamaForCausalLM with DISCO enabled."""
        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            use_disco=True,
            disco_cache_size=32,
        )
        
        model = LlamaForCausalLM(config)
        model.eval()
        
        # Test generation doesn't fail
        input_ids = paddle.randint(0, config.vocab_size, shape=[1, 8])
        with paddle.no_grad():
            outputs = model(input_ids, use_cache=True)
        
        self.assertEqual(outputs[0].shape[-1], config.vocab_size)


if __name__ == "__main__":
    unittest.main()