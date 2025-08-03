#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple test script for DISCO functionality without full PaddleNLP import."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paddle
import numpy as np

# Import only the necessary modules
from paddlenlp.transformers.llama.disco_cache import DISCOCache, LayerwiseEvictionManager, get_scores_with_kv_fusion
from paddlenlp.transformers.llama.configuration import LlamaConfig


def test_disco_cache():
    """Test basic DISCO cache functionality."""
    print("Testing DISCO Cache...")
    
    # Test initialization
    cache = DISCOCache(
        num_layers=4,
        cache_size=128,
        window_size=16,
        gamma=0.1,
    )
    
    assert len(cache.layer_budget) == 4
    assert sum(cache.layer_budget) == 128
    print("✓ DISCO cache initialization successful")
    
    # Test cache update
    batch_size = 2
    num_heads = 8
    seq_length = 32
    head_dim = 64
    
    key_states = paddle.randn([batch_size, num_heads, seq_length, head_dim])
    value_states = paddle.randn([batch_size, num_heads, seq_length, head_dim])
    
    updated_keys, updated_values = cache.update(key_states, value_states, layer_idx=0)
    
    assert updated_keys.shape == key_states.shape
    assert updated_values.shape == value_states.shape
    print("✓ Cache update successful")
    
    # Test eviction
    for i in range(5):
        new_keys = paddle.randn([batch_size, num_heads, 16, head_dim])
        new_values = paddle.randn([batch_size, num_heads, 16, head_dim])
        updated_keys, updated_values = cache.update(new_keys, new_values, layer_idx=0)
    
    actual_size = updated_keys.shape[2]
    budget = cache.layer_budget[0]
    assert actual_size <= budget
    print(f"✓ Eviction working correctly: {actual_size} <= {budget}")


def test_eviction_manager():
    """Test eviction manager scoring."""
    print("\nTesting Eviction Manager...")
    
    manager = LayerwiseEvictionManager(
        cache_size=32,
        window_size=8,
        gamma=0.1
    )
    
    batch_size = 2
    num_heads = 8
    seq_length = 32
    head_dim = 64
    window_size = 8
    
    key_states = paddle.randn([batch_size, num_heads, seq_length, head_dim])
    value_states = paddle.randn([batch_size, num_heads, seq_length, head_dim])
    attention_weights = paddle.softmax(
        paddle.randn([batch_size, num_heads, window_size, seq_length]),
        axis=-1
    )
    
    scores = manager.compute_scores(key_states, value_states, attention_weights)
    assert scores.shape[0] == batch_size
    assert scores.shape[1] == num_heads
    print("✓ Attention-based scoring successful")
    
    distance_scores = manager.compute_distance_scores(key_states, value_states)
    assert distance_scores.shape[0] == batch_size
    assert distance_scores.shape[-1] == seq_length
    print("✓ Distance-based scoring successful")


def test_kv_fusion():
    """Test KV fusion scoring."""
    print("\nTesting KV Fusion Scoring...")
    
    batch_size = 2
    num_heads = 8
    seq_length = 32
    head_dim = 64
    
    key_states = paddle.randn([batch_size, num_heads, seq_length, head_dim])
    value_states = paddle.randn([batch_size, num_heads, seq_length, head_dim])
    
    scores = get_scores_with_kv_fusion(key_states, value_states, window_size=16)
    
    assert scores.shape[0] == batch_size
    assert scores.shape[1] == num_heads
    print("✓ KV fusion scoring successful")


def test_llama_config():
    """Test LLaMA configuration with DISCO parameters."""
    print("\nTesting LLaMA Configuration...")
    
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        # DISCO parameters
        use_disco=True,
        disco_cache_size=128,
        disco_window_size=16,
        disco_gamma=0.15,
    )
    
    assert config.use_disco == True
    assert config.disco_cache_size == 128
    assert config.disco_window_size == 16
    assert config.disco_gamma == 0.15
    print("✓ DISCO configuration successful")
    
    # Test default config (DISCO disabled)
    default_config = LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
    )
    
    assert default_config.use_disco == False
    print("✓ Default configuration (DISCO disabled) successful")


def main():
    """Run all tests."""
    print("DISCO (Dynamic Score-based Cache Optimization) Tests")
    print("=" * 60)
    
    try:
        test_disco_cache()
        test_eviction_manager()
        test_kv_fusion()
        test_llama_config()
        
        print("\n" + "=" * 60)
        print("✅ All DISCO tests passed successfully!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()