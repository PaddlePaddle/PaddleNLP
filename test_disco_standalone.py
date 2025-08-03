#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone test for DISCO implementation without full PaddleNLP imports."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paddle
import numpy as np


def test_disco_core():
    """Test core DISCO functionality by importing modules directly."""
    print("Testing DISCO Core Implementation...")
    
    # Import DISCO cache directly
    exec(open('paddlenlp/transformers/llama/disco_cache.py').read(), globals())
    
    # Test DISCOCache
    print("\n1. Testing DISCOCache initialization...")
    cache = DISCOCache(
        num_layers=4,
        cache_size=128,
        window_size=16,
        gamma=0.1,
    )
    print(f"   - Number of layers: {cache.num_layers}")
    print(f"   - Total cache size: {cache.cache_size}")
    print(f"   - Layer budgets: {cache.layer_budget}")
    assert len(cache.layer_budget) == 4
    assert sum(cache.layer_budget) == 128
    print("   ✓ DISCOCache initialization successful")
    
    # Test cache operations
    print("\n2. Testing cache operations...")
    batch_size = 2
    num_heads = 8
    seq_length = 32
    head_dim = 64
    
    key_states = paddle.randn([batch_size, num_heads, seq_length, head_dim])
    value_states = paddle.randn([batch_size, num_heads, seq_length, head_dim])
    
    # Update cache
    updated_keys, updated_values = cache.update(key_states, value_states, layer_idx=0)
    print(f"   - Input shape: {key_states.shape}")
    print(f"   - Updated shape: {updated_keys.shape}")
    assert updated_keys.shape == key_states.shape
    print("   ✓ Cache update successful")
    
    # Test eviction by adding more data
    print("\n3. Testing eviction mechanism...")
    initial_seq_len = cache.get_seq_length(0)
    print(f"   - Initial sequence length: {initial_seq_len}")
    
    # Add more sequences to trigger eviction
    for i in range(5):
        new_keys = paddle.randn([batch_size, num_heads, 16, head_dim])
        new_values = paddle.randn([batch_size, num_heads, 16, head_dim])
        cache.update(new_keys, new_values, layer_idx=0)
    
    final_seq_len = cache.get_seq_length(0)
    budget = cache.layer_budget[0]
    print(f"   - Final sequence length: {final_seq_len}")
    print(f"   - Layer 0 budget: {budget}")
    assert final_seq_len <= budget
    print("   ✓ Eviction mechanism working correctly")
    
    # Test LayerwiseEvictionManager
    print("\n4. Testing LayerwiseEvictionManager...")
    manager = LayerwiseEvictionManager(
        cache_size=32,
        window_size=8,
        gamma=0.1
    )
    
    # Test distance-based scoring
    scores = manager.compute_distance_scores(key_states, value_states)
    print(f"   - Distance scores shape: {scores.shape}")
    assert scores.shape[0] == batch_size
    assert scores.shape[-1] == seq_length
    print("   ✓ Distance-based scoring successful")
    
    # Test with attention weights
    attention_weights = paddle.softmax(
        paddle.randn([batch_size, num_heads, 8, seq_length]),
        axis=-1
    )
    attn_scores = manager.compute_scores(key_states, value_states, attention_weights)
    print(f"   - Attention scores shape: {attn_scores.shape}")
    print("   ✓ Attention-based scoring successful")
    
    # Test KV fusion scoring
    print("\n5. Testing KV fusion scoring...")
    fusion_scores = get_scores_with_kv_fusion(key_states, value_states, window_size=16)
    print(f"   - Fusion scores shape: {fusion_scores.shape}")
    assert fusion_scores.shape[0] == batch_size
    assert fusion_scores.shape[1] == num_heads
    print("   ✓ KV fusion scoring successful")
    
    print("\n" + "="*60)
    print("✅ All DISCO core tests passed successfully!")
    print("="*60)


def test_config_integration():
    """Test configuration integration."""
    print("\n\nTesting Configuration Integration...")
    
    # Read and check configuration file
    config_file = 'paddlenlp/transformers/llama/configuration.py'
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Check if DISCO parameters are present
    disco_params = [
        'use_disco',
        'disco_window_size',
        'disco_gamma',
        'disco_cache_size',
        'disco_score_func_path',
        'disco_layer_budget'
    ]
    
    print("\nChecking DISCO parameters in configuration:")
    for param in disco_params:
        if param in content:
            print(f"   ✓ {param} found in configuration")
        else:
            print(f"   ✗ {param} NOT found in configuration")
            
    # Check modeling file
    print("\nChecking modeling.py modifications:")
    modeling_file = 'paddlenlp/transformers/llama/modeling.py'
    with open(modeling_file, 'r') as f:
        modeling_content = f.read()
    
    checks = [
        ('DISCO import', 'from .disco_cache import DISCOCache'),
        ('DISCO in attention', 'self.use_disco ='),
        ('DISCO cache init', 'self.disco_cache ='),
        ('DISCO in forward', 'if self.use_disco and use_cache:'),
    ]
    
    for check_name, check_str in checks:
        if check_str in modeling_content:
            print(f"   ✓ {check_name} found")
        else:
            print(f"   ✗ {check_name} NOT found")
    
    print("\n✅ Configuration integration check completed!")


def main():
    """Run all standalone tests."""
    print("DISCO Standalone Test Suite")
    print("=" * 80)
    
    try:
        test_disco_core()
        test_config_integration()
        
        print("\n" + "=" * 80)
        print("🎉 All DISCO standalone tests completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()