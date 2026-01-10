#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test DISCO cache eviction fix"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paddle
import numpy as np

# Direct import to avoid circular dependency
exec(open('paddlenlp/transformers/llama/disco_cache.py').read())

def test_eviction():
    """Test the eviction mechanism fix."""
    print("Testing DISCO eviction fix...")
    
    # Create cache with small budget to trigger eviction
    cache = DISCOCache(
        num_layers=4,
        cache_size=64,  # Total budget
        window_size=8,
        gamma=0.1,
    )
    
    # Set smaller per-layer budget to force eviction
    cache.layer_budget = [16, 16, 16, 16]  # 16 tokens per layer
    
    batch_size = 2
    num_heads = 8
    head_dim = 64
    layer_idx = 0
    
    # First update with 32 tokens (exceeds budget of 16)
    seq_length = 32
    key_states = paddle.randn([batch_size, num_heads, seq_length, head_dim])
    value_states = paddle.randn([batch_size, num_heads, seq_length, head_dim])
    
    print(f"  Initial update with {seq_length} tokens (budget: {cache.layer_budget[layer_idx]})")
    updated_keys, updated_values = cache.update(key_states, value_states, layer_idx)
    
    # Check that eviction happened
    actual_seq_len = updated_keys.shape[2]
    expected_max = cache.layer_budget[layer_idx]
    
    print(f"  After eviction: {actual_seq_len} tokens (expected <= {expected_max})")
    
    if actual_seq_len <= expected_max:
        print("  ✅ Eviction working correctly!")
    else:
        print(f"  ❌ Eviction failed: {actual_seq_len} > {expected_max}")
        return False
    
    # Add more tokens to test continuous eviction
    new_tokens = 10
    new_keys = paddle.randn([batch_size, num_heads, new_tokens, head_dim])
    new_values = paddle.randn([batch_size, num_heads, new_tokens, head_dim])
    
    print(f"  Adding {new_tokens} more tokens...")
    updated_keys, updated_values = cache.update(new_keys, new_values, layer_idx)
    
    actual_seq_len = updated_keys.shape[2]
    print(f"  After second update: {actual_seq_len} tokens (expected <= {expected_max})")
    
    if actual_seq_len <= expected_max:
        print("  ✅ Continuous eviction working correctly!")
        return True
    else:
        print(f"  ❌ Continuous eviction failed: {actual_seq_len} > {expected_max}")
        return False

if __name__ == "__main__":
    print("DISCO Eviction Fix Test")
    print("=" * 60)
    
    try:
        success = test_eviction()
        print("\n" + "=" * 60)
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Tests failed!")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()