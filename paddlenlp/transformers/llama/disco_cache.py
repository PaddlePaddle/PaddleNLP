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

"""
DISCO (Dynamic Score-based Cache Optimization) implementation for PaddleNLP.
Based on the CAKE algorithm, this provides adaptive KV cache eviction strategies.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import Tensor

# Avoid circular import issues
import logging
logger = logging.getLogger(__name__)


class DISCOCache:
    """
    DISCO Cache implementation for efficient KV cache management.
    This cache adaptively allocates memory across layers based on attention patterns.
    """

    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        window_size: int = 32,
        gamma: float = 0.1,
        score_func_path: Optional[str] = None,
        layer_budget: Optional[List[int]] = None,
    ):
        self.num_layers = num_layers
        self.cache_size = cache_size
        self.window_size = window_size
        self.gamma = gamma
        
        # Layer-wise cache allocation
        if layer_budget is None:
            # Default: equal allocation across layers
            self.layer_budget = [cache_size // num_layers] * num_layers
        else:
            self.layer_budget = layer_budget
            
        # Initialize score function
        self.score_func = None
        if score_func_path:
            try:
                coeffs = np.load(score_func_path)
                from numpy.polynomial import Polynomial
                self.score_func = Polynomial(coeffs)
            except Exception as e:
                logger.warning(f"Failed to load score function from {score_func_path}: {e}")
        
        # Cache storage
        self.key_cache = {}
        self.value_cache = {}
        self.attention_scores = {}
        self.dispersion_scores = {}
        
        # Layer-wise eviction managers
        self.layer_eviction_managers = {}
        for i in range(num_layers):
            self.layer_eviction_managers[i] = LayerwiseEvictionManager(
                cache_size=self.layer_budget[i],
                window_size=window_size,
                gamma=gamma
            )

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        attention_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Update cache with new key-value pairs."""
        
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # Concatenate new states
            self.key_cache[layer_idx] = paddle.concat([self.key_cache[layer_idx], key_states], axis=2)
            self.value_cache[layer_idx] = paddle.concat([self.value_cache[layer_idx], value_states], axis=2)
            
        # Apply eviction if needed
        if self.key_cache[layer_idx].shape[2] > self.layer_budget[layer_idx]:
            self._evict_tokens(layer_idx, attention_weights)
            
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _evict_tokens(self, layer_idx: int, attention_weights: Optional[Tensor] = None):
        """Evict tokens based on DISCO scoring strategy."""
        eviction_manager = self.layer_eviction_managers[layer_idx]
        
        # Get current cache
        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]
        
        # Compute eviction scores
        if attention_weights is not None:
            scores = eviction_manager.compute_scores(
                key_cache, value_cache, attention_weights
            )
        else:
            # Fallback to distance-based scoring
            scores = eviction_manager.compute_distance_scores(key_cache, value_cache)
            
        # Keep top-k tokens
        k = self.layer_budget[layer_idx]
        
        # Flatten scores to 1D for topk operation
        flat_scores = scores.flatten()
        _, keep_indices = paddle.topk(flat_scores, k=min(k, flat_scores.shape[0]))
        
        # Ensure indices are 1D
        keep_indices = keep_indices.flatten()
        
        # Update cache
        self.key_cache[layer_idx] = paddle.index_select(key_cache, keep_indices, axis=2)
        self.value_cache[layer_idx] = paddle.index_select(value_cache, keep_indices, axis=2)

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Get sequence length for a specific layer or maximum across all layers."""
        if layer_idx is not None:
            return self.key_cache.get(layer_idx, paddle.zeros([1, 1, 0, 1])).shape[2]
        else:
            if not self.key_cache:
                return 0
            return max(cache.shape[2] for cache in self.key_cache.values())

    def get_cache(self, layer_idx: int) -> Tuple[Tensor, Tensor]:
        """Get key-value cache for a specific layer."""
        if layer_idx not in self.key_cache:
            return None, None
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class LayerwiseEvictionManager:
    """Manages token eviction for a single layer."""
    
    def __init__(self, cache_size: int, window_size: int, gamma: float):
        self.cache_size = cache_size
        self.window_size = window_size
        self.gamma = gamma
        
    def compute_scores(
        self,
        key_states: Tensor,
        value_states: Tensor,
        attention_weights: Tensor
    ) -> Tensor:
        """Compute eviction scores based on attention patterns."""
        
        # Extract attention scores for recent window
        window_attn = attention_weights[..., -self.window_size:, :]
        
        # Compute mean and variance of attention
        attn_mean = window_attn.mean(axis=-2)
        attn_var = window_attn.var(axis=-2)
        
        # Combined score: mean + gamma * variance
        scores = attn_mean + self.gamma * attn_var
        
        # Apply smoothing
        if scores.shape[-1] > 5:
            # Simple moving average
            kernel_size = min(5, scores.shape[-1])
            # Reshape for avg_pool1d: [batch*heads, 1, seq_len]
            original_shape = scores.shape
            scores_reshaped = scores.reshape([-1, 1, scores.shape[-1]])
            scores = F.avg_pool1d(
                scores_reshaped,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            )
            # Reshape back
            scores = scores.squeeze(1).reshape(original_shape)
            
        return scores
        
    def compute_distance_scores(
        self,
        key_states: Tensor,
        value_states: Tensor
    ) -> Tensor:
        """Compute scores based on key-value distances (fallback method)."""
        
        # Compute L2 norm of keys and values
        key_norm = paddle.norm(key_states, p=2, axis=-1)
        value_norm = paddle.norm(value_states, p=2, axis=-1)
        
        # Combined score
        scores = key_norm + value_norm
        
        # Average across heads
        scores = scores.mean(axis=1)
        
        return scores


def get_scores_with_kv_fusion(
    key_states: Tensor,
    value_states: Tensor,
    window_size: int = 32
) -> Tensor:
    """
    Compute fusion scores based on key-value similarity patterns.
    This is used for dispersion-based scoring in DISCO.
    """
    
    # Extract recent window
    if key_states.shape[2] > window_size:
        recent_keys = key_states[:, :, -window_size:]
        recent_values = value_states[:, :, -window_size:]
    else:
        recent_keys = key_states
        recent_values = value_states
    
    # Compute key-value similarity
    key_norm = F.normalize(recent_keys, p=2, axis=-1)
    value_norm = F.normalize(recent_values, p=2, axis=-1)
    
    # Cosine similarity between keys and values
    similarity = paddle.sum(key_norm * value_norm, axis=-1)
    
    # Aggregate across window
    scores = similarity.mean(axis=-1)
    
    return scores