# DISCO (Dynamic Score-based Cache Optimization) for PaddleNLP

This branch implements the DISCO algorithm (originally called CAKE/Score) for efficient KV cache management in Large Language Models, specifically integrated into the LLaMA model architecture in PaddleNLP.

## Overview

DISCO is an adaptive KV cache eviction strategy that:
- Reduces memory usage to ~3.2% of the original KV cache
- Maintains model performance through intelligent token selection
- Uses layer-wise cache allocation based on attention patterns
- Employs information-theoretic scoring for eviction decisions

## Features

- **Adaptive Layer-wise Allocation**: Different layers receive different cache budgets based on their importance
- **Attention-based Scoring**: Uses attention patterns to determine token importance
- **Variance-aware Selection**: Considers both mean attention and variance (controlled by gamma parameter)
- **Efficient Implementation**: Seamlessly integrated into PaddleNLP's LLaMA implementation

## Configuration

To enable DISCO in your LLaMA model, use the following configuration parameters:

```python
from paddlenlp.transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    # Standard LLaMA parameters
    vocab_size=32000,
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    
    # DISCO parameters
    use_disco=True,                    # Enable DISCO algorithm
    disco_cache_size=1024,            # Total KV cache budget (tokens)
    disco_window_size=32,             # Recent token window for scoring
    disco_gamma=0.1,                  # Variance weight in scoring function
    disco_score_func_path=None,       # Optional: Path to learned scoring function
    disco_layer_budget=None,          # Optional: Per-layer budget allocation
)

model = LlamaForCausalLM(config)
```

## Usage Example

```python
# Initialize model with DISCO
model = LlamaForCausalLM(config)

# Use normally - DISCO will automatically manage the cache
outputs = model.generate(
    input_ids=input_ids,
    max_length=2048,
    use_cache=True,  # DISCO manages this cache
)
```

## Implementation Details

### Core Components

1. **`disco_cache.py`**: Core DISCO implementation
   - `DISCOCache`: Main cache management class
   - `LayerwiseEvictionManager`: Per-layer eviction logic
   - Scoring functions for token importance

2. **Modified `configuration.py`**: Added DISCO configuration parameters
   - `use_disco`: Enable/disable flag
   - Cache size and window parameters
   - Layer-specific configurations

3. **Modified `modeling.py`**: Integration into LLaMA attention
   - DISCO cache initialization in `LlamaModel`
   - Cache updates in `LlamaAttention.forward()`
   - Attention weight tracking for scoring

### Algorithm Flow

1. **Initialization**: Create DISCO cache with layer budgets
2. **Token Processing**: For each new token:
   - Compute attention scores
   - Update cache with new key-value pairs
   - If cache exceeds budget, evict lowest-scoring tokens
3. **Scoring**: Combine attention mean and variance
   - Score = mean_attention + γ * variance_attention
   - Apply smoothing for stability

## Performance

Based on the original research:
- **Memory**: Uses only 3.2% of original KV cache
- **Quality**: Maintains comparable model performance
- **Speed**: Efficient eviction with O(n log k) complexity

## Citation

If you use DISCO in your research, please cite:

```bibtex
@article{cake2024,
  title={Cascading and Adaptive KV Cache Eviction with Layer Preferences},
  author={[Authors]},
  year={2024}
}
```

## Future Work

- [ ] Multi-GPU optimization
- [ ] Dynamic budget reallocation
- [ ] Integration with other models (Qwen, Mistral)
- [ ] Learned scoring functions for specific tasks

## License

This implementation follows PaddleNLP's Apache 2.0 license.