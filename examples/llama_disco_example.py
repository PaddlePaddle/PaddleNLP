#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script demonstrating DISCO algorithm usage with LLaMA model in PaddleNLP.

DISCO (Dynamic Score-based Cache Optimization) provides adaptive KV cache management
for efficient LLM inference.
"""

import paddle
from paddlenlp.transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

def run_disco_example():
    """Run a simple example with DISCO-enabled LLaMA model."""
    
    # Configuration with DISCO enabled
    config = LlamaConfig(
        # Model parameters
        vocab_size=32000,
        hidden_size=1024,  # Smaller size for example
        num_hidden_layers=12,
        num_attention_heads=16,
        intermediate_size=2816,
        
        # DISCO parameters
        use_disco=True,
        disco_cache_size=512,  # Total cache budget
        disco_window_size=32,  # Recent token window
        disco_gamma=0.1,  # Variance weight factor
        disco_score_func_path=None,  # Optional: path to learned scoring function
        disco_layer_budget=None,  # Optional: per-layer budget allocation
    )
    
    # Initialize model with DISCO
    model = LlamaForCausalLM(config)
    model.eval()
    
    # Mock tokenizer for demonstration
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 1
            
        def __call__(self, text, return_tensors="pd"):
            # Simple mock tokenization
            tokens = list(range(2, 102))  # 100 tokens
            return {"input_ids": paddle.to_tensor([tokens])}
            
        def decode(self, token_ids):
            return f"Generated {len(token_ids)} tokens"
    
    tokenizer = MockTokenizer()
    
    # Example input
    input_text = "This is a long document that will test the DISCO cache management..."
    inputs = tokenizer(input_text, return_tensors="pd")
    
    print("Running LLaMA with DISCO cache management...")
    print(f"- Cache size: {config.disco_cache_size}")
    print(f"- Window size: {config.disco_window_size}")
    print(f"- Gamma: {config.disco_gamma}")
    
    # Generate with DISCO-managed cache
    with paddle.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_length=200,
            num_beams=1,
            do_sample=False,
            use_cache=True,  # DISCO will manage the cache
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0])
    print(f"\nGenerated: {generated_text}")
    
    # Check cache usage
    if hasattr(model.llama, 'disco_cache') and model.llama.disco_cache is not None:
        disco_cache = model.llama.disco_cache
        print("\nDISCO Cache Statistics:")
        for layer_idx in range(config.num_hidden_layers):
            seq_len = disco_cache.get_seq_length(layer_idx)
            budget = disco_cache.layer_budget[layer_idx]
            print(f"  Layer {layer_idx}: {seq_len}/{budget} tokens cached")


def compare_with_standard_cache():
    """Compare DISCO with standard KV cache."""
    
    print("\n" + "="*60)
    print("Comparing DISCO vs Standard Cache")
    print("="*60)
    
    # Standard configuration
    standard_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=16,
        intermediate_size=2816,
        use_disco=False,  # Standard cache
    )
    
    # DISCO configuration
    disco_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=16,
        intermediate_size=2816,
        use_disco=True,
        disco_cache_size=256,  # Smaller cache for comparison
        disco_window_size=16,
        disco_gamma=0.15,
    )
    
    # Initialize models
    standard_model = LlamaForCausalLM(standard_config)
    disco_model = LlamaForCausalLM(disco_config)
    
    print("\nStandard Cache: Full KV storage")
    print(f"DISCO Cache: Adaptive storage with {disco_config.disco_cache_size} token budget")
    
    # You can add actual performance comparisons here
    print("\nDISCO advantages:")
    print("- Reduced memory usage (3.2% of original in paper)")
    print("- Adaptive layer-wise allocation")
    print("- Maintains model quality with intelligent eviction")


if __name__ == "__main__":
    print("DISCO (Dynamic Score-based Cache Optimization) Example")
    print("======================================================\n")
    
    # Run basic example
    run_disco_example()
    
    # Compare with standard cache
    compare_with_standard_cache()
    
    print("\n✅ DISCO example completed successfully!")