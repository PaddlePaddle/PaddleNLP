# Mixtral

## 1. Model Overview

**Supported Model Weights:**

| Model                                |
|--------------------------------------|
| mistralai/Mixtral-8x7B-Instruct-v0.1 |

Usage:

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
```
