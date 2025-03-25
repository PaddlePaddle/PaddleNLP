# Mistral

## 1. Model Overview

**Supported Model Weights:**

| Model                                |
|--------------------------------------|
| mistralai/Mistral-7B-Instruct-v0.3   |
| mistralai/Mistral-7B-v0.1            |

Usage:

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
```
