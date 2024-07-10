# Mistral

## 1. 模型介绍

**支持模型权重:**

| Model                                |
|--------------------------------------|
| mistralai/Mistral-7B-Instruct-v0.3   |
| mistralai/Mistral-7B-v0.1            |

使用方法：

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
```
