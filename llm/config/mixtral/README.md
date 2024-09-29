# Mixtral

## 1. 模型介绍

**支持模型权重:**

| Model                                |
|--------------------------------------|
| mistralai/Mixtral-8x7B-Instruct-v0.1 |

使用方法：

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
```
