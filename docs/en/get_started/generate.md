# Quick Start to Large Model Generation

PaddleNLP provides convenient Auto APIs for quick model and tokenizer loading. Here's an example of text generation using Qwen/Qwen2-0.5B model:

```python
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", dtype="float16")
input_features = tokenizer("Hello! Please introduce yourself.", return_tensors="pd")
outputs = model.generate(**input_features, max_length=128)

print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))
# ['I am an AI language model. I can answer various questions, including but not limited to: weather, news, history, culture, science, education, entertainment, etc. What would you like to know?']
```
