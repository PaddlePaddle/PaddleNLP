# 详细介绍
# microsoft-DialoGPT-small
最先进的大规模预训练响应生成模型 (DialoGPT)
DialoGPT 是一种用于多轮对话的 SOTA 大规模预训练对话响应生成模型。 人类评估结果表明，DialoGPT 生成的响应与单轮对话图灵测试下的人类响应质量相当。 该模型是在来自 Reddit 讨论线程的 147M 多轮对话上训练的。


# 使用示例

```python
import paddle
from paddlenlp.transformers import GPTLMHeadModel, GPTTokenizer

path = "microsoft-DialoGPT-small"
model = GPTLMHeadModel.from_pretrained(path)
model.eval()
tokenizer = GPTTokenizer.from_pretrained(path)
text = "Welcome to paddlenlp!"
inputs = {
    k: paddle.to_tensor(
        v, dtype="int64").unsqueeze(0)
    for k, v in tokenizer(
        text, return_token_type_ids=False).items()
}
with paddle.no_grad():
    logits = model(**inputs)

print(logits.shape)
```

# 权重来源

https://huggingface.co/microsoft/DialoGPT-small
