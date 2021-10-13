# 详细介绍
# DistilGPT2
DistilGPT2 英语语言模型使用 OpenWebTextCorpus（OpenAI 的 WebText 数据集），使用 GPT2 的最小版本的进行了预训练。 该模型有 6 层、768 个维度和 12 个头，总计 82M 参数（相比之下 GPT2 的参数为 124M）。 平均而言，DistilGPT2 比 GPT2 快两倍。

在 WikiText-103 基准测试中，GPT2 在测试集上的困惑度为 16.3，而 DistilGPT2 的困惑度为 21.1（在训练集上进行微调后）。


# 使用示例

```python
import paddle
from paddlenlp.transformers import GPTLMHeadModel, GPTTokenizer

path = "distilgpt2"
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

https://huggingface.co/distilgpt2
