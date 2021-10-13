# 详细介绍
# Chinese Poem GPT2 Model
该模型用于生成中国古诗词。

# 训练数据
训练数据包含 80 万首中国古诗词，由 chinese-poetry 和 Poetry 项目收集。

# 使用示例

```python
import paddle
from paddlenlp.transformers import GPTLMHeadModel, BertTokenizer

path = "uer-gpt2-chinese-poem"
model = GPTLMHeadModel.from_pretrained(path)
model.eval()
tokenizer = BertTokenizer.from_pretrained(path)
text = "欢迎使用paddlenlp！"
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

https://huggingface.co/uer/gpt2-chinese-poem
