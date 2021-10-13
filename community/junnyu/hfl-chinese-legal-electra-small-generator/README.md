# 详细介绍
# Chinese ELECTRA
谷歌和斯坦福大学发布了一种名为 ELECTRA 的新预训练模型，与 BERT 及其变体相比，该模型具有非常紧凑的模型尺寸和相对具有竞争力的性能。 为进一步加快中文预训练模型的研究，HIT与科大讯飞联合实验室（HFL）发布了基于ELECTRA官方代码的中文ELECTRA模型。 与 BERT 及其变体相比，ELECTRA-small 只需 1/10 的参数就可以在几个 NLP 任务上达到相似甚至更高的分数。
这个项目依赖于官方ELECTRA代码: https://github.com/google-research/electra
该模型是small版本的generator，并且该模型专为法律领域而设计。

# 使用示例

```python
from paddlenlp.transformers import ElectraGenerator,ElectraTokenizer

path = "hfl-chinese-legal-electra-small-generator"
model = ElectraGenerator.from_pretrained(path)
tokenizer = ElectraTokenizer.from_pretrained(path)
model.eval()

text = "欢迎使用paddlenlp！"
inputs = {
    k: paddle.to_tensor(
        v, dtype="int64").unsqueeze(0)
    for k, v in tokenizer(text).items()
}

with paddle.no_grad():
    prediction_scores = pdmodel(**inputs)

print(prediction_scores.shape)

```

# 权重来源

https://huggingface.co/hfl/chinese-legal-electra-small-generator
