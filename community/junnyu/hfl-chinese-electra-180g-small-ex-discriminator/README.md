# 详细介绍
**介绍**：该模型是small版本的Electra discriminator模型，并且在180G的中文数据上进行训练。

**模型结构**： **`ElectraDiscriminator`**，带有判别器的中文Electra模型。

**适用下游任务**：**通用下游任务**，如：句子级别分类，token级别分类，抽取式问答等任务。

# 使用示例

```python
import paddle
from paddlenlp.transformers import ElectraDiscriminator,ElectraTokenizer

path = "junnyu/hfl-chinese-electra-180g-small-ex-discriminator"
model = ElectraDiscriminator.from_pretrained(path)
tokenizer = ElectraTokenizer.from_pretrained(path)
model.eval()

text = "欢迎使用paddlenlp！"
inputs = {
    k: paddle.to_tensor(
        v, dtype="int64").unsqueeze(0)
    for k, v in tokenizer(text).items()
}

with paddle.no_grad():
    logits = model(**inputs)

print(logits.shape)

```

# 权重来源

https://huggingface.co/hfl/chinese-electra-180g-small-ex-discriminator
谷歌和斯坦福大学发布了一种名为 ELECTRA 的新预训练模型，与 BERT 及其变体相比，该模型具有非常紧凑的模型尺寸和相对具有竞争力的性能。 为进一步加快中文预训练模型的研究，HIT与科大讯飞联合实验室（HFL）发布了基于ELECTRA官方代码的中文ELECTRA模型。 与 BERT 及其变体相比，ELECTRA-small 只需 1/10 的参数就可以在几个 NLP 任务上达到相似甚至更高的分数。
