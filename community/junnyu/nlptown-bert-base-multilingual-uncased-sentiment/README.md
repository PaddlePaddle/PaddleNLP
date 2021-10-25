# 详细介绍
**介绍**：nlptown-bert-base-multilingual-uncased-sentiment是一个带有序列分类头的多语言BERT模型，该模型可用于对英语、荷兰语、德语、法语、西班牙语和意大利语这六种语言的商品评论进行情感分析。其中评论的情感标签为1-5之间的星级。

**模型结构**： **`BertForSequenceClassification`**，带有序列分类头的Bert模型。

**适用下游任务**：**情感分类**，该权重已经在下游`Sentiment classification`任务上进行了微调，因此可直接使用。

## 训练数据
以下是用于微调模型的商品评论数量：
| Language | Number of reviews |
| -------- | ----------------- |
| English  | 150k           |
| Dutch    | 80k            |
| German   | 137k           |
| French   | 140k           |
| Italian  | 72k            |
| Spanish  | 50k            |

## 指标
微调后的模型在每种语言的 5,000 条商品评论中获得了以下准确率：
- Accuracy (exact) 完全匹配。
- Accuracy (off-by-1) 是模型预测的评分等级与人工给出的评分等级差值小于等于 1 所占的百分比。

| Language | Accuracy (exact) | Accuracy (off-by-1) |
| -------- | ---------------- | ------------------- |
| English  | 67%              | 95%                 |
| Dutch    | 57%              | 93%                 |
| German   | 61%              | 94%                 |
| French   | 59%              | 94%                 |
| Italian  | 59%              | 95%                 |
| Spanish  | 58%              | 95%                 |

## 联系方式
对于类似模型的问题、反馈和/或请求，请联系 [NLP Town](https://www.nlp.town)。


# 使用示例

```python
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
model = BertForSequenceClassification.from_pretrained(path)
model.eval()
path = "junnyu/nlptown-bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(path)
text = "I like you. I love you"
inputs = {
    k: paddle.to_tensor(
        v, dtype="int64").unsqueeze(0)
    for k, v in tokenizer(text).items()
}
with paddle.no_grad():
    score = F.softmax(model(**inputs), axis=-1)
id2label = {
    0: "1 star",
    1: "2 stars",
    2: "3 stars",
    3: "4 stars",
    4: "5 stars"
}
for i, s in enumerate(score[0].tolist()):
    label = id2label[i]
    print(f"{label} score {s}")

# 1 star score 0.0021950288210064173
# 2 stars score 0.0022533712908625603
# 3 stars score 0.015475980937480927
# 4 stars score 0.1935628354549408
# 5 stars score 0.7865128517150879

```

# 权重来源

https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
