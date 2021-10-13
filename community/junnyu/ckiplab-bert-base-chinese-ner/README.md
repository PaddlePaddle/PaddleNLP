# 详细介绍
# CKIP BERT Base Chinese
这个项目提供了繁体中文版transformer模型（包含ALBERT、BERT、GPT2）及自然语言处理工具（包含分词、词性标注、命名实体识别）。

关于完整使用方法及其他信息，请参考 https://github.com/ckiplab/ckip-transformers 。

# 使用示例

```python
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer
path = "ckiplab-bert-base-chinese-ner"
model = BertForTokenClassification.from_pretrained(path)
model.eval()
tokenizer = BertTokenizer.from_pretrained(path)
text = "我叫克拉拉，我住在加州伯克利。"
tokenized_text = tokenizer.tokenize(text)
inputs = {
    k: paddle.to_tensor(
        v, dtype="int64").unsqueeze(0)
    for k, v in tokenizer(text).items()
}
with paddle.no_grad():
    score = F.softmax(model(**inputs), axis=-1)
id2label = {
    "0": "O",
    "1": "B-CARDINAL",
    "2": "B-DATE",
    "3": "B-EVENT",
    "4": "B-FAC",
    "5": "B-GPE",
    "6": "B-LANGUAGE",
    "7": "B-LAW",
    "8": "B-LOC",
    "9": "B-MONEY",
    "10": "B-NORP",
    "11": "B-ORDINAL",
    "12": "B-ORG",
    "13": "B-PERCENT",
    "14": "B-PERSON",
    "15": "B-PRODUCT",
    "16": "B-QUANTITY",
    "17": "B-TIME",
    "18": "B-WORK_OF_ART",
    "19": "I-CARDINAL",
    "20": "I-DATE",
    "21": "I-EVENT",
    "22": "I-FAC",
    "23": "I-GPE",
    "24": "I-LANGUAGE",
    "25": "I-LAW",
    "26": "I-LOC",
    "27": "I-MONEY",
    "28": "I-NORP",
    "29": "I-ORDINAL",
    "30": "I-ORG",
    "31": "I-PERCENT",
    "32": "I-PERSON",
    "33": "I-PRODUCT",
    "34": "I-QUANTITY",
    "35": "I-TIME",
    "36": "I-WORK_OF_ART",
    "37": "E-CARDINAL",
    "38": "E-DATE",
    "39": "E-EVENT",
    "40": "E-FAC",
    "41": "E-GPE",
    "42": "E-LANGUAGE",
    "43": "E-LAW",
    "44": "E-LOC",
    "45": "E-MONEY",
    "46": "E-NORP",
    "47": "E-ORDINAL",
    "48": "E-ORG",
    "49": "E-PERCENT",
    "50": "E-PERSON",
    "51": "E-PRODUCT",
    "52": "E-QUANTITY",
    "53": "E-TIME",
    "54": "E-WORK_OF_ART",
    "55": "S-CARDINAL",
    "56": "S-DATE",
    "57": "S-EVENT",
    "58": "S-FAC",
    "59": "S-GPE",
    "60": "S-LANGUAGE",
    "61": "S-LAW",
    "62": "S-LOC",
    "63": "S-MONEY",
    "64": "S-NORP",
    "65": "S-ORDINAL",
    "66": "S-ORG",
    "67": "S-PERCENT",
    "68": "S-PERSON",
    "69": "S-PRODUCT",
    "70": "S-QUANTITY",
    "71": "S-TIME",
    "72": "S-WORK_OF_ART"
}
for t, s in zip(tokenized_text, score[0][1:-1]):
    index = paddle.argmax(s).item()
    label = id2label[str(index)]
    print(f"{label} {t} score {s[index].item()}")

# O 我 score 0.9999998807907104
# O 叫 score 1.0
# B-PERSON 克 score 0.9999995231628418
# I-PERSON 拉 score 0.9999992847442627
# E-PERSON 拉 score 0.9999995231628418
# O ， score 1.0
# O 我 score 1.0
# O 住 score 1.0
# O 在 score 1.0
# B-GPE 加 score 0.9999984502792358
# I-GPE 州 score 0.9999964237213135
# I-GPE 伯 score 0.9999923706054688
# I-GPE 克 score 0.999998927116394
# E-GPE 利 score 0.9999991655349731
# O 。 score 0.9999994039535522

```

# 权重来源

https://huggingface.co/ckiplab/bert-base-chinese-ner
