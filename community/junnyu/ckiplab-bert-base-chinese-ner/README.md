# 详细介绍
**介绍**：ckiplab-bert-base-chinese-ner 是一个带有token分类头的BERT模型，该模型已经在**命名实体识别任务**上进行了微调。

关于完整使用方法及其他信息，请参考 https://github.com/ckiplab/ckip-transformers 。

**模型结构**： **`BertForTokenClassification`**，带有token分类头的Bert模型。

**适用下游任务**：**命名实体识别**，该权重已经在下游`NER`任务上进行了微调，因此可直接使用。

# 使用示例

```python
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer
path = "junnyu/ckiplab-bert-base-chinese-ner"
model = BertForTokenClassification.from_pretrained(path)
model.eval()
tokenizer = BertTokenizer.from_pretrained(path)
text = "傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。"
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

# B-PERSON 傅 score 0.9999995231628418
# I-PERSON 達 score 0.9999994039535522
# E-PERSON 仁 score 0.9999995231628418
# B-DATE 今 score 0.9991734623908997
# O 將 score 0.9852147698402405
# O 執 score 1.0
# O 行 score 0.9999998807907104
# O 安 score 0.9999996423721313
# O 樂 score 0.9999997615814209
# O 死 score 0.9999997615814209
# O ， score 1.0
# O 卻 score 1.0
# O 突 score 1.0
# O 然 score 1.0
# O 爆 score 1.0
# O 出 score 1.0
# O 自 score 1.0
# O 己 score 1.0
# B-DATE 20 score 0.9999992847442627
# E-DATE 年 score 0.9999892711639404
# O 前 score 0.9999995231628418
# O 遭 score 1.0
# B-ORG 緯 score 0.9999990463256836
# I-ORG 來 score 0.9999986886978149
# I-ORG 體 score 0.999998927116394
# I-ORG 育 score 0.9999985694885254
# E-ORG 台 score 0.999998927116394
# O 封 score 1.0
# O 殺 score 1.0
# O ， score 1.0
# O 他 score 1.0
# O 不 score 1.0
# O 懂 score 1.0
# O 自 score 1.0
# O 己 score 1.0
# O 哪 score 1.0
# O 裡 score 1.0
# O 得 score 1.0
# O 罪 score 1.0
# O 到 score 1.0
# O 電 score 1.0
# O 視 score 1.0
# O 台 score 1.0
# O 。 score 0.9999960660934448

```

# 权重来源

https://huggingface.co/ckiplab/bert-base-chinese-ner
这个项目提供了繁体中文版transformer模型（包含ALBERT、BERT、GPT2）及自然语言处理工具（包含分词、词性标注、命名实体识别）。
