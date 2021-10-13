# 详细介绍
# CKIP BERT Base Chinese
这个项目提供了繁体中文版transformer模型（包含ALBERT、BERT、GPT2）及自然语言处理工具（包含分词、词性标注、命名实体识别）。

关于完整使用方法及其他信息，请参考 https://github.com/ckiplab/ckip-transformers 。

# 使用示例

```python
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer
path = "ckiplab-bert-base-chinese-pos"
model = BertForTokenClassification.from_pretrained(path)
model.eval()
tokenizer = BertTokenizer.from_pretrained(path)
text = "我叫沃尔夫冈，我住在柏林。"
tokenized_text = tokenizer.tokenize(text)
inputs = {
    k: paddle.to_tensor(
        v, dtype="int64").unsqueeze(0)
    for k, v in tokenizer(text).items()
}
with paddle.no_grad():
    score = F.softmax(model(**inputs), axis=-1)
id2label = {
    "0": "A",
    "1": "Caa",
    "2": "Cab",
    "3": "Cba",
    "4": "Cbb",
    "5": "D",
    "6": "Da",
    "7": "Dfa",
    "8": "Dfb",
    "9": "Di",
    "10": "Dk",
    "11": "DM",
    "12": "I",
    "13": "Na",
    "14": "Nb",
    "15": "Nc",
    "16": "Ncd",
    "17": "Nd",
    "18": "Nep",
    "19": "Neqa",
    "20": "Neqb",
    "21": "Nes",
    "22": "Neu",
    "23": "Nf",
    "24": "Ng",
    "25": "Nh",
    "26": "Nv",
    "27": "P",
    "28": "T",
    "29": "VA",
    "30": "VAC",
    "31": "VB",
    "32": "VC",
    "33": "VCL",
    "34": "VD",
    "35": "VF",
    "36": "VE",
    "37": "VG",
    "38": "VH",
    "39": "VHC",
    "40": "VI",
    "41": "VJ",
    "42": "VK",
    "43": "VL",
    "44": "V_2",
    "45": "DE",
    "46": "SHI",
    "47": "FW",
    "48": "COLONCATEGORY",
    "49": "COMMACATEGORY",
    "50": "DASHCATEGORY",
    "51": "DOTCATEGORY",
    "52": "ETCCATEGORY",
    "53": "EXCLAMATIONCATEGORY",
    "54": "PARENTHESISCATEGORY",
    "55": "PAUSECATEGORY",
    "56": "PERIODCATEGORY",
    "57": "QUESTIONCATEGORY",
    "58": "SEMICOLONCATEGORY",
    "59": "SPCHANGECATEGORY"
}
for t, s in zip(tokenized_text, score[0][1:-1]):
    index = paddle.argmax(s).item()
    label = id2label[str(index)]
    print(f"{label} {t} score {s[index].item()}")

# Nh 我 score 1.0
# VG 叫 score 0.9999830722808838
# Nc 沃 score 0.9999146461486816
# Nc 尔 score 0.9999760389328003
# Nc 夫 score 0.9984875917434692
# Na 冈 score 0.8717513680458069
# COMMACATEGORY ， score 1.0
# Nh 我 score 1.0
# VCL 住 score 0.9999992847442627
# P 在 score 0.9999998807907104
# Nc 柏 score 0.9999998807907104
# Nc 林 score 0.9891127943992615
# PERIODCATEGORY 。 score 1.0

```

# 权重来源

https://huggingface.co/ckiplab/bert-base-chinese-pos
