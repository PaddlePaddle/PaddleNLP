# 详细介绍
**介绍**：ckiplab-bert-base-chinese-pos 是一个带有token分类头的BERT模型，该模型已经在**词性标注任务**上进行了微调。

关于完整使用方法及其他信息，请参考 https://github.com/ckiplab/ckip-transformers 。

**模型结构**： **`BertForTokenClassification`**，带有token分类头的Bert模型。

**适用下游任务**：**词性标注**，该权重已经在下游`POS`任务上进行了微调，因此可直接使用。

# 使用示例

```python
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer
path = "junnyu/ckiplab-bert-base-chinese-pos"
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

# Nb 傅 score 0.9999998807907104
# Nb 達 score 0.9700667858123779
# Na 仁 score 0.9985846281051636
# Nd 今 score 0.9999947547912598
# D 將 score 0.9999957084655762
# VC 執 score 0.9999998807907104
# VC 行 score 0.9951109290122986
# Na 安 score 0.9999996423721313
# Na 樂 score 0.9999638795852661
# VH 死 score 0.9813857674598694
# COMMACATEGORY ， score 1.0
# D 卻 score 1.0
# D 突 score 1.0
# Cbb 然 score 0.9989008903503418
# VJ 爆 score 0.9999979734420776
# VC 出 score 0.9965670108795166
# Nh 自 score 1.0
# Nh 己 score 1.0
# Neu 20 score 0.9999995231628418
# Nf 年 score 0.9125530123710632
# Ng 前 score 0.9999992847442627
# P 遭 score 1.0
# Nb 緯 score 0.9999996423721313
# VA 來 score 0.9322434663772583
# Na 體 score 0.9846553802490234
# Nc 育 score 0.729569137096405
# Nc 台 score 0.9999841451644897
# VC 封 score 0.9999997615814209
# VC 殺 score 0.9999991655349731
# COMMACATEGORY ， score 1.0
# Nh 他 score 0.9999996423721313
# D 不 score 1.0
# VK 懂 score 1.0
# Nh 自 score 1.0
# Nh 己 score 0.9999978542327881
# Ncd 哪 score 0.9856181740760803
# Ncd 裡 score 0.9999995231628418
# VC 得 score 0.9999988079071045
# Na 罪 score 0.9994786381721497
# VCL 到 score 0.8332439661026001
# Nc 電 score 1.0
# Nc 視 score 0.9999986886978149
# Nc 台 score 0.9973978996276855
# PERIODCATEGORY 。 score 1.0

```

# 权重来源

https://huggingface.co/ckiplab/bert-base-chinese-pos
这个项目提供了繁体中文版transformer模型（包含ALBERT、BERT、GPT2）及自然语言处理工具（包含分词、词性标注、命名实体识别）。
