# 详细介绍
**介绍**：ckiplab-bert-base-chinese-ws 是一个带有token分类头的BERT模型，该模型已经在**分词任务**上进行了微调。

关于完整使用方法及其他信息，请参考 https://github.com/ckiplab/ckip-transformers 。

**模型结构**： **`BertForTokenClassification`**，带有token分类头的Bert模型。

**适用下游任务**：**分词**，该权重已经在下游`WS`任务上进行了微调，因此可直接使用。

# 使用示例

```python
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer
path = "junnyu/ckiplab-bert-base-chinese-ws"
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
id2label = {"0": "B", "1": "I"}
for t, s in zip(tokenized_text, score[0][1:-1]):
    index = paddle.argmax(s).item()
    label = id2label[str(index)]
    print(f"{label} {t} score {s[index].item()}")

# B 傅 score 0.9999865293502808
# I 達 score 0.999922513961792
# I 仁 score 0.9999332427978516
# B 今 score 0.9999370574951172
# B 將 score 0.9983423948287964
# B 執 score 0.9999731779098511
# I 行 score 0.9999544620513916
# B 安 score 0.9999713897705078
# I 樂 score 0.9999532699584961
# I 死 score 0.9998632669448853
# B ， score 0.9999871253967285
# B 卻 score 0.9999560117721558
# B 突 score 0.9999818801879883
# I 然 score 0.9999614953994751
# B 爆 score 0.9999759197235107
# I 出 score 0.9994433522224426
# B 自 score 0.9999866485595703
# I 己 score 0.9999630451202393
# B 20 score 0.9999810457229614
# B 年 score 0.9974608421325684
# B 前 score 0.8930220603942871
# B 遭 score 0.9999674558639526
# B 緯 score 0.999970555305481
# I 來 score 0.9999680519104004
# B 體 score 0.9997956156730652
# I 育 score 0.9999778270721436
# I 台 score 0.9980663657188416
# B 封 score 0.999984860420227
# I 殺 score 0.999974250793457
# B ， score 0.9999891519546509
# B 他 score 0.999988317489624
# B 不 score 0.9999889135360718
# B 懂 score 0.9997660517692566
# B 自 score 0.9999877214431763
# I 己 score 0.9999549388885498
# B 哪 score 0.9999915361404419
# I 裡 score 0.9980868101119995
# B 得 score 0.9999058246612549
# I 罪 score 0.9916028380393982
# I 到 score 0.8443355560302734
# B 電 score 0.9999363422393799
# I 視 score 0.9999769926071167
# I 台 score 0.999947190284729
# B 。 score 0.9999719858169556

```

# 权重来源

https://huggingface.co/ckiplab/bert-base-chinese-ws
这个项目提供了繁体中文版transformer模型（包含ALBERT、BERT、GPT2）及自然语言处理工具（包含分词、词性标注、命名实体识别）。
