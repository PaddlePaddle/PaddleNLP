# 详细介绍
# CKIP BERT Base Chinese
这个项目提供了繁体中文版transformer模型（包含ALBERT、BERT、GPT2）及自然语言处理工具（包含分词、词性标注、命名实体识别）。

关于完整使用方法及其他信息，请参考 https://github.com/ckiplab/ckip-transformers 。

# 使用示例

```python
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer
path = "ckiplab-bert-base-chinese-ws"
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
id2label = {0: "B", 1: "I"}
for t, s in zip(tokenized_text, score[0][1:-1]):
    index = paddle.argmax(s).item()
    label = id2label[str(index)]
    print(f"{label} {t} score {s[index].item()}")

# B 我 score 0.9999921321868896
# B 叫 score 0.9999772310256958
# B 克 score 0.9999295473098755
# I 拉 score 0.999772846698761
# I 拉 score 0.9999483823776245
# B ， score 0.9999879598617554
# B 我 score 0.9999914169311523
# B 住 score 0.9999860525131226
# B 在 score 0.6059999465942383
# B 加 score 0.9999884366989136
# I 州 score 0.9999697208404541
# B 伯 score 0.999879002571106
# I 克 score 0.9999772310256958
# I 利 score 0.9999678134918213
# B 。 score 0.9999856948852539

```

# 权重来源

https://huggingface.co/ckiplab/bert-base-chinese-ws
