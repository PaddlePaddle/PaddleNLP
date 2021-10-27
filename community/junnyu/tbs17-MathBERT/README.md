# 详细介绍
**介绍**：tbs17-MathBERT是一个数学领域的BERT模型，它以自监督的方式在大量英语数学语料库数据上进行了预训练。
预训练过程有两个目标：
- 掩码语言建模 (MLM)：取一个句子，模型随机掩码输入中 15% 的单词，然后通过模型运行整个掩码句子，并必须预测掩码单词。这与传统的循环神经网络 (RNN) 不同，后者通常一个接一个地看到单词，或者与 GPT 等自回归模型不同，后者在内部掩盖了未来的标记。它允许模型学习句子的双向表示。
- 下一句预测 (NSP)：模型在预训练期间连接两个掩码句子作为输入。有时它们对应于原文中相邻的句子，有时不对应。然后该模型必须预测两个句子是否相互跟随。通过这种方式，模型学习了数学语言的内部表示，然后可用于提取对下游任务有用的特征：例如，如果您有一个标记句子的数据集，您可以使用 MathBERT 生成的特征训练标准分类器模型作为输入。


**模型结构**： **`BertForPretraining`**，带有`MLM`和`NSP`任务的Bert模型。

**适用下游任务**：**数学领域相关的任务**，如：与数学领域相关的`句子级别分类`，`token级别分类`，`问答`等。

## 训练数据
- pre-k 到 HS 数学课程（engageNY、Utah Math、Illustrative Math）
- openculture.com 的大学数学书籍
- arxiv 数学论文摘要

# 使用示例

```python
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import BertForPretraining, BertTokenizer
path = "junnyu/tbs17-MathBERT"
model = BertForPretraining.from_pretrained(path)
tokenizer = BertTokenizer.from_pretrained(path)
model.eval()
text = "students apply these new understandings as they reason about and perform decimal [MASK] through the hundredths place."
tokens = ["[CLS]"]
text_list = text.split("[MASK]")
for i, t in enumerate(text_list):
    tokens.extend(tokenizer.tokenize(t))
    if i == len(text_list) - 1:
        tokens.extend(["[SEP]"])
    else:
        tokens.extend(["[MASK]"])

input_ids_list = tokenizer.convert_tokens_to_ids(tokens)
input_ids = paddle.to_tensor([input_ids_list])
with paddle.no_grad():
    pd_outputs = model(input_ids)[0][0]
pd_outputs_sentence = "paddle: "
for i, id in enumerate(input_ids_list):
    if id == tokenizer.convert_tokens_to_ids(["[MASK]"])[0]:
        scores, index = F.softmax(pd_outputs[i],
                                                        -1).topk(5)
        tokens = tokenizer.convert_ids_to_tokens(index.tolist())
        outputs = []
        for score, tk in zip(scores.tolist(), tokens):
            outputs.append(f"{tk}={score}")
        pd_outputs_sentence += "[" + "||".join(outputs) + "]" + " "
    else:
        pd_outputs_sentence += "".join(
            tokenizer.convert_ids_to_tokens(
                [id], skip_special_tokens=True)) + " "

print(pd_outputs_sentence)

paddle:  students apply these new understanding ##s as they reason about and perform decimal [numbers=0.8327996134757996||##s=0.0865364819765091||operations=0.0313422717154026||placement=0.019931407645344734||places=0.01254698634147644] through the hundred ##ths place .
```

# 权重来源

https://huggingface.co/tbs17/MathBERT
