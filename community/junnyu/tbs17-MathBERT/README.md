# 详细介绍
## MathBERT
MathBERT 是一个 Transformer 模型，它以自监督的方式在大量英语数学语料库数据上进行了预训练。
预训练过程有两个目标：
- 掩码语言建模 (MLM)：取一个句子，模型随机掩码输入中 15% 的单词，然后通过模型运行整个掩码句子，并必须预测掩码单词。这与传统的循环神经网络 (RNN) 不同，后者通常一个接一个地看到单词，或者与 GPT 等自回归模型不同，后者在内部掩盖了未来的标记。它允许模型学习句子的双向表示。
- 下一句预测 (NSP)：模型在预训练期间连接两个掩码句子作为输入。有时它们对应于原文中相邻的句子，有时不对应。然后该模型必须预测两个句子是否相互跟随。通过这种方式，模型学习了数学语言的内部表示，然后可用于提取对下游任务有用的特征：例如，如果您有一个标记句子的数据集，您可以使用 MathBERT 生成的特征训练标准分类器模型作为输入。


## 训练数据
- pre-k 到 HS 数学课程（engageNY、Utah Math、Illustrative Math）
- openculture.com 的大学数学书籍
- arxiv 数学论文摘要

# 使用示例

```python
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import BertForMaskedLM, BertTokenizer
path = "tbs17-MathBERT"
model = BertForMaskedLM.from_pretrained(path)
tokenizer = BertTokenizer.from_pretrained(path)
model.eval()
text = "The man worked as a [MASK]."
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
    pd_outputs = model(input_ids)[0]
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

# paddle: the man worked as a [book=0.6469274759292603||guide=0.07073356211185455||text=0.031362663954496384||man=0.023064589127898216||distance=0.02054688334465027] .  

```

# 权重来源

https://huggingface.co/tbs17/MathBERT
