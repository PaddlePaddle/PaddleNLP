# 详细介绍
**介绍**：该模型是small版本的Electra generator模型，该模型在法律领域数据上进行了预训练。

**模型结构**： **`ElectraGenerator`**，带有生成器的中文Electra模型。

**适用下游任务**：**法律领域的下游任务**，如：法律领域的句子级别分类，法律领域的token级别分类，法律领域的抽取式问答等任务。
（注：生成器的效果不好，通常使用判别器进行下游任务微调）


# 使用示例

```python
import paddle
from paddlenlp.transformers import ElectraGenerator, ElectraTokenizer

text = "本院经审查认为，本案[MASK]民间借贷纠纷申请再审案件，应重点审查二审判决是否存在错误的情形。"
path = "junnyu/hfl-chinese-legal-electra-small-generator"
model = ElectraGenerator.from_pretrained(path)
model.eval()
tokenizer = ElectraTokenizer.from_pretrained(path)

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
        scores, index = paddle.nn.functional.softmax(pd_outputs[i],
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
# paddle:  本 院 经 审 查 认 为 ， 本 案 [因=0.27444931864738464||经=0.18613006174564362||系=0.09408623725175858||的=0.07536833733320236||就=0.033634234219789505] 民 间 借 贷 纠 纷 申 请 再 审 案 件 ， 应 重 点 审 查 二 审 判 决 是 否 存 在 错 误 的 情 形 。
```

# 权重来源

https://huggingface.co/hfl/chinese-legal-electra-small-generator
谷歌和斯坦福大学发布了一种名为 ELECTRA 的新预训练模型，与 BERT 及其变体相比，该模型具有非常紧凑的模型尺寸和相对具有竞争力的性能。 为进一步加快中文预训练模型的研究，HIT与科大讯飞联合实验室（HFL）发布了基于ELECTRA官方代码的中文ELECTRA模型。 与 BERT 及其变体相比，ELECTRA-small 只需 1/10 的参数就可以在几个 NLP 任务上达到相似甚至更高的分数。
这个项目依赖于官方ELECTRA代码: https://github.com/google-research/electra
