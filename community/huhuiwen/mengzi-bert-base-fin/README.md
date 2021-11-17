# 详细介绍
**介绍**：基于mengzi-bert-base，继续在20G金融语料上训练，
包括金融新闻和研究报告。预训练目标有三个，掩码语言建模 (MLM)，句子顺序预测(SOP)
和词性标注(POS)。

**模型结构**： **`BertForMakedLM`**。

**适用下游任务**：**金融领域的自然语言理解类任务**，如：文本分类、实体识别、关系抽取、阅读理解等。

**训练数据**：20G金融语料，包括金融新闻和研究报告。

#使用示例：
```python
from paddlenlp import BertModel, BertTokenzier
tokenizer = BertTokenzier.from_pretrained('mengzi-bert-base-fin')
model = BertForMakedLM.from_pretrained('mengzi-bert-base-fin')

text  = "股市指某支新发行股票在定价和配置后的交易市场。"
inputs = tokenizer(text)
inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}
output = model(**inputs)
print(outputs)
```

# 权重来源
https://huggingface.co/Langboat/mengzi-bert-base-fin
