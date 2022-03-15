# 介绍
SciBERT是基于BERT，面向学术文本的预训练语言模型。
# 训练数据
SciBERT的训练数据来自[semanticscholar.org](https://www.semanticscholar.org/)语料库的论文数据。该语料库包含114万篇论文。本模型用语料库中的论文全文进行训练，而不仅仅是摘要。
# 模型结构
bert-base-cased
# 适合下游任务
面向学术文本的文本分类、实体识别、关系抽取、阅读理解等。
# 使用本模型
```
from paddlenlp import BertForMaskedLM, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('yaweisun/scibert-scivocab-uncased')
model = BertForMaskedLM.from_pretrained('yaweisun/scibert-scivocab-uncased')
```
# 权重来源
[https://github.com/allenai/scibert](https://github.com/allenai/scibert)
