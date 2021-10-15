# 模型介绍
tiny-distilbert-base-uncased在sst-2上finetune后的模型
# 模型来源
https://huggingface.co/sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english
# 模型使用
这个模型的命名方式用的是bert的前缀，转化成paddle时手动改成了distilbert。由于他的权重里有pooler而paddlenlpistilbert没有pooler实现，因此例子只显示如何用DistilBertModel加载权重。
```python 
import paddle
from paddlenlp.transformers import DistilBertModel, DistilBertTokenizer

model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
```