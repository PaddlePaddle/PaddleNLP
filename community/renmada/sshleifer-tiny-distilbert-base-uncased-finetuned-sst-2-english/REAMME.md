# 模型介绍
tiny-distilbert-base-uncased在sst-2上finetune后的模型
# 模型来源
https://huggingface.co/sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english
# 模型使用
```python
import paddle
from paddlenlp.transformers import DistilBertForSequenceClassification, DistilBertTokenizer

model = DistilBertForSequenceClassification.from_pretrained('renmada/sshleifer-tiny-distilbert-base-uncase-finetuned-sst-2-english')
tokenizer = DistilBertTokenizer.from_pretrained('renmada/sshleifer-tiny-distilbert-base-uncase-finetuned-sst-2-english')
inp = 'It is good'
ids = tokenizer.encode(inp)['input_ids']
ids = paddle.to_tensor([ids])
model.eval()
with paddle.no_grad():
    logtis = model(ids)
```
