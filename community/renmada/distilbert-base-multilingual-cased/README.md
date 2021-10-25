# 模型介绍
This model is a distilled version of the BERT base multilingual model. The code for the distillation process can be found here. This model is cased: it does make a difference between english and English.

The model is trained on the concatenation of Wikipedia in 104 different languages listed here. The model has 6 layers, 768 dimension and 12 heads, totalizing 134M parameters (compared to 177M parameters for mBERT-base). On average DistilmBERT is twice as fast as mBERT-base.
# 模型来源
https://huggingface.co/distilbert-base-multilingual-cased

# 模型使用
```python
import paddle
from paddlenlp.transformers import DistilBertForMaskedLM, DistilBertTokenizer

model = DistilBertForMaskedLM.from_pretrained('renmada/distilbert-base-multilingual-cased')
tokenizer = DistilBertTokenizer.from_pretrained('renmada/distilbert-base-multilingual-cased')

inp = '北京是中国的首都'
ids = tokenizer.encode(inp)['input_ids']  # [101, 10751, 13672, 16299, 10124, 10105, 12185, 10108, 50513, 119, 102]
print(ids)

# mask "北京"
ids[1] = 103
ids[2] = 103
ids = paddle.to_tensor([ids])

# Do mlm
model.eval()
with paddle.no_grad():
    mlm_logits = model(ids)
    mlm_pred = paddle.topk(mlm_logits, 1, -1)[1][0].unsqueeze(-1)

print(''.join(tokenizer.vocab.idx_to_token[int(x)] for x in mlm_pred[1:-1]))  # 汉阳是中国的首都
```
