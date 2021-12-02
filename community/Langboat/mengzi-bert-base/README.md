# 详细介绍
**介绍**：Mengzi-bert-base 是一个轻量级的中文语言BERT模型，
在300G中文语料库继续训练，预训练目标有三个，掩码语言建模 (MLM)，句子顺序预测(SOP)
和词性标注(POS)。

**模型结构**： **`BertForMakedLM`**。

**适用下游任务**：**自然语言理解类任务**，如：文本分类、实体识别、关系抽取、阅读理解等。

**训练数据**：300G中文语料库数据。

#使用示例：
```python
from paddlenlp import BertForMaskedLM, BertTokenizer
tokenizer = BertTokenizer.from_pretrained('Langboat/mengzi-bert-base')
model = BertForMaskedLM.from_pretrained('Langboat/mengzi-bert-base')


tokens = ['[CLS]', '我', '的', '[MASK]','很', '可', '爱','。', '[SEP]']
masked_ids = paddle.to_tensor([tokenizer.convert_tokens_to_ids(tokens)])
segment_ids = paddle.to_tensor([[0] * len(tokens)])

outputs = model(masked_ids, token_type_ids=segment_ids)
prediction_scores = outputs
prediction_index = paddle.argmax(prediction_scores[0, 3]).item()
predicted_token = tokenizer.convert_ids_to_tokens([prediction_index])[0]
print(tokens)
#['[CLS]', '我', '的', '[MASK]', '很', '可', '爱', '。', '[SEP]']
print(predicted_token)
#猫
```

# 权重来源
https://huggingface.co/Langboat/mengzi-bert-base
