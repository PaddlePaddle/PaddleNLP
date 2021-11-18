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
tokenizer = BertTokenzier.from_pretrained('huhuiwen/mengzi-bert-base-fin')
model = BertForMakedLM.from_pretrained('huhuiwen/mengzi-bert-base-fin')

text  = "股市指某支新发行股票在定价和配置后的交易市场。"
inputs = tokenizer(text)
tokens = ['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]']

for i in range(1, len(tokens)-1):
    tmp = tokens[:i] + ['[MASK]'] + tokens[i+1:]
    masked_ids = paddle.to_tensor([tokenizer.convert_tokens_to_ids(tmp)])
    segment_ids = paddle.to_tensor([[0]*len(tmp)])

    outputs = model(masked_ids, token_type_ids=segment_ids)
    prediction_scores = outputs
    print(tmp)
    # 打印被预测的字符
    prediction_index = paddle.argmax(prediction_scores[0, i]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([prediction_index])[0]
    print(predicted_token)
```

# 权重来源
https://huggingface.co/Langboat/mengzi-bert-base-fin
