# 详细介绍
本权重为使用PaddleNLP提供的[ERNIE-1.0预训练教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/language_model/ernie-1.0)，在CLUECorpusSmall 14g数据集上训练得到的权重。

本模型结构与ernie-1.0完全相同。使用训练配置`batch_size=512, max_steps=100w`, 训练得到。模型使用方法与原始ernie-1.0权重相同。

预训练全流程参见：https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/language_model/ernie-1.0/README.md

# 使用示例

示例一：
```python
import paddle
from paddlenlp.transformers import ErnieForMaskedLM, ErnieTokenizer
tokenizer = ErnieTokenizer.from_pretrained('zhui/ernie-1.0-cluecorpussmall')
model = ErnieForMaskedLM.from_pretrained('zhui/ernie-1.0-cluecorpussmall')

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

示例二：
```python
import paddle
from paddlenlp.transformers import *

tokenizer = AutoTokenizer.from_pretrained('zhui/ernie-1.0-cluecorpussmall')
text = tokenizer('自然语言处理')

# 语义表示
model = AutoModel.from_pretrained('zhui/ernie-1.0-cluecorpussmall')
sequence_output, pooled_output = model(input_ids=paddle.to_tensor([text['input_ids']]))
# 文本分类 & 句对匹配
model = AutoModelForSequenceClassification.from_pretrained('zhui/ernie-1.0-cluecorpussmall')
# 序列标注
model = AutoModelForTokenClassification.from_pretrained('zhui/ernie-1.0-cluecorpussmall')
# 问答
model = AutoModelForQuestionAnswering.from_pretrained('zhui/ernie-1.0-cluecorpussmall')
```
