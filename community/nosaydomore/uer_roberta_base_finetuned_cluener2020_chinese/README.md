## uer/roberta-base-finetuned-cluener2020-chinese

权重来源：https://huggingface.co/uer/roberta-base-finetuned-cluener2020-chinese

```python
from paddlenlp.transformers import (
    RobertaModel, RobertaForMaskedLM, RobertaForQuestionAnswering,
    RobertaForSequenceClassification, RobertaForTokenClassification)
from paddlenlp.transformers import RobertaBPETokenizer, RobertaTokenizer
import paddle
import os
import numpy as np


text = '江苏警方通报特斯拉冲进店铺'

tokenizer = RobertaTokenizer.from_pretrained('uer/roberta-base-finetuned-cluener2020-chinese')
token = tokenizer(text)

config = RobertaModel.pretrained_init_configuration[
    'uer/roberta-base-finetuned-cluener2020-chinese']
roberta = RobertaModel(**config)
model = RobertaForTokenClassification(roberta, 32)
model_state = paddle.load(os.path.join(path, "model_state.pdparams"))
model.load_dict(model_state)
model.eval()
input_ids = paddle.to_tensor(token['input_ids'], dtype='int64').unsqueeze(0)
with paddle.no_grad():
    output = model(input_ids)

import paddle.nn.functional as F
output = F.softmax(output)
id2label = {
    "0": "O",
    "1": "B-address",
    "2": "I-address",
    "3": "B-book",
    "4": "I-book",
    "5": "B-company",
    "6": "I-company",
    "7": "B-game",
    "8": "I-game",
    "9": "B-government",
    "10": "I-government",
    "11": "B-movie",
    "12": "I-movie",
    "13": "B-name",
    "14": "I-name",
    "15": "B-organization",
    "16": "I-organization",
    "17": "B-position",
    "18": "I-position",
    "19": "B-scene",
    "20": "I-scene",
    "21": "S-address",
    "22": "S-book",
    "23": "S-company",
    "24": "S-game",
    "25": "S-government",
    "26": "S-movie",
    "27": "S-name",
    "28": "S-organization",
    "29": "S-position",
    "30": "S-scene",
    "31": "[PAD]"
}
tokenized_text = tokenizer.tokenize(text)
scores = []
char_cn = []
for t, s in zip(tokenized_text, output[0][1:-1]):
    index = paddle.argmax(s).item()
    label = id2label[str(index)]
    if index != 0:
        scores.append(s[index].item())
        char_cn.append(t)
        print(f"{label} {t} score {s[index].item()}")
print("{}:{}".format("".join(char_cn[:2]), sum(scores[:2]) / 2))
print("{}:{}".format("".join(char_cn[2:]), sum(scores[2:]) / 3))

'''
B-address 江 score 0.6618999242782593
I-address 苏 score 0.5544551610946655
B-company 特 score 0.4227274954319
I-company 斯 score 0.45469844341278076
I-company 拉 score 0.5207833051681519
江苏:0.6081775426864624
特斯拉:0.46606974800427753
'''
