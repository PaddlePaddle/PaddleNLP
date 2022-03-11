## nosaydomore/roberta-en-large
权重来源
https://huggingface.co/roberta-large

在英文数据集上预训练的roberta-base MaskedLM模型

```python
from paddlenlp.transformers import (
    RobertaModel, RobertaForMaskedLM, RobertaForQuestionAnswering,
    RobertaForSequenceClassification, RobertaForTokenClassification)
from paddlenlp.transformers import RobertaBPETokenizer, RobertaTokenizer
import paddle
import os
import numpy as np

model = RobertaForMaskedLM.from_pretrained('nosaydomore/roberta-en-large')
tokenizer = RobertaBPETokenizer.from_pretrained('nosaydomore/roberta-en-large')
text = ["The man worked as a", "."]  #"The man worked as a <mask>."
tokens_list = []
for i in range(2):
    tokens_list.append(tokenizer.tokenize(text[i]))

tokens = ['<s>']
tokens.extend(tokens_list[0])
tokens.extend(['<mask>'])
tokens.extend(tokens_list[1])
tokens.extend(['</s>'])
token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(token_ids)

model.eval()
input_ids = paddle.to_tensor([token_ids])
with paddle.no_grad():
    pd_outputs = model(input_ids)

pd_outputs = pd_outputs[0]

pd_outputs_sentence = "paddle: "
for i, id in enumerate(token_ids):
    if id == 50264:
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

'''
paddle:  The Ġman Ġworked Ġas Ġa [Ġmechanic=0.08260241150856018||Ġdriver=0.05736102908849716||Ġteacher=0.047089140862226486||Ġbartender=0.04641677811741829||Ġwaiter=0.04239342361688614] .  
'''
