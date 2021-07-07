#！-*- coding: utf-8 -*-
# SimBERT 相似度任务测试
# 基于LCQMC语料
import paddle
import paddlenlp

import numpy as np
from paddlenlp.transformers.bert.modeling import *
from paddlenlp.transformers.bert.tokenizer import *

paddle_model_name = 'simbert-base-chinese'
paddle_model = BertModel.from_pretrained(paddle_model_name,with_pool="linear")
paddle_tokenizer = BertTokenizer.from_pretrained(paddle_model_name)
paddle_model.eval()

# 加载数据
data = [
    ['世界上什么东西最小', '世界上什么东西最小？'],
    ['光眼睛大就好看吗', '眼睛好看吗？'],
    ['小蝌蚪找妈妈怎么样', '小蝌蚪找妈妈是谁画的'],
]

text_dictA, text_dictB, labels = [], [], []
texts = []

for d in data:
    text_dictA.append(d[0])
    text_dictB.append(d[1])

result_A,result_B = [],[]

count = 0
for t in text_dictA:
    paddle_inputs = paddle_tokenizer(t)
    paddle_inputs = {k:paddle.to_tensor([v]) for (k, v) in paddle_inputs.items()}
    paddle_outputs = paddle_model(**paddle_inputs)
    paddle_logits = paddle_outputs[1]
    paddle_array = paddle_logits.numpy()
    result_A.append(paddle_array)

count = 0
for t in text_dictB:
    paddle_inputs = paddle_tokenizer(t)
    paddle_inputs = {k:paddle.to_tensor([v]) for (k, v) in paddle_inputs.items()}
    paddle_outputs = paddle_model(**paddle_inputs)
    paddle_logits = paddle_outputs[1]
    paddle_array = paddle_logits.numpy()
    result_B.append(paddle_array)
#计算相似度
sims = []
for a_vecs,b_vecs in zip(result_A,result_B):

    a_vecs = a_vecs / (a_vecs**2).sum()**0.5
    b_vecs = b_vecs / (b_vecs**2).sum()**0.5
    sim = (a_vecs * b_vecs).sum()
    sims.append(sim)
sims = np.array(sims)

for idx, text in enumerate(data):
        print('Data: {} \t similarity: {}'.format(text, sims[idx]))