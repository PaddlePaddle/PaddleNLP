# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Many thanks for following projects.
# https://github.com/TsinghuaAI/CPM-Generate
# https://github.com/jm12138/CPM-Generate-Paddle

import argparse
import numpy as np

import paddle
from paddlenlp.transformers import GPT2Model, GPT2ForPretraining
from paddlenlp.transformers import GPT2ChineseTokenizer, GPT2Tokenizer
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "gpt2-base-cn": (GPT2ForPretraining, GPT2ChineseTokenizer),
    "gpt2-medium-en": (GPT2ForPretraining, GPT2Tokenizer),
}


class Demo:
    def __init__(self, model_name_or_path="gpt2-base-cn"):
        model_class, tokenizer_class = MODEL_CLASSES[model_name_or_path]
        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        logger.info('Loading the model parameters, please wait...')
        self.model = model_class.from_pretrained(model_name_or_path)
        self.model.eval()
        logger.info('Model loaded.')

    # prediction function
    def predict(self, text, max_len=10):
        ids = self.tokenizer.encode(text)
        input_id = paddle.to_tensor(
            np.array(ids).reshape(1, -1).astype('int64'))
        output, cached_kvs = self.model(input_id, use_cache=True, cache=None)
        nid = int(np.argmax(output[0, -1].numpy()))
        ids.append(nid)
        out = [nid]
        for i in range(max_len):
            input_id = paddle.to_tensor(
                np.array([nid]).reshape(1, -1).astype('int64'))
            output, cached_kvs = self.model(
                input_id, use_cache=True, cache=cached_kvs)
            nid = int(np.argmax(output[0, -1].numpy()))
            ids.append(nid)
            # if nid is '\n', the predicion is over.
            if nid == 3:
                break
            out.append(nid)
        logger.info(text)
        logger.info(self.tokenizer.decode(out))

    # One shot example
    def ask_question(self, question, max_len=10):
        self.predict("问题：中国的首都是哪里？答案：北京。\n问题：%s 答案：" % question, max_len)

    # dictation poetry
    def dictation_poetry(self, front, max_len=10):
        self.predict('''默写古诗: 大漠孤烟直，长河落日圆。\n%s''' % front, max_len)


if __name__ == "__main__":
    demo = Demo("gpt2-base-cn")
    demo.ask_question("百度的厂长是谁?")
    demo.dictation_poetry("举杯邀明月，")
    del demo
    # demo = Demo("gpt2-medium-en")
    # demo.predict("Question: Where is the capital of China? Answer: Beijing. \nQuestion: Who is the CEO of Apple? Answer:", 20)
