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
from paddlenlp.transformers import GPT2Model, GPT2ForGreedyGeneration
from paddlenlp.transformers import GPT2ChineseTokenizer, GPT2Tokenizer
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "gpt2-base-cn": (GPT2ForGreedyGeneration, GPT2ChineseTokenizer),
    "gpt2-medium-en": (GPT2ForGreedyGeneration, GPT2Tokenizer),
}


class Demo:
    def __init__(self, model_name_or_path="gpt2-base-cn", max_predict_len=32):
        model_class, tokenizer_class = MODEL_CLASSES[model_name_or_path]
        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        logger.info('Loading the model parameters, please wait...')
        self.model = model_class.from_pretrained(
            model_name_or_path, max_predict_len=max_predict_len)
        self.model.eval()
        logger.info('Model loaded.')

    # prediction function
    def predict(self, text):
        ids = self.tokenizer.encode(text)
        input_ids = paddle.to_tensor(
            np.array(ids).reshape(1, -1).astype('int64'))
        out = self.model(input_ids, self.tokenizer.command_name_map["stop"].Id)
        out = [int(x) for x in out.numpy().reshape([-1])]
        logger.info(self.tokenizer.decode(out))

    # One shot example
    def ask_question(self, question):
        self.predict("问题：中国的首都是哪里？答案：北京。\n问题：%s 答案：" % question)

    # dictation poetry
    def dictation_poetry(self, front):
        self.predict('''默写古诗: 大漠孤烟直，长河落日圆。\n%s''' % front)


if __name__ == "__main__":
    # demo = Demo("gpt2-base-cn")
    # demo.ask_question("苹果的CEO是谁?")
    # demo.dictation_poetry("举杯邀明月，")
    # del demo
    demo = Demo("gpt2-medium-en")
    demo.predict(
        "Question: Where is the capital of China? Answer: Beijing. \nQuestion: Who is the CEO of Apple? Answer:"
    )
