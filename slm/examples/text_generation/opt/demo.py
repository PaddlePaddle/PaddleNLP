# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from paddlenlp.transformers.gpt.tokenizer import GPTTokenizer
from paddlenlp.transformers.opt.modeling import OPTForCausalLM
from paddlenlp.utils.log import logger


class Demo:
    def __init__(self, model_name_or_path, max_predict_len=128):
        self.tokenizer = GPTTokenizer.from_pretrained(model_name_or_path)
        logger.info("Loading the model parameters, please wait...")
        self.model = OPTForCausalLM.from_pretrained(model_name_or_path)
        self.model.eval()
        self.max_predict_len = max_predict_len
        logger.info("Model loaded.")

    @paddle.no_grad()
    def generate(self, inputs):
        ids = self.tokenizer(inputs)["input_ids"]
        input_ids = paddle.to_tensor([ids], dtype="int64")
        outputs = self.model.generate(input_ids, max_length=self.max_predict_len)[0][0]
        decode_outputs = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(outputs.cpu()))

        print(f"input text: \n{inputs}")
        print(f"output text: \n{decode_outputs}")
        print("=" * 50)


if __name__ == "__main__":

    demo = Demo(model_name_or_path="facebook/opt-1.3b", max_predict_len=10)
    input_text_list = [
        "Question:If x is 2 and y is 5, what is x+y?\n"
        "Answer: 7\n\n"
        "Question: if x is 12 and y is 9, what is x+y?\n"
        "Answer:21\n\n"
        "Question: if x is 3 and y is 4, what is x+y?\n",
        "a chat between a curious human and Statue of Liberty.\n"
        "Human: What is your name?\n"
        "Statue: I am statue of liberty.\n\n"
        "Human: where do you live?\n"
        "Statue: New york city.\n\n"
        "Human: how long have you lived there?\n",
        "Chinese: 我想回家。\n"
        "English: I want to go home.\n\n"
        "Chinese: 我不知道。\n"
        "English: I don't know.\n\n"
        "Chinese: 我饿了。\n"
        "English: I am hungry.\n\n"
        "Chinese: 我累了。\n",
    ]
    for text in input_text_list:
        demo.generate(text)
