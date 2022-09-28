# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddlenlp.transformers import T5ForConditionalGeneration, T5Tokenizer


class Demo:

    def __init__(self, model_name_or_path="t5-base", max_predict_len=5):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        print("Loading the model parameters, please wait...")
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name_or_path)
        self.model.eval()
        self.max_predict_len = max_predict_len
        print("Model loaded.")

    # prediction function
    @paddle.no_grad()
    def generate(self, inputs, max_predict_len=None):
        max_predict_len = max_predict_len if max_predict_len is not None else self.max_predict_len

        ids = self.tokenizer(inputs)["input_ids"]
        input_ids = paddle.to_tensor([ids], dtype="int64")
        outputs = self.model.generate(input_ids,
                                      max_length=max_predict_len)[0][0]
        decode_outputs = self.tokenizer.decode(
            outputs, skip_special_tokens=True).strip()
        print(f"input text: {inputs}")
        print(f"label: {decode_outputs}")
        print("=" * 50)


if __name__ == "__main__":
    label_length_map = {
        "cola": 4,
        "sst2": 1,
        "mrpc": 5,
        "stsb": 5,
        "qqp": 5,
        "mnli": 4,
        "qnli": 5,
        "rte": 5,
    }
    demo = Demo(model_name_or_path="t5-base")
    input_text_list = [
        "sst2 sentence: contains no wit , only labored gags ",
        "sst2 sentence: that loves its characters and communicates something rather beautiful about human nature ",
        "cola sentence: Mickey looked it up.",
        "sst2 sentence: remains utterly satisfied to remain the same throughout ",
        "sst2 sentence: a well-made and often lovely depiction of the mysteries of friendship "
    ]
    for text in input_text_list:
        max_predict_len = label_length_map[text.split()[0]]
        demo.generate(text, max_predict_len=max_predict_len)

    # input text: sst2 sentence: contains no wit , only labored gags
    # label: negative
    # ==================================================
    # input text: sst2 sentence: that loves its characters and communicates something rather beautiful about human nature
    # label: positive
    # ==================================================
    # input text: cola sentence: Mickey looked it up.
    # label: acceptable
    # ==================================================
    # input text: sst2 sentence: remains utterly satisfied to remain the same throughout
    # label: positive
    # ==================================================
    # input text: sst2 sentence: a well-made and often lovely depiction of the mysteries of friendship
    # label: positive
    # ==================================================
