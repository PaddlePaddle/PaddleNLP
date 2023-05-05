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
from __future__ import annotations

from paddlenlp.transformers import GPTTokenizer, OPTForCausalLM


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-125m", help="The directory of model.")
    parser.add_argument("--batch_size", type=int, default=2, help="The batch size of data.")
    parser.add_argument("--src_length", type=int, default=200, help="The batch size of data.")
    parser.add_argument("--tgt_length", type=int, default=20, help="The batch size of data.")
    return parser.parse_args()


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class Predictor(object):
    def __init__(self, args):
        self.tokenizer = GPTTokenizer.from_pretrained(args.model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.batch_size = args.batch_size
        self.args = args
        self.model = OPTForCausalLM.from_pretrained(args.model_name_or_path)
        self.model.eval()

    def preprocess(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="pd",
            padding=True,
            truncation=True,
            truncation_side="left",
            return_position_ids=False,
            return_token_type_ids=False,
        )
        return inputs

    def infer(self, inputs):
        result = self.model.generate(
            **inputs,
            decode_strategy="sampling",
            top_k=1,
            max_length=self.args.tgt_length,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        result = result[0]
        return result

    def postprocess(self, infer_data):
        result = []
        for x in infer_data.tolist():
            res = self.tokenizer.decode(x, skip_special_tokens=True)
            result.append(res)
        out_dict = {"result": result}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output


def predict():
    args = parse_arguments()
    predictor = Predictor(args)
    all_texts = ["Hello, I am conscious and", "The woman worked as a"]
    batch_texts = batchfy_text(all_texts, args.batch_size)
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for text, result in zip(texts, outputs["result"]):
            print("{}\n{}".format(text, result))


if __name__ == "__main__":
    predict()
