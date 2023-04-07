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

import paddle
from utils import load_model

from paddlenlp.transformers import AutoTokenizer, BloomForCausalLM


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="bigscience/bloom-560m", help="The directory of model.")
    parser.add_argument("--save_onepiece_model_path", default=None, help="The directory of model.")
    parser.add_argument("--batch_size", type=int, default=2, help="The batch size of data.")
    parser.add_argument("--max_length", type=int, default=200, help="The batch size of data.")
    parser.add_argument("--seed", type=int, default=20, help="the seed of parameter initialization")
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
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.batch_size = args.batch_size
        self.args = args
        self.model = load_model(args, BloomForCausalLM)
        self.model.eval()

    def preprocess(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="np",
            padding=True,
            max_length="max_length",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        inputs_tensor = {}
        for key, value in inputs.items():
            inputs_tensor[key] = paddle.to_tensor(value)
        return inputs_tensor

    def infer(self, inputs):
        if self.model.config.dtype == "float32" or self.model.config.dtype is None:
            result = self.model.generate(
                **inputs,
                max_length=self.args.max_length,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                decode_strategy="sampling",
                top_k=1,
            )
        else:
            with paddle.amp.auto_cast(False, level="O2", dtype=self.model.config.dtype):
                result = self.model.generate(
                    **inputs,
                    max_length=self.args.max_length,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    decode_strategy="sampling",
                    top_k=1,
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

    def save_onepiece_model(self, save_onepiece_model_path):
        self.model.save_pretrained(save_dir=save_onepiece_model_path, merge_tensor_parallel=True)
        paddle.distributed.barrier()
        self.tokenizer.save_pretrained(save_onepiece_model_path)
        paddle.distributed.barrier()


def predict():
    args = parse_arguments()
    predictor = Predictor(args)
    tokenizer = predictor.tokenizer
    all_texts = [
        f"答案：年基准利率4.35% {tokenizer.eos_token}上下文：从实际看,贷款的基本条件是: 一是中国大陆居民,年龄在60岁以下; 二是有稳定的住址和工作或经营地点; 三是有稳定的收入来源; 四是无不良信用记录,贷款用途不能作为炒股,赌博等行为; 五是具有完全民事行为能力。{tokenizer.eos_token}在已知答案的前提下，问题：",
        f"答案：U系列{tokenizer.eos_token}上下文：U系列是最好的，采用国际顶尖技术（由格力自主研发）双级变频压缩机，提高压缩机运转效率，制冷制热能力更强劲；1赫兹变频技术，使空调相当于一个15 W电灯泡，更加节能省电；送风面积广，风力大；生态风，净化空气。非常不错，现在国美在做活动，可以了解一下。{tokenizer.eos_token}在已知答案的前提下，问题：",
    ]
    batch_texts = batchfy_text(all_texts, args.batch_size)
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for text, result in zip(texts, outputs["result"]):
            print("{}\n{}".format(text, result))
    if args.save_onepiece_model_path is not None:
        predictor.save_onepiece_model(args.save_onepiece_model_path)


if __name__ == "__main__":
    predict()
