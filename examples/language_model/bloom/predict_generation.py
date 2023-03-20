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

import numpy as np
import paddle
from configuration import BloomConfig
from model_split_merge import merge_model_parallel
from modeling import BloomForGeneration
from transformers import AutoTokenizer


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="The directory of model.")
    parser.add_argument("--batch_size", type=int, default=2, help="The batch size of data.")
    return parser.parse_args()


def left_padding(inputs, pad_id, padding="longest"):
    assert "input_ids" in inputs, "input_ids should be in inputs!"
    max_length = 0
    for ids in inputs["input_ids"]:
        max_length = max(max_length, len(ids))

    def extend_max_lenth(value, max_length, to_pad_id):
        return [to_pad_id] * (max_length - len(value)) + value

    def extend_filed(name, max_length, to_pad_id):
        values = inputs[name]
        res = []
        for index, value in enumerate(values):
            res.append(extend_max_lenth(value, max_length, to_pad_id))
        inputs[name] = res

    extend_filed("input_ids", max_length, pad_id)
    if "attention_mask" in inputs:
        extend_filed("attention_mask", max_length, 0)
    if "position_ids" in inputs:
        extend_filed("position_ids", max_length, 0)

    return inputs


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class Predictor(object):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.model = self.create_model(args)
        self.batch_size = args.batch_size

    def create_model(self, args):
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        config = BloomConfig.from_pretrained(args.model_path)

        # Set the generaiton the hyperparameter
        config.max_dec_len = 20
        config.temperature = 0.5
        config.decode_strateg = "sampling"
        config.eos_token_id = tokenizer.eos_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.pad_token_id = tokenizer.pad_token_id
        config.use_cache = True
        config.top_k = 1
        config.use_recompute = False

        # Merge the model splits to a total model
        merge_model_path = merge_model_parallel(args.model_path, config)

        # Load the model and parameter
        config.mp_degree = 1
        model = BloomForGeneration.from_pretrained(merge_model_path, config=config)
        model.eval()
        return model

    def preprocess(self, input_text):
        inputs = self.tokenizer(input_text)
        inputs = left_padding(inputs, self.tokenizer.pad_token_id)
        input_map = {
            "input_ids": np.array(inputs["input_ids"], dtype="int64"),
        }
        return input_map

    def infer(self, input_map):
        results = self.model(paddle.to_tensor(input_map["input_ids"]))
        return results

    def postprocess(self, infer_data):
        result = []
        for x in infer_data[0].tolist():
            tokens = self.tokenizer.convert_ids_to_tokens(x)
            sentence = self.tokenizer.convert_tokens_to_string(tokens)
            result.append(sentence)
        out_dict = {"result": result}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output


if __name__ == "__main__":
    args = parse_arguments()
    predictor = Predictor(args)
    all_texts = [
        "答案：年基准利率4.35%</s>上下文：从实际看,贷款的基本条件是: 一是中国大陆居民,年龄在60岁以下; 二是有稳定的住址和工作或经营地点; 三是有稳定的收入来源; 四是无不良信用记录,贷款用途不能作为炒股,赌博等行为; 五是具有完全民事行为能力。</s>在已知答案的前提下，问题：",
        "答案：U系列</s>上下文：U系列是最好的，采用国际顶尖技术（由格力自主研发）双级变频压缩机，提高压缩机运转效率，制冷制热能力更强劲；1赫兹变频技术，使空调相当于一个15 W电灯泡，更加节能省电；送风面积广，风力大；生态风，净化空气。非常不错，现在国美在做活动，可以了解一下，问题：",
    ]
    batch_texts = batchfy_text(all_texts, args.batch_size)
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for text, result in zip(texts, outputs["result"]):
            print("{} \n {}".format(text, result))
