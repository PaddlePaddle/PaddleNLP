
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

import argparse
import os

import paddle
from transformers import AutoTokenizer 
from modeling import BloomForGeneration
from configuration import BloomConfig

MODEL_CLASSES = {
    "bigscience/bloom-560m":(BloomForGeneration)
}


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_type",
        default="bigscience/bloom-560m",
        type=str,
        # required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_path",
        default="bigscience/bloom-560m",
        type=str,
        required=False,
        help="Path of the trained model to be exported.",
    )
    parser.add_argument(
        "--output_path",
        default="inference/gpt",
        type=str,
        # required=True,
        help="The output file prefix used to save the exported inference model.",
    )
    args = parser.parse_args()
    return args


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


def main():
    args = parse_args()

    args.model_type = args.model_type.lower()
    model_class = MODEL_CLASSES[args.model_type]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = BloomConfig.from_pretrained(args.model_path)

    config.max_dec_len = 20
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id  = tokenizer.bos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.use_cache = True
    config.top_k = 1

    model = model_class.from_pretrained(args.model_path, config=config)
    #state_dict = paddle.load('/root/.paddlenlp/models/bigscience/bloom-560m/model_state.pdparams')

    #keys_load = state_dict.keys()
    #keys_raw = model.state_dict().keys()
    #for key1, key2 in zip(keys_load, keys_raw):
    #    print("{}\t{}\n".format(key1, key2))
    #model.set_state_dict(state_dict)
    
    model.eval()
    # Convert to static graph with specific input description
    input_text = ["Nice to meet"]#, "Hello "]
    inputs = tokenizer(input_text)
    # input_ids = tokenizer.encode(input_text)['input_ids']
    inputs = tokenizer(input_text)
    #inputs = left_padding(inputs, tokenizer.bos_token_id)
    input_ids = inputs["input_ids"]

    input_ids = paddle.to_tensor(input_ids, dtype="int64")
    ret = model(input_ids=input_ids)
    print(ret)

    # ret =  model.generate(input_ids = data["input_ids"])
    for x in ret[0].tolist():
        print("==" * 30)
        print(tokenizer.convert_ids_to_tokens(x))

if __name__ == "__main__":
    main()
