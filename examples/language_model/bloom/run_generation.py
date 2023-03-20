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
from modeling import BloomConfig, BloomForGeneration
from tap import Tap
from transformers import AutoTokenizer


class Config(Tap):
    model_name_or_path: str  # bloom model name, eg: bloom-560m
    device: str = "gpu"


def left_padding(inputs, pad_id):
    assert "input_ids" in inputs, "input_ids should be in inputs!"
    max_length = 0
    for ids in inputs["input_ids"]:
        max_length = max(max_length, len(ids))

    def extend_max_lenth(value, max_length, to_pad_id):
        return [to_pad_id] * (max_length - len(value)) + value

    def extend_filed(name, max_length, to_pad_id):
        values = inputs[name]
        res = [extend_max_lenth(value, max_length, to_pad_id) for value in values]
        inputs[name] = res

    extend_filed("input_ids", max_length, pad_id)
    if "attention_mask" in inputs:
        extend_filed("attention_mask", max_length, 0)
    if "position_ids" in inputs:
        extend_filed("position_ids", max_length, 0)

    return inputs


def convert_examples(sentences: list[str], tokenizer, max_length: int = 20):
    features = tokenizer.batch_encode_plus(
        sentences,
        padding="max_length",
        max_length=max_length,
    )
    features = left_padding(features, tokenizer.pad_token_id)
    return paddle.to_tensor(features["input_ids"], dtype="int64")


def generate(args: Config):
    paddle.set_device(args.device)
    paddle.set_default_dtype("float16")

    config = BloomConfig.from_pretrained(args.model_name_or_path)
    config.use_cache = True
    config.use_pure_fp16 = False
    model = BloomForGeneration.from_pretrained(args.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    sentences = [
        "Where is the captial of China?",
        "I love you, but",
        "What is your problem ?",
    ]
    input_ids = convert_examples(sentences, tokenizer, None)
    print("start to forward model ...")

    with paddle.amp.auto_cast(True):
        decoded_ids, _ = model(input_ids, use_cache=True)

    decoded_ids = decoded_ids.detach().tolist()
    for ids in decoded_ids:
        print("========================================")
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(tokenizer.convert_tokens_to_string(tokens))


if __name__ == "__main__":
    args = Config().parse_args(known_only=True)
    generate(args)
