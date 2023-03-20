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


def cnn_dm_convert_example(example, tokenizer, data_args, is_test=True):
    def punctuation_standardization(string: str):
        punctuation_dict = {"\u201c": '"', "\u201d": '"', "\u2019": "'", "\u2018": "'", "\u2013": "-"}
        for key, value in punctuation_dict.items():
            string = string.replace(key, value)
        return string

    def detokenize(string, is_target=False):
        _tok_dict = {"(": "-LRB-", ")": "-RRB-", "[": "-LSB-", "]": "-RSB-", "{": "-LCB-", "}": "-RCB-"}
        if not is_target:
            string = string.replace("<S_SEP>", "")
        else:
            string = string.replace("<S_SEP>", "[SEP]")
        for key, value in _tok_dict.items():
            string = string.replace(value, key)
        string = string.replace("''", '"')
        string = string.replace("``", '"')
        string = string.replace("`", "'")
        string = string.replace(" n't", "n't")
        string = string.replace(" 's", "'s")
        string = string.replace(" 'd", "'d")
        string = string.replace(" 'll", "'ll")
        return string

    def preprocess(string):
        string = string.strip()
        string = punctuation_standardization(string)
        string = detokenize(string)
        return string

    example["article"] = preprocess(example["article"])
    example["highlights"] = preprocess(example["highlights"])
    prompt = [tokenizer.convert_tokens_to_ids(x) for x in ("[CLS]", "[sMASK]", "ĠContent", ":")]
    inputs = tokenizer(
        " " + example["article"], add_special_tokens=False, max_length=data_args.src_length - len(prompt)
    )
    pad_length = data_args.src_length - len(inputs["input_ids"]) - len(prompt)
    inputs["input_ids"] = np.array([prompt + inputs["input_ids"] + [tokenizer.pad_token_id] * pad_length])
    inputs["attention_mask"] = np.array([[1] * len(prompt) + inputs["attention_mask"] + [0] * pad_length])
    sep = inputs["input_ids"].shape[1]
    inputs = tokenizer.build_inputs_for_generation(
        inputs,
        max_gen_length=data_args.tgt_length,
        targets=" " + example["highlights"] if not is_test else None,
        padding=True,
    )
    for input_name in inputs.keys():
        inputs[input_name] = inputs[input_name].squeeze(0)
    inputs["attention_mask"] = sep
    if is_test:
        inputs["position_ids"] = inputs["position_ids"][:, : inputs["input_ids"].shape[-1]]
        labels = tokenizer.encode(" " + example["highlights"], add_special_tokens=False)["input_ids"]
        loss_mask = [0] * sep + [1] * len(labels) + [0] * (data_args.tgt_length - len(labels))
        labels = (
            [0] * sep
            + labels
            + [tokenizer.eop_token_id]
            + [tokenizer.pad_token_id] * (data_args.tgt_length - len(labels) - 1)
        )
        inputs["label_ids"] = labels
        inputs["loss_mask"] = loss_mask
    return inputs
