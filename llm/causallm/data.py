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
import json

import numpy as np

from paddlenlp.peft import LoRAModel, PrefixModelForCausalLM


def get_convert_example(model):
    if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
        base_model_prefix = model.model.base_model_prefix
    else:
        base_model_prefix = model.base_model_prefix

    if base_model_prefix == "chatglm":
        return convert_example_chatglm
    elif base_model_prefix in ["chatglm_v2", "llama", "bloom"]:
        return convert_example_common
    else:
        raise ValueError(
            f"Unknown base_model_prefix: {model.base_model_prefix}. Supported base_model_prefix list: chatglm, bloom, llama."
        )


def read_local_dataset(path):
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            yield json.loads(line.strip())


class DataFormatError(ValueError):
    pass


def tokenize_example(tokenizer, example, data_args):
    if "src" in example and "tgt" in example:
        source = example["src"][0] if isinstance(example["src"], list) else example["src"]
        target = example["tgt"][0] if isinstance(example["tgt"], list) else example["tgt"]
    else:
        raise DataFormatError(
            f"Example format is wrong, please check: {example} or rewrite tokenize_example in data.py "
        )
    tokenized_source = tokenizer(
        source,
        max_length=data_args.src_length,
        truncation=True,
        truncation_side="left",
        add_special_tokens=True,
    )
    tokenized_target = tokenizer(
        target,
        max_length=data_args.tgt_length,
        truncation=True,
        truncation_side="right",
        add_special_tokens=False,
    )

    tokenized_target_input_ids = tokenized_target["input_ids"]
    # Add eos_token_id at the end of sequence if the sentence is not truncated.
    # Attention! In some cases(ex. ChatGLMv2), tokenized eos_token is not equal to eos_token_id.
    if len(tokenized_target_input_ids) < data_args.tgt_length:
        tokenized_target_input_ids += [tokenizer.eos_token_id]

    return tokenized_source, tokenized_target_input_ids


def convert_example_common(example, tokenizer, data_args, is_test=True, intokens=False):
    tokenized_source, tokenized_target_input_ids = tokenize_example(tokenizer, example, data_args)

    if is_test:
        return dict(
            input_ids=tokenized_source["input_ids"],
            labels=tokenized_target_input_ids,
        )
    else:
        input_ids = tokenized_source["input_ids"] + tokenized_target_input_ids
        source_length = len(tokenized_source["input_ids"])
        labels = [-100] * source_length + input_ids[source_length:]
        # shift labels
        input_ids, labels = input_ids[:-1], labels[1:]
        features = {
            "input_ids": input_ids,
            "labels": labels,
        }
        seq_length = len(input_ids)
        if intokens:
            features["position_ids"] = list(range(seq_length))
            features["attention_mask"] = np.tril(np.ones((seq_length, seq_length), dtype="bool"))

        return features


def convert_example_chatglm(example, tokenizer, data_args, is_test=True, intokens=False):

    tokenized_source, tokenized_target_input_ids = tokenize_example(tokenizer, example, data_args)
    if is_test:
        return dict(
            input_ids=tokenized_source["input_ids"],
            position_ids=tokenized_source["position_ids"],
            attention_mask=tokenized_source["attention_mask"],
            labels=tokenized_target_input_ids,
        )
    else:
        input_ids = tokenized_source["input_ids"] + tokenized_target_input_ids
        bos_position = len(tokenized_source["input_ids"]) - 1

        attention_mask = np.tri(len(input_ids), len(input_ids))
        attention_mask[:, :bos_position] = 1
        attention_mask = attention_mask[None, :, :]
        attention_mask = (attention_mask < 0.5).astype("int64")

        labels = [-100] * bos_position + input_ids[bos_position:]

        # shift labels
        input_ids, labels = input_ids[:-1], labels[1:]
        attention_mask = attention_mask[..., :-1, :-1]

        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
