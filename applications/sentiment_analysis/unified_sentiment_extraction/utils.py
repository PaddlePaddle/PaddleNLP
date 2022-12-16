# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import random
import re

import numpy as np
import paddle


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_txt(file_path):
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            texts.append(line.strip())
    return texts


def load_json_file(path):
    exmaples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            example = json.loads(line)
            exmaples.append(example)
    return exmaples


def write_json_file(examples, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for example in examples:
            line = json.dumps(example, ensure_ascii=False)
            f.write(line + "\n")


def create_data_loader(dataset, mode="train", batch_size=1, trans_fn=None):
    """
    Create dataloader.
    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a data sample to input ids, etc.
    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train" else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler, return_list=True)
    return dataloader


def convert_example(example, tokenizer, max_seq_len):
    """
    example: {
        title
        prompt
        content
        result_list
    }
    """
    encoded_inputs = tokenizer(
        text=[example["prompt"]],
        text_pair=[example["content"]],
        truncation=True,
        max_seq_len=max_seq_len,
        pad_to_max_seq_len=True,
        return_attention_mask=True,
        return_position_ids=True,
        return_dict=False,
        return_offsets_mapping=True,
    )
    encoded_inputs = encoded_inputs[0]
    offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"]]
    bias = 0
    for index in range(1, len(offset_mapping)):
        mapping = offset_mapping[index]
        if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
            bias = offset_mapping[index - 1][1] + 1  # Includes [SEP] token
        if mapping[0] == 0 and mapping[1] == 0:
            continue
        offset_mapping[index][0] += bias
        offset_mapping[index][1] += bias
    start_ids = [0 for x in range(max_seq_len)]
    end_ids = [0 for x in range(max_seq_len)]
    for item in example["result_list"]:
        # Positioning at char granularity，offset_mapping indicates offset by char.
        start = map_offset(item["start"] + bias, offset_mapping)
        end = map_offset(item["end"] - 1 + bias, offset_mapping)
        start_ids[start] = 1.0
        end_ids[end] = 1.0

    tokenized_output = [
        encoded_inputs["input_ids"],
        encoded_inputs["token_type_ids"],
        encoded_inputs["position_ids"],
        encoded_inputs["attention_mask"],
        start_ids,
        end_ids,
    ]
    tokenized_output = [np.array(x, dtype="int64") for x in tokenized_output]
    return tuple(tokenized_output)


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def reader(data_path, max_seq_len=512):
    """
    read json
    """
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            content = json_line["content"].strip()
            prompt = json_line["prompt"]
            # Model Input is aslike: [CLS] Prompt [SEP] Content [SEP]
            # It include three summary tokens.
            if max_seq_len <= len(prompt) + 3:
                raise ValueError("The value of max_seq_len is too small, please set a larger value")
            max_content_len = max_seq_len - len(prompt) - 3
            if len(content) <= max_content_len:
                yield json_line
            else:
                result_list = json_line["result_list"]
                json_lines = []
                accumulate = 0
                while True:
                    cur_result_list = []

                    for result in result_list:
                        if result["start"] + 1 <= max_content_len < result["end"]:
                            max_content_len = result["start"]
                            break

                    cur_content = content[:max_content_len]
                    res_content = content[max_content_len:]

                    while True:
                        if len(result_list) == 0:
                            break
                        elif result_list[0]["end"] <= max_content_len:
                            if result_list[0]["end"] > 0:
                                cur_result = result_list.pop(0)
                                cur_result_list.append(cur_result)
                            else:
                                cur_result_list = [result for result in result_list]
                                break
                        else:
                            break

                    json_line = {"content": cur_content, "result_list": cur_result_list, "prompt": prompt}
                    json_lines.append(json_line)

                    for result in result_list:
                        if result["end"] <= 0:
                            break
                        result["start"] -= max_content_len
                        result["end"] -= max_content_len
                    accumulate += max_content_len
                    max_content_len = max_seq_len - len(prompt) - 3
                    if len(res_content) == 0:
                        break
                    elif len(res_content) < max_content_len:
                        json_line = {"content": res_content, "result_list": result_list, "prompt": prompt}
                        json_lines.append(json_line)
                        break
                    else:
                        content = res_content

                for json_line in json_lines:
                    yield json_line


def unify_prompt_name(prompt):
    # The classification labels are shuffled during finetuning, so they need
    # to be unified during evaluation.
    if re.search(r"\[.*?\]$", prompt):
        prompt_prefix = prompt[: prompt.find("[", 1)]
        cls_options = re.search(r"\[.*?\]$", prompt).group()[1:-1].split(",")
        cls_options = sorted(list(set(cls_options)))
        cls_options = ",".join(cls_options)
        prompt = prompt_prefix + "[" + cls_options + "]"
        return prompt
    return prompt


def get_relation_type_dict(relation_data):
    def compare(a, b):
        a = a[::-1]
        b = b[::-1]
        res = ""
        for i in range(min(len(a), len(b))):
            if a[i] == b[i]:
                res += a[i]
            else:
                break
        if res == "":
            return res
        elif res[::-1][0] == "的":
            return res[::-1][1:]
        return ""

    relation_type_dict = {}
    added_list = []
    for i in range(len(relation_data)):
        added = False
        if relation_data[i][0] not in added_list:
            for j in range(i + 1, len(relation_data)):
                match = compare(relation_data[i][0], relation_data[j][0])
                if match != "":
                    match = unify_prompt_name(match)
                    if relation_data[i][0] not in added_list:
                        added_list.append(relation_data[i][0])
                        relation_type_dict.setdefault(match, []).append(relation_data[i][1])
                    added_list.append(relation_data[j][0])
                    relation_type_dict.setdefault(match, []).append(relation_data[j][1])
                    added = True
            if not added:
                added_list.append(relation_data[i][0])
                suffix = relation_data[i][0].rsplit("的", 1)[1]
                suffix = unify_prompt_name(suffix)
                relation_type_dict[suffix] = relation_data[i][1]
    return relation_type_dict
