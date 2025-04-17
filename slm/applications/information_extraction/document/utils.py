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

import base64
import json
from typing import List, Optional

import numpy as np

from paddlenlp.utils.ie_utils import map_offset, pad_image_data
from paddlenlp.utils.log import logger


def reader(data_path, max_seq_len=512):
    """
    read json
    """
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            content = json_line["content"].strip()
            prompt = json_line["prompt"]
            boxes = json_line.get("bbox", None)
            image = json_line.get("image", None)
            # Model Input is aslike: [CLS] prompt [SEP] [SEP] text [SEP] for UIE-X
            if boxes is not None and image is not None:
                summary_token_num = 4
            else:
                summary_token_num = 3
            if max_seq_len <= len(prompt) + summary_token_num:
                raise ValueError("The value of max_seq_len is too small, please set a larger value")
            max_content_len = max_seq_len - len(prompt) - summary_token_num
            if len(content) <= max_content_len:
                yield json_line
            else:
                result_list = json_line["result_list"]
                json_lines = []
                accumulate = 0
                while True:
                    cur_result_list = []
                    for result in result_list:
                        if result["end"] - result["start"] > max_content_len:
                            logger.warning(
                                "result['end'] - result ['start'] exceeds max_content_len, which will result in no valid instance being returned"
                            )
                        if (
                            result["start"] + 1 <= max_content_len < result["end"]
                            and result["end"] - result["start"] <= max_content_len
                        ):
                            max_content_len = result["start"]
                            break

                    cur_content = content[:max_content_len]
                    res_content = content[max_content_len:]
                    if boxes is not None and image is not None:
                        cur_boxes = boxes[:max_content_len]
                        res_boxes = boxes[max_content_len:]

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

                    if boxes is not None and image is not None:
                        json_line = {
                            "content": cur_content,
                            "result_list": cur_result_list,
                            "prompt": prompt,
                            "bbox": cur_boxes,
                            "image": image,
                        }
                    else:
                        json_line = {
                            "content": cur_content,
                            "result_list": cur_result_list,
                            "prompt": prompt,
                        }
                    json_lines.append(json_line)

                    for result in result_list:
                        if result["end"] <= 0:
                            break
                        result["start"] -= max_content_len
                        result["end"] -= max_content_len
                    accumulate += max_content_len
                    max_content_len = max_seq_len - len(prompt) - summary_token_num
                    if len(res_content) == 0:
                        break
                    elif len(res_content) < max_content_len:
                        if boxes is not None and image is not None:
                            json_line = {
                                "content": res_content,
                                "result_list": result_list,
                                "prompt": prompt,
                                "bbox": res_boxes,
                                "image": image,
                            }
                        else:
                            json_line = {"content": res_content, "result_list": result_list, "prompt": prompt}

                        json_lines.append(json_line)
                        break
                    else:
                        content = res_content
                        boxes = res_boxes

                for json_line in json_lines:
                    yield json_line


def get_dynamic_max_len(examples, default_max_len: int, dynamic_max_length: List[int]) -> int:
    """get max_length by examples which you can change it by examples in batch"""
    cur_length = len(examples[0]["input_ids"])
    max_length = default_max_len
    for max_length_option in sorted(dynamic_max_length):
        if cur_length <= max_length_option:
            max_length = max_length_option
            break
    return max_length


def convert_example(
    example,
    tokenizer,
    max_seq_len,
    pad_id=1,
    c_sep_id=2,
    summary_token_num=4,
    dynamic_max_length: Optional[List[int]] = None,
):

    content = example["content"]
    prompt = example["prompt"]
    bbox_lines = example.get("bbox", None)
    image_buff_string = example.get("image", None)
    # Text
    if bbox_lines is None or image_buff_string is None:
        if dynamic_max_length is not None:
            temp_encoded_inputs = tokenizer(
                text=[example["prompt"]],
                text_pair=[example["content"]],
                truncation=True,
                max_seq_len=max_seq_len,
                return_attention_mask=True,
                return_position_ids=True,
                return_dict=False,
                return_offsets_mapping=True,
            )
            max_length = get_dynamic_max_len(
                examples=temp_encoded_inputs, default_max_len=max_seq_len, dynamic_max_length=dynamic_max_length
            )
            # always pad to max_length
            encoded_inputs = tokenizer(
                text=[example["prompt"]],
                text_pair=[example["content"]],
                truncation=True,
                max_seq_len=max_length,
                pad_to_max_seq_len=True,
                return_attention_mask=True,
                return_position_ids=True,
                return_dict=False,
                return_offsets_mapping=True,
            )
            max_seq_len = max_length
        else:
            encoded_inputs = tokenizer(
                text=[example["prompt"]],
                text_pair=[example["content"]],
                truncation=True,
                max_seq_len=max_seq_len,
                pad_to_max_seq_len=True,
                return_attention_mask=True,
                return_position_ids=True,
                return_offsets_mapping=True,
                return_dict=False,
            )

        encoded_inputs = encoded_inputs[0]

        inputs_ids = encoded_inputs["input_ids"]
        position_ids = encoded_inputs["position_ids"]
        attention_mask = encoded_inputs["attention_mask"]

        q_sep_index = inputs_ids.index(2, 1)
        c_sep_index = attention_mask.index(0)

        offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"]]

        bias = 0
        for index in range(len(offset_mapping)):
            if index == 0:
                continue
            mapping = offset_mapping[index]
            if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                # bias = index
                bias = offset_mapping[index - 1][-1] + 1

            if mapping[0] == 0 and mapping[1] == 0:
                continue
            offset_mapping[index][0] += bias
            offset_mapping[index][1] += bias

        offset_bias = bias

        bbox_list = [[0, 0, 0, 0] for x in range(len(inputs_ids))]
        token_type_ids = [
            1 if token_index <= q_sep_index or token_index > c_sep_index else 0 for token_index in range(max_seq_len)
        ]
        padded_image = np.zeros([3, 224, 224])

    # Doc
    else:
        inputs_ids = []
        prev_bbox = [-1, -1, -1, -1]
        this_text_line = ""
        q_sep_index = -1
        offset_mapping = []
        last_offset = 0
        for char_index, (char, bbox) in enumerate(zip(content, bbox_lines)):
            if char_index == 0:
                prev_bbox = bbox
                this_text_line = char
                continue

            if all([bbox[x] == prev_bbox[x] for x in range(4)]):
                this_text_line += char
            else:
                offset_mapping, last_offset, q_sep_index, inputs_ids = _encode_doc(
                    tokenizer,
                    offset_mapping,
                    last_offset,
                    prompt,
                    this_text_line,
                    inputs_ids,
                    q_sep_index,
                    max_seq_len,
                )
                this_text_line = char
            prev_bbox = bbox

        if len(this_text_line) > 0:
            offset_mapping, last_offset, q_sep_index, inputs_ids = _encode_doc(
                tokenizer, offset_mapping, last_offset, prompt, this_text_line, inputs_ids, q_sep_index, max_seq_len
            )

        if len(inputs_ids) > max_seq_len:
            inputs_ids = inputs_ids[: (max_seq_len - 1)] + [c_sep_id]
            offset_mapping = offset_mapping[: (max_seq_len - 1)] + [[0, 0]]
        else:
            inputs_ids += [c_sep_id]
            offset_mapping += [[0, 0]]

        offset_bias = offset_mapping[q_sep_index - 1][-1] + 1

        seq_len = len(inputs_ids)
        inputs_ids += [pad_id] * (max_seq_len - seq_len)
        token_type_ids = [1] * (q_sep_index + 1) + [0] * (seq_len - q_sep_index - 1)
        token_type_ids += [pad_id] * (max_seq_len - seq_len)

        bbox_list = _process_bbox(inputs_ids, bbox_lines, offset_mapping, offset_bias)

        offset_mapping += [[0, 0]] * (max_seq_len - seq_len)

        position_ids = list(range(seq_len))

        position_ids = position_ids + [0] * (max_seq_len - seq_len)
        attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)

        image_data = base64.b64decode(image_buff_string.encode("utf8"))
        padded_image = pad_image_data(image_data)

    start_ids = np.array([0.0 for x in range(max_seq_len)], dtype="int64")
    end_ids = np.array([0.0 for x in range(max_seq_len)], dtype="int64")

    for item in example["result_list"]:
        start = map_offset(item["start"] + offset_bias, offset_mapping)
        end = map_offset(item["end"] - 1 + offset_bias, offset_mapping)
        start_ids[start] = 1.0
        end_ids[end] = 1.0

    assert len(inputs_ids) == max_seq_len
    assert len(token_type_ids) == max_seq_len
    assert len(position_ids) == max_seq_len
    assert len(attention_mask) == max_seq_len
    assert len(bbox_list) == max_seq_len
    tokenized_output = {
        "input_ids": inputs_ids,
        "token_type_ids": token_type_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "bbox": bbox_list,
        "image": padded_image,
        "start_positions": start_ids,
        "end_positions": end_ids,
    }
    return tokenized_output


def _process_bbox(tokens, bbox_lines, offset_mapping, offset_bias):
    bbox_list = [[0, 0, 0, 0] for x in range(len(tokens))]

    for index, bbox in enumerate(bbox_lines):
        index_token = map_offset(index + offset_bias, offset_mapping)
        if 0 <= index_token < len(bbox_list):
            bbox_list[index_token] = bbox
    return bbox_list


def _encode_doc(tokenizer, offset_mapping, last_offset, prompt, this_text_line, inputs_ids, q_sep_index, max_seq_len):
    if len(offset_mapping) == 0:
        content_encoded_inputs = tokenizer(
            text=[prompt],
            text_pair=[this_text_line],
            max_seq_len=max_seq_len,
            return_dict=False,
            return_offsets_mapping=True,
        )
        content_encoded_inputs = content_encoded_inputs[0]
        inputs_ids = content_encoded_inputs["input_ids"][:-1]
        sub_offset_mapping = [list(x) for x in content_encoded_inputs["offset_mapping"]]
        q_sep_index = content_encoded_inputs["input_ids"].index(2, 1)

        bias = 0
        for i in range(len(sub_offset_mapping)):
            if i == 0:
                continue
            mapping = sub_offset_mapping[i]
            if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                bias = sub_offset_mapping[i - 1][-1] + 1
            if mapping[0] == 0 and mapping[1] == 0:
                continue
            if mapping == sub_offset_mapping[i - 1]:
                continue
            sub_offset_mapping[i][0] += bias
            sub_offset_mapping[i][1] += bias

        offset_mapping = sub_offset_mapping[:-1]
        last_offset = offset_mapping[-1][-1]
    else:
        content_encoded_inputs = tokenizer(
            text=this_text_line, max_seq_len=max_seq_len, return_dict=False, return_offsets_mapping=True
        )
        inputs_ids += content_encoded_inputs["input_ids"][1:-1]
        sub_offset_mapping = [list(x) for x in content_encoded_inputs["offset_mapping"]]

        for i, sub_list in enumerate(sub_offset_mapping[1:-1]):
            if i == 0:
                org_offset = sub_list[1]
            else:
                if sub_list[0] != org_offset and sub_offset_mapping[1:-1][i - 1] != sub_list:
                    last_offset += 1
                org_offset = sub_list[1]
            offset_mapping += [[last_offset, sub_list[1] - sub_list[0] + last_offset]]
            last_offset = offset_mapping[-1][-1]
    return offset_mapping, last_offset, q_sep_index, inputs_ids
