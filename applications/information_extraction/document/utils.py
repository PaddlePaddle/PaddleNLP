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
import re
import numpy as np
import json
import base64
from io import BytesIO
from PIL import Image

import paddle
from paddlenlp.utils.image_utils import ResizeImage, NormalizeImage, Permute
from paddlenlp.metrics import SpanEvaluator

resize_func = ResizeImage(target_size=224, interp=1)
norm_func = NormalizeImage(is_channel_first=False,
                           mean=[123.675, 116.280, 103.530],
                           std=[58.395, 57.120, 57.375])
permute_func = Permute(to_bgr=False)

MODEL_MAP = {
    "uie-x-base": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_x_base_v0.1/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_x_base/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_x_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_x_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_x_base/tokenizer_config.json",
            "sentencepiece.bpe.model":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_x_base/sentencepiece.bpe.model"
        },
    }
}


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
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            content = json_line['content'].strip()
            prompt = json_line['prompt']
            bboxes = json_line.get("bboxes", None)
            image = json_line.get("image", None)
            # Model Input is aslike: [CLS] prompt [SEP] [SEP] text [SEP] for UIE-X
            # It include three summary tokens.
            if max_seq_len <= len(prompt) + 3:
                raise ValueError(
                    "The value of max_seq_len is too small, please set a larger value"
                )
            max_content_len = max_seq_len - len(prompt) - 3
            if len(content) <= max_content_len:
                yield json_line
            else:
                result_list = json_line['result_list']
                json_lines = []
                accumulate = 0
                while True:
                    cur_result_list = []
                    for result in result_list:
                        if result['end'] - result['start'] > max_content_len:
                            logger.warning(
                                "result['end'] - result ['start'] exceeds max_content_len, which will result in no valid instance being returned"
                            )
                        if result['start'] + 1 <= max_content_len < result[
                                'end'] and result['end'] - result[
                                    'start'] <= max_content_len:
                            max_content_len = result['start']
                            break

                    cur_content = content[:max_content_len]
                    res_content = content[max_content_len:]

                    while True:
                        if len(result_list) == 0:
                            break
                        elif result_list[0]['end'] <= max_content_len:
                            if result_list[0]['end'] > 0:
                                cur_result = result_list.pop(0)
                                cur_result_list.append(cur_result)
                            else:
                                cur_result_list = [
                                    result for result in result_list
                                ]
                                break
                        else:
                            break
                    if bboxes is not None and image is not None:
                        json_line = {
                            'content': cur_content,
                            'result_list': cur_result_list,
                            'prompt': prompt,
                            'bboxes': bboxes,
                            'image': image,
                        }
                    else:
                        json_line = {
                            'content': cur_content,
                            'result_list': cur_result_list,
                            'prompt': prompt,
                        }

                    json_lines.append(json_line)

                    for result in result_list:
                        if result['end'] <= 0:
                            break
                        result['start'] -= max_content_len
                        result['end'] -= max_content_len
                    accumulate += max_content_len
                    max_content_len = max_seq_len - len(prompt) - 3
                    if len(res_content) == 0:
                        break
                    elif len(res_content) < max_content_len:
                        json_line = {
                            'content': res_content,
                            'result_list': result_list,
                            'prompt': prompt
                        }
                        json_lines.append(json_line)
                        break
                    else:
                        content = res_content

                for json_line in json_lines:
                    yield json_line


def convert_example(example,
                    tokenizer,
                    max_seq_len,
                    pad_id=1,
                    c_sep_id=2,
                    summary_token_num=4):

    content = example["content"]
    prompt = example["prompt"]
    bbox_lines = example.get("bboxes", None)
    image_buff_string = example.get("image", None)
    # Text
    if bbox_lines is None:
        encoded_inputs = tokenizer(text=[example["prompt"]],
                                   text_pair=[example["content"]],
                                   truncation=True,
                                   max_seq_len=max_seq_len,
                                   pad_to_max_seq_len=True,
                                   return_attention_mask=True,
                                   return_position_ids=True,
                                   return_offsets_mapping=True,
                                   return_dict=False)

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
            1 if token_index <= q_sep_index or token_index > c_sep_index else 0
            for token_index in range(max_seq_len)
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
                    tokenizer, offset_mapping, last_offset, prompt,
                    this_text_line, inputs_ids, q_sep_index, max_seq_len)
                this_text_line = char
            prev_bbox = bbox

        if len(this_text_line) > 0:
            offset_mapping, last_offset, q_sep_index, inputs_ids = _encode_doc(
                tokenizer, offset_mapping, last_offset, prompt, this_text_line,
                inputs_ids, q_sep_index, max_seq_len)

        if len(inputs_ids) > max_seq_len:
            inputs_ids = inputs_ids[:(max_seq_len - 1)] + [c_sep_id]
            offset_mapping = offset_mapping[:(max_seq_len - 1)] + [[0, 0]]
        else:
            inputs_ids += [c_sep_id]
            offset_mapping += [[0, 0]]

        offset_bias = offset_mapping[q_sep_index - 1][-1] + 1

        seq_len = len(inputs_ids)
        inputs_ids += [pad_id] * (max_seq_len - seq_len)
        token_type_ids = [1] * (q_sep_index + 1) + [0] * (seq_len -
                                                          q_sep_index - 1)
        token_type_ids += [pad_id] * (max_seq_len - seq_len)

        bbox_list = _process_bbox(inputs_ids, bbox_lines, offset_mapping,
                                  offset_bias)

        offset_mapping += [[0, 0]] * (max_seq_len - seq_len)

        # Reindex the text
        text_start_idx = offset_mapping[1:].index([0, 0
                                                   ]) + summary_token_num - 1
        for idx in range(text_start_idx, max_seq_len):
            offset_mapping[idx][0] -= offset_bias
            offset_mapping[idx][1] -= offset_bias

        position_ids = list(range(seq_len))

        position_ids = position_ids + [0] * (max_seq_len - seq_len)
        attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)

        image_data = base64.b64decode(image_buff_string.encode("utf8"))
        padded_image = _pad_image_data(image_data)

    start_ids = np.array([0.0 for x in range(max_seq_len + 49)], dtype="int64")
    end_ids = np.array([0.0 for x in range(max_seq_len + 49)], dtype="int64")

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
    prev_bbox = [0, 0, 0, 0]

    for index, bbox in enumerate(bbox_lines):
        index_token = map_offset(index + offset_bias, offset_mapping)
        if 0 <= index_token < len(bbox_list):
            bbox_list[index_token] = bbox
        prev_bbox = bbox
    return bbox_list


def _pad_image_data(image_data):
    if not image_data:
        image = np.zeros([3, 224, 224])
        return image
    # decode image
    data = np.frombuffer(bytearray(image_data), dtype="uint8")
    image = np.array(Image.open(BytesIO(data)).convert('RGB'))
    sample = {"image": image}
    # resize image
    sample = resize_func(sample)
    # norm image
    sample = norm_func(sample)
    # permute
    sample = permute_func(sample)
    return sample['image']


def _encode_doc(tokenizer, offset_mapping, last_offset, prompt, this_text_line,
                inputs_ids, q_sep_index, max_seq_len):
    if len(offset_mapping) == 0:
        content_encoded_inputs = tokenizer(text=[prompt],
                                           text_pair=[this_text_line],
                                           max_seq_len=max_seq_len,
                                           return_dict=False,
                                           return_offsets_mapping=True)
        content_encoded_inputs = content_encoded_inputs[0]
        inputs_ids = content_encoded_inputs["input_ids"][:-1]
        sub_offset_mapping = [
            list(x) for x in content_encoded_inputs["offset_mapping"]
        ]
        q_sep_index = content_encoded_inputs["input_ids"].index(2, 1)

        bias = 0
        for index in range(len(sub_offset_mapping)):
            if index == 0:
                continue
            mapping = sub_offset_mapping[index]
            if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                bias = sub_offset_mapping[index - 1][-1] + 1
            if mapping[0] == 0 and mapping[1] == 0:
                continue
            sub_offset_mapping[index][0] += bias
            sub_offset_mapping[index][1] += bias

        offset_mapping = sub_offset_mapping[:-1]
        last_offset = offset_mapping[-1][-1]
    else:
        content_encoded_inputs = tokenizer(text=this_text_line,
                                           max_seq_len=max_seq_len,
                                           return_dict=False,
                                           return_offsets_mapping=True)
        inputs_ids += content_encoded_inputs["input_ids"][1:-1]
        sub_offset_mapping = [
            list(x) for x in content_encoded_inputs["offset_mapping"]
        ]

        for sub_list in sub_offset_mapping[1:-1]:
            offset_mapping += [[
                last_offset, sub_list[1] - sub_list[0] + last_offset
            ]]
            last_offset = offset_mapping[-1][-1]
    return offset_mapping, last_offset, q_sep_index, inputs_ids


def unify_prompt_name(prompt):
    # The classification labels are shuffled during finetuning, so they need
    # to be unified during evaluation.
    if re.search(r'\[.*?\]$', prompt):
        prompt_prefix = prompt[:prompt.find("[", 1)]
        cls_options = re.search(r'\[.*?\]$', prompt).group()[1:-1].split(",")
        cls_options = sorted(list(set(cls_options)))
        cls_options = ",".join(cls_options)
        prompt = prompt_prefix + "[" + cls_options + "]"
        return prompt
    return prompt


def get_relation_type_dict(relation_data, schema_lang="ch"):

    def compare(a, b, schema_lang="ch"):
        if schema_lang == "ch":
            a = a[::-1]
            b = b[::-1]

        res = ''
        for i in range(min(len(a), len(b))):
            if a[i] == b[i]:
                res += a[i]
            else:
                break
        if res == "":
            return res
        if schema_lang == "ch" and res[::-1][0] == "的":
            return res[::-1][1:]
        elif schema_lang == "en" and res[-3:] == " of":
            return res[:-3]
        return ""

    relation_type_dict = {}
    added_list = []
    for i in range(len(relation_data)):
        added = False
        if relation_data[i][0] not in added_list:
            for j in range(i + 1, len(relation_data)):
                match = compare(relation_data[i][0],
                                relation_data[j][0],
                                schema_lang=schema_lang)
                if match != "":
                    match = unify_prompt_name(match)
                    if relation_data[i][0] not in added_list:
                        added_list.append(relation_data[i][0])
                        relation_type_dict.setdefault(match, []).append(
                            relation_data[i][1])
                    added_list.append(relation_data[j][0])
                    relation_type_dict.setdefault(match, []).append(
                        relation_data[j][1])
                    added = True
            if not added:
                added_list.append(relation_data[i][0])
                if schema_lang == "ch":
                    suffix = relation_data[i][0].rsplit("的", 1)[1]
                    suffix = unify_prompt_name(suffix)
                    relation_type = suffix
                else:
                    prefix = relation_data[i][0].split(" of ", 1)[0]
                    prefix = unify_prompt_name(prefix)
                    relation_type = prefix
                relation_type_dict.setdefault(relation_type,
                                              []).append(relation_data[i][1])
    return relation_type_dict


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

    shuffle = True if mode == 'train' else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle)
    dataloader = paddle.io.DataLoader(dataset,
                                      batch_sampler=sampler,
                                      return_list=True)
    return dataloader


def uie_loss_func(outputs, labels):
    criterion = paddle.nn.BCELoss()
    start_ids, end_ids = labels
    start_prob, end_prob = outputs
    start_ids = paddle.cast(start_ids, 'float32')
    end_ids = paddle.cast(end_ids, 'float32')
    loss_start = criterion(start_prob, start_ids)
    loss_end = criterion(end_prob, end_ids)
    loss = (loss_start + loss_end) / 2.0
    return loss


def compute_metrics(p):
    metric = SpanEvaluator()
    start_prob, end_prob = p.predictions
    start_ids, end_ids = p.label_ids
    metric.reset()

    num_correct, num_infer, num_label = metric.compute(start_prob, end_prob,
                                                       start_ids, end_ids)
    metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    metric.reset()
    return {"precision": precision, "recall": recall, "f1": f1}
