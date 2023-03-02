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
import re
from io import BytesIO

import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image

from ..metrics import SpanEvaluator
from .image_utils import NormalizeImage, Permute, ResizeImage

resize_func = ResizeImage(target_size=224, interp=1)
norm_func = NormalizeImage(is_channel_first=False, mean=[123.675, 116.280, 103.530], std=[58.395, 57.120, 57.375])
permute_func = Permute(to_bgr=False)


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def pad_image_data(image_data):
    if not image_data:
        image = np.zeros([3, 224, 224])
        return image
    # decode image
    data = np.frombuffer(bytearray(image_data), dtype="uint8")
    image = np.array(Image.open(BytesIO(data)).convert("RGB"))
    sample = {"image": image}
    # resize image
    sample = resize_func(sample)
    # norm image
    sample = norm_func(sample)
    # permute
    sample = permute_func(sample)
    return sample["image"]


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


def get_relation_type_dict(relation_data, schema_lang="ch"):
    def compare(a, b, schema_lang="ch"):
        if schema_lang == "ch":
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
                match = compare(relation_data[i][0], relation_data[j][0], schema_lang=schema_lang)
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
                if schema_lang == "ch":
                    suffix = relation_data[i][0].rsplit("的", 1)[1]
                    suffix = unify_prompt_name(suffix)
                    relation_type = suffix
                else:
                    prefix = relation_data[i][0].split(" of ", 1)[0]
                    prefix = unify_prompt_name(prefix)
                    relation_type = prefix
                relation_type_dict.setdefault(relation_type, []).append(relation_data[i][1])
    return relation_type_dict


def uie_loss_func(outputs, labels):
    criterion = paddle.nn.BCELoss()
    start_ids, end_ids = labels
    start_prob, end_prob = outputs
    start_ids = paddle.cast(start_ids, "float32")
    end_ids = paddle.cast(end_ids, "float32")
    loss_start = criterion(start_prob, start_ids)
    loss_end = criterion(end_prob, end_ids)
    loss = (loss_start + loss_end) / 2.0
    return loss


def compute_metrics(p):
    metric = SpanEvaluator()
    start_prob, end_prob = p.predictions
    start_ids, end_ids = p.label_ids
    metric.reset()

    num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)
    metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    metric.reset()

    return {"precision": precision, "recall": recall, "f1": f1}


class ClosedDomainIEProcessor:
    """Closed Domain IE task data processor"""

    def __init__(self):
        pass

    @staticmethod
    def preprocess_text(
        examples,
        tokenizer=None,
        max_seq_len=512,
        doc_stride=256,
        label_maps=None,
        with_label=True,
    ):
        cnt = 0
        tokenized_examples = []
        for example in examples:
            content = example["text"]

            all_tokens = tokenizer.tokenize(content)
            all_offset_mapping = tokenizer.get_offset_mapping(content)

            start_offset = 0
            doc_spans = []
            max_tokens = max_seq_len - 2
            while start_offset < len(all_tokens):
                length = len(all_tokens) - start_offset
                if length > max_tokens:
                    length = max_tokens
                doc_spans.append({"start": start_offset, "length": length})
                if start_offset + length == len(all_tokens):
                    break
                start_offset += min(length, doc_stride, max_tokens)

            for doc_span in doc_spans:
                tokens = [tokenizer.cls_token]
                offset_mapping = [(0, 0)]
                doc_start = doc_span["start"]
                doc_end = doc_span["start"] + doc_span["length"] - 1
                text_offset = all_offset_mapping[doc_start][0]
                text_length = text_offset + all_offset_mapping[doc_end][1]

                for i in range(doc_span["length"]):
                    split_org_index = doc_span["start"] + i
                    tokens.append(all_tokens[split_org_index])
                    offset_mapping.append(
                        (
                            all_offset_mapping[doc_start + i][0] - text_offset,
                            all_offset_mapping[doc_start + i][1] - text_offset,
                        )
                    )

                tokens.append(tokenizer.sep_token)
                offset_mapping.append((0, 0))

                input_mask = [1] * len(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                input_list = {
                    "input_ids": input_ids,
                    "attention_mask": input_mask,
                    "offset_mapping": offset_mapping,
                    "text": content[text_offset:text_length],
                    "doc_id": cnt,
                    "text_offset": text_offset,
                }

                cnt += 1

                if with_label:
                    entity_labels = []
                    for e in example["entity_list"]:
                        _start, _end = e["start_index"], e["start_index"] + len(e["text"]) - 1
                        start = map_offset(_start, all_offset_mapping)
                        end = map_offset(_end, all_offset_mapping)
                        if not (start >= doc_start and end <= doc_end):
                            continue
                        if start == -1 or end == -1:
                            continue
                        label = label_maps["entity_label2id"][e["type"]]
                        entity_labels.append([label, start - doc_start + 1, end - doc_start + 1])

                    relation_labels = []
                    for r in example["spo_list"]:
                        _sh, _oh = r["subject_start_index"], r["object_start_index"]
                        _st, _ot = _sh + len(r["subject"]) - 1, _oh + len(r["object"]) - 1
                        sh = map_offset(_sh, all_offset_mapping)
                        st = map_offset(_st, all_offset_mapping)
                        oh = map_offset(_oh, all_offset_mapping)
                        ot = map_offset(_ot, all_offset_mapping)
                        if not (sh >= doc_start and st <= doc_end) or not (oh >= doc_start and ot <= doc_end):
                            continue
                        if sh == -1 or st == -1 or oh == -1 or ot == -1:
                            continue
                        p = label_maps["relation_label2id"][r["predicate"]]
                        relation_labels.append(
                            [
                                sh - doc_start + 1,
                                st - doc_start + 1,
                                p,
                                oh - doc_start + 1,
                                ot - doc_start + 1,
                            ]
                        )

                    input_list["labels"] = {"entity_labels": entity_labels, "relation_labels": relation_labels}
                tokenized_examples.append(input_list)
        return tokenized_examples

    @staticmethod
    def preprocess_doc(
        examples,
        tokenizer=None,
        max_seq_len=512,
        doc_stride=256,
        label_maps=None,
        with_label=True,
    ):
        def _process_bbox(tokens, bbox_lines, offset_mapping, offset_bias):
            bbox_list = [[0, 0, 0, 0] for x in range(len(tokens))]

            for index, bbox in enumerate(bbox_lines):
                index_token = map_offset(index + offset_bias, offset_mapping)
                if 0 <= index_token < len(bbox_list):
                    bbox_list[index_token] = bbox
            return bbox_list

        cnt = 0
        tokenized_examples = []
        for example in examples:
            content = example["text"]
            bbox_lines = example["bbox"]
            image_buff_string = example["image"]

            image_data = base64.b64decode(image_buff_string.encode("utf8"))
            padded_image = pad_image_data(image_data)

            all_doc_tokens = []
            all_offset_mapping = []
            last_offset = 0
            for char_index, (char, bbox) in enumerate(zip(content, bbox_lines)):
                if char_index == 0:
                    prev_bbox = bbox
                    this_text_line = char
                    continue

                if all(bbox[x] == prev_bbox[x] for x in range(4)):
                    this_text_line += char
                else:
                    cur_offset_mapping = tokenizer.get_offset_mapping(this_text_line)
                    for i, sub_list in enumerate(cur_offset_mapping):
                        if i == 0:
                            org_offset = sub_list[1]
                        else:
                            if sub_list[0] != org_offset:
                                last_offset += 1
                            org_offset = sub_list[1]
                        all_offset_mapping += [[last_offset, sub_list[1] - sub_list[0] + last_offset]]
                        last_offset = all_offset_mapping[-1][-1]
                    all_doc_tokens += tokenizer.tokenize(this_text_line)
                    this_text_line = char
                prev_bbox = bbox
            if len(this_text_line) > 0:
                cur_offset_mapping = tokenizer.get_offset_mapping(this_text_line)
                for i, sub_list in enumerate(cur_offset_mapping):
                    if i == 0:
                        org_offset = sub_list[1]
                    else:
                        if sub_list[0] != org_offset:
                            last_offset += 1
                        org_offset = sub_list[1]
                    all_offset_mapping += [[last_offset, sub_list[1] - sub_list[0] + last_offset]]
                    last_offset = all_offset_mapping[-1][-1]
                all_doc_tokens += tokenizer.tokenize(this_text_line)

            all_doc_token_boxes = _process_bbox(all_doc_tokens, bbox_lines, all_offset_mapping, 0)

            start_offset = 0
            doc_spans = []
            max_tokens_for_doc = max_seq_len - 2
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append({"start": start_offset, "length": length})
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride, max_tokens_for_doc)

            for doc_span in doc_spans:
                tokens = [tokenizer.cls_token]
                token_boxes = [[0, 0, 0, 0]]
                offset_mapping = [(0, 0)]
                doc_start = doc_span["start"]
                doc_end = doc_span["start"] + doc_span["length"] - 1
                text_offset = all_offset_mapping[doc_start][0]
                text_length = text_offset + all_offset_mapping[doc_end][1]

                for i in range(doc_span["length"]):
                    split_org_index = doc_span["start"] + i
                    tokens.append(all_doc_tokens[split_org_index])
                    token_boxes.append(all_doc_token_boxes[split_org_index])
                    offset_mapping.append(
                        (
                            all_offset_mapping[doc_start + i][0] - text_offset,
                            all_offset_mapping[doc_start + i][1] - text_offset,
                        )
                    )

                tokens.append(tokenizer.sep_token)
                token_boxes.append([0, 0, 0, 0])
                offset_mapping.append((0, 0))

                input_mask = [1] * len(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                input_list = {
                    "input_ids": input_ids,
                    "attention_mask": input_mask,
                    "bbox": token_boxes,
                    "image": padded_image,
                    "offset_mapping": offset_mapping,
                    "text": content[text_offset:text_length],
                    "doc_id": cnt,
                    "text_offset": text_offset,
                }

                cnt += 1

                if with_label:
                    entity_labels = []
                    for e in example["entity_list"]:
                        _start, _end = e["start_index"], e["start_index"] + len(e["text"]) - 1
                        start = map_offset(_start, all_offset_mapping)
                        end = map_offset(_end, all_offset_mapping)
                        if not (start >= doc_start and end <= doc_end):
                            continue
                        if start == -1 or end == -1:
                            continue
                        label = label_maps["entity_label2id"][e["type"]]
                        entity_labels.append([label, start - doc_start + 1, end - doc_start + 1])

                    relation_labels = []
                    for r in example["spo_list"]:
                        _sh, _oh = r["subject_start_index"], r["object_start_index"]
                        _st, _ot = _sh + len(r["subject"]) - 1, _oh + len(r["object"]) - 1
                        sh = map_offset(_sh, all_offset_mapping)
                        st = map_offset(_st, all_offset_mapping)
                        oh = map_offset(_oh, all_offset_mapping)
                        ot = map_offset(_ot, all_offset_mapping)
                        if not (sh >= doc_start and st <= doc_end) or not (oh >= doc_start and ot <= doc_end):
                            continue
                        if sh == -1 or st == -1 or oh == -1 or ot == -1:
                            continue
                        p = label_maps["relation_label2id"][r["predicate"]]
                        relation_labels.append(
                            [
                                sh - doc_start + 1,
                                st - doc_start + 1,
                                p,
                                oh - doc_start + 1,
                                ot - doc_start + 1,
                            ]
                        )

                    input_list["labels"] = {"entity_labels": entity_labels, "relation_labels": relation_labels}
                tokenized_examples.append(input_list)
        return tokenized_examples

    @staticmethod
    def postprocess(
        logits,
        all_preds,
        doc_ids,
        cur_doc_id,
        offset_mappings,
        texts,
        text_offsets,
        label_maps,
        with_prob=False,
    ):
        if len(logits) == 1:
            batch_ent_outputs = []
            for entity_output, offset_mapping, text, text_offset in zip(
                logits[0].numpy(),
                offset_mappings,
                texts,
                text_offsets,
            ):
                entity_output[:, [0, -1]] -= np.inf
                entity_output[:, :, [0, -1]] -= np.inf
                if with_prob:
                    entity_probs = F.softmax(paddle.to_tensor(entity_output, dtype="float64"), axis=1).numpy()

                ent_list = []
                for l, start, end in zip(*np.where(entity_output > 0.0)):
                    _start, _end = (offset_mapping[start][0], offset_mapping[end][-1])
                    ent = {
                        "text": text[_start:_end],
                        "type": label_maps["entity_id2label"][l],
                        "start_index": _start + text_offset,
                    }
                    if with_prob:
                        ent_prob = entity_probs[l, start, end]
                        ent["probability"] = ent_prob
                    ent_list.append(ent)
                batch_ent_outputs.append(ent_list)
            for doc_id, batch_ent_output in zip(doc_ids, batch_ent_outputs):
                if doc_id == cur_doc_id:
                    for ent_pred in batch_ent_output:
                        if ent_pred not in all_preds["entity_preds"][doc_id]:
                            all_preds["entity_preds"][doc_id].append(ent_pred)
                else:
                    all_preds["entity_preds"].append(batch_ent_output)
                    cur_doc_id = doc_id
        else:
            batch_ent_outputs = []
            batch_rel_outputs = []
            for entity_output, head_output, tail_output, offset_mapping, text, text_offset in zip(
                logits[0].numpy(),
                logits[1].numpy(),
                logits[2].numpy(),
                offset_mappings,
                texts,
                text_offsets,
            ):
                entity_output[:, [0, -1]] -= np.inf
                entity_output[:, :, [0, -1]] -= np.inf
                if with_prob:
                    entity_probs = F.softmax(paddle.to_tensor(entity_output, dtype="float64"), axis=1).numpy()
                    head_probs = F.softmax(paddle.to_tensor(head_output, dtype="float64"), axis=1).numpy()
                    tail_probs = F.softmax(paddle.to_tensor(tail_output, dtype="float64"), axis=1).numpy()

                ents = set()
                ent_list = []
                for l, start, end in zip(*np.where(entity_output > 0.0)):
                    ents.add((start, end))
                    _start, _end = (offset_mapping[start][0], offset_mapping[end][-1])
                    ent = {
                        "text": text[_start:_end],
                        "type": label_maps["entity_id2label"][l],
                        "start_index": _start + text_offset,
                    }
                    if with_prob:
                        ent_prob = entity_probs[l, start, end]
                        ent["probability"] = ent_prob
                    ent_list.append(ent)
                batch_ent_outputs.append(ent_list)

                rel_list = []
                for sh, st in ents:
                    for oh, ot in ents:
                        p1s = np.where(head_output[:, sh, oh] > 0.0)[0]
                        p2s = np.where(tail_output[:, st, ot] > 0.0)[0]
                        ps = set(p1s) & set(p2s)
                        for p in ps:
                            rel = {
                                "subject": text[offset_mapping[sh][0] : offset_mapping[st][1]],
                                "predicate": label_maps["relation_id2label"][p],
                                "object": text[offset_mapping[oh][0] : offset_mapping[ot][1]],
                                "subject_start_index": offset_mapping[sh][0] + text_offset,
                                "object_start_index": offset_mapping[oh][0] + text_offset,
                            }
                            if with_prob:
                                rel_prob = head_probs[p, sh, oh] * tail_probs[p, st, ot]
                                rel["probability"] = rel_prob
                            rel_list.append(rel)
                batch_rel_outputs.append(rel_list)
            for doc_id, batch_ent_output, batch_rel_output in zip(doc_ids, batch_ent_outputs, batch_rel_outputs):
                if doc_id == cur_doc_id:
                    for ent_pred in batch_ent_output:
                        if ent_pred not in all_preds["entity_preds"][doc_id]:
                            all_preds["entity_preds"][doc_id].append(ent_pred)
                    for rel_pred in batch_rel_output:
                        if rel_pred not in all_preds["spo_preds"][doc_id]:
                            all_preds["spo_preds"][doc_id].append(rel_pred)
                else:
                    all_preds["entity_preds"].append(batch_ent_output)
                    all_preds["spo_preds"].append(batch_rel_output)
                    cur_doc_id = doc_id
        return all_preds, cur_doc_id
