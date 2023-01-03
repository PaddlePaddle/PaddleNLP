# encoding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The HuggingFace Inc. team.
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
import collections
import hashlib
import random

import cv2
import datasets
import editdistance
import numpy as np
import scipy
import six
from PIL import Image
from seqeval.metrics.sequence_labeling import get_entities

from paddlenlp.trainer import EvalPrediction


def _get_md5(string):
    """Get md5 value for string"""
    hl = hashlib.md5()
    hl.update(string.encode(encoding="utf-8"))
    return hl.hexdigest()


def _decode_image(im_base64):
    """Decode image"""
    if im_base64 is not None:
        image = base64.b64decode(im_base64.encode("utf-8"))
        im = np.frombuffer(image, dtype="uint8")
        im = cv2.imdecode(im, 1)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im
    else:
        return np.zeros([224, 224, 3], dtype=np.uint8)


def _resize_image(
    im,
    target_size=0,
    interp=cv2.INTER_LINEAR,
    resize_box=False,
):
    """Resize the image numpy."""
    if not isinstance(im, np.ndarray):
        raise TypeError("image type is not numpy.")
    if len(im.shape) != 3:
        raise ValueError("image is not 3-dimensional.")
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    if isinstance(target_size, list):
        # Case for multi-scale training
        selected_size = random.choice(target_size)
    else:
        selected_size = target_size
    if float(im_size_min) == 0:
        raise ZeroDivisionError("min size of image is 0")
    resize_w = selected_size
    resize_h = selected_size

    im = im.astype("uint8")
    im = Image.fromarray(im)
    im = im.resize((int(resize_w), int(resize_h)), interp)
    im = np.array(im)
    return im


def _scale_same_as_image(boxes, width, height, target_size):
    """
    Scale the bounding box of each character within maximum boundary.
    """
    scale_x = target_size / width
    scale_y = target_size / height

    new_boxes = [
        [
            int(max(0, min(box[0] * scale_x, target_size - 1))),
            int(max(0, min(box[1] * scale_y, target_size - 1))),
            int(max(0, min(box[2] * scale_x, target_size - 1))),
            int(max(0, min(box[3] * scale_y, target_size - 1))),
        ]
        for box in boxes
    ]
    return new_boxes, (scale_x, scale_y)


def _permute(im, channel_first=True, to_bgr=False):
    """Permute"""
    if channel_first:
        im = np.swapaxes(im, 1, 2)
        im = np.swapaxes(im, 1, 0)
    if to_bgr:
        im = im[[2, 1, 0], :, :]
    return im


def _str2im(
    im_base64,
    target_size=224,
):
    # Step1: decode image
    origin_im = _decode_image(im_base64)
    # Step2: resize image
    im = _resize_image(origin_im, target_size=target_size, interp=1, resize_box=False)
    return im, origin_im


def get_label_ld(qas, scheme="bio"):
    if scheme == "cls":
        unique_labels = set()
        for qa in qas:
            label_text = qa["answers"][0]["text"][0]
            unique_labels.add(label_text)

        label_list = list(unique_labels)
        label_list.sort()
    else:
        unique_keys = set()
        for qa in qas:
            for key in qa["question"]:
                unique_keys.add(key)
        key_list = list(unique_keys)
        key_list.sort()

        label_list = ["O"]
        for key in key_list:
            if scheme == "bio":
                label_list.append("B-" + key)
                label_list.append("I-" + key)
            elif scheme == "bioes":
                label_list.append("B-" + key)
                label_list.append("I-" + key)
                label_list.append("E-" + key)
                label_list.append("S-" + key)
            else:
                raise NotImplementedError

    label_dict = {l: i for i, l in enumerate(label_list)}
    return label_list, label_dict


def anls_score(labels, predictions):
    def get_anls(prediction, ground_truth):
        prediction = prediction.strip().lower()
        ground_truth = ground_truth.strip().lower()
        iou = 1 - editdistance.eval(prediction, ground_truth) / max(len(prediction), len(ground_truth), 1e-5)
        anls = iou if iou >= 0.5 else 0.0
        return anls

    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        if len(ground_truths) == 0:
            return 0
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    anls, total = 0, 0
    assert labels.keys() == predictions.keys()
    for _id in labels.keys():
        assert labels[_id].keys() == predictions[_id].keys()
        for question in labels[_id]:
            if len(predictions[_id][question]) > 0:
                prediction_text = predictions[_id][question][0]
            else:
                prediction_text = ""
            ground_truths = labels[_id][question]
            total += 1
            anls += metric_max_over_ground_truths(get_anls, prediction_text, ground_truths)

    anls = 100.0 * anls / total
    return {"anls": anls}


class PreProcessor:
    def __init__(self):
        pass

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span["start"] + doc_span["length"] - 1
            if position < doc_span["start"]:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span["start"]
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index
        return cur_span_index == best_span_index

    def preprocess_ner(
        self,
        examples,
        tokenizer=None,
        label_dict=None,
        max_seq_length=512,
        doc_stride=128,
        target_size=1000,
        max_size=1000,
        other_label="O",
        ignore_label_id=-100,
        use_segment_box=False,
        preprocessing_num_workers=1,
        scheme="bio",
        lang="en",
    ):
        """
        Adapt to NER task.
        """
        tokenized_examples = collections.defaultdict(list)
        for example_idx, example_text in enumerate(examples["text"]):
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            all_doc_token_boxes = []
            all_doc_token_labels = []
            cls_token_box = [0, 0, 0, 0]
            sep_token_box = [0, 0, 0, 0]
            pad_token_box = [0, 0, 0, 0]

            im_base64 = examples["image"][example_idx]
            image, _ = _str2im(im_base64)
            image = _permute(image, to_bgr=False)

            if use_segment_box:
                bboxes = examples["segment_bbox"][example_idx]
            else:
                bboxes = examples["bbox"][example_idx]
            bboxes, _s = _scale_same_as_image(
                bboxes,
                examples["width"][example_idx],
                examples["height"][example_idx],
                target_size,
            )

            qas = examples["qas"][example_idx]
            orig_labels = [other_label] * len(example_text)
            for question, answers in zip(qas["question"], qas["answers"]):
                for answer_start, answer_end in zip(
                    answers["answer_start"],
                    answers["answer_end"],
                ):
                    if scheme == "bio":
                        orig_labels[answer_start] = "B-" + question
                        orig_labels[answer_start + 1 : answer_end] = ["I-" + question] * (
                            answer_end - answer_start - 1
                        )
                    elif scheme == "bioes":
                        orig_labels[answer_start] = "B-" + question
                        if answer_end - answer_start - 1 > 1:
                            orig_labels[answer_end - 1] = "E-" + question
                            orig_labels[answer_start + 1 : answer_end - 1] = ["I-" + question] * (
                                answer_end - answer_start - 2
                            )
                        else:
                            orig_labels[answer_start] = "S-" + question

            for (i, token) in enumerate(example_text):
                orig_to_tok_index.append(len(all_doc_tokens))
                if lang == "ch":
                    sub_tokens = tokenizer.tokenize("&" + token)[1:]
                else:
                    sub_tokens = tokenizer.tokenize(token)
                label = orig_labels[i]
                box = bboxes[i]
                for j, sub_token in enumerate(sub_tokens):
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
                    all_doc_token_boxes.append(box)
                    if "B-" in label[:2]:
                        if j == 0:
                            all_doc_token_labels.append(label)
                        else:
                            all_doc_token_labels.append("I-" + label[2:])
                    elif "E-" in label[:2]:
                        if len(sub_tokens) - 1 == j:
                            all_doc_token_labels.append("E-" + label[2:])
                        else:
                            all_doc_token_labels.append("I-" + label[2:])
                    elif "S-" in label[:2]:
                        if len(sub_tokens) == 1:
                            all_doc_token_labels.append(label)
                        else:
                            if j == 0:
                                all_doc_token_labels.append("B-" + label[2:])
                            elif len(sub_tokens) - 1 == j:
                                all_doc_token_labels.append("E-" + label[2:])
                            else:
                                all_doc_token_labels.append("I-" + label[2:])
                    else:
                        all_doc_token_labels.append(label)

            max_tokens_for_doc = max_seq_length - 2
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append({"start": start_offset, "length": length})
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride, max_tokens_for_doc)

            for (doc_span_index, doc_span) in enumerate(doc_spans):

                tokens = []
                token_boxes = []
                token_label_ids = []
                token_to_orig_map = {}
                token_is_max_context = {}
                sentence_ids = []
                tokens.append(tokenizer.cls_token)
                token_boxes.append(cls_token_box)
                token_label_ids.append(ignore_label_id)
                sentence_ids.append(0)

                for i in range(doc_span["length"]):
                    split_token_index = doc_span["start"] + i
                    token_to_orig_map[str(len(tokens))] = tok_to_orig_index[split_token_index]

                    is_max_context = self._check_is_max_context(doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[str(len(tokens))] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    token_boxes.append(all_doc_token_boxes[split_token_index])
                    token_label_ids.append(label_dict[all_doc_token_labels[split_token_index]])
                    sentence_ids.append(0)

                token_is_max_context[str(len(tokens))] = False
                token_to_orig_map[str(len(tokens))] = -1
                tokens.append(tokenizer.sep_token)
                token_boxes.append(sep_token_box)
                token_label_ids.append(ignore_label_id)
                sentence_ids.append(0)
                input_mask = [1] * len(tokens)

                while len(tokens) < max_seq_length:
                    token_is_max_context[str(len(tokens))] = False
                    token_to_orig_map[str(len(tokens))] = -1
                    tokens.append(tokenizer.pad_token)
                    input_mask.append(0)
                    sentence_ids.append(0)
                    token_boxes.append(pad_token_box)
                    token_label_ids.append(ignore_label_id)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                position_ids = list(range(len(input_ids)))

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(token_boxes) == max_seq_length
                assert len(sentence_ids) == max_seq_length
                assert len(token_label_ids) == max_seq_length

                feature_id = examples["name"][example_idx] + "__" + str(examples["page_no"][example_idx])
                tokenized_examples["id"].append(feature_id)
                tokenized_examples["tokens"].append(tokens)
                tokenized_examples["input_ids"].append(input_ids)
                tokenized_examples["attention_mask"].append(input_mask)
                tokenized_examples["token_type_ids"].append(sentence_ids)
                tokenized_examples["bbox"].append(token_boxes)
                tokenized_examples["position_ids"].append(position_ids)
                tokenized_examples["image"].append(image)
                # tokenized_examples["orig_image"].append(origin_image)
                tokenized_examples["labels"].append(token_label_ids)
                tokenized_examples["token_is_max_context"].append(token_is_max_context)
                tokenized_examples["token_to_orig_map"].append(token_to_orig_map)
        return tokenized_examples

    def _improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        tok_answer_text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(orig_answer_text))
        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = tokenizer.convert_tokens_to_string(doc_tokens[new_start : (new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def preprocess_mrc(
        self,
        examples,
        tokenizer=None,
        max_seq_length=512,
        doc_stride=128,
        max_query_length=64,
        target_size=1000,
        max_size=1000,
        use_segment_box=False,
        preprocessing_num_workers=1,
        is_training=False,
        lang="en",
    ):
        """
        Adapt to MRC task.
        """

        tokenized_examples = collections.defaultdict(list)
        for example_idx, example_text in enumerate(examples["text"]):
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            all_doc_token_boxes = []
            cls_token_box = [0, 0, 0, 0]
            sep_token_box = [0, 0, 0, 0]
            pad_token_box = [0, 0, 0, 0]
            query_token_box = [0, 0, 0, 0]

            im_base64 = examples["image"][example_idx]
            image, _ = _str2im(im_base64)
            image = _permute(image, to_bgr=False)

            if use_segment_box:
                bboxes = examples["segment_bbox"][example_idx]
            else:
                bboxes = examples["bbox"][example_idx]
            bboxes, _s = _scale_same_as_image(
                bboxes,
                examples["width"][example_idx],
                examples["height"][example_idx],
                target_size,
            )

            for (i, token) in enumerate(example_text):
                orig_to_tok_index.append(len(all_doc_tokens))
                if lang == "ch":
                    sub_tokens = tokenizer.tokenize("&" + token)[1:]
                else:
                    sub_tokens = tokenizer.tokenize(token)
                box = bboxes[i]
                for j, sub_token in enumerate(sub_tokens):
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
                    all_doc_token_boxes.append(box)

            qas = examples["qas"][example_idx]
            for qid, question, answers in zip(qas["question_id"], qas["question"], qas["answers"]):

                query_tokens = tokenizer.tokenize(
                    question, add_special_tokens=False, truncation=False, max_length=max_query_length
                )

                start_offset = 0
                doc_spans = []
                max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
                while start_offset < len(all_doc_tokens):
                    length = len(all_doc_tokens) - start_offset
                    if length > max_tokens_for_doc:
                        length = max_tokens_for_doc
                    doc_spans.append({"start": start_offset, "length": length})
                    if start_offset + length == len(all_doc_tokens):
                        break
                    start_offset += min(length, doc_stride, max_tokens_for_doc)

                for (doc_span_index, doc_span) in enumerate(doc_spans):

                    tokens = []
                    token_boxes = []
                    token_to_orig_map = {}
                    token_is_max_context = {}
                    sentence_ids = []
                    seg_a = 0
                    seg_b = 1

                    token_is_max_context[str(len(tokens))] = False
                    token_to_orig_map[str(len(tokens))] = -1
                    tokens.append(tokenizer.cls_token)
                    token_boxes.append(cls_token_box)
                    sentence_ids.append(seg_a)

                    for i in range(doc_span["length"]):
                        split_token_index = doc_span["start"] + i
                        token_to_orig_map[str(len(tokens))] = tok_to_orig_index[split_token_index]

                        is_max_context = self._check_is_max_context(doc_spans, doc_span_index, split_token_index)
                        token_is_max_context[str(len(tokens))] = is_max_context
                        tokens.append(all_doc_tokens[split_token_index])
                        token_boxes.append(all_doc_token_boxes[split_token_index])
                        sentence_ids.append(seg_a)

                    token_is_max_context[str(len(tokens))] = False
                    token_to_orig_map[str(len(tokens))] = -1
                    tokens.append(tokenizer.sep_token)
                    token_boxes.append(sep_token_box)
                    sentence_ids.append(seg_a)
                    input_mask = [1] * len(tokens)

                    while len(tokens) < max_seq_length - len(query_tokens) - 1:
                        token_is_max_context[str(len(tokens))] = False
                        token_to_orig_map[str(len(tokens))] = -1
                        tokens.append(tokenizer.pad_token)
                        input_mask.append(0)
                        sentence_ids.append(seg_b)
                        token_boxes.append(pad_token_box)

                    for idx, token in enumerate(query_tokens):
                        token_is_max_context[str(len(tokens))] = False
                        token_to_orig_map[str(len(tokens))] = -1
                        tokens.append(token)
                        input_mask.append(1)
                        sentence_ids.append(seg_b)
                        token_boxes.append(query_token_box)

                    token_is_max_context[str(len(tokens))] = False
                    token_to_orig_map[str(len(tokens))] = -1
                    tokens.append(tokenizer.sep_token)
                    input_mask.append(1)
                    token_boxes.append(sep_token_box)
                    sentence_ids.append(seg_b)

                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    position_ids = list(range(len(tokens) - len(query_tokens) - 1)) + list(
                        range(len(query_tokens) + 1)
                    )

                    assert len(input_ids) == max_seq_length
                    assert len(input_mask) == max_seq_length
                    assert len(token_boxes) == max_seq_length
                    assert len(sentence_ids) == max_seq_length

                    answer_rcd = []
                    for answer_text, answer_start, answer_end in zip(
                        answers["text"],
                        answers["answer_start"],
                        answers["answer_end"],
                    ):

                        if is_training and answer_start == -1 and answer_end == -1:
                            continue

                        start_position = -1
                        end_position = -1

                        if is_training:

                            if [answer_start, answer_end] in answer_rcd:
                                continue
                            answer_rcd.append([answer_start, answer_end])

                            tok_start_position = orig_to_tok_index[answer_start]
                            if answer_end < len(example_text) - 1:
                                tok_end_position = orig_to_tok_index[answer_end] - 1
                            else:
                                tok_end_position = len(all_doc_tokens) - 1
                            (tok_start_position, tok_end_position) = self._improve_answer_span(
                                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, answer_text
                            )
                            # If the answer is outside the span, set start_position == end_position == 0

                            # For training, if our document chunk does not contain an annotation
                            # we throw it out, since there is nothing to predict.
                            doc_start = doc_span["start"]
                            doc_end = doc_span["start"] + doc_span["length"] - 1
                            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                                start_position = 0
                                end_position = 0
                            else:
                                doc_offset = 1
                                start_position = tok_start_position - doc_start + doc_offset
                                end_position = tok_end_position - doc_start + doc_offset

                        start_labels = [0] * len(input_ids)
                        end_labels = [0] * len(input_ids)
                        start_labels[start_position] = 1
                        end_labels[end_position] = 1
                        answer_rcd.append([start_position, end_position])

                        feature_id = examples["name"][example_idx] + "__" + str(examples["page_no"][example_idx])
                        tokenized_examples["id"].append(feature_id)
                        tokenized_examples["question_id"].append(qid)
                        tokenized_examples["questions"].append(question)
                        tokenized_examples["tokens"].append(tokens)
                        tokenized_examples["input_ids"].append(input_ids)
                        tokenized_examples["attention_mask"].append(input_mask)
                        tokenized_examples["token_type_ids"].append(sentence_ids)
                        tokenized_examples["bbox"].append(token_boxes)
                        tokenized_examples["position_ids"].append(position_ids)
                        tokenized_examples["image"].append(image)
                        tokenized_examples["start_positions"].append(start_position)
                        tokenized_examples["end_positions"].append(end_position)
                        tokenized_examples["start_labels"].append(start_labels)
                        tokenized_examples["end_labels"].append(end_labels)
                        tokenized_examples["token_is_max_context"].append(token_is_max_context)
                        tokenized_examples["token_to_orig_map"].append(token_to_orig_map)

                        if not is_training:
                            break
        return tokenized_examples

    def preprocess_cls(
        self,
        examples,
        tokenizer=None,
        label_dict=None,
        max_seq_length=512,
        doc_stride=128,
        target_size=1000,
        max_size=1000,
        other_label="O",
        ignore_label_id=-100,
        use_segment_box=False,
        preprocessing_num_workers=1,
    ):
        """
        Adapt to CLS task.
        """

        tokenized_examples = collections.defaultdict(list)
        for example_idx, example_text in enumerate(examples["text"]):
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            all_doc_token_boxes = []
            cls_token_box = [0, 0, 0, 0]
            sep_token_box = [0, 0, 0, 0]
            pad_token_box = [0, 0, 0, 0]

            im_base64 = examples["image"][example_idx]
            image, _ = _str2im(im_base64)
            image = _permute(image, to_bgr=False)

            if use_segment_box:
                bboxes = examples["segment_bbox"][example_idx]
            else:
                bboxes = examples["bbox"][example_idx]
            bboxes, _s = _scale_same_as_image(
                bboxes,
                examples["width"][example_idx],
                examples["height"][example_idx],
                target_size,
            )

            qas = examples["qas"][example_idx]
            label = label_dict[qas["answers"][0]["text"][0]]

            for (i, token) in enumerate(example_text):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                box = bboxes[i]
                for j, sub_token in enumerate(sub_tokens):
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
                    all_doc_token_boxes.append(box)

            max_tokens_for_doc = max_seq_length - 2
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append({"start": start_offset, "length": length})
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride, max_tokens_for_doc)

            for doc_span in doc_spans:

                tokens = []
                token_boxes = []
                sentence_ids = []
                tokens.append(tokenizer.cls_token)
                token_boxes.append(cls_token_box)
                sentence_ids.append(0)

                for i in range(doc_span["length"]):
                    split_token_index = doc_span["start"] + i
                    tokens.append(all_doc_tokens[split_token_index])
                    token_boxes.append(all_doc_token_boxes[split_token_index])
                    sentence_ids.append(0)

                tokens.append(tokenizer.sep_token)
                token_boxes.append(sep_token_box)
                sentence_ids.append(0)
                input_mask = [1] * len(tokens)

                while len(tokens) < max_seq_length:
                    tokens.append(tokenizer.pad_token)
                    input_mask.append(0)
                    sentence_ids.append(0)
                    token_boxes.append(pad_token_box)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                position_ids = list(range(len(input_ids)))

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(token_boxes) == max_seq_length
                assert len(sentence_ids) == max_seq_length

                feature_id = examples["name"][example_idx] + "__" + str(examples["page_no"][example_idx])
                tokenized_examples["id"].append(feature_id)
                tokenized_examples["tokens"].append(tokens)
                tokenized_examples["input_ids"].append(input_ids)
                tokenized_examples["attention_mask"].append(input_mask)
                tokenized_examples["token_type_ids"].append(sentence_ids)
                tokenized_examples["bbox"].append(token_boxes)
                tokenized_examples["position_ids"].append(position_ids)
                tokenized_examples["image"].append(image)
                # tokenized_examples["orig_image"].append(origin_image)
                tokenized_examples["labels"].append(label)
        return tokenized_examples


class PostProcessor:
    def __init__(self):
        """init post processor"""

        self.examples_cache = collections.defaultdict(list)
        self.features_cache = collections.defaultdict(list)
        self._PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
        )

    def get_predictions(self, pred, label_list, with_crf=False):
        if not with_crf:
            pred = scipy.special.softmax(pred, axis=-1)
            pred_ids = np.argmax(pred, axis=1)
        else:
            pred_ids = pred
        prediction_score = [pred[idx][i] for idx, i in enumerate(pred_ids)]
        predictions = [label_list[i] for i in pred_ids]
        return predictions, prediction_score

    def postprocess_ner(
        self,
        examples: datasets.Dataset,
        features: datasets.Dataset,
        preds,
        labels,
        label_list,
        tokenizer=None,
        with_crf=False,
        lang="en",
    ):
        if "name" not in self.examples_cache:
            self.examples_cache["name"] = [item for item in examples["name"]]
        if "page_no" not in self.examples_cache:
            self.examples_cache["page_no"] = [item for item in examples["page_no"]]
        if "text" not in self.examples_cache:
            self.examples_cache["text"] = [item for item in examples["text"]]
        if "id" not in self.features_cache:
            self.features_cache["id"] = [item for item in features["id"]]
        if "tokens" not in self.features_cache:
            self.features_cache["tokens"] = [item for item in features["tokens"]]
        if "token_is_max_context" not in self.features_cache:
            self.features_cache["token_is_max_context"] = [item for item in features["token_is_max_context"]]
        if "token_to_orig_map" not in self.features_cache:
            self.features_cache["token_to_orig_map"] = [item for item in features["token_to_orig_map"]]
        separator = "" if lang == "ch" else " "

        feature_id_to_features = collections.defaultdict(list)
        for idx, feature_id in enumerate(self.features_cache["id"]):
            feature_id_to_features[feature_id].append(idx)

        references = collections.defaultdict(list)
        predictions = collections.defaultdict(list)
        recover_preds = []
        recover_labels = []

        for eid, example_id in enumerate(self.examples_cache["name"]):
            feature_map = example_id + "__" + str(self.examples_cache["page_no"][eid])
            features_ids = feature_id_to_features[feature_map]
            gather_pred = []
            gather_label = []
            gather_tokens = []
            gather_score = []
            gather_map = []
            for idx in features_ids:
                pred, label = preds[idx], labels[idx]
                prediction, prediction_score = self.get_predictions(pred, label_list, with_crf=with_crf)

                token_is_max_context = self.features_cache["token_is_max_context"][idx]
                token_to_orig_map = self.features_cache["token_to_orig_map"][idx]
                for token_idx in range(len(token_is_max_context)):
                    token_idx += 1
                    if token_is_max_context[str(token_idx)]:
                        gather_tokens.append(self.features_cache["tokens"][idx][token_idx])
                        gather_pred.append(prediction[token_idx])
                        gather_score.append(prediction_score[token_idx])
                        gather_label.append(label[token_idx])
                        gather_map.append(token_to_orig_map[str(token_idx)])

            recover_pred = [p for (p, l) in zip(gather_pred, gather_label) if l != -100]
            recover_label = [label_list[l] for l in gather_label if l != -100]

            pred_entities = get_entities(recover_pred)
            gt_entities = get_entities(recover_label)
            recover_preds.append(recover_pred)
            recover_labels.append(recover_label)

            for item in pred_entities:
                entity = tokenizer.convert_tokens_to_string(gather_tokens[item[1] : (item[2] + 1)]).strip()
                orig_doc_start = gather_map[item[1]]
                orig_doc_end = gather_map[item[2]]
                orig_tokens = self.examples_cache["text"][eid][orig_doc_start : (orig_doc_end + 1)]
                orig_text = separator.join(orig_tokens)
                final_text = self.get_final_text(entity, orig_text, False, tokenizer)
                predictions[example_id].append(
                    [
                        item[0],
                        final_text,
                        sum(gather_score[item[1] : item[2] + 1]) / (item[2] - item[1] + 1),
                        [item[1], item[2]],
                        ", ".join(recover_pred[item[1] : item[2] + 1]),
                    ]
                )

            for item in gt_entities:
                entity = tokenizer.convert_tokens_to_string(gather_tokens[item[1] : (item[2] + 1)]).strip()
                orig_doc_start = gather_map[item[1]]
                orig_doc_end = gather_map[item[2]]
                orig_tokens = self.examples_cache["text"][eid][orig_doc_start : (orig_doc_end + 1)]
                orig_text = separator.join(orig_tokens)
                final_text = self.get_final_text(entity, orig_text, False, tokenizer)
                references[example_id].append(
                    [item[0], final_text, 1, [item[1], item[2]], ", ".join(recover_label[item[1] : item[2] + 1])]
                )
            if example_id not in predictions:
                predictions[example_id].append(["", "", -1, [], ""])

        return predictions, references, EvalPrediction(predictions=recover_preds, label_ids=recover_labels)

    def _get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def get_final_text(self, pred_text, orig_text, do_lower_case, tokenizer):
        """Project the tokenized prediction back to the original text."""

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        tok_text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            return orig_text

        output_text = orig_text[orig_start_position : (orig_end_position + 1)]
        return output_text

    def postprocess_mrc(
        self,
        examples: datasets.Dataset,
        features: datasets.Dataset,
        preds,
        labels,
        tokenizer,
        max_answer_length=64,
        n_best_size=5,
        lang="en",
    ):
        if "name" not in self.examples_cache:
            self.examples_cache["name"] = [item for item in examples["name"]]
        if "page_no" not in self.examples_cache:
            self.examples_cache["page_no"] = [item for item in examples["page_no"]]
        if "text" not in self.examples_cache:
            self.examples_cache["text"] = [item for item in examples["text"]]
        if "qas" not in self.examples_cache:
            self.examples_cache["qas"] = [item for item in examples["qas"]]

        if "id" not in self.features_cache:
            self.features_cache["id"] = [item for item in features["id"]]
        if "tokens" not in self.features_cache:
            self.features_cache["tokens"] = [item for item in features["tokens"]]
        if "question_id" not in self.features_cache:
            self.features_cache["question_id"] = [item for item in features["question_id"]]
        if "questions" not in self.features_cache:
            self.features_cache["questions"] = [item for item in features["questions"]]
        if "token_is_max_context" not in self.features_cache:
            self.features_cache["token_is_max_context"] = [item for item in features["token_is_max_context"]]
        if "token_to_orig_map" not in self.features_cache:
            self.features_cache["token_to_orig_map"] = [item for item in features["token_to_orig_map"]]

        separator = "" if lang == "ch" else " "

        feature_id_to_features = collections.defaultdict(list)
        for idx, feature_id in enumerate(self.features_cache["id"]):
            feature_id_to_features[feature_id].append(idx)

        predictions, references = collections.defaultdict(
            lambda: collections.defaultdict(list)
        ), collections.defaultdict(lambda: collections.defaultdict(list))
        for ei, example_id in enumerate(self.examples_cache["name"]):
            feature_map = example_id + "__" + str(self.examples_cache["page_no"][ei])
            features_ids = feature_id_to_features[feature_map]
            prelim_predictions = []
            for i, idx in enumerate(features_ids):

                start_logits = preds[0][idx]
                end_logits = preds[1][idx]

                start_indexes = self._get_best_indexes(start_logits, n_best_size)
                end_indexes = self._get_best_indexes(end_logits, n_best_size)
                token_is_max_context = self.features_cache["token_is_max_context"][idx]

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if not token_is_max_context.get(str(start_index), False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            self._PrelimPrediction(
                                feature_index=idx,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=start_logits[start_index],
                                end_logit=end_logits[end_index],
                            )
                        )

            prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

            for rcd in prelim_predictions:

                question_id = self.features_cache["question_id"][rcd.feature_index]
                question = self.features_cache["questions"][rcd.feature_index]
                if question_id in predictions[example_id]:
                    continue

                if rcd.start_index > 0:
                    tok_tokens = self.features_cache["tokens"][rcd.feature_index][
                        rcd.start_index : (rcd.end_index + 1)
                    ]
                    orig_doc_start = self.features_cache["token_to_orig_map"][rcd.feature_index][str(rcd.start_index)]
                    orig_doc_end = self.features_cache["token_to_orig_map"][rcd.feature_index][str(rcd.end_index)]
                    orig_tokens = self.examples_cache["text"][ei][orig_doc_start : (orig_doc_end + 1)]
                    orig_text = separator.join(orig_tokens)

                    tok_text = tokenizer.convert_tokens_to_string(tok_tokens).strip()
                    final_text = self.get_final_text(tok_text, orig_text, False, tokenizer)
                else:
                    continue
                if question_id in predictions[example_id]:
                    predictions[example_id][question_id]["answers"].append(final_text)
                else:
                    predictions[example_id][question_id] = {"question": question, "answers": [final_text]}

        for example_index, example in enumerate(examples):
            eid = self.examples_cache["name"][example_index]
            qas = self.examples_cache["qas"][example_index]
            for question_id, question, answers in zip(qas["question_id"], qas["question"], qas["answers"]):
                references[eid][question_id] = {
                    "question": question,
                    "answers": [answer_text for answer_text in answers["text"]],
                }
                if eid not in predictions or question_id not in predictions[eid]:
                    predictions[eid][question_id] = {"question": question, "answers": [""]}

        formatted_predictions = [
            {
                "id": k,
                "annotations": [
                    {"qid": str(qid), "question": qa["question"], "value": qa["answers"]} for qid, qa in v.items()
                ],
            }
            for k, v in predictions.items()
        ]
        formated_references = [
            {
                "id": k,
                "annotations": [
                    {"qid": str(qid), "question": qa["question"], "value": qa["answers"]} for qid, qa in v.items()
                ],
            }
            for k, v in references.items()
        ]
        return (
            predictions,
            references,
            EvalPrediction(predictions=formatted_predictions, label_ids=formated_references),
        )

    def postprocess_cls(
        self,
        examples: datasets.Dataset,
        features: datasets.Dataset,
        preds,
        labels,
        label_list,
        tokenizer=None,
    ):
        if "name" not in self.examples_cache:
            self.examples_cache["name"] = [item for item in examples["name"]]
        if "page_no" not in self.examples_cache:
            self.examples_cache["page_no"] = [item for item in examples["page_no"]]
        if "id" not in self.features_cache:
            self.features_cache["id"] = [item for item in features["id"]]

        feature_id_to_features = collections.defaultdict(list)
        for idx, feature_id in enumerate(self.features_cache["id"]):
            feature_id_to_features[feature_id].append(idx)

        references = {}
        predictions = {}
        recover_preds = []
        recover_labels = []

        for eid, example_id in enumerate(self.examples_cache["name"]):
            feature_map = example_id + "__" + str(self.examples_cache["page_no"][eid])
            features_ids = feature_id_to_features[feature_map]

            max_rcd = [0, -1]
            for i, idx in enumerate(features_ids):
                pred, label = preds[idx], labels[idx]
                pred = scipy.special.softmax(pred, axis=-1)
                pred_id = int(np.argmax(pred, axis=-1))
                if pred[pred_id] > max_rcd[0]:
                    max_rcd = [pred[pred_id], pred_id]

            recover_preds.append(max_rcd[1])
            recover_labels.append(label)
            predictions[example_id] = label_list[max_rcd[1]]
            references[example_id] = label_list[label]
        return predictions, references, EvalPrediction(predictions=recover_preds, label_ids=recover_labels)
