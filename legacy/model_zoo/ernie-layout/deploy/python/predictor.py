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
import collections

import cv2
import numpy as np
import paddle
import scipy
import six
from paddleocr import PaddleOCR
from PIL import Image
from seqeval.metrics.sequence_labeling import get_entities

from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.image_utils import ppocr2example
from paddlenlp.utils.log import logger


class InferBackend(object):
    def __init__(self, model_path_prefix, device="cpu"):
        logger.info(">>> [InferBackend] Creating Engine ...")
        config = paddle.inference.Config(
            model_path_prefix + ".pdmodel",
            model_path_prefix + ".pdiparams",
        )
        if device == "gpu":
            config.enable_use_gpu(100, 0)
            config.switch_ir_optim(False)
        else:
            config.disable_gpu()
            config.enable_mkldnn()
        config.switch_use_feed_fetch_ops(False)
        config.disable_glog_info()
        config.enable_memory_optim()
        self.predictor = paddle.inference.create_predictor(config)
        self.input_names = [name for name in self.predictor.get_input_names()]
        self.input_handles = [self.predictor.get_input_handle(name) for name in self.predictor.get_input_names()]
        self.output_handles = [self.predictor.get_output_handle(name) for name in self.predictor.get_output_names()]
        logger.info(">>> [InferBackend] Engine Created ...")

    def infer(self, input_dict: dict):
        for idx, input_name in enumerate(self.input_names):
            self.input_handles[idx].copy_from_cpu(input_dict[input_name])
        self.predictor.run()
        outputs = [output_handle.copy_to_cpu() for output_handle in self.output_handles]
        return outputs


class Predictor(object):
    def __init__(self, args):
        use_gpu = True if args.device == "gpu" else False
        self.tokenizer = AutoTokenizer.from_pretrained("ernie-layoutx-base-uncased")
        self.batch_size = args.batch_size
        self.max_seq_length = args.max_seq_length
        self.task_type = args.task_type
        self.lang = args.lang
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False, use_gpu=use_gpu)

        self.examples_cache = collections.defaultdict(list)
        self.features_cache = collections.defaultdict(list)
        self._PrelimPrediction = collections.namedtuple(
            "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
        )
        self.inference_backend = InferBackend(args.model_path_prefix, device=args.device)
        if self.task_type == "ner":
            self.label_dict = {
                "O": 0,
                "B-ANSWER": 1,
                "I-ANSWER": 2,
                "B-HEADER": 3,
                "I-HEADER": 4,
                "B-QUESTION": 5,
                "I-QUESTION": 6,
            }
            self.preprocess = self.preprocess_ner
            self.postprocess = self.postprocess_ner
        elif self.task_type == "cls":
            self.label_dict = {
                "advertisement": 0,
                "budget": 1,
                "email": 2,
                "file folder": 3,
                "form": 4,
                "handwritten": 5,
                "invoice": 6,
                "letter": 7,
                "memo": 8,
                "news article": 9,
                "presentation": 10,
                "questionnaire": 11,
                "resume": 12,
                "scientific publication": 13,
                "scientific report": 14,
                "specification": 15,
            }
            self.preprocess = self.preprocess_cls
            self.postprocess = self.postprocess_cls
        elif self.task_type == "mrc":
            self.questions = args.questions
            self.preprocess = self.preprocess_mrc
            self.postprocess = self.postprocess_mrc
        else:
            raise ValueError("Unspport task type: {}".format(args.task_type))

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

    def _get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def get_predictions(self, pred, label_list):
        pred = scipy.special.softmax(pred, axis=-1)
        pred_ids = np.argmax(pred, axis=1)
        prediction_score = [pred[idx][i] for idx, i in enumerate(pred_ids)]
        predictions = [label_list[i] for i in pred_ids]
        return predictions, prediction_score

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

    def preprocess_ner(self, examples, doc_stride=128, target_size=1000, max_size=1000):
        ignore_label_id = -100
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

            bboxes = examples["bbox"][example_idx]
            bboxes, _s = _scale_same_as_image(
                bboxes,
                examples["width"][example_idx],
                examples["height"][example_idx],
                target_size,
            )

            orig_labels = ["O"] * len(example_text)

            for (i, token) in enumerate(example_text):
                orig_to_tok_index.append(len(all_doc_tokens))
                if self.lang == "ch":
                    sub_tokens = self.tokenizer.tokenize("&" + token)[1:]
                else:
                    sub_tokens = self.tokenizer.tokenize(token)
                label = orig_labels[i]
                box = bboxes[i]
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
                    all_doc_token_boxes.append(box)
                    all_doc_token_labels.append(label)

            max_tokens_for_doc = self.max_seq_length - 2
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
                tokens.append(self.tokenizer.cls_token)
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
                    token_label_ids.append(self.label_dict[all_doc_token_labels[split_token_index]])
                    sentence_ids.append(0)

                token_is_max_context[str(len(tokens))] = False
                token_to_orig_map[str(len(tokens))] = -1
                tokens.append(self.tokenizer.sep_token)
                token_boxes.append(sep_token_box)
                token_label_ids.append(ignore_label_id)
                sentence_ids.append(0)
                input_mask = [1] * len(tokens)

                while len(tokens) < self.max_seq_length:
                    token_is_max_context[str(len(tokens))] = False
                    token_to_orig_map[str(len(tokens))] = -1
                    tokens.append(self.tokenizer.pad_token)
                    input_mask.append(0)
                    sentence_ids.append(0)
                    token_boxes.append(pad_token_box)
                    token_label_ids.append(ignore_label_id)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                position_ids = list(range(len(input_ids)))

                tokenized_examples["id"].append(example_idx)
                tokenized_examples["tokens"].append(tokens)
                tokenized_examples["input_ids"].append(input_ids)
                tokenized_examples["attention_mask"].append(input_mask)
                tokenized_examples["token_type_ids"].append(sentence_ids)
                tokenized_examples["bbox"].append(token_boxes)
                tokenized_examples["position_ids"].append(position_ids)
                tokenized_examples["image"].append(image)
                tokenized_examples["labels"].append(token_label_ids)
                tokenized_examples["token_is_max_context"].append(token_is_max_context)
                tokenized_examples["token_to_orig_map"].append(token_to_orig_map)
        for input_id in tokenized_examples["input_ids"]:
            input_id = input_id + [1 * self.tokenizer.tokens_to_ids[self.tokenizer.pad_token]] * (
                self.max_seq_length - len(input_id)
            )

        for att_mask in tokenized_examples["attention_mask"]:
            att_mask = att_mask + [1 * self.tokenizer.tokens_to_ids[self.tokenizer.pad_token]] * (
                self.max_seq_length - len(att_mask)
            )

        for bbox in tokenized_examples["bbox"]:
            bbox = bbox + [[0, 0, 0, 0] for _ in range(self.max_seq_length - len(bbox))]

        for label in tokenized_examples["labels"]:
            label = label + [1 * ignore_label_id] * (self.max_seq_length - len(label))

        self.examples_cache["name"] = list(range(len(examples["text"])))
        self.examples_cache["text"] = [item for item in examples["text"]]
        self.features_cache["id"] = [item for item in tokenized_examples["id"]]
        self.features_cache["labels"] = [item for item in tokenized_examples["labels"]]
        self.features_cache["tokens"] = [item for item in tokenized_examples["tokens"]]
        self.features_cache["token_is_max_context"] = [item for item in tokenized_examples["token_is_max_context"]]
        self.features_cache["token_to_orig_map"] = [item for item in tokenized_examples["token_to_orig_map"]]
        return tokenized_examples

    def postprocess_ner(self, preds):
        separator = "" if self.lang == "ch" else " "
        feature_id_to_features = collections.defaultdict(list)
        for idx, feature_id in enumerate(self.features_cache["id"]):
            feature_id_to_features[feature_id].append(idx)

        predictions = []
        recover_preds = []

        for eid, example_id in enumerate(self.examples_cache["name"]):
            prediction_tags = []
            feature_map = example_id
            features_ids = feature_id_to_features[feature_map]
            gather_pred = []
            gather_label = []
            gather_tokens = []
            gather_score = []
            gather_map = []
            for idx in features_ids:
                pred, label = preds[idx], self.features_cache["labels"][idx]
                prediction, prediction_score = self.get_predictions(pred, list(self.label_dict.keys()))

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

            pred_entities = get_entities(recover_pred)
            recover_preds.append(recover_pred)

            for item in pred_entities:
                entity = self.tokenizer.convert_tokens_to_string(gather_tokens[item[1] : (item[2] + 1)]).strip()
                orig_doc_start = gather_map[item[1]]
                orig_doc_end = gather_map[item[2]]
                orig_tokens = self.examples_cache["text"][eid][orig_doc_start : (orig_doc_end + 1)]
                orig_text = separator.join(orig_tokens)
                final_text = self.get_final_text(entity, orig_text, False, self.tokenizer)
                final_text = final_text.replace("   ", " ")

                res = {
                    "text": final_text,
                    "label": item[0],
                    "start": item[1],
                    "end": item[2],
                    "probability": sum(gather_score[item[1] : item[2] + 1]) / (item[2] - item[1] + 1),
                }
                prediction_tags.append(res)

            predictions.append(prediction_tags)
        return predictions

    def preprocess_cls(self, examples, doc_stride=128, target_size=1000, max_size=1000):
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

            bboxes = examples["bbox"][example_idx]
            bboxes, _s = _scale_same_as_image(
                bboxes,
                examples["width"][example_idx],
                examples["height"][example_idx],
                target_size,
            )

            for (i, token) in enumerate(example_text):
                orig_to_tok_index.append(len(all_doc_tokens))
                if self.lang == "ch":
                    sub_tokens = self.tokenizer.tokenize("&" + token)[1:]
                else:
                    sub_tokens = self.tokenizer.tokenize(token)
                box = bboxes[i]
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
                    all_doc_token_boxes.append(box)

            max_tokens_for_doc = self.max_seq_length - 2
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
                tokens.append(self.tokenizer.cls_token)
                token_boxes.append(cls_token_box)
                sentence_ids.append(0)

                for i in range(doc_span["length"]):
                    split_token_index = doc_span["start"] + i
                    tokens.append(all_doc_tokens[split_token_index])
                    token_boxes.append(all_doc_token_boxes[split_token_index])
                    sentence_ids.append(0)

                tokens.append(self.tokenizer.sep_token)
                token_boxes.append(sep_token_box)
                sentence_ids.append(0)
                input_mask = [1] * len(tokens)

                while len(tokens) < self.max_seq_length:
                    tokens.append(self.tokenizer.pad_token)
                    input_mask.append(0)
                    sentence_ids.append(0)
                    token_boxes.append(pad_token_box)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                position_ids = list(range(len(input_ids)))

                tokenized_examples["id"].append(example_idx)
                tokenized_examples["tokens"].append(tokens)
                tokenized_examples["input_ids"].append(input_ids)
                tokenized_examples["attention_mask"].append(input_mask)
                tokenized_examples["token_type_ids"].append(sentence_ids)
                tokenized_examples["bbox"].append(token_boxes)
                tokenized_examples["position_ids"].append(position_ids)
                tokenized_examples["image"].append(image)
        for input_id in tokenized_examples["input_ids"]:
            input_id = input_id + [1 * self.tokenizer.tokens_to_ids[self.tokenizer.pad_token]] * (
                self.max_seq_length - len(input_id)
            )

        for att_mask in tokenized_examples["attention_mask"]:
            att_mask = att_mask + [1 * self.tokenizer.tokens_to_ids[self.tokenizer.pad_token]] * (
                self.max_seq_length - len(att_mask)
            )

        for bbox in tokenized_examples["bbox"]:
            bbox = bbox + [[0, 0, 0, 0] for _ in range(self.max_seq_length - len(bbox))]

        self.examples_cache["name"] = list(range(len(examples["text"])))
        self.features_cache["id"] = [item for item in tokenized_examples["id"]]
        return tokenized_examples

    def postprocess_cls(self, preds):
        feature_id_to_features = collections.defaultdict(list)
        for idx, feature_id in enumerate(self.features_cache["id"]):
            feature_id_to_features[feature_id].append(idx)

        predictions = []

        for example_id in self.examples_cache["name"]:
            features_ids = feature_id_to_features[example_id]

            max_rcd = [0, -1]
            for idx in features_ids:
                pred = preds[idx]
                pred = scipy.special.softmax(pred, axis=-1)
                pred_id = int(np.argmax(pred, axis=-1))
                if pred[pred_id] > max_rcd[0]:
                    max_rcd = [pred[pred_id], pred_id]

            predictions.append(list(self.label_dict.keys())[max_rcd[1]])
        return predictions

    def preprocess_mrc(self, examples, doc_stride=128, max_query_length=64, target_size=1000, max_size=1000):
        qid = -1
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

            bboxes = examples["bbox"][example_idx]
            bboxes, _s = _scale_same_as_image(
                bboxes,
                examples["width"][example_idx],
                examples["height"][example_idx],
                target_size,
            )

            for (i, token) in enumerate(example_text):
                orig_to_tok_index.append(len(all_doc_tokens))
                if self.lang == "ch":
                    sub_tokens = self.tokenizer.tokenize("&" + token)[1:]
                else:
                    sub_tokens = self.tokenizer.tokenize(token)
                box = bboxes[i]
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
                    all_doc_token_boxes.append(box)

            for question in self.questions[example_idx]:
                qid += 1
                query_tokens = self.tokenizer.tokenize(
                    question, add_special_tokens=False, truncation=False, max_length=max_query_length
                )

                start_offset = 0
                doc_spans = []
                max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3
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
                    tokens.append(self.tokenizer.cls_token)
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
                    tokens.append(self.tokenizer.sep_token)
                    token_boxes.append(sep_token_box)
                    sentence_ids.append(seg_a)
                    input_mask = [1] * len(tokens)

                    while len(tokens) < self.max_seq_length - len(query_tokens) - 1:
                        token_is_max_context[str(len(tokens))] = False
                        token_to_orig_map[str(len(tokens))] = -1
                        tokens.append(self.tokenizer.pad_token)
                        input_mask.append(0)
                        sentence_ids.append(seg_b)
                        token_boxes.append(pad_token_box)

                    for token in query_tokens:
                        token_is_max_context[str(len(tokens))] = False
                        token_to_orig_map[str(len(tokens))] = -1
                        tokens.append(token)
                        input_mask.append(1)
                        sentence_ids.append(seg_b)
                        token_boxes.append(query_token_box)

                    token_is_max_context[str(len(tokens))] = False
                    token_to_orig_map[str(len(tokens))] = -1
                    tokens.append(self.tokenizer.sep_token)
                    input_mask.append(1)
                    token_boxes.append(sep_token_box)
                    sentence_ids.append(seg_b)

                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    position_ids = list(range(len(tokens) - len(query_tokens) - 1)) + list(
                        range(len(query_tokens) + 1)
                    )

                    answer_rcd = []
                    start_position = -1
                    end_position = -1

                    start_labels = [0] * len(input_ids)
                    end_labels = [0] * len(input_ids)
                    start_labels[start_position] = 1
                    end_labels[end_position] = 1
                    answer_rcd.append([start_position, end_position])

                    tokenized_examples["id"].append(example_idx)
                    tokenized_examples["question_id"].append(qid)
                    tokenized_examples["questions"].append(question)
                    tokenized_examples["tokens"].append(tokens)
                    tokenized_examples["input_ids"].append(input_ids)
                    tokenized_examples["attention_mask"].append(input_mask)
                    tokenized_examples["token_type_ids"].append(sentence_ids)
                    tokenized_examples["bbox"].append(token_boxes)
                    tokenized_examples["position_ids"].append(position_ids)
                    tokenized_examples["image"].append(image)
                    tokenized_examples["token_is_max_context"].append(token_is_max_context)
                    tokenized_examples["token_to_orig_map"].append(token_to_orig_map)
        for input_id in tokenized_examples["input_ids"]:
            input_id = input_id + [1 * self.tokenizer.tokens_to_ids[self.tokenizer.pad_token]] * (
                self.max_seq_length - len(input_id)
            )

        for att_mask in tokenized_examples["attention_mask"]:
            att_mask = att_mask + [1 * self.tokenizer.tokens_to_ids[self.tokenizer.pad_token]] * (
                self.max_seq_length - len(att_mask)
            )

        for bbox in tokenized_examples["bbox"]:
            bbox = bbox + [[0, 0, 0, 0] for _ in range(self.max_seq_length - len(bbox))]
        self.examples_cache["name"] = list(range(len(examples["text"])))
        self.examples_cache["text"] = [item for item in examples["text"]]
        self.features_cache["id"] = [item for item in tokenized_examples["id"]]
        self.features_cache["question_id"] = [item for item in tokenized_examples["question_id"]]
        self.features_cache["tokens"] = [item for item in tokenized_examples["tokens"]]
        self.features_cache["questions"] = [item for item in tokenized_examples["questions"]]
        self.features_cache["token_is_max_context"] = [item for item in tokenized_examples["token_is_max_context"]]
        self.features_cache["token_to_orig_map"] = [item for item in tokenized_examples["token_to_orig_map"]]
        return tokenized_examples

    def postprocess_mrc(self, preds, max_answer_length=64, n_best_size=5):
        separator = "" if self.lang == "ch" else " "
        feature_id_to_features = collections.defaultdict(list)
        for idx, feature_id in enumerate(self.features_cache["id"]):
            feature_id_to_features[feature_id].append(idx)

        predictions = collections.defaultdict(lambda: collections.defaultdict(list))
        for ei, example_id in enumerate(self.examples_cache["name"]):
            feature_map = example_id
            features_ids = feature_id_to_features[feature_map]
            prelim_predictions = []
            for idx in features_ids:
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

                    tok_text = self.tokenizer.convert_tokens_to_string(tok_tokens).strip()
                    final_text = self.get_final_text(tok_text, orig_text, False, self.tokenizer)
                else:
                    continue
                if question_id in predictions[example_id]:
                    predictions[example_id][question_id]["answer"].append(final_text)
                else:
                    predictions[example_id][question_id] = {"question": question, "answer": [final_text]}
        formatted_predictions = []
        for v in predictions.values():
            formatted_predictions.append([{"question": qa["question"], "answer": qa["answer"]} for qa in v.values()])
        return formatted_predictions

    def infer(self, data):
        return self.inference_backend.infer(data)

    def predict(self, docs):
        input_data = []
        for doc in docs:
            ocr_result = self.ocr.ocr(doc, cls=True)
            # Compatible with paddleocr>=2.6.0.2
            ocr_result = ocr_result[0] if len(ocr_result) == 1 else ocr_result
            example = ppocr2example(ocr_result, doc)
            input_data.append(example)

        inputs = collections.defaultdict(list)
        for data in input_data:
            for k in data.keys():
                inputs[k].append(data[k])

        preprocess_result = self.preprocess(inputs)
        preds = [[], []] if self.task_type == "mrc" else []
        for idx in range(0, len(preprocess_result["id"]), self.batch_size):
            l, r = idx, idx + self.batch_size
            input_dict = {}
            for input_name in self.inference_backend.input_names:
                input_dict[input_name] = np.array(preprocess_result[input_name][l:r], dtype="int64")
            output = self.infer(input_dict)
            if self.task_type != "mrc":
                preds.extend(output[0].tolist())
            else:
                preds[0].extend(output[0].tolist())
                preds[1].extend(output[1].tolist())
        results = self.postprocess(preds)
        formatted_results = []
        for doc, res in zip(docs, results):
            formatted_result = {"doc": doc, "result": res}
            formatted_results.append(formatted_result)
        return formatted_results


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
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
):
    # step1: decode image
    origin_im = _decode_image(im_base64)
    # step2: resize image
    im = _resize_image(origin_im, target_size=target_size, interp=1, resize_box=False)
    return im, origin_im
