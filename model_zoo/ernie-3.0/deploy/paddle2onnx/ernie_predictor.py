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

import six
import os
import numpy as np
import paddle
import onnxruntime as ort
from paddlenlp.transformers import AutoTokenizer


class InferBackend(object):

    def __init__(self, model_path, use_fp16):
        print(">>> [InferBackend] Creating Engine ...")
        providers = ['CUDAExecutionProvider']
        sess_options = ort.SessionOptions()
        predictor = ort.InferenceSession(model_path,
                                         sess_options=sess_options,
                                         providers=providers)
        if "CUDAExecutionProvider" in predictor.get_providers():
            print(">>> [InferBackend] Use GPU to inference ...")
            if use_fp16:
                from onnxconverter_common import float16
                import onnx
                print(">>> [InferBackend] Use FP16 to inference ...")
                fp16_model = "fp16_model.onnx"
                onnx_model = onnx.load_model(model_path)
                trans_model = float16.convert_float_to_float16(
                    onnx_model, keep_io_types=True)
                onnx.save_model(trans_model, fp16_model)
                sess_options = ort.SessionOptions()
                predictor = ort.InferenceSession(fp16_model,
                                                 sess_options=sess_options,
                                                 providers=providers)
        else:
            print(">>> [InferBackend] Use CPU to inference ...")
            if use_fp16:
                print(
                    ">>> [InferBackend] use_fp16 only takes effect when deploying on gpu ..."
                )
        self.predictor = predictor
        input_name1 = self.predictor.get_inputs()[1].name
        input_name2 = self.predictor.get_inputs()[0].name
        self.input_handles = [input_name1, input_name2]
        print(">>> [InferBackend] Engine Created ...")

    def infer(self, input_dict: dict):
        result = self.predictor.run(None, input_dict)
        return result


def token_cls_print_ret(infer_result, input_data):
    rets = infer_result["value"]
    for i, ret in enumerate(rets):
        print("input data:", input_data[i])
        print("The model detects all entities:")
        for iterm in ret:
            print("entity:", iterm["entity"], "  label:", iterm["label"],
                  "  pos:", iterm["pos"])
        print("-----------------------------")


def seq_cls_print_ret(infer_result, input_data):
    label_list = [
        "news_story", "news_culture", "news_entertainment", "news_sports",
        "news_finance", "news_house", "news_car", "news_edu", "news_tech",
        "news_military", "news_travel", "news_world", "news_stock",
        "news_agriculture", "news_game"
    ]
    label = infer_result["label"].squeeze().tolist()
    confidence = infer_result["confidence"].squeeze().tolist()
    for i, ret in enumerate(infer_result):
        print("input data:", input_data[i])
        print("seq cls result:")
        print("label:", label_list[label[i]], "  confidence:", confidence[i])
        print("-----------------------------")


class ErniePredictor(object):

    def __init__(self, args):
        self.task_name = args.task_name
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                       use_faster=True)
        if args.task_name == 'seq_cls':
            self.label_names = []
            self.preprocess = self.seq_cls_preprocess
            self.postprocess = self.seq_cls_postprocess
            self.printer = seq_cls_print_ret
        elif args.task_name == 'token_cls':
            self.label_names = [
                'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'
            ]
            self.preprocess = self.token_cls_preprocess
            self.postprocess = self.token_cls_postprocess
            self.printer = token_cls_print_ret
        else:
            print(
                "[ErniePredictor]: task_name only support seq_cls and token_cls now."
            )
            exit(0)

        self.max_seq_length = args.max_seq_length
        self.inference_backend = InferBackend(args.model_path, args.use_fp16)

    def seq_cls_preprocess(self, input_data: list):
        data = input_data
        # tokenizer + pad
        data = self.tokenizer(data,
                              max_length=self.max_seq_length,
                              padding=True,
                              truncation=True)
        input_ids = data["input_ids"]
        token_type_ids = data["token_type_ids"]
        return {
            "input_ids": np.array(input_ids, dtype="int64"),
            "token_type_ids": np.array(token_type_ids, dtype="int64")
        }

    def seq_cls_postprocess(self, infer_data, input_data):
        logits = np.array(infer_data[0])
        max_value = np.max(logits, axis=1, keepdims=True)
        exp_data = np.exp(logits - max_value)
        probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        out_dict = {
            "label": probs.argmax(axis=-1),
            "confidence": probs.max(axis=-1)
        }
        return out_dict

    def token_cls_preprocess(self, data: list):
        # tokenizer + pad
        is_split_into_words = False
        if isinstance(data[0], list):
            is_split_into_words = True
        data = self.tokenizer(data,
                              max_length=self.max_seq_length,
                              padding=True,
                              truncation=True,
                              is_split_into_words=is_split_into_words)

        input_ids = data["input_ids"]
        token_type_ids = data["token_type_ids"]
        return {
            "input_ids": np.array(input_ids, dtype="int64"),
            "token_type_ids": np.array(token_type_ids, dtype="int64")
        }

    def token_cls_postprocess(self, infer_data, input_data):
        result = np.array(infer_data[0])
        tokens_label = result.argmax(axis=-1).tolist()
        # 获取batch中每个token的实体
        value = []
        for batch, token_label in enumerate(tokens_label):
            start = -1
            label_name = ""
            items = []
            for i, label in enumerate(token_label):
                if (self.label_names[label] == "O"
                        or "B-" in self.label_names[label]) and start >= 0:
                    entity = input_data[batch][start:i - 1]
                    if isinstance(entity, list):
                        entity = "".join(entity)
                    items.append({
                        "pos": [start, i - 2],
                        "entity": entity,
                        "label": label_name,
                    })
                    start = -1
                if "B-" in self.label_names[label]:
                    start = i - 1
                    label_name = self.label_names[label][2:]
            if start >= 0:
                items.append({
                    "pos": [start, len(token_label) - 1],
                    "entity":
                    input_data[batch][start:len(token_label) - 1],
                    "label":
                    ""
                })
            value.append(items)

        out_dict = {"value": value, "tokens_label": tokens_label}
        return out_dict

    def infer(self, data):
        return self.inference_backend.infer(data)

    def predict(self, input_data: list):
        preprocess_result = self.preprocess(input_data)
        infer_result = self.infer(preprocess_result)
        result = self.postprocess(infer_result, input_data)
        self.printer(result, input_data)
        return result
