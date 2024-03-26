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

import os
import re

import fastdeploy as fd
import numpy as np

from paddlenlp.trainer.argparser import strtobool
from paddlenlp.transformers import AutoTokenizer


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="The directory of model.")
    parser.add_argument("--slot_label_path", type=str, default="", help="Path of the slot label file.")
    parser.add_argument("--intent_label_path", type=str, default="", help="Path of the intent label file.")
    parser.add_argument("--model_prefix", type=str, default="infer_model", help="The model and params file prefix.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["gpu", "cpu"],
        help="Type of inference device, support 'cpu' or 'gpu'.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle",
        choices=["onnx_runtime", "paddle", "openvino", "tensorrt", "paddle_tensorrt"],
        help="The inference runtime backend.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument("--max_length", type=int, default=16, help="The max length of sequence.")
    parser.add_argument("--cpu_num_threads", type=int, default=1, help="The number of threads when inferring on cpu.")
    parser.add_argument("--use_trt_fp16", type=strtobool, default=False, help="Wheter to use FP16 mode")
    parser.add_argument(
        "--use_fast",
        type=strtobool,
        default=True,
        help="Whether to use fast_tokenizer to accelarate the tokenization.",
    )
    return parser.parse_args()


def batchify_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class Predictor(object):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=args.use_fast)
        self.runtime = self.create_fd_runtime(args)
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.slot_label_map = {}
        self.intent_label_map = {}

        slot_label_path = self.get_actual_path(args.slot_label_path, "slots_label.txt", args)
        if not os.path.exists(slot_label_path):
            raise ValueError("Slot label path doesn't exist")
        with open(slot_label_path, "r") as f:
            for i, label in enumerate(f):
                self.slot_label_map[i] = label.rstrip("\n")

        intent_label_path = self.get_actual_path(args.intent_label_path, "intent_label.txt", args)
        if not os.path.exists(intent_label_path):
            raise ValueError("Intent label path doesn't exist")
        with open(intent_label_path, "r") as f:
            for i, label in enumerate(f):
                self.intent_label_map[i] = label.rstrip("\n")

    def get_actual_path(self, path, default_path, args):
        if os.path.exists(path):
            return path
        return os.path.join(args.model_dir, default_path)

    def create_fd_runtime(self, args):
        option = fd.RuntimeOption()
        model_path = os.path.join(args.model_dir, args.model_prefix + ".pdmodel")
        params_path = os.path.join(args.model_dir, args.model_prefix + ".pdiparams")
        option.set_model_path(model_path, params_path)
        if args.device == "cpu":
            option.use_cpu()
            option.set_cpu_thread_num(args.cpu_num_threads)
        else:
            option.use_gpu()
        if args.backend == "paddle":
            option.use_paddle_infer_backend()
        elif args.backend == "onnx_runtime":
            option.use_ort_backend()
        elif args.backend == "openvino":
            option.use_openvino_backend()
        else:
            option.use_trt_backend()
            if args.backend == "paddle_tensorrt":
                option.enable_paddle_to_trt()
                option.enable_paddle_trt_collect_shape()
            trt_file = os.path.join(args.model_dir, "infer.trt")
            option.set_trt_input_shape(
                "input_ids",
                min_shape=[1, 1],
                opt_shape=[args.batch_size, args.max_length],
                max_shape=[args.batch_size, args.max_length],
            )
            if args.use_trt_fp16:
                option.enable_trt_fp16()
                trt_file = trt_file + ".fp16"
            option.set_trt_cache_file(trt_file)
        return fd.Runtime(option)

    def preprocess(self, data):
        data = self.tokenizer(
            data, max_length=self.max_length, padding=True, truncation=True, return_attention_mask=True
        )
        input_ids_name = self.runtime.get_input_info(0).name
        input_map = {
            input_ids_name: np.array(data["input_ids"], dtype="int32"),
        }
        attention_mask = data["attention_mask"]

        return input_map, attention_mask

    def infer(self, input_map):
        results = self.runtime.infer(input_map)
        return results

    def intent_cls_postprocess(self, intent_logits):
        max_value = np.max(intent_logits, axis=1, keepdims=True)
        exp_data = np.exp(intent_logits - max_value)
        probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        out_dict = {"intent": probs.argmax(axis=-1), "confidence": probs.max(axis=-1)}
        return out_dict

    def slot_cls_postprocess(self, slot_logits, input_data, attention_masks):
        batch_preds = slot_logits.argmax(axis=-1).tolist()

        value = []
        for batch, preds in enumerate(batch_preds):
            start = -1
            label_name = ""
            items = []
            token_length = len(input_data[batch])

            mask_token_ids = self.tokenizer.get_special_tokens_mask(input_data[batch], already_has_special_tokens=True)
            attention_mask = attention_masks[batch]

            for i, pred in enumerate(preds):
                slot_label_name = self.slot_label_map[pred]

                if (slot_label_name[:2] in ["O", "B-"] or i == token_length - 1) and start >= 0:

                    entity = []
                    for index in range(start, i):
                        if mask_token_ids[index] == 1 or not attention_mask[index]:
                            continue

                        token_char = self.tokenizer.decode(input_data[batch][index])

                        # tokenizer's result may contain '#' subword, and there may be spaces between digits
                        token_char = re.sub(r"(?<=\d) +(?=\d)", "", token_char.replace("#", ""))

                        entity.append(token_char.strip())

                    if len(entity) > 0:
                        items.append(
                            {
                                "slot": label_name,
                                "entity": "".join(entity),
                                "pos": [start, i - 1],
                            }
                        )
                    start = -1

                # if "B-" in self.slot_label_map[pred]:
                if slot_label_name[:2] == "B-":
                    start = i
                    label_name = self.slot_label_map[pred][2:]

            value.append(items)
        out_dict = {"value": value}
        return out_dict

    def postprocess(self, infer_data, data, attention_mask):
        intent_logits = np.array(infer_data[0])
        intent_out = self.intent_cls_postprocess(intent_logits)
        slot_logits = np.array(infer_data[1])
        slot_out = self.slot_cls_postprocess(slot_logits, data, attention_mask)
        out_list = [
            {
                "intent": self.intent_label_map[intent_out["intent"][i]],
                "confidence": intent_out["confidence"][i],
                "slot": slot_out["value"][i],
            }
            for i in range(len(data))
        ]
        return out_list

    def predict(self, data):
        input_map, attention_mask = self.preprocess(data)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result, input_map["input_ids"], attention_mask)
        return output


if __name__ == "__main__":
    args = parse_arguments()
    predictor = Predictor(args)
    data = ["来一首周华健的花心", "播放我们都一样", "到信阳市汽车配件城", "打电话给18866668888"]
    batch_data = batchify_text(data, args.batch_size)
    j = 0
    for batch in batch_data:
        output = predictor.predict(batch)
        for out in output:
            print(f"No. {j} text = {data[j]}")
            print(out)
            j += 1
