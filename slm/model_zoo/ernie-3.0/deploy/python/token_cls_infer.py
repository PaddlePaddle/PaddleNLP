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
import distutils.util
import os

import fastdeploy as fd
import numpy as np

from paddlenlp.transformers import AutoTokenizer


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="The directory of model.")
    parser.add_argument("--vocab_path", type=str, default="", help="The path of tokenizer vocab.")
    parser.add_argument("--model_prefix", type=str, default="model", help="The model and params file prefix.")
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
    parser.add_argument("--max_length", type=int, default=128, help="The max length of sequence.")
    parser.add_argument("--log_interval", type=int, default=10, help="The interval of logging.")
    parser.add_argument("--use_fp16", type=distutils.util.strtobool, default=False, help="Wheter to use FP16 mode")
    return parser.parse_args()


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class ErnieForTokenClassificationPredictor(object):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.runtime = self.create_fd_runtime(args)
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.label_names = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]

    def create_fd_runtime(self, args):
        option = fd.RuntimeOption()
        model_path = os.path.join(args.model_dir, args.model_prefix + ".pdmodel")
        params_path = os.path.join(args.model_dir, args.model_prefix + ".pdiparams")
        option.set_model_path(model_path, params_path)
        if args.device == "cpu":
            option.use_cpu()
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
            option.set_trt_input_shape(
                "token_type_ids",
                min_shape=[1, 1],
                opt_shape=[args.batch_size, args.max_length],
                max_shape=[args.batch_size, args.max_length],
            )
            if args.use_fp16:
                option.enable_trt_fp16()
                trt_file = trt_file + ".fp16"
            option.set_trt_cache_file(trt_file)
        return fd.Runtime(option)

    def preprocess(self, texts):
        is_split_into_words = False
        if isinstance(texts[0], list):
            is_split_into_words = True
        data = self.tokenizer(
            texts, max_length=self.max_length, padding=True, truncation=True, is_split_into_words=is_split_into_words
        )
        input_ids_name = self.runtime.get_input_info(0).name
        token_type_ids_name = self.runtime.get_input_info(1).name
        input_map = {
            input_ids_name: np.array(data["input_ids"], dtype="int64"),
            token_type_ids_name: np.array(data["token_type_ids"], dtype="int64"),
        }
        return input_map

    def infer(self, input_map):
        results = self.runtime.infer(input_map)
        return results

    def postprocess(self, infer_data, input_data):
        result = np.array(infer_data[0])
        tokens_label = result.argmax(axis=-1).tolist()
        value = []
        for batch, token_label in enumerate(tokens_label):
            start = -1
            label_name = ""
            items = []
            for i, label in enumerate(token_label):
                if (self.label_names[label] == "O" or "B-" in self.label_names[label]) and start >= 0:
                    entity = input_data[batch][start : i - 1]
                    if isinstance(entity, list):
                        entity = "".join(entity)
                    if len(entity) == 0:
                        break
                    items.append(
                        {
                            "pos": [start, i - 2],
                            "entity": entity,
                            "label": label_name,
                        }
                    )
                    start = -1
                if "B-" in self.label_names[label]:
                    start = i - 1
                    label_name = self.label_names[label][2:]
            value.append(items)

        out_dict = {"value": value, "tokens_label": tokens_label}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result, texts)
        return output


def token_cls_print_ret(infer_result, input_data):
    rets = infer_result["value"]
    for i, ret in enumerate(rets):
        print("input data:", input_data[i])
        print("The model detects all entities:")
        for iterm in ret:
            print("entity:", iterm["entity"], "  label:", iterm["label"], "  pos:", iterm["pos"])
        print("-----------------------------")


if __name__ == "__main__":
    args = parse_arguments()
    predictor = ErnieForTokenClassificationPredictor(args)
    texts = ["北京的涮肉，重庆的火锅，成都的小吃都是极具特色的美食。", "乔丹、科比、詹姆斯和姚明都是篮球界的标志性人物。"]
    batch_data = batchfy_text(texts, args.batch_size)
    for data in batch_data:
        outputs = predictor.predict(data)
        token_cls_print_ret(outputs, data)
