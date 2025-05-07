# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle.inference as paddle_infer

from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.env import (
    PADDLE_INFERENCE_MODEL_SUFFIX,
    PADDLE_INFERENCE_WEIGHTS_SUFFIX,
)


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Directory containing model and tokenizer files.")
    parser.add_argument("--slot_label_path", type=str, default="", help="Slot label file path.")
    parser.add_argument("--intent_label_path", type=str, default="", help="Intent label file path.")
    parser.add_argument("--model_prefix", type=str, default="infer_model", help="Model prefix (default: infer_model).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--max_length", type=int, default=16, help="Max sequence length.")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Device for inference.")
    return parser.parse_args()


def batchify_text(texts, batch_size):
    return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]


class PaddlePredictor:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.config = self._create_config(args)
        self.predictor = paddle_infer.create_predictor(self.config)
        self.input_handle = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        self.intent_output = self.predictor.get_output_handle(self.predictor.get_output_names()[0])
        self.slot_output = self.predictor.get_output_handle(self.predictor.get_output_names()[1])

        self.slot_label_map = self._load_label_map(self._resolve_path(args.slot_label_path, "slots_label.txt", args))
        self.intent_label_map = self._load_label_map(
            self._resolve_path(args.intent_label_path, "intent_label.txt", args)
        )

    def _resolve_path(self, path, default_filename, args):
        return path if os.path.exists(path) else os.path.join(args.model_dir, default_filename)

    def _load_label_map(self, filepath):
        with open(filepath, "r") as f:
            return {i: line.strip() for i, line in enumerate(f)}

    def _create_config(self, args):
        model_path = os.path.join(args.model_dir, f"{args.model_prefix}{PADDLE_INFERENCE_MODEL_SUFFIX}")
        params_path = os.path.join(args.model_dir, f"{args.model_prefix}{PADDLE_INFERENCE_WEIGHTS_SUFFIX}")
        config = paddle_infer.Config(model_path, params_path)

        if args.device == "gpu":
            config.enable_use_gpu(100, 0)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(2)

        config.switch_ir_optim(True)
        config.enable_memory_optim()
        return config

    def preprocess(self, texts):
        encoded = self.tokenizer(texts, max_length=self.max_length, padding=True, truncation=True)
        return np.array(encoded["input_ids"]).astype("int32"), texts

    def infer(self, input_ids):
        self.input_handle.copy_from_cpu(input_ids)
        self.predictor.run()
        intent_result = self.intent_output.copy_to_cpu()
        slot_result = self.slot_output.copy_to_cpu()
        return intent_result, slot_result

    def intent_cls_postprocess(self, intent_logits):
        probs = np.exp(intent_logits - np.max(intent_logits, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return {
            "intent": np.argmax(probs, axis=-1),
            "confidence": np.max(probs, axis=-1),
        }

    def slot_cls_postprocess(self, slot_logits, raw_texts):
        preds = slot_logits.argmax(axis=-1)
        results = []
        for i, pred_seq in enumerate(preds):
            items = []
            start, label_name = -1, ""
            for j, label_id in enumerate(pred_seq):
                label = self.slot_label_map.get(label_id, "O")
                if label.startswith("B-"):
                    if start != -1:
                        items.append({"slot": label_name, "entity": "".join(raw_texts[i][start:j])})
                    start = j
                    label_name = label[2:]
                elif label == "O" and start != -1:
                    items.append({"slot": label_name, "entity": "".join(raw_texts[i][start:j])})
                    start = -1
            if start != -1:
                items.append({"slot": label_name, "entity": "".join(raw_texts[i][start:])})
            results.append(items)
        return results

    def predict(self, texts):
        input_ids, raw_texts = self.preprocess(texts)
        intent_logits, slot_logits = self.infer(input_ids)
        intent_result = self.intent_cls_postprocess(intent_logits)
        slot_result = self.slot_cls_postprocess(slot_logits, raw_texts)
        return [
            {
                "intent": self.intent_label_map[intent_result["intent"][i]],
                "confidence": float(intent_result["confidence"][i]),
                "slot": slot_result[i],
            }
            for i in range(len(texts))
        ]


if __name__ == "__main__":
    args = parse_arguments()
    predictor = PaddlePredictor(args)

    # 示例输入
    data = ["来一首周华健的花心", "播放我们都一样", "到信阳市汽车配件城"]
    batches = batchify_text(data, args.batch_size)

    idx = 0
    for batch in batches:
        result = predictor.predict(batch)
        for r in result:
            print(f"No. {idx} text = {data[idx]}")
            print(r)
            idx += 1
