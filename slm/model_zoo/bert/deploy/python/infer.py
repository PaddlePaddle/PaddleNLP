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

import argparse
import os

import numpy as np
from paddle import inference
from scipy.special import softmax

from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.env import (
    PADDLE_INFERENCE_MODEL_SUFFIX,
    PADDLE_INFERENCE_WEIGHTS_SUFFIX,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="The directory of model.")
    parser.add_argument("--model_prefix", type=str, default="model", help="Prefix of the model file (no extension).")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu", help="Device for inference.")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID if using GPU.")
    parser.add_argument("--cpu_threads", type=int, default=4, help="CPU threads if using CPU.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference.")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length.")
    return parser.parse_args()


def batchfy_text(texts, batch_size):
    return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]


class Predictor(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.max_length = args.max_length

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

        model_file = os.path.join(args.model_dir, args.model_prefix + f"{PADDLE_INFERENCE_MODEL_SUFFIX}")
        params_file = os.path.join(args.model_dir, args.model_prefix + f"{PADDLE_INFERENCE_WEIGHTS_SUFFIX}")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Params file not found: {params_file}")

        config = inference.Config(model_file, params_file)
        if args.device == "gpu":
            config.enable_use_gpu(100, args.device_id)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(args.cpu_threads)

        config.switch_use_feed_fetch_ops(False)
        self.predictor = inference.create_predictor(config)
        self.input_handles = [self.predictor.get_input_handle(name) for name in self.predictor.get_input_names()]
        self.output_handle = self.predictor.get_output_handle(self.predictor.get_output_names()[0])

    def preprocess(self, texts):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=True,
        )
        input_ids = np.array(encoded["input_ids"], dtype="int64")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int64")
        return input_ids, token_type_ids

    def infer(self, input_ids, token_type_ids):
        self.input_handles[0].copy_from_cpu(input_ids)
        self.input_handles[1].copy_from_cpu(token_type_ids)
        self.predictor.run()
        return self.output_handle.copy_to_cpu()

    def postprocess(self, logits):
        probs = softmax(logits, axis=1)
        return {"label": probs.argmax(axis=1), "confidence": probs}

    def predict(self, texts):
        input_ids, token_type_ids = self.preprocess(texts)
        logits = self.infer(input_ids, token_type_ids)
        return self.postprocess(logits)


if __name__ == "__main__":
    args = parse_arguments()
    predictor = Predictor(args)

    texts_ds = [
        "against shimmering cinematography that lends the setting the ethereal beauty of an asian landscape painting",
        "the situation in a well-balanced fashion",
        "at achieving the modest , crowd-pleasing goals it sets for itself",
        "so pat it makes your teeth hurt",
        "this new jangle of noise , mayhem and stupidity must be a serious contender for the title .",
    ]
    label_map = {0: "negative", 1: "positive"}

    batch_texts = batchfy_text(texts_ds, args.batch_size)

    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for i, sentence in enumerate(texts):
            label = outputs["label"][i]
            confidence = outputs["confidence"][i]
            print(
                f"Batch id: {bs}, example id: {i}, sentence: {sentence}, "
                f"label: {label_map[label]}, negative prob: {confidence[0]:.4f}, positive prob: {confidence[1]:.4f}."
            )
