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

import numpy as np
import paddle

from paddlenlp.trainer.argparser import strtobool
from paddlenlp.transformers import AutoTokenizer


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="inference", help="The directory of model.")
    parser.add_argument("--model_prefix", type=str, default="llama", help="The model and params file prefix.")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Type of inference device, support 'cpu' or 'gpu'.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument("--src_length", type=int, default=800, help="decoding length")
    parser.add_argument("--top_k", type=int, default=1, help="top_p parameter for decoding")
    parser.add_argument("--temperature", type=float, default=1, help="temperature parameter for decoding")
    parser.add_argument("--top_p", type=int, default=0, help="top_p parameter for decoding")
    parser.add_argument(
        "--use_pre_caches",
        default="False",
        type=strtobool,
        help="whether use pre_caches",
    )
    return parser.parse_args()


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class Predictor(object):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.batch_size = args.batch_size
        self.src_length = args.src_length

        model_path = os.path.join(args.model_dir, args.model_prefix + ".pdmodel")
        params_path = os.path.join(args.model_dir, args.model_prefix + ".pdiparams")
        config = paddle.inference.Config(model_path, params_path)

        if args.device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
        elif args.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
        config.disable_glog_info()
        self.args = args
        self.predictor = paddle.inference.create_predictor(config)

    def preprocess(self, input_text):
        inputs = self.tokenizer(
            input_text,
            padding=True,
            return_tensors="np",
            max_length=self.src_length,
            return_attention_mask=True,
            return_position_ids=True,
        )
        inputs["max_length"] = np.array(args.src_length, dtype="int64")
        inputs["top_p"] = np.array(args.top_p, dtype="float32")
        inputs["top_k"] = np.array(args.top_k, dtype="int64")
        inputs["temperature"] = np.array(args.temperature, dtype="float32")

        input_ids_shape = inputs["input_ids"].shape

        # create 4-d attention-mask
        attention_mask = (
            paddle.tril(paddle.ones([input_ids_shape[-1], input_ids_shape[-1]], dtype="float32"))
            .unsqueeze_(0)
            .unsqueeze_(0)
        )
        attention_mask = (1 - attention_mask) * paddle.finfo(paddle.float16).min

        inputs["attention_mask"] = attention_mask
        use_pre_caches = self.args.use_pre_caches

        if use_pre_caches:
            pre_caches_numpy = np.load(os.path.join(self.args.model_dir, "pre_caches.npy"))

            pre_caches = np.split(pre_caches_numpy, self.config.num_hidden_layers)
            for i in range(self.config.num_hidden_layers):
                inputs["pre_caches_{}".format(i)] = pre_caches[i].transpose(1, 0, 2, 3, 4).astype("float16")
        else:
            for i in range(self.config.num_hidden_layers):
                inputs["pre_caches_{}".format(i)] = np.zeros([2, 1, 32, 128, 128]).astype("float16")

        # append pre_cache attention_mask
        if use_pre_caches:
            pre_caches_length = pre_caches[0].shape[-2]
            batch_size = inputs["input_ids"].shape[0]
            pre_cache_attention_mask = paddle.zeros(
                [batch_size, 1, inputs["input_ids"].shape[-1], pre_caches_length], dtype=attention_mask.dtype
            )
            attention_mask = paddle.concat([pre_cache_attention_mask, attention_mask], axis=3)
        else:
            pre_caches_length = 128
            batch_size = inputs["input_ids"].shape[0]
            pre_cache_attention_mask = paddle.full(
                [batch_size, 1, inputs["input_ids"].shape[-1], pre_caches_length],
                paddle.finfo(paddle.float16).min,
                dtype=attention_mask.dtype,
            )
            attention_mask = paddle.concat([pre_cache_attention_mask, attention_mask], axis=3)

        inputs["attention_mask"] = attention_mask.numpy()
        inputs["use_pre_caches"] = np.array(True, dtype=np.bool_)

        return inputs

    def infer(self, inputs):
        input_handles = {}
        # print("inputs", inputs)
        for name in self.predictor.get_input_names():
            # print("start to infer based on:", name)
            print("predictor input -> ", name)
            print(inputs[name])

            input_handles[name] = self.predictor.get_input_handle(name)
            input_handles[name].copy_from_cpu(inputs[name])

        print("start to run")
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results = output_handle.copy_to_cpu()
        print(results)
        return results

    def postprocess(self, infer_data):
        result = []
        for x in infer_data.tolist():
            sentence = self.tokenizer.decode(x, skip_special_tokens=True)
            result.append(sentence)
        out_dict = {"result": result}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output


if __name__ == "__main__":
    args = parse_arguments()
    paddle.seed(100)
    predictor = Predictor(args)
    all_texts = [
        "answer: linebacker context: The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into three turnovers, including a fumble which they recovered for a touchdown. Denver linebacker Von Miller was named Super Bowl MVP, recording five solo tackles, 2½ sacks, and two forced fumbles. </s>",
        "answer: five context: The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into three turnovers, including a fumble which they recovered for a touchdown. Denver linebacker Von Miller was named Super Bowl MVP, recording five solo tackles, 2½ sacks, and two forced fumbles. </s>",
    ]
    batch_texts = batchfy_text(all_texts, args.batch_size)
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for text, result in zip(texts, outputs["result"]):
            print("{} \n\n {}".format(text, result))
