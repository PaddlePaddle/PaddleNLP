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
import distutils.util
import os

import numpy as np
import paddle
from tokenizer import LLaMATokenizer
from utils import left_padding


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="The directory of model.")
    parser.add_argument("--model_prefix", type=str, default="llama", help="The model and params file prefix.")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Type of inference device, support 'cpu' or 'gpu'.",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="The batch size of data.")
    parser.add_argument("--max_length", type=int, default=128, help="The max length of sequence.")
    parser.add_argument("--log_interval", type=int, default=10, help="The interval of logging.")
    parser.add_argument("--use_fp16", type=distutils.util.strtobool, default=False, help="Wheter to use FP16 mode")
    parser.add_argument("--cpu_threads", type=int, default=1, help="Number of threads to predict when using cpu.")
    parser.add_argument("--device_id", type=int, default=0, help="Select which gpu device to train model.")
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
        self.tokenizer = LLaMATokenizer.from_pretrained(args.model_dir)
        self.batch_size = args.batch_size
        self.max_length = args.max_length

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
        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config)

        self.input_handles = [self.predictor.get_input_handle(name) for name in self.predictor.get_input_names()]

        self.output_handle = self.predictor.get_output_handle(self.predictor.get_output_names()[0])

    def preprocess(self, input_text):
        inputs = self.tokenizer(input_text)
        inputs = left_padding(inputs, self.tokenizer.pad_token_id)
        input_map = {
            "input_ids": np.array(inputs["input_ids"], dtype="int64"),
        }
        return input_map

    def infer(self, input_map):
        input_ids = input_map["input_ids"]
        self.input_handles[0].copy_from_cpu(input_ids)
        self.predictor.run()
        results = self.output_handle.copy_to_cpu()
        return results

    def postprocess(self, infer_data):
        result = []
        for x in infer_data[0].tolist():
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
    predictor = Predictor(args)
    all_texts = [
        "answer: Denver Broncos context: Super Bowl 50 was an American football game to "
        "determine the champion of the National Football League (NFL) for the 2015 season. "
        "The American Football Conference (AFC) champion Denver Broncos defeated the National "
        "Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. "
        "The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. "
        'As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, '
        "as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals "
        '(under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50. </s>'
        "question: "
    ]
    all_texts = ["My name is"]
    batch_texts = batchfy_text(all_texts, args.batch_size)
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for text, result in zip(texts, outputs["result"]):
            print("{} \n {}".format(text, result))
