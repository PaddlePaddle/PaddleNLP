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

import argparse

from predictor import Predictor


def parse_args():
    # yapf: disable
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_path_prefix", type=str, required=True, help="The path prefix of inference model to be used.")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size per GPU for inference.")
    parser.add_argument("--max_seq_length", default=512, type=int, help="The maximum input sequence length. Sequences longer than this will be split automatically.")
    parser.add_argument("--task_type", default="ner", type=str, choices=["ner", "cls", "mrc"], help="Specify the task type.")
    parser.add_argument("--lang", default="en", type=str, choices=["ch", "en"], help="Specify the task type.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    args = parser.parse_args()
    # yapf: enable
    return args


def main():
    args = parse_args()
    if args.task_type == "mrc":
        args.questions = [
            [
                "公司的类型属于什么？",
                "杨小峰的住所是在哪里？",
                "这个公司的法定代表人叫什么？",
                "花了多少钱进行注册的这个公司？",
                "公司在什么时候成立的？",
                "杨小峰是什么身份？",
                "91510107749745776R代表的是什么？",
            ],
        ]
        docs = ["./images/mrc_sample.jpg"]
    elif args.task_type == "cls":
        docs = ["./images/cls_sample.jpg"]
    elif args.task_type == "ner":
        docs = ["./images/ner_sample.jpg"]
    else:
        raise ValueError("Unspport task type: {}".format(args.task_type))

    predictor = Predictor(args)

    outputs = predictor.predict(docs)
    import pprint

    pprint.sorted = lambda x, key=None: x
    pprint.pprint(outputs)


if __name__ == "__main__":
    main()
