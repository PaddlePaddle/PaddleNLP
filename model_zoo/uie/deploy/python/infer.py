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

import argparse
import distutils.util
import os
from pprint import pprint

import fastdeploy as fd
from fastdeploy.text import SchemaLanguage, UIEModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="The directory of model, params and vocab file.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Type of inference device, support 'cpu' or 'gpu'.",
    )
    parser.add_argument("--multilingual", action="store_true", help="Whether is the multilingual model.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument("--device_id", type=int, default=0, help="device(gpu) id")
    parser.add_argument("--max_length", type=int, default=128, help="The max length of sequence.")
    parser.add_argument(
        "--position_prob",
        default=0.5,
        type=float,
        help="Probability threshold for start/end index probabiliry.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle_inference",
        choices=["onnx_runtime", "paddle_inference", "openvino", "paddle_tensorrt"],
        help="The inference runtime backend.",
    )
    parser.add_argument(
        "--cpu_threads", type=int, default=1, help="The number of threads to execute inference in cpu device."
    )
    parser.add_argument("--use_fp16", type=distutils.util.strtobool, default=False, help="Use FP16 mode")
    return parser.parse_args()


def create_option(args):
    option = fd.RuntimeOption()
    # Set device
    if args.device == "cpu":
        option.use_cpu()
        option.set_cpu_thread_num(args.cpu_threads)
    else:
        option.use_gpu(args.device_id)

    # Set backend
    if args.backend == "onnx_runtime":
        option.use_ort_backend()
    elif args.backend == "paddle":
        option.use_paddle_infer_backend()
    elif args.backend == "openvino":
        option.use_openvino_backend()
    elif args.backend == "paddle_tensorrt":
        option.use_paddle_infer_backend()
        option.paddle_infer_option.collect_trt_shape = True
        option.paddle_infer_option.enable_trt = True
        # Only useful for single stage predict
        option.trt_option.set_shape(
            "input_ids", [1, 1], [args.batch_size, args.max_length], [args.batch_size, args.max_length]
        )
        option.trt_option.set_shape(
            "token_type_ids", [1, 1], [args.batch_size, args.max_length], [args.batch_size, args.max_length]
        )
        option.trt_option.set_shape(
            "position_ids", [1, 1], [args.batch_size, args.max_length], [args.batch_size, args.max_length]
        )
        option.trt_option.set_shape(
            "attention_mask", [1, 1], [args.batch_size, args.max_length], [args.batch_size, args.max_length]
        )
        trt_file = os.path.join(args.model_dir, "inference.trt")
        if args.use_fp16:
            option.trt_option.enable_fp16 = True
            trt_file = trt_file + ".fp16"
        option.trt_option.serialize_file = trt_file
    return option


if __name__ == "__main__":
    args = parse_arguments()
    option = create_option(args)

    model_path = os.path.join(args.model_dir, "model.pdmodel")
    param_path = os.path.join(args.model_dir, "model.pdiparams")
    vocab_path = os.path.join(args.model_dir, "vocab.txt")
    texts = [
        '"北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。"',
        "原告赵六，2022年5月29日生\n委托代理人孙七，深圳市C律师事务所律师。\n被告周八，1990年7月28日出生\n委托代理人吴九，山东D律师事务所律师",
    ]
    schema1 = ["法院", {"原告": "委托代理人"}, {"被告": "委托代理人"}]
    schema2 = [{"原告": ["出生日期", "委托代理人"]}, {"被告": ["出生日期", "委托代理人"]}]
    uie = UIEModel(
        model_path,
        param_path,
        vocab_path,
        position_prob=args.position_prob,
        max_length=args.max_length,
        schema=schema1,
        batch_size=args.batch_size,
        runtime_option=option,
        schema_language=SchemaLanguage.ZH,
    )
    print("-----------------------------")
    outputs = uie.predict(texts, return_dict=True)
    print(outputs)
    for text, output in zip(texts, outputs):
        print("1. Input text: ")
        print(text)
        print("2. Input schema: ")
        print(schema1)
        print("3. Result: ")
        pprint(output)
        print("-----------------------------")

    uie.set_schema(schema2)
    outputs = uie.predict(texts, return_dict=True)
    for text, output in zip(texts, outputs):
        print("1. Input text: ")
        print(text)
        print("2. Input schema: ")
        print(schema2)
        print("3. Result: ")
        pprint(output)
        print("-----------------------------")
