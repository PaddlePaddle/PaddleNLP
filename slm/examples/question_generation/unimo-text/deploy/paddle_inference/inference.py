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
import os
import time
from pprint import pprint

import numpy as np
import paddle
from infer_utils import create_data_loader, postprocess_response, select_sum
from paddle import inference

from paddlenlp.datasets import load_dataset
from paddlenlp.ops.ext_utils import load
from paddlenlp.transformers import UNIMOTokenizer


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_model_dir", default="./infer_model", type=str, help="Path to save inference model of UNIMOText. "
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="unimo-text-1.0", help="The path or shortcut name of the tokenizer."
    )
    parser.add_argument(
        "--device", default="gpu", choices=["gpu", "cpu", "xpu"], help="Device selected for inference."
    )
    parser.add_argument(
        "--use_tensorrt",
        default=False,
        type=eval,
        choices=[True, False],
        help="Whether to use inference engin TensorRT when using gpu.",
    )
    parser.add_argument(
        "--enable_mkldnn",
        default=False,
        type=eval,
        choices=[True, False],
        help="Enable to use mkldnn to speed up when using cpu.",
    )
    parser.add_argument("--cpu_threads", default=10, type=int, help="Number of threads to predict when using cpu.")
    parser.add_argument(
        "--precision", default="fp32", type=str, choices=["fp32", "fp16", "int8"], help="The tensorrt precision."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--output_path", type=str, default="./predict.txt", help="The file path where the infer result will be saved."
    )
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--dataset_name", type=str, default="dureader_qg", help="The name of the dataset to load.")
    parser.add_argument("--predict_file", type=str, required=False, default=None, help="Predict data path.")
    parser.add_argument("--max_dec_len", type=int, default=20, help="The maximum sequence length of decoding.")
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The numbers of returned sequences for one input in generation.",
    )

    args = parser.parse_args()
    return args


def setup_predictor(args):
    """Setup inference predictor."""
    # Load FastGeneration lib.
    load("FastGeneration", verbose=True)
    model_file = os.path.join(args.inference_model_dir, "unimo_text.pdmodel")
    params_file = os.path.join(args.inference_model_dir, "unimo_text.pdiparams")
    if not os.path.exists(model_file):
        raise ValueError("not find model file path {}".format(model_file))
    if not os.path.exists(params_file):
        raise ValueError("not find params file path {}".format(params_file))
    config = inference.Config(model_file, params_file)
    if args.device == "gpu":
        config.enable_use_gpu(100, 0)
        config.switch_ir_optim()
        config.enable_memory_optim()
        config.disable_glog_info()

        precision_map = {
            "fp16": inference.PrecisionType.Half,
            "fp32": inference.PrecisionType.Float32,
            "int8": inference.PrecisionType.Int8,
        }
        precision_mode = precision_map[args.precision]
        if args.use_tensorrt:
            config.enable_tensorrt_engine(
                max_batch_size=args.batch_size, min_subgraph_size=30, precision_mode=precision_mode
            )
    elif args.device == "cpu":
        config.disable_gpu()
        if args.enable_mkldnn:
            config.enable_mkldnn()
            config.set_mkldnn_cache_capacity(10)

        config.set_cpu_math_library_num_threads(args.cpu_threads)
    elif args.device == "xpu":
        config.enable_xpu(100)
    predictor = inference.create_predictor(config)
    return predictor


@paddle.no_grad()
def infer_one(args, predictor, inputs=None):
    """Use predictor to inference."""
    tokenizer = UNIMOTokenizer.from_pretrained("unimo-text-1.0")

    if not inputs:
        inputs = {
            "context": "奇峰黄山千米以上的山峰有77座，整座黄山就是一座花岗岩的峰林，自古有36大峰，36小峰，最高峰莲花峰、最险峰天都峰和观日出的最佳点光明顶构成黄山的三大主峰。",
            "answer": "莲花峰",
        }

    inputs = "答案：" + inputs["answer"] + tokenizer.sep_token + "上下文：" + inputs["context"]
    data = tokenizer.gen_encode(
        inputs, add_start_token_for_decoding=True, return_length=True, is_split_into_words=False
    )

    input_handles = {}
    for name in predictor.get_input_names():
        input_handles[name] = predictor.get_input_handle(name)
        if name == "attention_mask":
            input_handles[name].copy_from_cpu(np.expand_dims(np.asarray(data[name], dtype="float32"), axis=(0, 1)))
        else:
            input_handles[name].copy_from_cpu(np.asarray(data[name], dtype="int32").reshape([1, -1]))

    output_handles = [predictor.get_output_handle(name) for name in predictor.get_output_names()]

    predictor.run()

    output = [output_handle.copy_to_cpu() for output_handle in output_handles]

    for sample in output[0][:, :, 0].tolist():
        print("".join(postprocess_response(sample, tokenizer)))


@paddle.no_grad()
def infer(args, predictor, data_loader, tokenizer):
    print("Infer begin...")
    pred_ref = []
    total_time = 0.0
    start_time = time.time()
    for step, inputs in enumerate(data_loader, 1):
        input_ids, token_type_ids, position_ids, attention_mask, seq_len = inputs
        data = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "seq_len": seq_len,
        }

        input_handles = {}
        for name in predictor.get_input_names():
            input_handles[name] = predictor.get_input_handle(name)
            if name == "attention_mask":
                input_handles[name].copy_from_cpu(np.asarray(data[name], dtype="float32"))
            else:
                input_handles[name].copy_from_cpu(np.asarray(data[name], dtype="int32"))

        output_handles = [predictor.get_output_handle(name) for name in predictor.get_output_names()]

        predictor.run()

        output = [output_handle.copy_to_cpu() for output_handle in output_handles]

        ids = output[0]
        scores = output[1]

        ids = paddle.to_tensor(ids, dtype="int32")[:, 0, :]
        scores = paddle.to_tensor(scores, dtype="float32")

        total_time += time.time() - start_time
        if step % args.logging_steps == 0:
            print("step %d - %.3fs/step" % (step, total_time / args.logging_steps))
            total_time = 0.0

        results = select_sum(ids, scores, tokenizer, args.max_dec_len, args.num_return_sequences)

        pred_ref.extend(results)
        start_time = time.time()

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for ref in pred_ref:
            fout.write(ref + "\n")

    print("\nSave inference result into: %s" % args.output_path)

    if "target" in data_loader.dataset[0].keys():
        with open(args.output_path + ".reference.txt", "w", encoding="utf-8") as fout:
            targets = [example["target"] for example in data_loader.dataset]
            for target in targets:
                fout.write(target + "\n")


if __name__ == "__main__":
    args = setup_args()
    pprint(args)

    predictor = setup_predictor(args)
    tokenizer = UNIMOTokenizer.from_pretrained(args.model_name_or_path)
    ds = load_dataset(args.dataset_name, splits="dev", data_files=args.predict_file)
    ds, data_loader = create_data_loader(ds, tokenizer, args, "test")

    time_begin = time.time()
    infer(args, predictor, data_loader, tokenizer)
    print("inference cost time:", time.time() - time_begin)
