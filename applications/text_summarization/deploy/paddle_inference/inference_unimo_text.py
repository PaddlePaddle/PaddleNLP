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
import numpy as np
from pprint import pprint

import paddle
from paddle import inference

from paddlenlp.transformers import UNIMOLMHeadModel, UNIMOTokenizer
from paddlenlp.ops.ext_utils import load
import os


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_model_dir",
                        default="./infer_model",
                        type=str,
                        help="Path to save inference model of UNIMOText. ")
    parser.add_argument("--device",
                        default="gpu",
                        choices=["gpu", "cpu", "xpu"],
                        help="Device selected for inference.")
    parser.add_argument(
        "--use_tensorrt",
        default=False,
        type=eval,
        choices=[True, False],
        help="Whether to use inference engin TensorRT when using gpu.")
    parser.add_argument('--enable_mkldnn',
                        default=False,
                        type=eval,
                        choices=[True, False],
                        help='Enable to use mkldnn to speed up when using cpu.')
    parser.add_argument('--cpu_threads',
                        default=10,
                        type=int,
                        help='Number of threads to predict when using cpu.')
    parser.add_argument("--precision",
                        default="fp32",
                        type=str,
                        choices=["fp32", "fp16", "int8"],
                        help='The tensorrt precision.')
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="Batch size per GPU/CPU for training.")

    args = parser.parse_args()
    return args


def setup_predictor(args):
    """Setup inference predictor."""
    # Load FasterTransformer lib.
    load("FasterTransformer", verbose=True)
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
        # #### simulate pipeline_service
        # config.enable_memory_optim()
        # config.switch_ir_optim(False)
        # config.switch_use_feed_fetch_ops(False)
        # config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        # config.set_cpu_math_library_num_threads(12)
        # config.enable_use_gpu(100, 0)
        # #### simulate pipeline_service
        precision_map = {
            "fp16": inference.PrecisionType.Half,
            "fp32": inference.PrecisionType.Float32,
            "int8": inference.PrecisionType.Int8
        }
        precision_mode = precision_map[args.precision]
        if args.use_tensorrt:
            config.enable_tensorrt_engine(max_batch_size=args.batch_size,
                                          min_subgraph_size=30,
                                          precision_mode=precision_mode)
    elif args.device == "cpu":
        config.disable_gpu()
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            # config.enable_oneDNN()
            # config.set_oneDNN_cache_capacity(10)
            config.enable_mkldnn()
            config.set_mkldnn_cache_capacity(10)

        config.set_cpu_math_library_num_threads(args.cpu_threads)
    elif args.device == "xpu":
        config.enable_xpu(100)
    predictor = inference.create_predictor(config)
    return predictor


def postprocess_response(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.mask_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    return tokens


def infer(args, predictor):
    """Use predictor to inference."""
    tokenizer = UNIMOTokenizer.from_pretrained('unimo-text-1.0')

    inputs = "深度学习是人工智能的核心技术领域。百度飞桨作为中国首个自主研发、功能丰富、开源开放的产业级深度学习平台,将从多层次技术产品、产业AI人才培养和强大的生态资源支持三方面全面护航企业实现快速AI转型升级。"

    data = tokenizer.gen_encode(inputs,
                                add_start_token_for_decoding=True,
                                return_length=True,
                                is_split_into_words=False)

    input_handles = {}
    for name in predictor.get_input_names():
        input_handles[name] = predictor.get_input_handle(name)
        if name == "attention_mask":
            input_handles[name].copy_from_cpu(
                np.expand_dims(np.asarray(data[name], dtype="float32"),
                               axis=(0, 1)))
        else:
            input_handles[name].copy_from_cpu(
                np.asarray(data[name], dtype="int32").reshape([1, -1]))
            # print(name, np.asarray(data[name], dtype="int32").reshape([1, -1]))

    output_handles = [
        predictor.get_output_handle(name)
        for name in predictor.get_output_names()
    ]

    predictor.run()

    output = [output_handle.copy_to_cpu() for output_handle in output_handles]

    # print('output', output)
    for sample in output[0].transpose([1, 0]).tolist():
        print("".join(postprocess_response(sample, tokenizer)))


if __name__ == "__main__":
    args = setup_args()
    pprint(args)
    predictor = setup_predictor(args)
    infer(args, predictor)
