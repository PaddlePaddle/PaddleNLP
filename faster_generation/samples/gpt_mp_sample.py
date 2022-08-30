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
import time
import paddle
import numpy as np
from pprint import pprint
from paddlenlp.ops import enable_ft_para, get_ft_para_conf
from paddlenlp.transformers import GPTLMHeadModel, GPTChineseTokenizer, GPTTokenizer

MODEL_CLASSES = {
    "gpt-cpm-large-cn": (GPTLMHeadModel, GPTChineseTokenizer),
    "gpt-cpm-small-cn-distill": (GPTLMHeadModel, GPTChineseTokenizer),
    "gpt2-en": (GPTLMHeadModel, GPTTokenizer),
    "gpt2-medium-en": (GPTLMHeadModel, GPTTokenizer),
    "gpt2-large-en": (GPTLMHeadModel, GPTTokenizer),
    "gpt2-xl-en": (GPTLMHeadModel, GPTTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="gpt-cpm-large-cn",
        choices=list(MODEL_CLASSES.keys()),
        help="The model name to specify which gpt to use. It can be " +
        ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--max_length",
                        default=50,
                        type=int,
                        help="Maximum output length.")
    parser.add_argument(
        "--topk",
        default=1,
        type=int,
        help=
        "The number of highest probability tokens to keep for top-k-sampling.")
    parser.add_argument("--topp",
                        default=1.0,
                        type=float,
                        help="The cumulative probability for top-p-filtering.")
    parser.add_argument("--temperature",
                        default=1.0,
                        type=float,
                        help="The temperature to set.")
    parser.add_argument("--tensor_para_size",
                        default=2,
                        type=int,
                        help="The size for tensor parallel.")
    parser.add_argument("--layer_para_size",
                        default=1,
                        type=int,
                        help="The size for layer parallel.")
    parser.add_argument(
        "--layer_para_batch_size",
        default=None,
        type=int,
        help="The local batch size for pipeline parallel."
        "It is suggested to use `batch_size // layer_para_size`.")
    parser.add_argument("--use_fp16",
                        action="store_true",
                        help="Whether to use fp16 to predict.")
    parser.add_argument("--profile",
                        action="store_true",
                        help="Whether to profile.")
    args = parser.parse_args()
    return args


def profile(batch_size, total_step=50, warmup_step=10, rank=0):

    def _wrapper(func):

        def _impl(*args, **kwargs):
            for i in range(total_step):
                if i == warmup_step:
                    paddle.device.cuda.synchronize()
                    start_time = time.time()
                out = func(*args, **kwargs)
            paddle.device.cuda.synchronize()
            end_time = time.time()
            if rank is None or get_ft_para_conf().rank == rank:
                time_interval = end_time - start_time
                num_batch = total_step - warmup_step
                print("Latency: %2fs, QPS: %2f" %
                      (time_interval / num_batch,
                       num_batch * batch_size / time_interval))
            return out

        return _impl

    return _wrapper


def main(args):
    if args.use_fp16:
        paddle.set_default_dtype("float16")
    enable_ft_para(
        args.tensor_para_size, args.layer_para_size,
        args.batch_size // args.layer_para_size
        if args.layer_para_batch_size is None else args.layer_para_batch_size)
    # TODO(guosheng): Maybe device can be set in `enable_ft_para`
    paddle.set_device("gpu:" + str(get_ft_para_conf().rank))

    model_name = args.model_name
    if args.profile:
        MODEL_CLASSES[model_name][0].generate = profile(args.batch_size)(
            MODEL_CLASSES[model_name][0].generate)
    tokenizer = MODEL_CLASSES[model_name][-1].from_pretrained(model_name)
    model = MODEL_CLASSES[model_name][0].from_pretrained(model_name,
                                                         load_state_as_np=True)
    model.eval()

    # NOTE: When using prompt, open this and replace the text with what you want.
    input = '花间一壶酒，独酌无相亲。举杯邀明月，'
    # input = '一时黛玉进了荣府，下了车。众嬷嬷引着，便往东转弯，'
    # input = '爱因斯坦曾经说过：'
    input_ids = tokenizer(input)["input_ids"]
    # NOTE: When generating from the beginning, open this.
    # input_ids = [tokenizer.eos_token_id]
    input_ids = [input_ids] * args.batch_size

    inputs_ids = paddle.to_tensor(input_ids, dtype='int32')
    outputs, _ = model.generate(input_ids=inputs_ids,
                                max_length=args.max_length,
                                decode_strategy='sampling',
                                top_k=args.topk,
                                top_p=args.topp,
                                temperature=args.temperature,
                                use_faster=True)

    # Only make the first process to output.
    if get_ft_para_conf().rank == 0:
        for i in range(len(outputs)):
            result = tokenizer.convert_ids_to_string(
                outputs[i].numpy().tolist())
            print("Result:", result)


if __name__ == "__main__":
    args = parse_args()
    pprint(args)
    main(args)
