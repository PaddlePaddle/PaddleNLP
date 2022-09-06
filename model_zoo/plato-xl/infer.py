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
import paddle
import numpy as np
from distutils.util import strtobool
from pprint import pprint
from paddlenlp.ops import enable_ft_para, get_ft_para_conf
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.transformers import UnifiedTransformerLMHeadModel, UnifiedTransformerTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_role",
                        type=strtobool,
                        default=True,
                        help="Whether to use role embeddings.")
    parser.add_argument(
        "--position_style",
        default="relative",
        choices=["continuous", "relative"],
        type=str,
        help="The type for positional embedding. Default is relative.")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
    parser.add_argument(
        "--num_return_sequences",
        default=1,
        type=int,
        help="The number of returned sequences for each sample.")
    parser.add_argument("--max_out_len",
                        default=64,
                        type=int,
                        help="Maximum output sequence length.")
    parser.add_argument("--min_out_len",
                        default=1,
                        type=int,
                        help="Minimum output sequence length.")
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
    parser.add_argument("--use_faster",
                        action="store_true",
                        help="Whether to use faster generation. ")
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help=
        "Whether to use fp16 to predict. Only available when `use_faster` is True."
    )
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


def postprocess_response(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.sep_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    response = " ".join(tokens)
    return response


def main(args):
    # For memory saving when using FasterGeneration:
    # If environment variable `PPFG_QKV_MEM_OPT` is set and the weights of q/k/v
    # is fused, it will try to delete the original unfused weights. Note the
    # rollback to original model would not be guarantee anymore when the faster
    # model failed if the original weights are deleted.
    os.environ["PPFG_QKV_MEM_OPT"] = "1"
    if args.use_fp16:
        assert args.use_faster, "Only supports FP16 when using FasterGeneration."
        paddle.set_default_dtype("float16")
    enable_ft_para()
    # TODO(guosheng): Maybe device can be set in `enable_ft_para`
    paddle.set_device("gpu:" + str(get_ft_para_conf().rank))

    if args.profile:
        UnifiedTransformerLMHeadModel.generate = profile(args.batch_size)(
            UnifiedTransformerLMHeadModel.generate)
    tokenizer = UnifiedTransformerTokenizer.from_pretrained("plato-xl")
    model = UnifiedTransformerLMHeadModel.from_pretrained("plato-xl",
                                                          load_state_as_np=True)
    model.eval()

    history = [
        "hi , Mary ! What do you usually like to do in your spare time ?",
        "well , I spend a lot of time watching movies .",
        "what a confidence ! I always watch a lot of movies , too ."
        "oh really , Frank ? What kind of movies do you like ?"
    ]
    inputs = [history] * args.batch_size
    inputs = list(
        map(
            lambda history: tokenizer.dialogue_encode(
                history=history,
                add_start_token_as_response=True,
                return_length=True,
                return_role_ids=args.use_role,
                position_style=args.position_style), inputs))
    collator = DataCollatorWithPadding(tokenizer)
    data = collator(inputs)

    outputs, _ = model.generate(
        input_ids=data['input_ids'],
        token_type_ids=data['token_type_ids'],
        position_ids=data['position_ids'],
        attention_mask=data['attention_mask'].cast(
            "float32"),  # TODO(guosheng): remove this cast
        role_ids=data.get('role_ids', None),
        seq_len=data['seq_len'],
        max_length=args.max_out_len,
        min_length=args.min_out_len,
        decode_strategy='sampling',
        top_k=args.topk,
        top_p=args.topp,
        temperature=args.temperature,
        num_return_sequences=args.num_return_sequences,
        use_faster=args.use_faster,
        use_fp16_decoding=args.use_fp16)

    # Only make the first process to output.
    if get_ft_para_conf().rank == 0:
        for i in range(len(outputs)):
            result = postprocess_response(outputs[i].numpy(), tokenizer)
            print("Result:", result)


if __name__ == "__main__":
    args = parse_args()
    pprint(args)
    main(args)
