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

import time

import numpy as np
import paddle
from paddle.distributed import fleet
from utils import dybatch_preprocess, get_state_dict, load_real_time_tokens

from paddlenlp.transformers import ChatGLMConfig, ChatGLMTokenizer
from paddlenlp.transformers.chatglm.modeling import (
    ChatGLMForConditionalGenerationDyBatch,
)


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="THUDM/chatglm-6b",
        help="The directory of model.",
    )
    parser.add_argument(
        "--merge_tensor_parallel_path",
        default=None,
        help="The directory of model to merge tensor parallel parts.",
    )
    parser.add_argument("--batch_size", type=int, default=3, help="The batch size of data.")
    parser.add_argument("--src_length", type=int, default=1000, help="The batch size of data.")
    parser.add_argument("--tgt_length", type=int, default=100, help="The batch size of data.")
    parser.add_argument(
        "--lora_path",
        default=None,
        help="The directory of LoRA parameters. Default to None",
    )
    parser.add_argument(
        "--prefix_path",
        default=None,
        help="The directory of Prefix Tuning parameters. Default to None",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use fp16 16-bit (mixed) precision " "training instead of 32-bit training.",
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
    def __init__(self, args=None, tokenizer=None, model=None, **kwargs):
        if args is None:
            self.tokenizer = tokenizer
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.model = model
            self.src_length = kwargs["src_length"]
            self.tgt_length = kwargs["tgt_length"]
        else:
            self.tokenizer = ChatGLMTokenizer.from_pretrained(args.model_name_or_path)
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.batch_size = args.batch_size
            self.args = args
            self.src_length = self.args.src_length
            self.tgt_length = self.args.tgt_length

            tensor_parallel_degree = paddle.distributed.get_world_size()
            tensor_parallel_rank = 0
            if tensor_parallel_degree > 1:
                strategy = fleet.DistributedStrategy()
                strategy.hybrid_configs = {
                    "dp_degree": 1,
                    "mp_degree": tensor_parallel_degree,
                    "pp_degree": 1,
                    "sharding_degree": 1,
                }
                fleet.init(is_collective=True, strategy=strategy)
                hcg = fleet.get_hybrid_communicate_group()
                tensor_parallel_rank = hcg.get_model_parallel_rank()

            self.config = ChatGLMConfig.from_pretrained(args.model_name_or_path)
            self.config.tensor_parallel_degree = tensor_parallel_degree
            self.config.tensor_parallel_rank = tensor_parallel_rank

            dtype = "float32"
            if args.fp16:
                dtype = "float16"
            paddle.set_default_dtype(dtype)
            self.model = ChatGLMForConditionalGenerationDyBatch.from_pretrained(
                args.model_name_or_path,
                config=self.config,
                dtype=dtype,
            )
            state_dict = get_state_dict(
                args.model_name_or_path, ChatGLMForConditionalGenerationDyBatch, self.model.config
            )
            self.model.model.transformer.set_state_dict(state_dict)

        self.model.eval()
        self.pre_ids = paddle.to_tensor(np.full((args.batch_size, args.tgt_length + 1), -1, dtype="int64"))
        self.attention_mask = paddle.ones(
            shape=(args.batch_size, 1, args.src_length, args.src_length),
            dtype="float16",
        )
        self.tgt_generation_mask = paddle.zeros(
            shape=[args.batch_size, 1, 1, args.src_length + args.tgt_length + 1],
            dtype="float16",
        )

        self.tgt_pos = paddle.ones(
            shape=[args.batch_size, 2, 1],
            dtype="int64",
        )

        self.cache_kvs = []
        for _ in range(self.config.num_hidden_layers):
            self.cache_kvs.append(
                paddle.cast(
                    paddle.to_tensor(
                        np.zeros(
                            (
                                2,
                                args.batch_size,
                                self.config.num_attention_heads // tensor_parallel_degree,
                                args.src_length + args.tgt_length + 1,
                                self.config.hidden_size // self.config.num_attention_heads,
                            ),
                            dtype="float32",
                        )
                    ),
                    "float16",
                )
            )

    def preprocess(self, input_text):
        inputs = dybatch_preprocess(self.tokenizer, input_text, self.model.config, self.args)

        for i in range(inputs["input_ids"].shape[0]):
            length = inputs["seq_len_encoder"][i][0]
            self.attention_mask[i, 0, :length, :length] = 0
            self.attention_mask[i, 0, : length - 1, length - 1] = 1
            self.tgt_generation_mask[i, 0, 0, :length] = paddle.ones(shape=[1, length], dtype="float16")
            self.tgt_pos[i, 0, 0] = paddle.to_tensor([length], dtype="int64")

        inputs["attention_mask"] = self.attention_mask
        inputs["tgt_generation_mask"] = self.tgt_generation_mask
        inputs["cache_kvs"] = self.cache_kvs
        inputs["pre_ids"] = self.pre_ids
        inputs["tgt_pos"] = self.tgt_pos

        inputs_tensor = {}
        for key, value in inputs.items():
            if key != "cache_kvs":
                print("Input", key, value.dtype, value.shape)
                inputs_tensor[key] = paddle.to_tensor(value)
            else:
                print("CacheKV", value[0].dtype, value[0].shape)
                inputs_tensor[key] = value

        return inputs_tensor

    def infer(self, inputs):
        for i in range(1):
            start = time.perf_counter()
            with paddle.no_grad():
                with paddle.amp.auto_cast(False, level="O2", dtype=self.model.config.dtype):
                    result = self.model.generate(
                        **inputs,
                    )
            hf_cost = (time.perf_counter() - start) * 1000
            print("Speed Paddle:", hf_cost)
            print(result[0].shape)

        result = result[0]
        return result

    def postprocess(self, infer_data):
        if paddle.distributed.get_rank() == 0:
            tokens = load_real_time_tokens()
            result = []
            for x in tokens.tolist():
                res = self.tokenizer.decode(x, skip_special_tokens=True)
                result.append(res)
            out_dict = {"result": result}
        else:
            out_dict = {"result": "not first rank"}
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
        "你好",
        "你好啊，请问你叫什么名字",
        "你好啊，你在干什么",
    ]

    outputs = predictor.predict(all_texts)
    if paddle.distributed.get_rank() == 0:
        for text, result in zip(all_texts, outputs["result"]):
            print("=" * 20)
            print("Query: {}\nAnswer: {}".format(text, result))
