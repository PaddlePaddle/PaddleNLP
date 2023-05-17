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
from __future__ import annotations

import paddle
from paddle.distributed import fleet

from paddlenlp.layers import LoRAConfig, LoRAModel
from paddlenlp.prompt import PrefixConfig, PrefixModelForCausalLM
from paddlenlp.prompt.prefix import bloom_postprocess_past_key_value
from paddlenlp.transformers import AutoConfig, AutoTokenizer, BloomForCausalLM


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="bigscience/bloom-560m", help="The directory of model.")
    parser.add_argument("--save_onepiece_model_path", default=None, help="The directory of model.")
    parser.add_argument("--batch_size", type=int, default=2, help="The batch size of data.")
    parser.add_argument("--src_length", type=int, default=200, help="The batch size of data.")
    parser.add_argument("--tgt_length", type=int, default=200, help="The batch size of data.")
    parser.add_argument("--seed", type=int, default=20, help="the seed of parameter initialization")
    parser.add_argument("--lora_path", default=None, help="The directory of LoRA parameters. Default to None")
    parser.add_argument(
        "--prefix_path", default=None, help="The directory of Prefix Tuning parameters. Default to None"
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
            self.model = model
            self.src_length = kwargs["src_length"]
            self.tgt_length = kwargs["tgt_length"]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            self.tokenizer.padding_side = "left"
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

            if self.args.lora_path is not None:
                lora_config = LoRAConfig.from_pretrained(self.args.lora_path)
                dtype = lora_config.dtype
            elif self.args.prefix_path is not None:
                prefix_config = PrefixConfig.from_pretrained(self.args.prefix_path)
                dtype = prefix_config.dtype
            else:
                config = AutoConfig.from_pretrained(args.model_name_or_path)
                dtype = config.dtype if config.dtype is not None else "float16"

            self.model = BloomForCausalLM.from_pretrained(
                args.model_name_or_path,
                load_state_as_np=True,
                low_cpu_mem_usage=True,
                dtype=dtype,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
            )
            if self.args.lora_path is not None:
                self.model = LoRAModel.from_pretrained(self.model, self.args.lora_path)
            if self.args.prefix_path is not None:
                self.model = PrefixModelForCausalLM.from_pretrained(
                    self.model, self.args.prefix_path, bloom_postprocess_past_key_value
                )

        self.model.eval()

    def preprocess(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="np",
            padding=True,
            max_length=self.src_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        inputs_tensor = {}
        for key, value in inputs.items():
            inputs_tensor[key] = paddle.to_tensor(value)
        return inputs_tensor

    def infer(self, inputs):
        if self.model.config.dtype == "float32" or self.model.config.dtype is None:
            with paddle.no_grad():
                result = self.model.generate(
                    **inputs,
                    max_length=self.tgt_length,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    decode_strategy="sampling",
                    top_k=1,
                )
        else:
            with paddle.no_grad():
                with paddle.amp.auto_cast(False, level="O2", dtype=self.model.config.dtype):
                    result = self.model.generate(
                        **inputs,
                        max_length=self.tgt_length,
                        bos_token_id=self.tokenizer.bos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        decode_strategy="sampling",
                        top_k=1,
                    )
        result = result[0]
        return result

    def postprocess(self, infer_data):
        result = []
        for x in infer_data.tolist():
            res = self.tokenizer.decode(x, skip_special_tokens=True)
            result.append(res)
        out_dict = {"result": result}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output

    def save_onepiece_model(self, save_onepiece_model_path):
        self.model.save_pretrained(save_dir=save_onepiece_model_path, merge_tensor_parallel=True)
        paddle.distributed.barrier()
        self.tokenizer.save_pretrained(save_onepiece_model_path)
        paddle.distributed.barrier()


def predict():
    args = parse_arguments()
    predictor = Predictor(args)
    all_texts = [
        "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？ ",
        "以下是一道小学数学题：小明有三个苹果，小红有两个苹果，他们一共有多少个苹果？",
        "题目：小明家里有5只猫，其中3只是黑猫，其他都是橘猫。小红去小明家玩，看到了其中2只橘猫。请问小明家还剩几只黑猫？",
        "题目：小明有4个橙子，他想把橙子平均分给他的好朋友小红、小绿和小蓝。问每个人可以分到几个橙子，是否有剩余的橙子？",
        "以下是一道小学数学题：有一个小商店正在做促销活动。如果你购买5个玩具车，可以获得2个免费的玩具车。现在小明要买23个玩具车，他需要买多少个才可以获得免费的玩具车呢？",
    ]
    batch_texts = batchfy_text(all_texts, args.batch_size)
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for text, result in zip(texts, outputs["result"]):
            print("{}\n{}".format(text, result))
    if args.save_onepiece_model_path is not None:
        predictor.save_onepiece_model(args.save_onepiece_model_path)


if __name__ == "__main__":
    predict()
