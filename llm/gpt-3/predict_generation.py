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
from utils import get_hcg, init_dist_env, set_seed

from paddlenlp.transformers import (
    GPTChineseTokenizer,
    GPTConfig,
    GPTForCausalLM,
    GPTTokenizer,
)

MODEL_CLASSES = {
    "gpt2": (GPTForCausalLM, GPTTokenizer),
    "gpt2-cn": (GPTForCausalLM, GPTChineseTokenizer),
}


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="gpt2-cn", help="The directory of model.")
    parser.add_argument("--model_name_or_path", default="gpt-cpm-large-cn", help="The directory of model.")
    parser.add_argument("--save_onepiece_model_path", default=None, help="The directory of model.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument("--src_length", type=int, default=200, help="The batch size of data.")
    parser.add_argument("--tgt_length", type=int, default=200, help="The batch size of data.")
    parser.add_argument("--seed", type=int, default=20, help="the seed of parameter initialization")
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
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.batch_size = args.batch_size
        self.args = args
        self.src_length = self.args.src_length
        self.tgt_length = self.args.tgt_length

        tensor_parallel_degree = paddle.distributed.get_world_size()
        tensor_parallel_rank = 0
        if tensor_parallel_degree > 1:
            hcg = get_hcg()
            tensor_parallel_rank = hcg.get_model_parallel_rank()

        config = GPTConfig.from_pretrained(args.model_name_or_path)
        dtype = config.dtype if config.dtype is not None else "float16"

        self.model = GPTForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_state_as_np=True,
            low_cpu_mem_usage=True,
            dtype=dtype,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.model.config.pad_token_id
        self.model.eval()

    def preprocess(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="np",
            padding=True,
            max_length=self.src_length,
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
                    eos_token_id=self.tokenizer.eol_token_id,
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
                        eos_token_id=self.tokenizer.eol_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        decode_strategy="sampling",
                        top_k=1,
                    )
        result = result[0]
        return result

    def postprocess(self, infer_data):
        result = []
        for x in infer_data.tolist():
            res = self.tokenizer.convert_ids_to_string(x)
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

    # Init the fleet config
    tensor_parallel_degree = paddle.distributed.get_world_size()
    if tensor_parallel_degree > 1:
        init_dist_env(tensor_parallel_degree=tensor_parallel_degree, seed=args.seed)
    set_seed(args.seed)

    predictor = Predictor(args)
    all_texts = ["问题：中国的首都是哪里？答案：北京。\n问题：苹果的CEO是谁? 答案：", "问题：中国的首都是哪里？答案：北京。\n问题：广东的省会是哪个城市? 答案："]
    batch_texts = batchfy_text(all_texts, args.batch_size)
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for text, result in zip(texts, outputs["result"]):
            print(result)
    if args.save_onepiece_model_path is not None:
        predictor.save_onepiece_model(args.save_onepiece_model_path)


if __name__ == "__main__":
    predict()
