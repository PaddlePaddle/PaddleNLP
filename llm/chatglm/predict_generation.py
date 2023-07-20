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
import json

import paddle
from paddle.distributed import fleet

from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.peft.prefix import (
    chatglm_pad_attention_mask,
    chatglm_postprocess_past_key_value,
)
from paddlenlp.transformers import ChatGLMConfig, ChatGLMForCausalLM, ChatGLMTokenizer


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="THUDM/chatglm-6b", help="The directory of model.")
    parser.add_argument(
        "--merge_tensor_parallel_path", default=None, help="The directory of model to merge tensor parallel parts."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument("--src_length", type=int, default=1024, help="The batch size of data.")
    parser.add_argument("--tgt_length", type=int, default=1024, help="The batch size of data.")
    parser.add_argument("--lora_path", default=None, help="The directory of LoRA parameters. Default to None")
    parser.add_argument(
        "--prefix_path", default=None, help="The directory of Prefix Tuning parameters. Default to None"
    )
    parser.add_argument("--data_file", default=None, help="data file directory")
    parser.add_argument("--predict_file", default="prediction.json", help="predict result file directory")
    parser.add_argument("--device", type=str, default="gpu", help="Device")
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
            self.tokenizer = ChatGLMTokenizer.from_pretrained(args.model_name_or_path)
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
                config = ChatGLMConfig.from_pretrained(args.model_name_or_path)
                dtype = config.dtype if config.dtype is not None else config.paddle_dtype

            self.model = ChatGLMForCausalLM.from_pretrained(
                args.model_name_or_path,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
                load_state_as_np=True,
                dtype=dtype,
            )
            if self.args.lora_path is not None:
                self.model = LoRAModel.from_pretrained(self.model, self.args.lora_path)
                self.model.mark_only_lora_as_trainable()
            if self.args.prefix_path is not None:
                self.model = PrefixModelForCausalLM.from_pretrained(
                    self.model, self.args.prefix_path, chatglm_postprocess_past_key_value, chatglm_pad_attention_mask
                )

        self.model.eval()

    def preprocess(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="np",
            padding=True,
            max_length=self.src_length,
            truncation=True,
            truncation_side="left",
        )
        inputs_tensor = {}
        for key in inputs:
            inputs_tensor[key] = paddle.to_tensor(inputs[key])
        return inputs_tensor

    def infer(self, inputs):
        result = self.model.generate(
            **inputs,
            decode_strategy="sampling",
            top_k=1,
            max_length=self.tgt_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        result = result[0]
        return result

    def postprocess(self, infer_data):
        result = []
        for x in infer_data.tolist():
            res = self.tokenizer.decode(x, skip_special_tokens=True)
            res = res.strip("\n")
            result.append(res)
        out_dict = {"result": result}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output


if __name__ == "__main__":
    args = parse_arguments()
    paddle.set_device(args.device)
    predictor = Predictor(args)

    if args.data_file is None:
        all_texts = [
            "你好",
            "[Round 0]\n问：你好\n答：你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。\n[Round 1]\n问：晚上睡不着应该怎么办\n答：",
        ]
    else:
        all_texts = []
        with open(args.data_file, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                if "src" in example:
                    context = example["src"][0] if isinstance(example["src"], list) else example["src"]
                elif "content" in example:
                    context = example["content"]
                all_texts.append(context)

    batch_texts = batchfy_text(all_texts, args.batch_size)
    with open(args.predict_file, "w", encoding="utf-8") as f:
        for bs, texts in enumerate(batch_texts):
            outputs = predictor.predict(texts)
            for text, result in zip(texts, outputs["result"]):
                print("{}\n{}".format(text, result))
                out = {"src": text, "output": result}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    if args.merge_tensor_parallel_path is not None:
        predictor.model.save_pretrained(
            save_dir=args.merge_tensor_parallel_path,
            merge_tensor_parallel=True,
        )
        predictor.tokenizer.save_pretrained(args.merge_tensor_parallel_path)
