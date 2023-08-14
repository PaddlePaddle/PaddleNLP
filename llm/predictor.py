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

import json
import os
import sys
from abc import abstractmethod

import numpy as np
import paddle
from paddle.distributed import fleet
from utils import get_prefix_tuning_params

from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.taskflow.utils import static_mode_guard
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, required=True, help="The directory of model.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument("--src_length", type=int, default=1024, help="The max length of source text.")
    parser.add_argument("--tgt_length", type=int, default=1024, help="The max length of target text.")
    parser.add_argument("--lora_path", default=None, help="The directory of LoRA parameters. Default to None")
    parser.add_argument(
        "--prefix_path", default=None, help="The directory of Prefix Tuning parameters. Default to None"
    )
    parser.add_argument("--top_k", type=int, default=1, help="top_k parameter for generation")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p parameter for generation")
    parser.add_argument("--temperature", type=float, default=0.95, help="top_p parameter for generation")
    parser.add_argument("--data_file", default=None, help="data file directory")
    parser.add_argument("--output_file", default="output.json", help="predict result file directory")
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    parser.add_argument("--dtype", type=str, default=None, help="Model dtype")
    parser.add_argument("--gpt", type=bool, default=False, help="GPTForCausalLM")
    return parser


def parse_arguments():
    parser = get_parser()
    return parser.parse_args()


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class BasePredictor:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        if isinstance(self.tokenizer, LlamaTokenizer):
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "<pad>"

    def preprocess(self, source):
        tokenized_source = self.tokenizer(
            source,
            max_length=self.args.src_length,
            truncation=True,
            truncation_side="left",
            return_tensors="pd",
            padding=True,
            add_special_tokens=True,
        )
        return tokenized_source

    @abstractmethod
    def infer(self, inputs):
        raise NotImplementedError

    def postprocess(self, predictions):
        decoded_predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded_predictions

    def predict(self, source):
        tokenized_source = self.preprocess(source)
        predictions = self.infer(tokenized_source)
        decoded_predictions = self.postprocess(predictions)
        return decoded_predictions


class DygraphPredictor(BasePredictor):
    def __init__(self, args):
        super().__init__(args)
        tensor_parallel_degree = paddle.distributed.get_world_size()
        self.tensor_parallel_rank = 0
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
            self.tensor_parallel_rank = hcg.get_model_parallel_rank()

        if self.args.lora_path is not None:
            lora_config = LoRAConfig.from_pretrained(self.args.lora_path)
            dtype = lora_config.dtype
            lora_config.merge_weights = True
        elif self.args.prefix_path is not None:
            prefix_config = PrefixConfig.from_pretrained(self.args.prefix_path)
            dtype = prefix_config.dtype
        elif self.args.dtype is not None:
            dtype = self.args.dtype
        else:
            raise ValueError("Please specific the model dtype.")
        if self.args.gpt:
            sys.path.append("./gpt-3")
            from modeling import GPTForCausalLM

            self.model = GPTForCausalLM.from_pretrained(
                args.model_name_or_path,
                dtype=dtype,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=self.tensor_parallel_rank,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                dtype=dtype,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=self.tensor_parallel_rank,
            )

        if self.args.lora_path is not None:
            self.model = LoRAModel.from_pretrained(
                model=self.model, lora_path=self.args.lora_path, lora_config=lora_config
            )
        if self.args.prefix_path is not None:
            prefix_tuning_params = get_prefix_tuning_params(self.model)
            self.model = PrefixModelForCausalLM.from_pretrained(
                model=self.model,
                prefix_path=self.args.prefix_path,
                postprocess_past_key_value=prefix_tuning_params["postprocess_past_key_value"],
                pad_attention_mask=prefix_tuning_params["pad_attention_mask"],
            )
        self.model.eval()

    def infer(self, inputs):
        with paddle.no_grad():
            result = self.model.generate(
                **inputs,
                max_length=self.args.tgt_length,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                decode_strategy="sampling",
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
            )
        result = result[0]
        return result


class StaticGraphPredictor(BasePredictor):
    def __init__(self, args):
        super().__init__(args)

        model_path = os.path.join(args.model_dir, args.model_prefix + ".pdmodel")
        params_path = os.path.join(args.model_dir, args.model_prefix + ".pdiparams")
        config = paddle.inference.Config(model_path, params_path)

        if args.device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
        elif args.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
        config.disable_glog_info()

        with static_mode_guard():
            self.predictor = paddle.inference.create_predictor(config)

    def preprocess(self, input_text):
        inputs = super().preprocess(input_text)

        # reduce the max_length to prevent length overflow
        max_length = max(self.args.max_length - inputs["input_ids"].shape[-1], 1)
        inputs["max_length"] = np.array(max_length, dtype="int64")

        inputs["top_p"] = np.array(self.args.top_p, dtype="float32")
        inputs["temperature"] = np.array(self.args.temperature, dtype="float32")
        inputs["top_k"] = np.array(self.args.top_k, dtype="int64")
        inputs["repetition_penalty"] = np.array(self.args.repetition_penalty, dtype="float32")

        return inputs


def create_predictor(args):
    """create predictor instance by args, and support dygraph model and static inference model

    Args:
        args (Argument): the argument instance from commandline

    Returns:
        Predictor: the instance of predictor
    """
    if os.path.isdir(args.model_name_or_path):

        # if there is model_state.pdparams file, it should be DygraphPredictor
        if os.path.exists(os.path.join(args.model_name_or_path, "model_state.pdparams")):
            return DygraphPredictor(args)

        if any(
            [file.endswith("pdiparams") and file.endswith("pdmodel") for file in os.listdir(args.model_name_or_path)]
        ):
            return StaticGraphPredictor(args)

    return DygraphPredictor(args)


def predict():
    args = parse_arguments()
    paddle.set_device(args.device)

    predictor = create_predictor(args)

    source_texts = []
    target_texts = []
    if args.data_file:
        with open(args.data_file, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                source_texts.append(example["src"])
                target_texts.append(example["tgt"])
    else:
        source_texts = ["hello world, how are you?", "你好，请问你是谁?"]
        target_texts = ["", ""]

    batch_source_texts = batchfy_text(source_texts, args.batch_size)
    batch_target_texts = batchfy_text(target_texts, args.batch_size)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for bs, batch_source_text in enumerate(batch_source_texts):
            outputs = predictor.predict(batch_source_text)

            for output, source, target in zip(outputs, batch_source_texts[bs], batch_target_texts[bs]):
                print("***********Source**********")
                print(source)
                print("***********Target**********")
                print(target)
                print("***********Output**********")
                print(output)
                out = {"src": source, "tgt": target, "output": output}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    predict()
