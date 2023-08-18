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
from dataclasses import dataclass, field

import numpy as np
import paddle
from paddle.distributed import fleet
from utils import get_prefix_tuning_params

from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.taskflow.utils import static_mode_guard
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    PretrainedModel,
    PretrainedTokenizer,
)


@dataclass
class PredictorArgument:
    model_name_or_path: str = field(default=None, metadata={"help": "The directory of model."})
    model_prefix: str = field(default="model", metadata={"help": "the prefix name of static model"})
    src_length: int = field(default=1024, metadata={"help": "The max length of source text."})
    max_length: int = field(default=1024, metadata={"help": "the max length for decoding."})
    top_k: int = field(default=1, metadata={"help": "top_k parameter for generation"})
    top_p: float = field(default=1.0, metadata={"help": "top_p parameter for generation"})
    temperature: float = field(default=0.95, metadata={"help": "top_p parameter for generation"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "repetition penalty parameter for generation"})
    device: str = field(default="gpu", metadata={"help": "Device"})
    dtype: str = field(default=None, metadata={"help": "Model dtype"})
    lora_path: str = field(default=None, metadata={"help": "The directory of LoRA parameters. Default to None"})
    prefix_path: str = field(
        default=None, metadata={"help": "The directory of Prefix Tuning parameters. Default to None"}
    )
    type: str = field(
        default="dygraph", metadata={"help": "the type of predictor, it should be one of [dygraph, static]"}
    )


@dataclass
class ModelArgument:
    gpt: bool = field(default=False, metadata={"help": "GPTForCausalLM"})
    ernie: bool = field(default=False, metadata={"help": "Ernie35ForCausalLM"})
    data_file: None = field(default=None, metadata={"help": "data file directory"})
    output_file: str = field(default="output.json", metadata={"help": "predict result file directory"})
    batch_size: int = field(default=1, metadata={"help": "The batch size of data."})


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class BasePredictor:
    def __init__(self, config: PredictorArgument, tokenizer: PretrainedTokenizer = None):
        self.config: PredictorArgument = config
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, padding_side="left")

        self.tokenizer = tokenizer
        self.return_tensors = "pd"
        self._init_dist_env()

    def _preprocess(self, source):
        tokenized_source = self.tokenizer(
            source,
            max_length=self.config.src_length,
            truncation=True,
            truncation_side="left",
            return_tensors=self.return_tensors,
            padding=True,
            add_special_tokens=True,
        )
        return tokenized_source

    def _init_dist_env(self):
        tensor_parallel_degree = paddle.distributed.get_world_size()
        tensor_parallel_rank = paddle.distributed.get_rank()
        print("tensor_parallel_degree ===>", tensor_parallel_degree)
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
        self.tensor_parallel_rank = tensor_parallel_rank
        self.tensor_parallel_degree = tensor_parallel_degree

    @abstractmethod
    def _infer(self, inputs):
        raise NotImplementedError

    def _postprocess(self, predictions):
        decoded_predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded_predictions

    def predict(self, input_texts: str | list[str]):
        tokenized_source = self._preprocess(input_texts)
        predictions = self._infer(tokenized_source)
        decoded_predictions = self._postprocess(predictions)
        return decoded_predictions


class DygraphPredictor(BasePredictor):
    def __init__(
        self, config: PredictorArgument, model: PretrainedModel = None, tokenizer: PretrainedTokenizer = None
    ):
        super().__init__(config, tokenizer)
        self.model = model
        if config.lora_path is not None:
            lora_config = LoRAConfig.from_pretrained(config.lora_path)
            dtype = lora_config.dtype
            lora_config.merge_weights = True
        elif config.prefix_path is not None:
            prefix_config = PrefixConfig.from_pretrained(config.prefix_path)
            dtype = prefix_config.dtype
        elif config.dtype is not None:
            dtype = config.dtype
        else:
            raise ValueError("Please specific the model dtype.")

        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                dtype=dtype,
                tensor_parallel_degree=self.tensor_parallel_degree,
                tensor_parallel_rank=self.tensor_parallel_rank,
            )

        if config.lora_path is not None:
            self.model = LoRAModel.from_pretrained(
                model=self.model, lora_path=config.lora_path, lora_config=lora_config
            )
        if config.prefix_path is not None:
            prefix_tuning_params = get_prefix_tuning_params(self.model)
            self.model = PrefixModelForCausalLM.from_pretrained(
                model=self.model,
                prefix_path=config.prefix_path,
                postprocess_past_key_value=prefix_tuning_params["postprocess_past_key_value"],
            )
        self.model.eval()

    @paddle.no_grad()
    def _infer(self, inputs: dict[str, paddle.Tensor]):
        # the `max_length` of generate is: max_new_length, it will occur error when `max_length` + sequence_length > max_position_embeddings.
        # so change max_length to control the length of decoding.
        max_length = max(self.config.max_length - inputs["input_ids"].shape[-1], 1)
        result = self.model.generate(
            **inputs,
            max_length=max_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            decode_strategy="sampling",
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
        )
        result = result[0]
        return result


class StaticGraphPredictor(BasePredictor):
    def __init__(self, config: PredictorArgument, tokenizer: PretrainedTokenizer = None):
        super().__init__(config, tokenizer)

        params_path = os.path.join(self.config.model_name_or_path, self.config.model_prefix + ".pdiparams")
        model_path = os.path.join(self.config.model_name_or_path, self.config.model_prefix + ".pdmodel")
        inference_config = paddle.inference.Config(model_path, params_path)

        if self.config.device == "gpu":
            # set GPU configs accordingly
            inference_config.enable_use_gpu(100, 0)
        elif self.config.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            inference_config.disable_gpu()
        inference_config.disable_glog_info()

        with static_mode_guard():
            self.predictor = paddle.inference.create_predictor(inference_config)

        self.return_tensors = "np"

    def _preprocess(self, input_text: str | list[str]):
        inputs = super()._preprocess(input_text)

        # reduce the max_length to prevent length overflow
        # same as DygraphPredictor
        max_length = max(self.config.max_length - inputs["input_ids"].shape[-1], 1)
        inputs["max_length"] = np.array(max_length, dtype="int64")

        inputs["top_p"] = np.array(self.config.top_p, dtype="float32")
        inputs["temperature"] = np.array(self.config.temperature, dtype="float32")
        inputs["top_k"] = np.array(self.config.top_k, dtype="int64")
        inputs["repetition_penalty"] = np.array(self.config.repetition_penalty, dtype="float32")

        return inputs

    def _infer(self, inputs: dict[str, np.ndarray]):
        for name in self.predictor.get_input_names():
            self.predictor.get_input_handle(name).copy_from_cpu(inputs[name])

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results = output_handle.copy_to_cpu()
        # the first result is decoding_ids
        decoded_ids = results.tolist()
        return decoded_ids


def create_predictor(predictor_args: PredictorArgument, model_args: ModelArgument):
    tokenizer = AutoTokenizer.from_pretrained(predictor_args.model_name_or_path)
    # TODO(wj-Mcat): fix llama tokenzier pad_token bug
    if isinstance(tokenizer, LlamaTokenizer):
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"

    if predictor_args.type == "dygraph":
        model = None
        if model_args.gpt:
            sys.path.append("./gpt-3")
            from modeling import GPTForCausalLM

            tensor_parallel_degree = paddle.distributed.get_world_size()
            tensor_parallel_rank = paddle.distributed.get_rank()
            model = GPTForCausalLM.from_pretrained(
                predictor_args.model_name_or_path,
                dtype=predictor_args.dtype,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
            )
        elif model_args.ernie:
            sys.path.append("./ernie-3.5-se")
            from modeling import Ernie35ForCausalLM

            tensor_parallel_degree = paddle.distributed.get_world_size()
            tensor_parallel_rank = paddle.distributed.get_rank()
            model = Ernie35ForCausalLM.from_pretrained(
                predictor_args.model_name_or_path,
                dtype=predictor_args.dtype,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
            )

        predictor = DygraphPredictor(predictor_args, model=model, tokenizer=tokenizer)
    elif predictor_args.type == "static":
        predictor = StaticGraphPredictor(predictor_args, tokenizer=tokenizer)
    else:
        raise ValueError(
            f"receive unexpected predictor type: {predictor_args.type}, it should be one of [dygraph, static]"
        )
    return predictor


def predict():
    parser = PdArgumentParser((PredictorArgument, ModelArgument))
    predictor_args, model_args = parser.parse_args_into_dataclasses()
    paddle.set_device(predictor_args.device)

    predictor = create_predictor(predictor_args, model_args)
    source_texts = []
    target_texts = []
    if model_args.data_file:
        with open(model_args.data_file, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                source_texts.append(example["src"])
                target_texts.append(example["tgt"])
    else:
        source_texts = ["hello world, how are you?", "你好，请问你是谁?"]
        target_texts = ["", ""]

    batch_source_texts = batchfy_text(source_texts, model_args.batch_size)
    batch_target_texts = batchfy_text(target_texts, model_args.batch_size)

    with open(model_args.output_file, "w", encoding="utf-8") as f:
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
