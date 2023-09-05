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
import time
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
import paddle
import paddle.distributed.fleet.base.topology as tp
from paddle.distributed import fleet
from utils import (
    dybatch_preprocess,
    get_infer_model_path,
    get_prefix_tuning_params,
    load_real_time_tokens,
)

from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.taskflow.utils import static_mode_guard
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    PretrainedModel,
    PretrainedTokenizer,
)
from paddlenlp.utils.import_utils import import_module, is_paddlenlp_ops_available


@dataclass
class PredictorArgument:
    model_name_or_path: str = field(default=None, metadata={"help": "The directory of model."})
    model_prefix: str = field(default="model", metadata={"help": "the prefix name of static model"})
    src_length: int = field(default=1024, metadata={"help": "The max length of source text."})
    max_length: int = field(default=2048, metadata={"help": "the max length for decoding."})
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
    decode_strategy: str = field(
        default="sampling",
        metadata={
            "help": "the decoding strategy of generation, which should be one of ['sampling', 'greedy_search', 'beam_search']. Default to sampling"
        },
    )

    mode: str = field(
        default="dynamic", metadata={"help": "the type of predictor, it should be one of [dynamic, static]"}
    )
    inference_model: bool = field(default=False, metadata={"help": "whether use InferenceModel to do generation"})
    batch_size: int = field(default=1, metadata={"help": "The batch size of data."})
    max_batch_size: int = field(default=None, metadata={"help": "The max batch size of data during serving."})
    benchmark: bool = (
        field(
            default=False,
            metadata={
                "help": "If benchmark set as `True`, we will force model decode to max_length, which is helpful to compute throughput. "
            },
        ),
    )


@dataclass
class ModelArgument:
    model_type: str = field(
        default=None,
        metadata={"help": "the type of the model, which can be one of ['gpt-3', 'ernie-3.5-se', 'llama-img2txt']"},
    )
    data_file: str = field(default=None, metadata={"help": "data file directory"})
    output_file: str = field(default="output.json", metadata={"help": "predict result file directory"})


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


def init_dist_env():
    tensor_parallel_degree = paddle.distributed.get_world_size()
    tensor_parallel_rank = paddle.distributed.get_rank()

    if tensor_parallel_degree > 1:
        # refer to: https://github.com/PaddlePaddle/Paddle/blob/4abea956ee852ce52791a1e08fa92ed4d3be150d/python/paddle/distributed/fleet/fleet.py#L298C23-L298C45
        hcg = tp._HYBRID_PARALLEL_GROUP
        if hcg is None:
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
    return tensor_parallel_rank, tensor_parallel_degree


class BasePredictor:
    def __init__(self, config: PredictorArgument, tokenizer: PretrainedTokenizer = None):
        self.config: PredictorArgument = config
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, padding_side="left")

        self.tokenizer = tokenizer
        self.return_tensors = "pd"
        self.tensor_parallel_rank, self.tensor_parallel_degree = init_dist_env()

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
            decode_strategy=self.config.decode_strategy,
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


class StaticInferencePredictor(BasePredictor):
    def __init__(
        self,
        config: PredictorArgument,
        cache_kv_shapes: list[list[int]],
        tokenizer: PretrainedTokenizer = None,
    ):
        super().__init__(config, tokenizer)
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

        self.model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        self.dtype = dtype
        self.architectures = self.model_config.architectures[0].lower()
        self.cache_kvs = [paddle.zeros(shape, dtype=dtype) for shape in cache_kv_shapes]
        self.pre_ids = paddle.full([config.batch_size, config.max_length + 1], -1, dtype="int64")

        if "chatglm" in self.architectures:
            self.attention_mask = paddle.ones(
                shape=(config.batch_size, 1, config.max_length, config.max_length),
                dtype=dtype,
            )
            self.tgt_pos = paddle.ones(
                shape=[config.batch_size, 2, 1],
                dtype="int64",
            )
        else:
            self.attention_mask = paddle.zeros(
                shape=(config.batch_size, 1, config.max_length, config.max_length),
                dtype=dtype,
            )

        self.tgt_generation_mask = paddle.zeros(
            shape=[config.batch_size, 1, 1, config.max_length + 1],
            dtype=dtype,
        )
        self.predictor = self._create_predictor(config)

    def _create_predictor(self, predictor_args: PredictorArgument):
        if not is_paddlenlp_ops_available():
            raise ValueError(
                "you should install the paddlenlp ops to run inference predictor, "
                "https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md"
            )

        # register the custome ops
        import_module("paddlenlp_ops.encode_rotary_qk")
        import_module("paddlenlp_ops.get_padding_offset")
        import_module("paddlenlp_ops.qkv_transpose_split")
        import_module("paddlenlp_ops.rebuild_padding")
        import_module("paddlenlp_ops.transpose_remove_padding")
        import_module("paddlenlp_ops.write_cache_kv")

        infer_model_path = get_infer_model_path(predictor_args.model_name_or_path, predictor_args.model_prefix)

        config = paddle.inference.Config(infer_model_path + ".pdmodel", infer_model_path + ".pdiparams")

        config.switch_ir_optim(True)
        device_id = int(os.environ.get("FLAGS_selected_gpus", 0))
        config.enable_use_gpu(100, device_id)
        # config.disable_glog_info()
        # config.enable_memory_optim()

        if self.tensor_parallel_degree > 1:
            trainer_endpoints = fleet.worker_endpoints()
            current_endpoint = trainer_endpoints[self.tensor_parallel_rank]

            dist_config = config.dist_config()
            dist_config.set_ranks(self.tensor_parallel_degree, self.tensor_parallel_rank)
            dist_config.set_endpoints(trainer_endpoints, current_endpoint)
            dist_config.enable_dist_model(True)

            dist_config.set_comm_init_config(os.path.join(predictor_args.model_name_or_path, "rank_mapping.csv"))
            config.set_dist_config(dist_config)

        predictor = paddle.inference.create_predictor(config)
        return predictor

    def _preprocess(self, source):
        if "chatglm" in self.architectures:
            inputs = dybatch_preprocess(self.tokenizer, source, self.config.max_length, self.architectures)

            for i in range(inputs["input_ids"].shape[0]):
                length = inputs["seq_len_encoder"][i][0]
                self.attention_mask[i, 0, :length, :length] = 0
                self.attention_mask[i, 0, : length - 1, length - 1] = 1
                self.tgt_generation_mask[i, 0, 0, :length] = paddle.ones(shape=[1, length], dtype="float16")
                self.tgt_pos[i, 0, 0] = paddle.to_tensor([length], dtype="int64")

            inputs["attention_mask"] = self.attention_mask
            inputs["tgt_generation_mask"] = self.tgt_generation_mask
            inputs["tgt_pos"] = self.tgt_pos.numpy()
        else:
            inputs = dybatch_preprocess(self.tokenizer, source, self.config.max_length, self.architectures)
            for i in range(inputs["input_ids"].shape[0]):
                length = inputs["seq_len_encoder"][i][0]
                self.attention_mask[i, 0, :length, :length] = paddle.tril(
                    paddle.ones(shape=(length, length), dtype="float16")
                )
                self.tgt_generation_mask[i, 0, 0, :length] = paddle.ones(shape=[1, length], dtype="float16")

            inputs["attention_mask"] = self.attention_mask
            inputs["tgt_generation_mask"] = self.tgt_generation_mask
        return inputs

    @paddle.no_grad()
    def _infer(self, inputs):
        for k, v in inputs.items():
            input_tensor = self.predictor.get_input_handle(k)
            if "mask" in k or "position" in k:
                input_tensor.share_external_data(v)
            else:
                input_tensor.copy_from_cpu(v)

        for i in range(self.model_config.num_hidden_layers):
            input_tensor = self.predictor.get_input_handle("cache_kvs_" + str(i))
            input_tensor.share_external_data(self.cache_kvs[i])
        input_tensor = self.predictor.get_input_handle("pre_ids")
        input_tensor.share_external_data(self.pre_ids)

        self.predictor.run()

    def _postprocess(self, predictions):
        if paddle.distributed.get_rank() == 0:
            tokens: np.ndarray = load_real_time_tokens()
            decoded_predictions = self.tokenizer.batch_decode(
                tokens.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return decoded_predictions
        else:
            return None


class DygraphInferencePredictor(BasePredictor):
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

        self.dtype = dtype
        self.architectures = self.model.config.architectures[0].lower()

        self.cache_kvs = [
            paddle.zeros(shape, dtype=dtype)
            for shape in self.model.get_cache_kvs_shape(self.model.config, config.max_batch_size, config.max_length)
        ]
        self.pre_ids = paddle.full([config.max_batch_size, config.max_length], -1, dtype="int64")
        if "chatglm" in self.architectures:
            self.attention_mask = paddle.ones(
                shape=(config.batch_size, 1, config.max_length, config.max_length),
                dtype=dtype,
            )
            self.tgt_pos = paddle.ones(
                shape=[config.batch_size, 2, 1],
                dtype="int64",
            )
        else:
            self.attention_mask = paddle.zeros(
                shape=(config.batch_size, 1, config.max_length, config.max_length),
                dtype=dtype,
            )

        self.tgt_generation_mask = paddle.zeros(
            shape=[config.max_batch_size, 1, 1, config.max_length],
            dtype=dtype,
        )

    def _preprocess(self, source):
        if "chatglm" in self.architectures:
            inputs = dybatch_preprocess(self.tokenizer, source, self.config.max_length, self.architectures)

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
        else:
            inputs = dybatch_preprocess(self.tokenizer, source, self.config.max_length, self.architectures)
            for i in range(inputs["input_ids"].shape[0]):
                length = inputs["seq_len_encoder"][i][0]
                self.attention_mask[i, 0, :length, :length] = paddle.tril(
                    paddle.ones(shape=(length, length), dtype="float16")
                )
                inputs["attention_mask"] = self.attention_mask
                self.tgt_generation_mask[i, 0, 0, :length] = paddle.ones(shape=[1, length], dtype="float16")
                inputs["tgt_generation_mask"] = self.tgt_generation_mask
            inputs["cache_kvs"] = self.cache_kvs
            inputs["pre_ids"] = self.pre_ids

        inputs_tensor = {}
        for key, value in inputs.items():
            if key != "cache_kvs":
                inputs_tensor[key] = paddle.to_tensor(value)
            else:
                inputs_tensor[key] = value
        return inputs_tensor

    @paddle.no_grad()
    def _infer(self, inputs: dict[str, paddle.Tensor]):
        # the `max_length` of generate is: max_new_length, it will occur error when `max_length` + sequence_length > max_position_embeddings.
        # so change max_length to control the length of decoding.
        self.model.generate(
            **inputs,
        )
        return None

    def _postprocess(self, predictions):
        if paddle.distributed.get_rank() == 0:
            tokens: np.ndarray = load_real_time_tokens()
            decoded_predictions = self.tokenizer.batch_decode(
                tokens.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return decoded_predictions
        else:
            return None


def create_predictor(
    predictor_args: PredictorArgument,
    model_args: ModelArgument,
    tensor_parallel_degree: int = 1,
    tensor_parallel_rank: int = 0,
):
    tokenizer = AutoTokenizer.from_pretrained(predictor_args.model_name_or_path)
    # TODO(wj-Mcat): fix llama tokenzier pad_token bug
    if isinstance(tokenizer, LlamaTokenizer):
        tokenizer.pad_token = tokenizer.eos_token

    tensor_parallel_rank, tensor_parallel_degree = init_dist_env()
    if not predictor_args.inference_model:
        if predictor_args.mode == "dynamic":
            if model_args.model_type == "gpt-3":
                sys.path.append("./gpt-3")
                from modeling import GPTForCausalLM

                model = GPTForCausalLM.from_pretrained(
                    predictor_args.model_name_or_path,
                    dtype=predictor_args.dtype,
                    tensor_parallel_degree=tensor_parallel_degree,
                    tensor_parallel_rank=tensor_parallel_rank,
                )
            elif model_args.model_type == "ernie-3.5-se":
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
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    predictor_args.model_name_or_path,
                    dtype=predictor_args.dtype,
                    low_cpu_mem_usage=True,
                    tensor_parallel_degree=tensor_parallel_degree,
                    tensor_parallel_rank=tensor_parallel_rank,
                )

            predictor = DygraphPredictor(predictor_args, model=model, tokenizer=tokenizer)
        elif predictor_args.mode == "static":
            predictor = StaticGraphPredictor(predictor_args, tokenizer=tokenizer)
        else:
            raise ValueError("the `mode` should be one of [dynamic, static]")
    else:
        if predictor_args.mode == "dynamic":
            # TODO(wj-Mcat): complete AutoInferenceModel & AutoPredictor
            config = AutoConfig.from_pretrained(predictor_args.model_name_or_path)
            if "llama" in config.architectures[0].lower():
                if model_args.model_type == "llama-img2txt":
                    # we use llama for img2txt.
                    from paddlenlp.experimental.transformers import (
                        LlamaForMiniGPT4InferenceModel as LlamaInferenceModel,
                    )
                else:
                    from paddlenlp.experimental.transformers import (
                        LlamaForCausalLMInferenceModel as LlamaInferenceModel,
                    )

                config.tensor_parallel_degree = tensor_parallel_degree
                config.tensor_parallel_rank = tensor_parallel_rank
                model = LlamaInferenceModel.from_pretrained(
                    predictor_args.model_name_or_path, config=config, dtype=predictor_args.dtype
                )
                model.eval()
            elif "chatglm" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    ChatGLMForCausalLMInferenceModel,
                )

                config.tensor_parallel_degree = tensor_parallel_degree
                config.tensor_parallel_rank = tensor_parallel_rank

                model = ChatGLMForCausalLMInferenceModel.from_pretrained(
                    predictor_args.model_name_or_path,
                    config=config,
                    dtype=predictor_args.dtype,
                )
                model.eval()
            predictor = DygraphInferencePredictor(predictor_args, model=model, tokenizer=tokenizer)
        elif predictor_args.mode == "static":
            config = AutoConfig.from_pretrained(predictor_args.model_name_or_path)
            if "llama" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    LlamaForCausalLMInferenceModel,
                )

                cache_kvs_shape = LlamaForCausalLMInferenceModel.get_cache_kvs_shape(config, predictor_args.batch_size)
                predictor = StaticInferencePredictor(predictor_args, cache_kvs_shape, tokenizer=tokenizer)
            elif "chatglm" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    ChatGLMForCausalLMInferenceModel,
                )

                cache_kvs_shape = ChatGLMForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size
                )
                predictor = StaticInferencePredictor(predictor_args, cache_kvs_shape, tokenizer=tokenizer)
        else:
            raise ValueError("the `mode` should be one of [dynamic, static]")
    return predictor


def predict():
    parser = PdArgumentParser((PredictorArgument, ModelArgument))
    predictor_args, model_args = parser.parse_args_into_dataclasses()
    # init `max_batch_size`

    predictor_args.max_batch_size = predictor_args.max_batch_size or predictor_args.batch_size
    paddle.set_device(predictor_args.device)
    paddle.set_default_dtype(predictor_args.dtype)

    tensor_parallel_degree = paddle.distributed.get_world_size()
    if tensor_parallel_degree > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tensor_parallel_degree,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

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

    batch_source_texts = batchfy_text(source_texts, predictor_args.batch_size)
    batch_target_texts = batchfy_text(target_texts, predictor_args.batch_size)

    with open(model_args.output_file, "w", encoding="utf-8") as f:
        for bs, batch_source_text in enumerate(batch_source_texts):
            outputs = predictor.predict(batch_source_text)

            if predictor.tensor_parallel_rank > 0:
                continue
            for output, source, target in zip(outputs, batch_source_texts[bs], batch_target_texts[bs]):
                print("***********Source**********")
                print(source)
                print("***********Target**********")
                print(target)
                print("***********Output**********")
                print(output)
                out = {"src": source, "tgt": target, "output": output}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    if predictor_args.benchmark:
        benchmark(predictor, predictor_args, model_args)


def benchmark(predictor, predictor_args, model_args):
    # Just construct a simple benchmark input. We pad input to the src_length.
    test_texts = "hello world, how are you?"
    benchmark_texts = [test_texts + "<pad>" * predictor_args.src_length for _ in range(predictor_args.batch_size)]

    benchmark_texts = [
        "<pad>" * (predictor_args.src_length // 2 - 3) + "My name is " for _ in range(predictor_args.batch_size)
    ]
    batch_benchmark_texts = batchfy_text(benchmark_texts, predictor_args.batch_size)
    print("***********Start Benchmark**********")

    warmup_time = 1
    test_time = 1

    print("***********Start Warmup**********")
    for _ in range(warmup_time):
        for bs, batch_source_text in enumerate(batch_benchmark_texts):
            outputs = predictor.predict(batch_source_text)

    print("***********Start Speed Test**********")
    start = time.perf_counter()
    for _ in range(test_time):
        for bs, batch_source_text in enumerate(batch_benchmark_texts):
            outputs = predictor.predict(batch_source_text)
    end = time.perf_counter()

    output_tokens = sum([len(output) for output in outputs])
    print(
        "Input length is: {}, Output length is: {}, bs is: {}, Generate speed is: {:.3f} tokens/s(ips), QPS: {:.3f} requests/s. ".format(
            predictor_args.src_length,
            predictor_args.max_length - predictor_args.src_length,
            predictor_args.batch_size,
            (output_tokens / (end - start) / test_time),
            (predictor_args.batch_size / (end - start) / test_time),
        )
    )


if __name__ == "__main__":
    predict()
