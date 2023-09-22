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
    get_alibi_slopes,
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
import paddle_custom_device.npu.passes as passes

@dataclass
class PredictorArgument:
    model_name_or_path: str = field(default=None, metadata={"help": "The directory of model."})
    model_prefix: str = field(default="model", metadata={"help": "the prefix name of static model"})
    src_length: int = field(default=1024, metadata={"help": "The max length of source text."})
    max_length: int = field(default=2048, metadata={"help": "the max length for decoding."})
    top_k: int = field(default=0, metadata={"help": "top_k parameter for generation"})
    top_p: float = field(default=0.7, metadata={"help": "top_p parameter for generation"})
    temperature: float = field(default=0.95, metadata={"help": "top_p parameter for generation"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "repetition penalty parameter for generation"})
    device: str = field(default="gpu", metadata={"help": "Device"})
    dtype: str = field(default=None, metadata={"help": "Model dtype"})
    lora_path: str = field(default=None, metadata={"help": "The directory of LoRA parameters. Default to None"})
    export_precache: bool = field(default=False, metadata={"help": "whether use prefix weight to do infer"})
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
    quant_type: str = field(
        default="None",
        metadata={"help": "The quant type of inference model, support `weight_only_int8`, `weight_only_int4`."},
    )
    batch_size: int = field(default=1, metadata={"help": "The batch size of data."})
    benchmark: bool = field(
        default=False,
        metadata={
            "help": "If benchmark set as `True`, we will force model decode to max_length, which is helpful to compute throughput. "
        },
    )

    enable_memory_optim: bool = field(
        default=True,
        metadata={"help": "whether use `enable_memory_optim` in inference predictor"},
    )
    init_fleet_worker: bool = field(
        default=True,
        metadata={"help": "whether use `init_fleet_worker` in inference predictor"},
    )

    @property
    def total_max_length(self):
        return self.src_length + self.max_length


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
        self.model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        self.config: PredictorArgument = config
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, padding_side="left")

        self.tokenizer = tokenizer
        self.return_tensors = "pd"
        self.tensor_parallel_rank, self.tensor_parallel_degree = init_dist_env()
        self.model_config.tensor_parallel_rank, self.model_config.tensor_parallel_degree = init_dist_env()

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
        result = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_length,
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
        inputs["max_new_tokens"] = np.array(self.config.max_length, dtype="int64")

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


class InferencePredictorMixin:
    def __init__(self, config: PredictorArgument, tokenizer: PretrainedTokenizer):

        self.architectures = self.model_config.architectures[0].lower()

        self.dtype = config.dtype or self.model_config
        self.cache_kvs = [paddle.zeros(shape, dtype=self.dtype) for shape in self.cache_kvs_shape]
        self.num_layers, self.num_attention_heads, self.head_dim = (
            len(self.cache_kvs),
            self.cache_kvs[0].shape[-3],
            self.cache_kvs[0].shape[-1],
        )
        self.pre_ids = paddle.full([config.batch_size, config.total_max_length], -1, dtype="int64")
        if "chatglm" in self.architectures:
            self.attention_mask = paddle.ones(
                shape=(config.batch_size, 1, config.total_max_length, config.total_max_length),
                dtype=self.dtype,
            )
            self.tgt_pos = paddle.ones(
                shape=[config.batch_size, 2, 1],
                dtype="int64",
            )
        else:
            self.attention_mask = paddle.zeros(
                shape=(config.batch_size, 1, config.total_max_length, config.total_max_length),
                dtype=self.dtype,
            )

        self.tgt_generation_mask = paddle.zeros(
            shape=[config.batch_size, 1, 1, config.total_max_length],
            dtype=self.dtype,
        )
        self.arange_tensor_encoder = paddle.zeros(
            shape=(config.batch_size, 1, config.total_max_length), dtype=self.dtype
        )

        if config.export_precache:
            if config.prefix_path:
                prefix_cache = (
                    paddle.to_tensor(np.load(f"{config.prefix_path}/pre_caches.npy")).astype(self.dtype).unsqueeze(2)
                )
                prefix_cache = paddle.expand(
                    prefix_cache,
                    [
                        self.num_layers,
                        2,
                        config.batch_size,
                        self.num_attention_heads,
                        prefix_cache.shape[-2],
                        self.head_dim,
                    ],
                )
                self.pre_caches = [item.squeeze_(0) for item in paddle.split(prefix_cache, self.num_layers, axis=0)]
            else:
                prefix_cache = paddle.zeros(
                    [self.num_layers, 2, config.batch_size, self.num_attention_heads, 128, self.head_dim],
                    dtype=self.dtype,
                )
                self.pre_caches = [item.squeeze_(0) for item in paddle.split(prefix_cache, self.num_layers, axis=0)]

    def _postprocess(self, predictions):
        if paddle.distributed.get_rank() == 0:
            tokens: np.ndarray = load_real_time_tokens()
            decoded_predictions = self.tokenizer.batch_decode(
                tokens.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return decoded_predictions
        else:
            return None

    def _preprocess(self, source):
        self.attention_mask[:] = 0
        self.tgt_generation_mask[:] = 0
        pre_caches_length = 0 if not self.config.export_precache else self.pre_caches[0].shape[-2]

        if "chatglm" in self.architectures:
            inputs = dybatch_preprocess(
                self.tokenizer,
                source,
                self.config.src_length,
                self.config.max_length,
                self.architectures,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                benchmark=self.config.benchmark,
            )
            for i in range(inputs["input_ids"].shape[0]):
                length = inputs["seq_len_encoder"][i][0]
                self.attention_mask[i, 0, :length, :length] = 0
                self.attention_mask[i, 0, : length - 1, length - 1] = 1
                self.tgt_generation_mask[i, 0, 0, :length] = paddle.ones(shape=[1, length], dtype=self.config.dtype)
                self.tgt_pos[i, 0, 0] = paddle.to_tensor([length], dtype="int64")

            inputs["tgt_pos"] = self.tgt_pos
        elif "bloom" in self.architectures:
            inputs = dybatch_preprocess(
                self.tokenizer,
                source,
                self.config.src_length,
                self.config.max_length,
                self.architectures,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                benchmark=self.config.benchmark,
            )
            for i in range(inputs["input_ids"].shape[0]):
                length = inputs["seq_len_encoder"][i][0]
                self.attention_mask[i, :, :length, :length] = paddle.tril(
                    paddle.ones(shape=(length, length), dtype=self.config.dtype)
                )
                self.arange_tensor_encoder[i, :, :length] = paddle.arange(length).astype(self.config.dtype)

                self.tgt_generation_mask[i, :, 0, :length] = paddle.ones(shape=[1, length], dtype=self.config.dtype)
            # alibi encoder
            alibi_slopes = get_alibi_slopes(self.model_config.n_head)
            inputs["position_ids"] = paddle.to_tensor(alibi_slopes, dtype="float32")

            alibi = alibi_slopes[..., None] * self.arange_tensor_encoder
            alibi = alibi[:, :, None, :]

            if self.model_config.tensor_parallel_degree > 1:
                block_size = self.model_config.n_head // self.model_config.tensor_parallel_degree
                alibi = alibi[
                    :,
                    self.model_config.tensor_parallel_rank
                    * block_size : (self.model_config.tensor_parallel_rank + 1)
                    * block_size,
                ]
                alibi = alibi.reshape([inputs["input_ids"].shape[0], block_size, 1, self.config.max_length])
                inputs["position_ids"] = inputs["position_ids"][
                    self.model_config.tensor_parallel_rank
                    * block_size : (self.model.config.tensor_parallel_rank + 1)
                    * block_size
                ]

            alibi_encoder = alibi.expand(
                [
                    inputs["input_ids"].shape[0],
                    self.model_config.n_head // self.model_config.tensor_parallel_degree,
                    self.config.total_max_length,
                    self.config.total_max_length,
                ]
            )
            alibi_decoder = alibi.expand(
                [
                    inputs["input_ids"].shape[0],
                    self.model_config.n_head // self.model_config.tensor_parallel_degree,
                    1,
                    self.config.total_max_length,
                ]
            )
            self.attention_mask = (
                alibi_encoder + (1 - self.attention_mask) * paddle.finfo(self.attention_mask.dtype).min
            )
            self.tgt_generation_mask = (
                alibi_decoder + (1 - self.tgt_generation_mask) * paddle.finfo(self.tgt_generation_mask.dtype).min
            )

        else:
            inputs = dybatch_preprocess(
                self.tokenizer,
                source,
                self.config.src_length,
                self.config.max_length,
                self.architectures,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                pre_caches_length=pre_caches_length,
                benchmark=self.config.benchmark,
            )

            for i in range(inputs["input_ids"].shape[0]):
                length = inputs["seq_len_encoder"][i][0]
                self.attention_mask[i, 0, :length, :length] = paddle.tril(
                    paddle.ones(shape=(length, length), dtype=self.config.dtype)
                )

                if pre_caches_length > 0:
                    if self.config.prefix_path is None:
                        prefix_attention_mask = paddle.zeros(
                            [1, length, pre_caches_length], dtype=self.attention_mask.dtype
                        )
                    else:
                        prefix_attention_mask = paddle.ones(
                            [1, length, pre_caches_length], dtype=self.attention_mask.dtype
                        )
                    post_attention_mask = paddle.tril(
                        paddle.ones(shape=(length, length), dtype=self.attention_mask.dtype)
                    ).unsqueeze_(axis=0)
                    self.attention_mask[i, 0, :length, : length + pre_caches_length] = paddle.concat(
                        [prefix_attention_mask, post_attention_mask], axis=2
                    )

                if self.config.prefix_path is None:
                    self.tgt_generation_mask[i, 0, 0, pre_caches_length : length + pre_caches_length] = paddle.ones(
                        shape=[1, length], dtype="float16"
                    )
                else:
                    self.tgt_generation_mask[i, 0, 0, : length + pre_caches_length] = paddle.ones(
                        shape=[1, length + pre_caches_length], dtype=self.config.dtype
                    )

        inputs["pre_ids"] = self.pre_ids
        inputs["attention_mask"] = self.attention_mask
        inputs["tgt_generation_mask"] = self.tgt_generation_mask

        if pre_caches_length > 0:
            if self.config.mode == "dynamic":
                inputs["pre_caches"] = self.pre_caches
            else:
                for i in range(len(self.pre_caches)):
                    inputs["pre_caches_{}".format(i)] = self.pre_caches[i].numpy()

        return inputs


class StaticInferencePredictor(InferencePredictorMixin, BasePredictor):
    def __init__(
        self,
        config: PredictorArgument,
        cache_kvs_shape: list[list[int]],
        tokenizer: PretrainedTokenizer = None,
    ):
        self.cache_kvs_shape = cache_kvs_shape
        BasePredictor.__init__(self, config, tokenizer)
        InferencePredictorMixin.__init__(self, config, tokenizer)

        self.predictor = self._create_predictor(config)

    def _create_predictor(self, predictor_args: PredictorArgument):
        # if not is_paddlenlp_ops_available():
        #     raise ValueError(
        #         "you should install the paddlenlp ops to run inference predictor, "
        #         "https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md"
        #     )

        # # register the custome ops
        # import_module("paddlenlp_ops.encode_rotary_qk")
        # import_module("paddlenlp_ops.get_padding_offset")
        # import_module("paddlenlp_ops.qkv_transpose_split")
        # import_module("paddlenlp_ops.rebuild_padding")
        # import_module("paddlenlp_ops.transpose_remove_padding")
        # import_module("paddlenlp_ops.write_cache_kv")

        infer_model_path = get_infer_model_path(predictor_args.model_name_or_path, predictor_args.model_prefix)

        config = paddle.inference.Config(infer_model_path + ".pdmodel", infer_model_path + ".pdiparams")

        config.switch_ir_optim(True)
        device_id = int(os.environ.get("FLAGS_selected_npus", 0))
        config.enable_custom_device("npu", device_id)

        config.enable_memory_optim()

        passes.setUp()

        # config.enable_profile()
        config.enable_save_optim_model(True)
        config.set_optim_cache_dir("./optim_cache")

        pass_builder = config.pass_builder()
        passes.addPasses(pass_builder, "llama7B_mp8_dynamic_batch")
        # passes.addPasses(pass_builder, "llama65B_mp8_dynamic_batch")
        
        pass_builder.turn_on_debug()
        # config.disable_glog_info()
        if predictor_args.enable_memory_optim:
            config.enable_memory_optim()

        # Note(zhengzekang): Force to use fleet executor
        if predictor_args.init_fleet_worker or self.tensor_parallel_degree > 1:
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

    @paddle.no_grad()
    def _infer(self, inputs):
        for k, v in inputs.items():
            input_tensor = self.predictor.get_input_handle(k)

            if "mask" in k or "position" in k:
                input_tensor.share_external_data(v)
            else:
                if paddle.is_tensor(v):
                    v = v.numpy()
                input_tensor.copy_from_cpu(v)

        for i in range(len(self.cache_kvs_shape)):
            input_tensor = self.predictor.get_input_handle("cache_kvs_" + str(i))
            input_tensor.share_external_data(self.cache_kvs[i])

        input_tensor = self.predictor.get_input_handle("pre_ids")
        input_tensor.share_external_data(self.pre_ids)

        self.predictor.run()


class DygraphInferencePredictor(InferencePredictorMixin, BasePredictor):
    def __init__(
        self,
        config: PredictorArgument,
        model: PretrainedModel = None,
        tokenizer: PretrainedTokenizer = None,
    ):
        self.cache_kvs_shape = model.get_cache_kvs_shape(model.config, config.batch_size)
        BasePredictor.__init__(self, config, tokenizer)
        InferencePredictorMixin.__init__(self, config, tokenizer)
        self.model = model

    @paddle.no_grad()
    def _infer(self, inputs: dict[str, paddle.Tensor]):
        for key in inputs.keys():
            if paddle.is_tensor(inputs[key]):
                continue
            if isinstance(inputs[key], list):
                if paddle.is_tensor(inputs[key]):
                    continue
                inputs[key] = [paddle.to_tensor(item) for item in inputs[key]]
            else:
                inputs[key] = paddle.to_tensor(inputs[key])

        inputs["cache_kvs"] = self.cache_kvs
        self.model.generate(
            **inputs,
        )
        return None


def create_predictor(
    predictor_args: PredictorArgument,
    model_args: ModelArgument,
    tensor_parallel_degree: int = 1,
    tensor_parallel_rank: int = 0,
):
    tokenizer = AutoTokenizer.from_pretrained(predictor_args.model_name_or_path)
    # TODO(wj-Mcat): fix llama tokenzier pad_token bug
    if isinstance(tokenizer, LlamaTokenizer) and not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.unk_token

    # update config parameter for inference predictor
    if predictor_args.decode_strategy == "greedy_search":
        predictor_args.top_p = 0.0
        predictor_args.temperature = 1.0

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
            config.tensor_parallel_degree = tensor_parallel_degree
            config.tensor_parallel_rank = tensor_parallel_rank

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

                    config.quant_bits = -1

                    if predictor_args.quant_type.startswith("weight_only_int"):
                        quant_bits = int(predictor_args.quant_type[-1])
                        config.quant_bits = quant_bits

                model = LlamaInferenceModel.from_pretrained(
                    predictor_args.model_name_or_path, config=config, dtype=predictor_args.dtype
                )
                model.eval()

            elif "opt" in config.architectures[0].lower():
                if model_args.model_type == "opt-img2txt":
                    # we use opt for img2txt.
                    from paddlenlp.experimental.transformers import (
                        OPTForBlip2InferenceModel as OPTInferenceModel,
                    )
                else:
                    from paddlenlp.experimental.transformers import (
                        OPTForCausalLMInferenceModel as OPTInferenceModel,
                    )

                model = OPTInferenceModel.from_pretrained(
                    predictor_args.model_name_or_path, config=config, dtype=predictor_args.dtype
                )
                model.eval()
            elif "chatglm" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    ChatGLMForCausalLMInferenceModel,
                )

                model = ChatGLMForCausalLMInferenceModel.from_pretrained(
                    predictor_args.model_name_or_path,
                    config=config,
                    dtype=predictor_args.dtype,
                )
                model.eval()
            elif "bloom" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    BloomForCausalLMInferenceModel,
                )

                model = BloomForCausalLMInferenceModel.from_pretrained(
                    predictor_args.model_name_or_path,
                    config=config,
                    dtype=predictor_args.dtype,
                )
                cache_kvs_shape = BloomForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
                model.eval()
            elif "gpt" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    GPTForCausalLMInferenceModel,
                )

                model = GPTForCausalLMInferenceModel.from_pretrained(
                    predictor_args.model_name_or_path,
                    config=config,
                    dtype=predictor_args.dtype,
                )
                model.eval()
            else:
                raise ValueError("the `model type` should be one of [llama, chatglm, bloom, gpt]")
            predictor = DygraphInferencePredictor(predictor_args, model=model, tokenizer=tokenizer)
        elif predictor_args.mode == "static":
            config = AutoConfig.from_pretrained(predictor_args.model_name_or_path)
            if "llama" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    LlamaForCausalLMInferenceModel,
                )

                cache_kvs_shape = LlamaForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            elif "chatglm" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    ChatGLMForCausalLMInferenceModel,
                )

                cache_kvs_shape = ChatGLMForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            elif "bloom" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    BloomForCausalLMInferenceModel,
                )

                cache_kvs_shape = BloomForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            elif "gpt" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    GPTForCausalLMInferenceModel,
                )

                cache_kvs_shape = GPTForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            else:
                raise ValueError("the `model type` should be one of [llama, chatglm, bloom, gpt]")
            predictor = StaticInferencePredictor(predictor_args, cache_kvs_shape, tokenizer=tokenizer)
        else:
            raise ValueError("the `mode` should be one of [dynamic, static]")
    return predictor


def predict():
    parser = PdArgumentParser((PredictorArgument, ModelArgument))
    predictor_args, model_args = parser.parse_args_into_dataclasses()

    paddle.set_device(predictor_args.device)
    paddle.set_default_dtype(predictor_args.dtype)

    tensor_parallel_degree = paddle.distributed.get_world_size()
    # Note(zhengzekang): force to use fleet executor.
    if predictor_args.init_fleet_worker or tensor_parallel_degree > 1:
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

    batch_benchmark_texts = batchfy_text(benchmark_texts, predictor_args.batch_size)
    print("***********Start Benchmark**********")

    warmup_time = 10
    test_time = 100

    print("***********Start Warmup**********")
    for _ in range(warmup_time):
        for bs, batch_source_text in enumerate(batch_benchmark_texts):
            outputs = predictor.predict(batch_source_text)

    print("***********Start Speed Test**********")
    start = time.perf_counter()
    output_tokens = 0
    for _ in range(test_time):
        for bs, batch_source_text in enumerate(batch_benchmark_texts):
            outputs = predictor.predict(batch_source_text)
            output_tokens += sum([len(output) for output in outputs])
    end = time.perf_counter()
    print("Avg Elapse time is: ", (end - start) / test_time)
    print("Output tokens is: ", output_tokens)
    print(
        "Input length is: {}, Output length is: {}, bs is: {}, Generate speed is: {:.3f} tokens/s(ips), QPS: {:.3f} requests/s. ".format(
            predictor_args.src_length,
            predictor_args.max_length,
            predictor_args.batch_size,
            (output_tokens / (end - start)),
            (predictor_args.batch_size * test_time / (end - start)),
        )
    )


if __name__ == "__main__":
    predict()
