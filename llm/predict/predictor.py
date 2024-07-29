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
from threading import Thread
from typing import List, Optional

import numpy as np
import paddle
import paddle.distributed.fleet.base.topology as tp
import paddle.incubate.multiprocessing as mp
from paddle.base.framework import in_cinn_mode, in_pir_executor_mode
from paddle.distributed import fleet
from utils.utils import (
    dybatch_preprocess,
    get_alibi_slopes,
    get_default_max_decoding_length,
    get_default_max_encoding_length,
    get_infer_model_path,
    get_model_max_position_embeddings,
    get_prefix_tuning_params,
    init_chat_template,
    load_real_time_tokens,
    read_res,
)

from paddlenlp.generation import GenerationConfig, TextIteratorStreamer
from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.taskflow.utils import static_mode_guard
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    ChatGLMTokenizer,
    ChatGLMv2Tokenizer,
    Llama3Tokenizer,
    LlamaTokenizer,
    PretrainedModel,
    PretrainedTokenizer,
)
from paddlenlp.utils.import_utils import import_module, is_paddlenlp_ops_available
from paddlenlp.utils.log import logger

# Note(@RochardWooSJTU): MAX_BSZ must be the same as definition in get_output / save_output
MAX_BSZ = 512


@dataclass
class PredictorArgument:
    model_name_or_path: str = field(default=None, metadata={"help": "The directory of model."})
    model_prefix: str = field(default="model", metadata={"help": "the prefix name of static model"})
    src_length: int = field(default=None, metadata={"help": "The max length of source text."})
    max_length: int = field(default=None, metadata={"help": "the max length for decoding."})
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
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention"},
    )

    mode: str = field(
        default="dynamic", metadata={"help": "the type of predictor, it should be one of [dynamic, static]"}
    )
    inference_model: bool = field(default=False, metadata={"help": "whether use InferenceModel to do generation"})
    quant_type: str = field(
        default=None,
        metadata={"help": "Quantization type. Supported values: a8w8, weight_only_int4, weight_only_int8"},
    )
    avx_model: bool = field(
        default=False, metadata={"help": "whether use AvxModel to do generation when using cpu inference"}
    )
    avx_type: str = field(
        default=None,
        metadata={"help": "avx compute type. Supported values: fp16, bf16"},
    )
    batch_size: int = field(default=1, metadata={"help": "The batch size of data."})
    benchmark: bool = field(
        default=False,
        metadata={
            "help": "If benchmark set as `True`, we will force model decode to max_length, which is helpful to compute throughput. "
        },
    )

    block_attn: bool = field(default=False, metadata={"help": "whether use block attention"})
    block_size: int = field(default=64, metadata={"help": "the block size for cache_kvs."})
    cachekv_int8: bool = field(
        default=False,
        metadata={"help": "If cachekv_int8 set as `True`, cache kv would be quantized to int8 dynamically. "},
    )

    chat_template: str = field(
        default=None,
        metadata={
            "help": "the path of `chat_template.json` file to handle multi-rounds conversation. "
            "If is None(do not set --chat_template argument), it will use the default `chat_template.json`;"
            "If is equal with `model_name_or_path`, it will use the default loading; "
            "If is directory, it will find the `chat_template.json` under the directory; If is file, it will load it."
            "If is none string, it will not use chat_template.json."
        },
    )

    @property
    def total_max_length(self):
        return self.src_length + self.max_length

    @property
    def use_cachekv_int8(self):
        return "dynamic" if self.cachekv_int8 else "None"


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


def get_eos_token_id(
    tokenizer: PretrainedTokenizer, generation_config: Optional[GenerationConfig] = None
) -> int | List[List[int]]:
    """get eos_token_id from generation_config or tokenizer

    Returns:
        int | List[int]: eos_token_id to stop the generation
    """
    eos_token_ids = []
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)

    if generation_config is not None and generation_config.eos_token_id is not None:
        if isinstance(generation_config.eos_token_id, int):
            eos_token_ids.append(generation_config.eos_token_id)
        else:
            eos_token_ids.extend(generation_config.eos_token_id)

    eos_token_ids_dict = {str(item): item for item in eos_token_ids}
    return list(eos_token_ids_dict.values())


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

        try:
            self.generation_config = GenerationConfig.from_pretrained(config.model_name_or_path)
        except:
            logger.warning(
                "Can't find generation config, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

    def _preprocess(self, source):
        if self.tokenizer.chat_template is not None:
            source = [source] if isinstance(source, str) else source
            source = [self.tokenizer.apply_chat_template(sentence, tokenize=False) for sentence in source]

        tokenized_source = self.tokenizer(
            source,
            max_length=self.config.src_length,
            truncation=True,
            truncation_side="left",
            return_tensors=self.return_tensors,
            padding=True,
            # when use chat_template, it should not add special tokens
            # chatglm2 prefix-tokens can not be tokenized into ids
            add_special_tokens=self.tokenizer.chat_template is None
            or isinstance(self.tokenizer, (ChatGLMv2Tokenizer, ChatGLMTokenizer)),
        )
        return tokenized_source

    @abstractmethod
    def _infer(self, inputs):
        raise NotImplementedError

    def _postprocess(self, predictions, return_tokens=False):
        decoded_predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if return_tokens:
            return decoded_predictions, predictions
        else:
            return decoded_predictions

    def predict(self, input_texts: str | list[str], return_tokens=False):
        tokenized_source = self._preprocess(input_texts)
        predictions = self._infer(tokenized_source)
        decoded_predictions = self._postprocess(predictions, return_tokens=return_tokens)
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
                use_flash_attention=config.use_flash_attention,
                dtype=dtype,
                tensor_parallel_degree=self.tensor_parallel_degree,
                tensor_parallel_rank=self.tensor_parallel_rank,
            )

        if config.lora_path is not None:
            self.model = LoRAModel.from_pretrained(
                model=self.model, lora_path=config.lora_path, lora_config=lora_config
            )
            self.model.merge()
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
            eos_token_id=get_eos_token_id(self.tokenizer, self.generation_config),
            pad_token_id=self.tokenizer.pad_token_id,
            decode_strategy=self.config.decode_strategy,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
        )
        result = result[0]
        return result

    def stream_predict(self, inputs: dict[str, paddle.Tensor]):
        text_streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        input_features = self._preprocess(inputs)
        generation_kwargs = dict(
            **input_features,
            streamer=text_streamer,
            max_new_tokens=self.config.max_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=get_eos_token_id(self.tokenizer, self.generation_config),
            pad_token_id=self.tokenizer.pad_token_id,
            decode_strategy=(
                "greedy_search" if self.config.top_k == 1 and self.config.top_p == 1.0 else self.config.decode_strategy
            ),
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        return text_streamer


class StaticGraphPredictor(BasePredictor):
    def __init__(self, config: PredictorArgument, tokenizer: PretrainedTokenizer = None):
        super().__init__(config, tokenizer)

        inference_config = paddle.inference.Config(self.config.model_name_or_path, self.config.model_prefix)

        if self.config.device == "gpu":
            # set GPU configs accordingly
            inference_config.enable_use_gpu(100, 0)
        elif self.config.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            inference_config.disable_gpu()
        inference_config.disable_glog_info()
        inference_config.enable_new_executor()
        if in_pir_executor_mode():
            inference_config.enable_new_ir()
            if in_cinn_mode():
                inference_config.enable_cinn()

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
        self.pre_ids = paddle.full([config.batch_size, config.total_max_length], -1, dtype="int64")

        if config.device == "cpu" and config.avx_model:
            assert (
                "llama" in self.architectures and self.model_config.model_type != "llama-img2txt"
            ), "avx_mode only support llama now"
            self.cache_kvs = None
            self.attention_mask = None
            self.tgt_generation_mask = None
            self.tgt_pos = None
        else:
            self.arange_tensor_encoder = paddle.arange(config.total_max_length, dtype=self.dtype)
            self.cache_kvs = [paddle.zeros(shape, dtype=self.dtype) for shape in self.cache_kvs_shape]
            self.num_layers, self.num_attention_heads, self.head_dim = (
                len(self.cache_kvs),
                self.cache_kvs[0].shape[-3],
                self.cache_kvs[0].shape[-1],
            )
            self.tgt_generation_mask = paddle.ones(
                shape=[config.batch_size, 1, 1, config.total_max_length],
                dtype=self.dtype,
            )
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
            if config.export_precache:
                if config.prefix_path:
                    prefix_cache = (
                        paddle.to_tensor(np.load(f"{config.prefix_path}/pre_caches.npy"))
                        .astype(self.dtype)
                        .unsqueeze(2)
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
                    self.pre_caches = [
                        item.squeeze_(0) for item in paddle.split(prefix_cache, self.num_layers, axis=0)
                    ]
                else:
                    prefix_cache = paddle.zeros(
                        [self.num_layers, 2, config.batch_size, self.num_attention_heads, 128, self.head_dim],
                        dtype=self.dtype,
                    )
                    self.pre_caches = [
                        item.squeeze_(0) for item in paddle.split(prefix_cache, self.num_layers, axis=0)
                    ]

        try:
            self.generation_config = GenerationConfig.from_pretrained(config.model_name_or_path)
        except:
            logger.warning(
                "Can't find generation config, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

    def _postprocess(self, predictions, return_tokens=False):
        if paddle.distributed.get_rank() == 0:
            tokens: np.ndarray = load_real_time_tokens()
            decoded_predictions = self.tokenizer.batch_decode(
                tokens.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if return_tokens:
                return decoded_predictions, tokens.tolist()
            else:
                return decoded_predictions
        else:
            return None

    def _preprocess(self, source):
        if self.attention_mask is not None:
            self.attention_mask[:] = 0
        if self.tgt_generation_mask is not None:
            self.tgt_generation_mask[:] = 1
        pre_caches_length = 0 if not self.config.export_precache else self.pre_caches[0].shape[-2]

        if self.tokenizer.chat_template is not None:
            source = [source] if isinstance(source, str) else source
            source = [self.tokenizer.apply_chat_template(sentence, tokenize=False) for sentence in source]

        inputs = dybatch_preprocess(
            self.tokenizer,
            source,
            self.config.src_length,
            self.config.max_length,
            self.architectures,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            eos_token_id=get_eos_token_id(self.tokenizer, self.generation_config),
            benchmark=self.config.benchmark,
            pre_caches_length=pre_caches_length,
        )

        if "chatglmforcausallm" == self.architectures.lower():
            if inputs["input_ids"].shape[0] < self.config.batch_size:
                self.tgt_pos = self.tgt_pos[: inputs["input_ids"].shape[0]]
            for i in range(inputs["input_ids"].shape[0]):
                length = inputs["seq_len_encoder"][i][0]
                if self.attention_mask is not None:
                    self.attention_mask[i, 0, :length, :length] = 1
                    self.attention_mask[i, 0, : length - 1, length - 1] = 0
                if self.tgt_pos is not None:
                    self.tgt_pos[i, 0, 0] = paddle.to_tensor([length], dtype="int64")

                if pre_caches_length > 0:
                    prefix_attention_mask = paddle.ones(
                        [1, length, pre_caches_length], dtype=self.attention_mask.dtype
                    )
                    post_attention_mask = paddle.ones(
                        shape=(length, length), dtype=self.attention_mask.dtype
                    ).unsqueeze_(axis=0)
                    post_attention_mask[0, : length - 1, length - 1] = 0
                    self.attention_mask[i, 0, :length, : length + pre_caches_length] = paddle.concat(
                        [prefix_attention_mask, post_attention_mask], axis=2
                    )

            inputs["tgt_pos"] = self.tgt_pos
        elif "bloom" in self.architectures:
            for i in range(inputs["input_ids"].shape[0]):
                length = inputs["seq_len_encoder"][i][0]
                if self.attention_mask is not None:
                    self.attention_mask[i, :, :length, :length] = paddle.tril(
                        paddle.ones(shape=(length, length), dtype=self.config.dtype)
                    )
                if pre_caches_length > 0:
                    if self.config.prefix_path is None:
                        prefix_attention_mask = paddle.zeros([1, length, pre_caches_length], dtype=self.config.dtype)
                    else:
                        prefix_attention_mask = paddle.ones([1, length, pre_caches_length], dtype=self.config.dtype)
                    post_attention_mask = paddle.tril(
                        paddle.ones(shape=(length, length), dtype=self.config.dtype)
                    ).unsqueeze_(axis=0)
                    if self.attention_mask is not None:
                        self.attention_mask[i, :, :length, : length + pre_caches_length] = paddle.concat(
                            [prefix_attention_mask, post_attention_mask], axis=2
                        )

            inputs["tgt_pos"] = inputs["tgt_pos"] + pre_caches_length
            # alibi encoder
            alibi_slopes = get_alibi_slopes(self.model_config.n_head)
            inputs["position_ids"] = paddle.to_tensor(alibi_slopes, dtype="float32")

            alibi = alibi_slopes[None, :, None, None] * self.arange_tensor_encoder

            if self.model_config.tensor_parallel_degree > 1:
                block_size = self.model_config.n_head // self.model_config.tensor_parallel_degree
                alibi = alibi[
                    :,
                    self.model_config.tensor_parallel_rank
                    * block_size : (self.model_config.tensor_parallel_rank + 1)
                    * block_size,
                ]
                alibi = alibi.reshape([self.config.batch_size, block_size, 1, self.config.max_length])
                inputs["position_ids"] = inputs["position_ids"][
                    self.model_config.tensor_parallel_rank
                    * block_size : (self.model.config.tensor_parallel_rank + 1)
                    * block_size
                ]

            alibi_encoder = alibi.expand(
                [
                    self.config.batch_size,
                    self.model_config.n_head // self.model_config.tensor_parallel_degree,
                    self.config.total_max_length,
                    self.config.total_max_length,
                ]
            )
            # only generate valid encoder attention mask, other place set 0.
            alibi_encoder[i, :, length:, length:] = 0

            alibi_decoder = alibi.expand(
                [
                    self.config.batch_size,
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
            for i in range(inputs["input_ids"].shape[0]):
                length = inputs["seq_len_encoder"][i][0]
                if self.attention_mask is not None:
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
                    if self.attention_mask is not None:
                        self.attention_mask[i, 0, :length, : length + pre_caches_length] = paddle.concat(
                            [prefix_attention_mask, post_attention_mask], axis=2
                        )

        inputs["pre_ids"] = self.pre_ids
        inputs["attention_mask"] = self.attention_mask
        inputs["tgt_generation_mask"] = self.tgt_generation_mask

        if self.config.device == "cpu" and self.config.avx_model:
            inputs.pop("position_ids")
            inputs.pop("tgt_pos")
            inputs.pop("attention_mask")
            inputs.pop("tgt_generation_mask")

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
        if not is_paddlenlp_ops_available():
            raise ValueError(
                "you should install the paddlenlp ops to run inference predictor, "
                "https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md"
            )

        # register the custome ops
        if predictor_args.device == "cpu" and predictor_args.avx_model:
            import_module("paddlenlp_ops.xft_llama_layer")
        else:
            import_module("paddlenlp_ops.encode_rotary_qk")
            import_module("paddlenlp_ops.get_padding_offset")
            import_module("paddlenlp_ops.qkv_transpose_split")
            import_module("paddlenlp_ops.rebuild_padding")
            import_module("paddlenlp_ops.transpose_remove_padding")
            import_module("paddlenlp_ops.write_cache_kv")

        infer_model_path = get_infer_model_path(predictor_args.model_name_or_path, predictor_args.model_prefix)

        config = paddle.inference.Config(infer_model_path + ".pdmodel", infer_model_path + ".pdiparams")

        config.switch_ir_optim(True)
        # remove `gpu_cpu_map_matmul_v2_to_matmul_pass` to avoid mapping matmul_v2 -> matmul op
        if predictor_args.dtype == "bfloat16":
            config.delete_pass("gpu_cpu_map_matmul_v2_to_matmul_pass")

        if predictor_args.device in paddle.device.get_all_custom_device_type():
            device_id = int(os.environ.get("FLAGS_selected_{}s".format(predictor_args.device), 0))
            config.enable_custom_device(predictor_args.device, device_id)
        elif predictor_args.device == "xpu":
            raise ValueError(
                "you should export xpu static model with --block_attn flag and use predictor with --block_attn too"
                "https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/inference.md"
            )
        elif predictor_args.device == "cpu" and predictor_args.avx_model:
            config.disable_gpu()
            config.switch_ir_optim(False)
        else:
            device_id = int(os.environ.get("FLAGS_selected_gpus", 0))
            config.enable_use_gpu(100, device_id)
        config.enable_new_executor()

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
        self.cache_kvs_shape = model.get_cache_kvs_shape(model.config, config.batch_size, config.total_max_length)
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


class BlockInferencePredictorMixin:
    def __init__(self, config: PredictorArgument, tokenizer: PretrainedTokenizer):

        self.num_layers = len(self.cache_kvs_shape) // 2
        self.num_attention_heads = self.cache_kvs_shape[0][-3]
        self.head_dim = self.cache_kvs_shape[0][-1]
        self.max_block_nums = self.cache_kvs_shape[0][0]
        self.batch_size = config.batch_size
        self.model_name_or_path = config.model_name_or_path

        self.architectures = self.model_config.architectures[0].lower()

        self.dtype = config.dtype or self.model_config.dtype

        self.total_max_length = config.src_length + config.max_length
        self.block_size = config.block_size
        self.pre_max_block_num = (self.total_max_length + config.block_size - 1) // config.block_size
        self.max_block_nums = config.batch_size * self.pre_max_block_num

        try:
            self.rope_theta = self.model_config.rope_theta
        except:
            self.rope_theta = 10000.0

        self.pre_cache_length = 0

        if config.export_precache:
            pre_cache_npy = np.load(config.prefix_path)
            self.pre_cache_length = pre_cache_npy.shape[-2]
            config.max_length -= self.pre_cache_length
            self.pre_caches = [
                paddle.zeros(
                    [config.batch_size, self.num_attention_heads, self.pre_cache_length, self.head_dim],
                    dtype=self.dtype,
                )
                for _ in range(2 * self.num_layers)
            ]
            for i in range(self.num_layers):
                self.pre_caches[2 * i][:, :, :, :] = paddle.to_tensor(pre_cache_npy[i][0], dtype=self.dtype).unsqueeze(
                    0
                )
                self.pre_caches[2 * i + 1][:, :, :, :] = paddle.to_tensor(
                    pre_cache_npy[i][1], dtype=self.dtype
                ).unsqueeze(0)

            self.pre_cache_mask = paddle.zeros(
                shape=[config.batch_size, 1, config.src_length, config.src_length + self.pre_cache_length],
                dtype=config.dtype,
            )
            self.pre_cache_mask[:, :, :, : self.pre_cache_length] = 1
            self.pre_cache_mask[:, :, :, self.pre_cache_length :] = paddle.tril(
                paddle.ones(shape=[config.batch_size, 1, config.src_length, config.src_length], dtype=config.dtype)
            )

        if config.use_cachekv_int8 == "dynamic":
            self.k_quant_scales = [
                paddle.zeros([config.batch_size, self.num_attention_heads], dtype="float32")
                for _ in range(self.num_layers)
            ]
            self.v_quant_scales = [
                paddle.zeros([config.batch_size, self.num_attention_heads], dtype="float32")
                for _ in range(self.num_layers)
            ]
            self.k_dequant_scales = [
                paddle.zeros([config.batch_size, self.num_attention_heads], dtype="float32")
                for _ in range(self.num_layers)
            ]
            self.v_dequant_scales = [
                paddle.zeros([config.batch_size, self.num_attention_heads], dtype="float32")
                for _ in range(self.num_layers)
            ]

        if config.benchmark:
            self.min_length = config.max_length
        else:
            self.min_length = 2

        self.free_list = [i for i in range(self.max_block_nums)][::-1]
        self.used_list = [[] for _ in range(config.batch_size)]

    def init_inputs(self, config: PredictorArgument):
        self.inputs = {}

        if config.export_precache:
            self.inputs["src_mask"] = (self.pre_cache_mask - 1) * 1e4
        self.inputs["pre_ids"] = paddle.full([config.batch_size, self.total_max_length], -1, dtype="int64")
        self.inputs["bad_tokens"] = paddle.to_tensor(
            [
                -1,
            ],
            dtype="int64",
        )
        self.inputs["penalty_score"] = paddle.full(shape=[config.batch_size, 1], fill_value=1.0, dtype="float32")
        self.inputs["frequency_score"] = paddle.full(shape=[config.batch_size, 1], fill_value=0.0, dtype="float32")
        self.inputs["presence_score"] = paddle.full(shape=[config.batch_size, 1], fill_value=0.0, dtype="float32")

        self.inputs["min_length"] = paddle.full(
            shape=[config.batch_size, 1], fill_value=self.min_length, dtype="int64"
        )
        self.inputs["max_length"] = paddle.full(
            shape=[config.batch_size, 1], fill_value=config.max_length, dtype="int64"
        )
        self.inputs["stop_nums"] = paddle.full(shape=[1], fill_value=config.batch_size, dtype="int64")
        self.inputs["rope_emb"] = self._get_rotary_position_embedding(
            paddle.arange(self.total_max_length).reshape((1, -1)), self.head_dim, self.rope_theta
        )
        eos_token_id = get_eos_token_id(self.tokenizer, self.generation_config)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.inputs["eos_token_id"] = paddle.to_tensor(
            np.array(eos_token_id * config.batch_size).reshape(-1, 1).astype("int64")
        )
        # bloom model needs src_mask and tgt_mask!
        if "bloom" in self.architectures:
            lower_one_tril = paddle.tril(
                paddle.ones(shape=(self.total_max_length, self.total_max_length), dtype=self.dtype)
            )
            lower_one_tril = lower_one_tril[None, None, :, :]
            self.inputs["src_mask"] = lower_one_tril.tile([self.batch_size, 1, 1, 1])
            self.inputs["tgt_mask"] = paddle.full(
                shape=[config.batch_size, 1, 1, self.total_max_length], fill_value=1, dtype=self.dtype
            )
            arange_tensor_encoder = paddle.arange(self.total_max_length).astype(self.dtype)
            alibi_slopes = get_alibi_slopes(self.num_attention_heads)
            alibi = alibi_slopes[None, :, None, None] * arange_tensor_encoder
            alibi_encoder = alibi.tile([self.batch_size, 1, self.total_max_length, 1])
            alibi_decoder = alibi.tile(
                [
                    self.batch_size,
                    1,
                    1,
                    1,
                ]
            )
            # self.inputs["src_mask/tgt_mask"] is read only, will not be updated!
            self.inputs["src_mask"] = (
                alibi_encoder + (1 - self.inputs["src_mask"]) * paddle.finfo(self.dtype).min
            ).cast(self.dtype)
            self.inputs["tgt_mask"] = (
                alibi_decoder + (1 - self.inputs["tgt_mask"]) * paddle.finfo(self.dtype).min
            ).cast(self.dtype)

        # need update
        self.inputs["block_tables"] = paddle.full(
            shape=[config.batch_size, self.pre_max_block_num], fill_value=-1, dtype="int32"
        )
        self.inputs["input_ids"] = paddle.full(
            shape=[config.batch_size, self.total_max_length], fill_value=-1, dtype="int64"
        )
        self.inputs["top_p"] = paddle.full(shape=[config.batch_size, 1], fill_value=config.top_p, dtype="float32")
        self.inputs["temperature"] = paddle.full(shape=[config.batch_size, 1], fill_value=1.0, dtype="float32")
        self.inputs["seq_lens_this_time"] = paddle.full(shape=[config.batch_size, 1], fill_value=0, dtype="int32")
        self.inputs["seq_lens_encoder"] = paddle.full(shape=[config.batch_size, 1], fill_value=0, dtype="int32")
        self.inputs["seq_lens_decoder"] = paddle.full(shape=[config.batch_size, 1], fill_value=0, dtype="int32")
        self.inputs["step_idx"] = paddle.full(shape=[config.batch_size, 1], fill_value=0, dtype="int64")
        self.inputs["not_need_stop"] = paddle.full(shape=[1], fill_value=False, dtype="bool")
        self.inputs["stop_flags"] = paddle.full(shape=[config.batch_size, 1], fill_value=True, dtype="bool")
        self.inputs["next_tokens"] = paddle.full(shape=[config.batch_size, 1], fill_value=-1, dtype="int64")
        self.inputs["is_block_step"] = paddle.full(shape=[config.batch_size], fill_value=False, dtype="bool")
        free_list = list(range(self.pre_max_block_num - 1, int(self.pre_max_block_num * 0.75) - 1, -1))
        self.inputs["encoder_block_lens"] = paddle.full(shape=[config.batch_size], fill_value=0, dtype="int32")
        self.inputs["step_block_list"] = paddle.full(shape=[config.batch_size], fill_value=-1, dtype="int32")
        self.inputs["step_lens"] = paddle.full(shape=[1], fill_value=0, dtype="int32")
        self.inputs["recover_block_list"] = paddle.full(shape=[config.batch_size], fill_value=-1, dtype="int32")
        self.inputs["recover_lens"] = paddle.full(shape=[1], fill_value=0, dtype="int32")
        self.inputs["need_block_list"] = paddle.full(shape=[config.batch_size], fill_value=-1, dtype="int32")
        self.inputs["need_block_len"] = paddle.full(shape=[1], fill_value=0, dtype="int32")
        self.inputs["used_list_len"] = paddle.full(shape=[config.batch_size], fill_value=0, dtype="int32")
        self.inputs["free_list"] = paddle.to_tensor(free_list, dtype="int32")
        self.inputs["free_list_len"] = paddle.full(shape=[1], fill_value=self.pre_max_block_num * 0.25, dtype="int32")

    def _get_rotary_position_embedding(self, position_ids, head_dim, rope_theta=10000.0):
        """
        Pre-calculate rotary position embedding for position_ids.

        Args:
            position_ids: [1, S]
            head_dim: D

        Returns:
            rot_emb: [2, 1, S, 1, D], cos + sin
        """
        bsz, max_seq_len = position_ids.shape[:2]
        rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, head_dim), dtype="float32")
        inv_freq = rope_theta ** (-paddle.arange(0, head_dim, 2, dtype="float32") / head_dim)

        # shape: [B, S, D/2]
        freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
        # shape: [B, S, 1, D]
        emb = paddle.concat([freqs, freqs], axis=-1).reshape((bsz, max_seq_len, 1, head_dim))

        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)
        return rot_emb

    def _preprocess(self, source):
        if self.tokenizer.chat_template is not None:
            source = [source] if isinstance(source, str) else source
            source = [self.tokenizer.apply_chat_template(sentence, tokenize=False) for sentence in source]

        for i, text in enumerate(source):
            tokens = self.tokenizer(
                text,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=self.config.src_length,
                # if use chat_template, it will not add special_tokens
                add_special_tokens=self.tokenizer.chat_template is None
                or isinstance(self.tokenizer, (ChatGLMv2Tokenizer, ChatGLMTokenizer)),
            )
            input_ids = tokens["input_ids"][0]
            length = len(input_ids)
            self.inputs["input_ids"][i : i + 1, :length] = input_ids
            self.inputs["penalty_score"][i : i + 1] = self.config.repetition_penalty
            self.inputs["frequency_score"][i : i + 1] = 0.0
            self.inputs["presence_score"][i : i + 1] = 0.0
            self.inputs["top_p"][i : i + 1] = self.config.top_p
            self.inputs["temperature"][i : i + 1] = self.config.temperature
            self.inputs["seq_lens_this_time"][i : i + 1] = length
            self.inputs["seq_lens_encoder"][i : i + 1] = length
            self.inputs["seq_lens_decoder"][i : i + 1] = 0
            self.inputs["step_idx"][i : i + 1] = 0
            self.inputs["stop_flags"][i : i + 1] = False
            self.inputs["not_need_stop"][0] = True
            need_block_nums = (
                length + self.config.max_length + self.pre_cache_length + self.block_size - 1
            ) // self.block_size
            for bi in range(need_block_nums):
                bi_now = self.free_list.pop()
                self.used_list[i].append(bi_now)
                self.inputs["block_tables"][i : i + 1, bi] = bi_now


class DygraphBlockInferencePredictor(BlockInferencePredictorMixin, BasePredictor):
    def __init__(
        self,
        config: PredictorArgument,
        model: PretrainedModel = None,
        tokenizer: PretrainedTokenizer = None,
    ):
        self.cache_kvs_shape = model.get_cache_kvs_shape(model.config, config.batch_size)
        BasePredictor.__init__(self, config, tokenizer)
        BlockInferencePredictorMixin.__init__(self, config, tokenizer)

        if config.use_cachekv_int8 == "dynamic" or config.use_cachekv_int8 == "static":
            self.cache_kvs = [paddle.zeros(shape, dtype="uint8") for shape in self.cache_kvs_shape]
        else:
            self.cache_kvs = [paddle.zeros(shape, dtype=self.dtype) for shape in self.cache_kvs_shape]

        self.model = model

        self.init_inputs(config)
        if config.export_precache:
            self.inputs["pre_caches"] = self.pre_caches
        if config.use_cachekv_int8 == "dynamic":
            self.inputs["k_quant_scales"] = self.k_quant_scales
            self.inputs["v_quant_scales"] = self.v_quant_scales
            self.inputs["k_dequant_scales"] = self.k_dequant_scales
            self.inputs["v_dequant_scales"] = self.v_dequant_scales

        self.inputs["cache_kvs"] = self.cache_kvs

    @paddle.no_grad()
    def _infer(self, inputs: dict[str, paddle.Tensor]):
        self.model.generate(
            **inputs,
        )

    @paddle.no_grad()
    def predict(self, input_texts: str | list[str], return_tokens=False):
        self._preprocess(input_texts)

        result_queue = mp.Queue()
        tensor_queue = mp.Queue()

        output_tensor = paddle.full(shape=[MAX_BSZ + 2, 1], fill_value=2, dtype="int64")
        output_tensor = output_tensor.cpu()
        tensor_queue.put(output_tensor)

        read_res_process = mp.Process(target=read_res, args=[self.model_name_or_path, tensor_queue, result_queue])
        read_res_process.start()

        while self.inputs["not_need_stop"]:
            self._infer(self.inputs)
        # reset free_list
        for i in range(self.config.batch_size):
            self.free_list.extend(self.used_list[i])
            self.used_list[i] = []

        outputs = []
        output_tokens = []
        while len(outputs) < self.batch_size:
            result = result_queue.get(timeout=1)
            outputs.append(result[-1])
            output_tokens.append(result[-2])
        if return_tokens:
            return outputs, output_tokens
        else:
            return outputs


class StaticBlockInferencePredictor(BlockInferencePredictorMixin, BasePredictor):
    def __init__(
        self,
        config: PredictorArgument,
        cache_kvs_shape: list[list[int]],
        tokenizer: PretrainedTokenizer = None,
    ):
        self.cache_kvs_shape = cache_kvs_shape
        BasePredictor.__init__(self, config, tokenizer)
        BlockInferencePredictorMixin.__init__(self, config, tokenizer)

        self.init_inputs(config)

        if config.export_precache:
            for i in range(self.num_layers):
                self.inputs["pre_caches_{}".format(i)] = self.pre_caches[i]

        self.cache_kvs = {}
        if config.use_cachekv_int8 == "dynamic" or config.use_cachekv_int8 == "static":
            for i in range(len(self.cache_kvs_shape) // 2):
                self.cache_kvs["key_caches_{}".format(i)] = paddle.zeros(self.cache_kvs_shape[2 * i], dtype="uint8")
                self.cache_kvs["value_caches_{}".format(i)] = paddle.zeros(
                    self.cache_kvs_shape[2 * i + 1], dtype="uint8"
                )
        else:
            for i in range(len(self.cache_kvs_shape) // 2):
                self.cache_kvs["key_caches_{}".format(i)] = paddle.zeros(
                    self.cache_kvs_shape[2 * i], dtype=config.dtype
                )
                self.cache_kvs["value_caches_{}".format(i)] = paddle.zeros(
                    self.cache_kvs_shape[2 * i + 1], dtype=config.dtype
                )

        for i in range(self.num_layers):
            if self.config.use_cachekv_int8 == "dynamic":
                self.inputs["k_quant_scales_" + str(i)] = self.k_quant_scales[i]
                self.inputs["v_quant_scales_" + str(i)] = self.v_quant_scales[i]
                self.inputs["k_dequant_scales_" + str(i)] = self.k_dequant_scales[i]
                self.inputs["v_dequant_scales_" + str(i)] = self.v_dequant_scales[i]

        self._create_predictor(config)
        self.input_names = self.predictor.get_input_names()

        self._share_data()
        self.seq_lens_handle = self.predictor.get_input_handle("seq_lens_this_time")

    def _create_predictor(self, predictor_args: PredictorArgument):
        if not is_paddlenlp_ops_available():
            raise ValueError(
                "you should install the paddlenlp ops to run inference predictor, "
                "https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md"
            )

        infer_model_path = get_infer_model_path(predictor_args.model_name_or_path, predictor_args.model_prefix)

        config = paddle.inference.Config(infer_model_path + ".pdmodel", infer_model_path + ".pdiparams")

        config.switch_ir_optim(False)
        if predictor_args.device in paddle.device.get_all_custom_device_type():
            device_id = int(os.environ.get("FLAGS_selected_{}s".format(predictor_args.device), 0))
            config.enable_custom_device(predictor_args.device, device_id)
        elif predictor_args.device == "xpu":
            config.enable_xpu()
            device_id = int(os.environ.get("FLAGS_selected_xpus", 0))
            config.set_xpu_device_id(device_id)
            xpu_config = paddle.inference.XpuConfig()
            xpu_config.device_id = device_id
            xpu_config.l3_size = 63 * 1024 * 1024
            xpu_config.l3_autotune_size = 63 * 1024 * 1024
            config.set_xpu_config(xpu_config)
            config.switch_ir_optim(True)
            config.enable_memory_optim()
        else:
            device_id = int(os.environ.get("FLAGS_selected_gpus", 0))
            config.enable_use_gpu(100, device_id)
        # config.disable_glog_info()
        # config.enable_memory_optim()

        if predictor_args.device == "npu":
            import paddle_custom_device.npu.passes as passes

            config.switch_ir_optim(True)
            pass_builder = config.pass_builder()
            passes.addPasses(pass_builder, self.model_config.model_type, self.model_config.quant_type)

        if self.tensor_parallel_degree > 1:
            trainer_endpoints = fleet.worker_endpoints()
            current_endpoint = trainer_endpoints[self.tensor_parallel_rank]

            dist_config = config.dist_config()
            dist_config.set_ranks(self.tensor_parallel_degree, self.tensor_parallel_rank)
            dist_config.set_endpoints(trainer_endpoints, current_endpoint)
            dist_config.enable_dist_model(True)

            dist_config.set_comm_init_config(os.path.join(predictor_args.model_name_or_path, "rank_mapping.csv"))
            config.set_dist_config(dist_config)

        self.predictor = paddle.inference.create_predictor(config)

    def _share_data(self):
        """
        Share external data for inference predictor.
        """
        for name in self.input_names:
            if "pre_key_" in name or "pre_value_" in name:
                input_tensor = self.predictor.get_input_handle(name)
                input_tensor.share_external_data(self.inputs[name])
                continue
            if "caches" in name:
                input_tensor = self.predictor.get_input_handle(name)
                input_tensor.share_external_data(self.cache_kvs[name])
                continue
            if "seq_lens_this_time" in name:
                continue
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.share_external_data(self.inputs[name])

    def _infer(self):
        self.predictor.run()

    def predict(self, input_texts: str | list[str], return_tokens=False):

        s_time = time.time()
        self._preprocess(input_texts)
        real_bsz = len(input_texts)

        import copy

        seq_lens_this_time = copy.deepcopy(self.inputs["seq_lens_this_time"][:real_bsz])
        self.seq_lens_handle.share_external_data(seq_lens_this_time)
        logger.info(f"preprocess spend {time.time()  -  s_time}")

        result_queue = mp.Queue()
        tensor_queue = mp.Queue()

        output_tensor = paddle.full(shape=[MAX_BSZ + 2, 1], fill_value=2, dtype="int64")
        output_tensor = output_tensor.cpu()
        tensor_queue.put(output_tensor)

        read_res_process = mp.Process(target=read_res, args=[self.model_name_or_path, tensor_queue, result_queue])
        read_res_process.start()

        s_time = time.time()
        while self.inputs["not_need_stop"]:
            self.predictor.run()
        logger.info(f"running spend {time.time()  -  s_time}")

        # reset free_list
        for i in range(self.config.batch_size):
            self.free_list.extend(self.used_list[i])
            self.used_list[i] = []

        outputs = []
        output_tokens = []
        while len(outputs) < self.batch_size:
            result = result_queue.get(timeout=1)
            outputs.append(result[-1])
            output_tokens.append(result[-2])
        if return_tokens:
            return outputs, output_tokens
        else:
            return outputs

    def _preprocess(self, source):
        BlockInferencePredictorMixin._preprocess(self, source)
        for i, text in enumerate(source):
            tokens = self.tokenizer(
                text, return_tensors="np", padding=False, truncation=True, max_length=(self.config.src_length)
            )
            input_ids = tokens["input_ids"][0]
            length = len(input_ids)
            need_block_nums = (
                length + self.config.max_length + self.pre_cache_length + self.block_size - 1
            ) // self.block_size
            self.inputs["encoder_block_lens"][i : i + 1] = need_block_nums


def get_ptq_multicards_num(directory):
    count = 0
    prefix = "act_scales_"
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            count += 1
    return count


def create_predictor(
    predictor_args: PredictorArgument,
    model_args: ModelArgument,
    tensor_parallel_degree: int = 1,
    tensor_parallel_rank: int = 0,
):
    tokenizer = AutoTokenizer.from_pretrained(
        predictor_args.model_name_or_path,
    )
    # init chat_template for tokenizer
    init_chat_template(tokenizer, predictor_args.model_name_or_path, predictor_args.chat_template)

    # TODO(wj-Mcat): fix llama tokenzier pad_token bug
    if (isinstance(tokenizer, (LlamaTokenizer, Llama3Tokenizer))) and not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(predictor_args.model_name_or_path)

    max_position_embeddings = get_model_max_position_embeddings(config)
    if max_position_embeddings is None:
        max_position_embeddings = 2048
        logger.warning("Can not retrieval `max_position_embeddings` from config.json, use default value 2048")

    if predictor_args.src_length is None:
        if predictor_args.max_length is None:
            predictor_args.src_length = get_default_max_encoding_length(config)
            predictor_args.max_length = get_default_max_decoding_length(config)
        else:
            predictor_args.src_length = max_position_embeddings - predictor_args.max_length
            if predictor_args.src_length <= 0:
                raise ValueError(
                    f"--max_length<{predictor_args.max_length}> param should be smaller "
                    f"than max_position_embeddings<{max_position_embeddings}>"
                )
    else:
        if predictor_args.max_length is None:
            predictor_args.max_length = max_position_embeddings - predictor_args.src_length
            if predictor_args.max_length <= 0:
                raise ValueError(
                    f"--src_length<{predictor_args.src_length}> param should be smaller "
                    f"than max_position_embeddings<{max_position_embeddings}>"
                )
        else:
            if predictor_args.src_length + predictor_args.max_length > max_position_embeddings:
                raise ValueError(
                    f"The sum of src_length<{predictor_args.src_length}> and "
                    f"max_length<{predictor_args.max_length}> should be smaller than or equal to "
                    f"the maximum position embedding size<{max_position_embeddings}>"
                )

    # update config parameter for inference predictor
    if predictor_args.decode_strategy == "greedy_search":
        predictor_args.top_p = 0.0
        predictor_args.temperature = 1.0

    tensor_parallel_rank, tensor_parallel_degree = init_dist_env()
    if not predictor_args.inference_model:
        tokenizer.padding_side = "left"
        if predictor_args.mode == "dynamic":
            if model_args.model_type == "gpt-3":
                sys.path.append("./gpt-3")
                from modeling import GPTForCausalLM

                model = GPTForCausalLM.from_pretrained(
                    predictor_args.model_name_or_path,
                    dtype=predictor_args.dtype,
                    tensor_parallel_degree=tensor_parallel_degree,
                    tensor_parallel_rank=tensor_parallel_rank,
                    tensor_parallel_output=False,
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
                    tensor_parallel_output=False,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    predictor_args.model_name_or_path,
                    dtype=predictor_args.dtype,
                    use_flash_attention=predictor_args.use_flash_attention,
                    tensor_parallel_degree=tensor_parallel_degree,
                    tensor_parallel_rank=tensor_parallel_rank,
                    tensor_parallel_output=False,
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
            config.weight_only_quant_bits = -1
            config.quant_type = None
            config.model_name_or_path = ""
            config.use_cachekv_int8 = predictor_args.use_cachekv_int8
            config.single_card_ptq = True
            if predictor_args.avx_model:
                config.avx_type = predictor_args.avx_type

            if predictor_args.quant_type is not None:
                if predictor_args.quant_type.startswith("weight_only_int"):
                    weight_only_quant_bits = int(predictor_args.quant_type[-1])
                    config.weight_only_quant_bits = weight_only_quant_bits
                    config.quant_type = predictor_args.quant_type
                elif predictor_args.quant_type == "a8w8":
                    config.quant_type = predictor_args.quant_type

            if config.quantization_config.quant_type is not None and "a8w8" in config.quantization_config.quant_type:
                config.model_name_or_path = predictor_args.model_name_or_path
                config.quant_type = config.quantization_config.quant_type

                ptq_multicards_num = get_ptq_multicards_num(config.model_name_or_path)
                logger.info(f"PTQ from {ptq_multicards_num} cards, so we will not split")
                if ptq_multicards_num > 1:
                    config.single_card_ptq = False

                # Turn on GEMM int8 kernel tuning
                paddle.base.core.enable_autotune()
                paddle.base.core.update_autotune_status()

            if "llama" in config.architectures[0].lower():
                if model_args.model_type == "llama-img2txt":
                    # we use llama for img2txt.
                    from paddlenlp.experimental.transformers import (
                        LlamaForMiniGPT4InferenceModel as LlamaInferenceModel,
                    )
                elif predictor_args.block_attn:
                    config.max_seq_len = predictor_args.total_max_length
                    config.block_size = predictor_args.block_size
                    from paddlenlp.experimental.transformers import (
                        LlamaForCausalLMBlockInferenceModel as LlamaInferenceModel,
                    )

                    model = LlamaInferenceModel.from_pretrained(
                        predictor_args.model_name_or_path,
                        config=config,
                        dtype=predictor_args.dtype,
                        tensor_parallel_degree=tensor_parallel_degree,
                        tensor_parallel_rank=tensor_parallel_rank,
                    )
                else:
                    if predictor_args.device == "xpu":
                        raise ValueError(
                            "you should run xpu dynamic model with --block_attn flag"
                            "https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/inference.md"
                        )
                    elif predictor_args.device == "cpu" and predictor_args.avx_model:
                        from paddlenlp.experimental.transformers import (
                            LlamaForCausalLMAvxInferenceModel as LlamaInferenceModel,
                        )
                    else:
                        from paddlenlp.experimental.transformers import (
                            LlamaForCausalLMInferenceModel as LlamaInferenceModel,
                        )

                    model = LlamaInferenceModel.from_pretrained(
                        predictor_args.model_name_or_path,
                        config=config,
                        dtype=predictor_args.dtype,
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

            elif "chatglmv2forcausallm" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    ChatGLMv2ForCausalLMInferenceModel as Model,
                )

                model = Model.from_pretrained(
                    predictor_args.model_name_or_path, config=config, dtype=predictor_args.dtype
                )
                model.eval()
            elif "chatglmforcausallm" in config.architectures[0].lower():
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
                if predictor_args.block_attn:
                    from paddlenlp.experimental.transformers import (
                        BlommForCausalBlockLMInferenceModel as BloomInferenceModel,
                    )

                    config.block_size = predictor_args.block_size
                    config.max_seq_len = predictor_args.total_max_length
                else:
                    from paddlenlp.experimental.transformers import (
                        BloomForCausalLMInferenceModel as BloomInferenceModel,
                    )
                model = BloomInferenceModel.from_pretrained(
                    predictor_args.model_name_or_path,
                    config=config,
                    dtype=predictor_args.dtype,
                )
                cache_kvs_shape = BloomInferenceModel.get_cache_kvs_shape(
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
            elif "qwen" in config.architectures[0].lower():
                if model_args.model_type == "qwen-img2txt":
                    # we use qwen for img2txt.
                    from paddlenlp.experimental.transformers import (
                        QWenForQWenVLInferenceModel as QWenInferenceModel,
                    )
                else:
                    from paddlenlp.experimental.transformers import (
                        QWenForCausalLMInferenceModel as QWenInferenceModel,
                    )
                model = QWenInferenceModel.from_pretrained(
                    predictor_args.model_name_or_path,
                    config=config,
                    dtype=predictor_args.dtype,
                )
                model.eval()
            else:
                raise ValueError("the `model type` should be one of [llama, chatglm, bloom, gpt, qwen]")
            if predictor_args.block_attn:
                predictor = DygraphBlockInferencePredictor(predictor_args, model=model, tokenizer=tokenizer)
            else:
                predictor = DygraphInferencePredictor(predictor_args, model=model, tokenizer=tokenizer)

        elif predictor_args.mode == "static":
            config = AutoConfig.from_pretrained(predictor_args.model_name_or_path)
            if "llama" in config.architectures[0].lower():
                if predictor_args.block_attn:
                    config.block_size = predictor_args.block_size
                    config.max_seq_len = predictor_args.total_max_length
                    config.use_dynamic_cachekv_quant = predictor_args.use_cachekv_int8 == "dynamic"
                    from paddlenlp.experimental.transformers import (
                        LlamaForCausalLMBlockInferenceModel as LlamaInferenceModel,
                    )
                elif predictor_args.avx_model and predictor_args.device == "cpu":
                    from paddlenlp.experimental.transformers import (
                        LlamaForCausalLMAvxInferenceModel as LlamaInferenceModel,
                    )
                else:
                    from paddlenlp.experimental.transformers import (
                        LlamaForCausalLMInferenceModel as LlamaInferenceModel,
                    )

                cache_kvs_shape = LlamaInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            elif "chatglmv2forcausallm" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    ChatGLMv2ForCausalLMInferenceModel,
                )

                cache_kvs_shape = ChatGLMv2ForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            elif "chatglmv2forcausallm" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    ChatGLMv2ForCausalLMInferenceModel,
                )

                cache_kvs_shape = ChatGLMv2ForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            elif "chatglmforcausallm" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    ChatGLMForCausalLMInferenceModel,
                )

                cache_kvs_shape = ChatGLMForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            elif "bloom" in config.architectures[0].lower():
                if predictor_args.block_attn:
                    from paddlenlp.experimental.transformers import (
                        BlommForCausalBlockLMInferenceModel as BloomInferenceModel,
                    )

                    config.block_size = predictor_args.block_size
                    config.max_seq_len = predictor_args.total_max_length
                else:
                    from paddlenlp.experimental.transformers import (
                        BloomForCausalLMInferenceModel as BloomInferenceModel,
                    )
                cache_kvs_shape = BloomInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            elif "gpt" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    GPTForCausalLMInferenceModel,
                )

                cache_kvs_shape = GPTForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            elif "qwen" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    QWenForCausalLMInferenceModel,
                )

                cache_kvs_shape = QWenForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            else:
                raise ValueError("the `model type` should be one of [llama, chatglm, bloom, gpt, qwen]")
            if predictor_args.block_attn:
                predictor = StaticBlockInferencePredictor(predictor_args, cache_kvs_shape, tokenizer=tokenizer)
            else:
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
                if isinstance(example["src"], str) or predictor.tokenizer.chat_template is None:
                    if isinstance(example["src"], str):
                        source_texts.append(example["src"])
                        target_texts.append(example["tgt"])
                    else:
                        # load multi-rounds dataset
                        source_texts.append(example["src"][0])
                        target_texts.append(example["tgt"][0])
                else:
                    source_texts.append(list(zip(example["src"], example["tgt"])))
                    target_texts.append("")

    else:
        source_texts = ["你好，请问你是谁?"] * predictor_args.batch_size
        target_texts = [""] * predictor_args.batch_size

    batch_source_texts = batchfy_text(source_texts, predictor_args.batch_size)
    batch_target_texts = batchfy_text(target_texts, predictor_args.batch_size)

    with open(model_args.output_file, "w", encoding="utf-8") as f:
        for bs, batch_source_text in enumerate(batch_source_texts):
            logger.info("Start predict")
            outputs = predictor.predict(batch_source_text)
            logger.info("End predict")

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
            outputs, batch_tokens = predictor.predict(batch_source_text, return_tokens=True)
            output_tokens += sum([len(tokens) for tokens in batch_tokens])
    end = time.perf_counter()
    print("Avg Elapse time is: ", (end - start) / test_time)
    print("Output tokens is: ", output_tokens)
    print(
        "Input length is: {}, Output length is: {}, bs is: {}, IPS: {:.3f} tokens/s, QPS: {:.3f} requests/s. ".format(
            predictor_args.src_length,
            predictor_args.max_length,
            predictor_args.batch_size,
            (output_tokens / (end - start)),
            (predictor_args.batch_size * test_time / (end - start)),
        )
    )


if __name__ == "__main__":
    predict()
