# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


from abc import abstractmethod
from threading import Thread

import numpy as np
import paddle

from paddlenlp.generation import GenerationConfig, TextIteratorStreamer
from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    ChatGLMTokenizer,
    ChatGLMv2Tokenizer,
    PretrainedModel,
    PretrainedTokenizer,
)
from paddlenlp.trl import llm_utils
from paddlenlp.utils.log import logger

from .utils import PredictorConfig


class BasePredictor:
    def __init__(self, config: PredictorConfig, tokenizer: PretrainedTokenizer = None, model: PretrainedModel = None):
        if model is not None and hasattr(model, "config"):
            self.model_config = model.config
        else:
            self.model_config = AutoConfig.from_pretrained(config.model_name_or_path)

        self.config: PredictorConfig = config
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, padding_side="left")

        self.tokenizer = tokenizer

        self.return_tensors = "pd"
        self.tensor_parallel_rank, self.tensor_parallel_degree = llm_utils.init_dist_env()
        self.model_config.tensor_parallel_rank, self.model_config.tensor_parallel_degree = (
            self.tensor_parallel_rank,
            self.tensor_parallel_degree,
        )

        try:
            self.generation_config = GenerationConfig.from_pretrained(config.model_name_or_path)
        except:
            logger.warning(
                "Can't find generation config, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

    def _preprocess(self, source):

        if self.tokenizer.chat_template is not None:
            # for str -> List[str] eg. "hello"
            # for List[str] -> List[str]  eg. ["hello", "hello new"]
            # for List[List[str]] -> List[List[List[str]]]  eg. 历史对话形式,一轮
            #             [ [ "Hello, how are you?", "I'm doing great. How can I help you today?"],
            #                ["I'd like to show off how chat templating works!"], ]
            # for List[Dict] -> List[List[Dict]]  [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'nice'}]
            #                                 ->  [[{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'nice'}]]
            if not isinstance(source, list) or not isinstance(source[0], str):
                source = [source]
            source = [self.tokenizer.apply_chat_template(sentence, tokenize=False) for sentence in source]

        tokenized_source = self.tokenizer(
            source,
            max_length=self.config.src_length,
            truncation=True,
            return_position_ids=True if not isinstance(self.tokenizer, ChatGLMTokenizer) else False,
            return_attention_mask=True,
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
        self, config: PredictorConfig, tokenizer: PretrainedTokenizer = None, model: PretrainedModel = None, **kwargs
    ):
        super().__init__(config, tokenizer, model)
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
            prefix_tuning_params = llm_utils.get_prefix_tuning_params(self.model)
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
            eos_token_id=llm_utils.get_eos_token_id(self.tokenizer, self.generation_config),
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
            eos_token_id=llm_utils.get_eos_token_id(self.tokenizer, self.generation_config),
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


class InferencePredictorMixin(BasePredictor):
    def __init__(self, config: PredictorConfig, tokenizer: PretrainedTokenizer, model: PretrainedModel = None):
        BasePredictor.__init__(self, config, tokenizer, model)
        self.architectures = self.model_config.architectures[0].lower()

        self.dtype = config.dtype or self.model_config.dtype
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
            self.cache_kvs = [paddle.zeros(shape, dtype=self.dtype) for shape in self.cache_kvs_shape]
            self.num_layers, self.num_key_value_heads, self.head_dim = (
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
                            self.num_key_value_heads,
                            prefix_cache.shape[-2],
                            self.head_dim,
                        ],
                    )
                    self.pre_caches = [
                        item.squeeze_(0) for item in paddle.split(prefix_cache, self.num_layers, axis=0)
                    ]
                else:
                    prefix_cache = paddle.zeros(
                        [self.num_layers, 2, config.batch_size, self.num_key_value_heads, 128, self.head_dim],
                        dtype=self.dtype,
                    )
                    self.pre_caches = [
                        item.squeeze_(0) for item in paddle.split(prefix_cache, self.num_layers, axis=0)
                    ]

    def _postprocess(self, predictions, return_tokens=False):
        if paddle.distributed.get_rank() == 0:
            tokens: np.ndarray = llm_utils.load_real_time_tokens()
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
            if not isinstance(source, list) or not isinstance(source[0], str):
                source = [source]
            source = [self.tokenizer.apply_chat_template(sentence, tokenize=False) for sentence in source]

        inputs = llm_utils.dybatch_preprocess(
            self.tokenizer,
            source,
            self.config.src_length,
            self.config.max_length,
            self.architectures,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            eos_token_id=llm_utils.get_eos_token_id(self.tokenizer, self.generation_config),
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
            alibi_slopes = llm_utils.get_alibi_slopes(self.model_config.n_head)
            inputs["position_ids"] = paddle.to_tensor(alibi_slopes, dtype="float32")
            arange_tensor_encoder = paddle.arange(self.config.total_max_length, dtype=self.config.dtype)
            alibi = (alibi_slopes[None, :, None, None] * arange_tensor_encoder).astype(self.config.dtype)

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


class BlockInferencePredictorMixin(BasePredictor):
    def __init__(
        self,
        config: PredictorConfig,
        tokenizer: PretrainedTokenizer = None,
        model: PretrainedModel = None,
    ):
        BasePredictor.__init__(self, config, tokenizer, model)

        self.num_layers = len(self.cache_k_shapes)
        self.num_key_value_heads = self.cache_k_shapes[0][-3]
        self.head_dim = self.cache_k_shapes[0][-1]
        self.max_block_nums = self.cache_k_shapes[0][0]
        self.batch_size = config.batch_size
        self.model_name_or_path = config.model_name_or_path

        self.architectures = self.model_config.architectures[0].lower()

        self.dtype = config.dtype or self.model_config.dtype

        self.rope_theta = self.model_config.get("rope_theta", 10000.0)
        self.rope_scaling = self.model_config.get("rope_scaling", None)

        self.pre_cache_length = 0

        if config.export_precache:
            pre_cache_npy = np.load(config.prefix_path)
            self.pre_cache_length = pre_cache_npy.shape[-2]
            config.max_length -= self.pre_cache_length
            self.pre_caches = [
                paddle.zeros(
                    [config.batch_size, self.num_key_value_heads, self.pre_cache_length, self.head_dim],
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

        if config.cachekv_int8_type == "dynamic":
            self.k_quant_scales = [
                paddle.zeros([config.batch_size, self.num_key_value_heads], dtype="float32")
                for _ in range(self.num_layers)
            ]
            self.v_quant_scales = [
                paddle.zeros([config.batch_size, self.num_key_value_heads], dtype="float32")
                for _ in range(self.num_layers)
            ]
            self.k_dequant_scales = [
                paddle.zeros([config.batch_size, self.num_key_value_heads], dtype="float32")
                for _ in range(self.num_layers)
            ]
            self.v_dequant_scales = [
                paddle.zeros([config.batch_size, self.num_key_value_heads], dtype="float32")
                for _ in range(self.num_layers)
            ]

    def pad_batch_data(self, insts):
        """Pad the instances to the max sequence length in batch."""
        seq_lens = []
        for i, inst in enumerate(insts):
            length = len(inst)
            seq_lens.append(length)
            self.input_ids[i, :length] = np.array(inst)
        return seq_lens

    def init_model_inputs(self, config: PredictorConfig):
        self.input_ids = paddle.full(
            shape=[config.batch_size, config.total_max_length], fill_value=self.tokenizer.pad_token_id, dtype="int64"
        )
        self.model_inputs = {}

        if config.export_precache:
            self.model_inputs["src_mask"] = (self.pre_cache_mask - 1) * 1e4

        self.model_inputs["block_tables"] = paddle.full(
            shape=[config.batch_size, (config.total_max_length + config.block_size - 1) // config.block_size],
            fill_value=-1,
            dtype="int32",
        )
        self.model_inputs["top_p"] = paddle.full(
            shape=[config.batch_size, 1], fill_value=config.top_p, dtype="float32"
        )
        self.model_inputs["temperature"] = paddle.full(
            shape=[config.batch_size, 1], fill_value=config.temperature, dtype="float32"
        )
        self.model_inputs["eos_token_id"] = paddle.to_tensor(
            np.array(llm_utils.get_eos_token_id(self.tokenizer, self.generation_config)).reshape(-1, 1).astype("int64")
        )
        self.model_inputs["penalty_score"] = paddle.full(
            shape=[config.batch_size, 1], fill_value=config.repetition_penalty, dtype="float32"
        )
        self.model_inputs["frequency_score"] = paddle.full(
            shape=[config.batch_size, 1], fill_value=0.0, dtype="float32"
        )
        self.model_inputs["presence_score"] = paddle.full(
            shape=[config.batch_size, 1], fill_value=0.0, dtype="float32"
        )
        self.model_inputs["min_length"] = paddle.full(
            shape=[config.batch_size, 1], fill_value=config.min_length, dtype="int64"
        )
        self.model_inputs["max_length"] = paddle.full(
            shape=[config.batch_size, 1], fill_value=config.max_length, dtype="int64"
        )
        self.model_inputs["rope_emb"] = llm_utils.get_rotary_position_embedding(
            paddle.arange(config.total_max_length).reshape((1, -1)), self.head_dim, self.rope_theta, self.rope_scaling
        )
        self.model_inputs["bad_tokens"] = paddle.to_tensor([-1], dtype="int64")
        self.model_inputs["is_block_step"] = paddle.full(shape=[config.batch_size], fill_value=False, dtype="bool")

        # bloom model needs src_mask and tgt_mask!
        if "bloom" in self.architectures:
            lower_one_tril = paddle.tril(
                paddle.ones(shape=(config.total_max_length, config.total_max_length), dtype=self.dtype)
            )
            lower_one_tril = lower_one_tril[None, None, :, :]
            self.model_inputs["src_mask"] = lower_one_tril.tile([config.batch_size, 1, 1, 1])
            self.model_inputs["tgt_mask"] = paddle.full(
                shape=[config.batch_size, 1, 1, config.total_max_length], fill_value=1, dtype=self.dtype
            )
            arange_tensor_encoder = paddle.arange(config.total_max_length).astype(self.dtype)
            alibi_slopes = llm_utils.get_alibi_slopes(self.num_key_value_heads)
            alibi = alibi_slopes[None, :, None, None] * arange_tensor_encoder
            alibi_encoder = alibi.tile([config.batch_size, 1, config.total_max_length, 1])
            alibi_decoder = alibi.tile(
                [
                    config.batch_size,
                    1,
                    1,
                    1,
                ]
            )
            # self.model_inputs["src_mask/tgt_mask"] is read only, will not be updated!
            self.model_inputs["src_mask"] = (
                alibi_encoder + (1 - self.model_inputs["src_mask"]) * paddle.finfo(self.dtype).min
            ).cast(self.dtype)
            self.model_inputs["tgt_mask"] = (
                alibi_decoder + (1 - self.model_inputs["tgt_mask"]) * paddle.finfo(self.dtype).min
            ).cast(self.dtype)
        elif config.device == "npu" and self.model_config.get("alibi", False):
            lower_one_tril = paddle.tril(
                paddle.ones(shape=(config.total_max_length, config.total_max_length), dtype=self.dtype)
            )
            lower_one_tril = lower_one_tril[None, None, :, :]
            src_mask = lower_one_tril.tile([config.batch_size, 1, 1, 1])
            tgt_mask = paddle.full(
                shape=[config.batch_size, 1, 1, config.total_max_length], fill_value=1, dtype=self.dtype
            )
            arange_tensor_encoder = paddle.arange(config.total_max_length).astype(self.dtype)
            alibi_slopes = llm_utils.get_alibi_slopes(self.num_key_value_heads)
            alibi = alibi_slopes[None, :, None, None] * arange_tensor_encoder
            alibi_encoder = alibi.tile([config.batch_size, 1, config.total_max_length, 1])
            alibi_decoder = alibi.tile(
                [
                    config.batch_size,
                    1,
                    1,
                    1,
                ]
            )
            # self.model_inputs["src_mask/tgt_mask"] is read only, will not be updated!
            src_mask = (alibi_encoder + (1 - src_mask) * paddle.finfo(self.dtype).min).cast(self.dtype)
            tgt_mask = (alibi_decoder + (1 - tgt_mask) * paddle.finfo(self.dtype).min).cast(self.dtype)
            self.model_inputs["rope_emb"] = paddle.concat([src_mask.reshape([-1]), tgt_mask.reshape([-1])])

    def _preprocess(self, input_text: list[str]):
        len_input_text = len(input_text)
        if len_input_text < self.batch_size:
            padding_len = self.batch_size - len_input_text
            input_text += [""] * padding_len
            assert len(input_text) == self.batch_size

        if self.tokenizer.chat_template is not None:
            if not isinstance(input_text, list) or not isinstance(input_text[0], str):
                input_text = [input_text]
            input_text = [self.tokenizer.apply_chat_template(sentence, tokenize=False) for sentence in input_text]

        input_ids = []
        for text in input_text:
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
            input_ids.append(tokens["input_ids"][0])

        self.seq_lens = self.pad_batch_data(input_ids)
        self.model_inputs["input_ids"] = self.input_ids

        self.model_inputs["block_tables"][:][:] = -1
        free_list = list(range(self.max_block_nums))
        for i in range(self.config.batch_size):
            for j in range(
                (self.seq_lens[i] + self.config.max_length + self.config.block_size - 1) // self.config.block_size
            ):
                used_block_id = free_list.pop()
                self.model_inputs["block_tables"][i, j] = used_block_id

        self.model_inputs["seq_lens_this_time"] = paddle.to_tensor(
            np.array(self.seq_lens).astype("int32").reshape(-1, 1)
        )
        self.model_inputs["seq_lens_encoder"] = paddle.to_tensor(
            np.array(self.seq_lens).astype("int32").reshape(-1, 1)
        )
        self.model_inputs["seq_lens_decoder"] = paddle.full(
            shape=[self.config.batch_size, 1], fill_value=0, dtype="int32"
        )
        self.model_inputs["step_idx"] = paddle.full(shape=[self.config.batch_size, 1], fill_value=0, dtype="int64")
        self.model_inputs["not_need_stop"] = paddle.full(shape=[1], fill_value=True, dtype="bool").cpu()  # cpu
        self.model_inputs["stop_flags"] = paddle.full(
            shape=[self.config.batch_size, 1], fill_value=False, dtype="bool"
        )
        self.model_inputs["stop_nums"] = paddle.full(shape=[1], fill_value=self.config.batch_size, dtype="int64")
        self.model_inputs["pre_ids"] = paddle.full(
            shape=[self.config.batch_size, self.config.max_length], fill_value=-1, dtype="int64"
        )
        self.model_inputs["next_tokens"] = paddle.full(shape=[self.config.batch_size, 1], fill_value=-1, dtype="int64")

        # speculative decoding related parameters
        if self.config.speculate_method is not None:
            self.model_inputs["accept_tokens"] = paddle.full(
                shape=[self.config.batch_size, self.config.speculate_max_draft_token_num + 1],
                fill_value=0,
                dtype="int64",
            )
            self.model_inputs["accept_num"] = paddle.full(shape=[self.config.batch_size], fill_value=0, dtype="int32")
            self.model_inputs["draft_tokens"] = paddle.full(
                shape=[self.config.batch_size, self.config.speculate_max_draft_token_num + 1],
                fill_value=0,
                dtype="int64",
            )
            self.model_inputs["actual_draft_token_num"] = paddle.full(
                shape=[self.config.batch_size], fill_value=self.config.speculate_max_draft_token_num, dtype="int32"
            )

            self.proposer.input_ids_cpu = self.model_inputs["input_ids"].to("cpu", blocking=False)
            for bid in range(self.config.batch_size):
                self.model_inputs["pre_ids"][bid, 0] = self.model_inputs["input_ids"][bid][
                    self.seq_lens[bid] - 1
                ]  # get the last token before padding of this batch
                if self.config.speculate_method == "inference_with_reference":
                    self.proposer.input_ids_len[bid, 0] = self.seq_lens[bid]

        if self.config.mode == "static":
            for k, v in self.model_inputs.items():
                v.name = k
