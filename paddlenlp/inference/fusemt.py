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

import time

import paddle
import paddle.incubate.multiprocessing as mp

from paddlenlp.experimental.transformers import (
    EagleProposer,
    InferenceWithReferenceProposer,
    SpeculateArgument,
)
from paddlenlp.transformers import PretrainedModel, PretrainedTokenizer
from paddlenlp.trl import llm_utils
from paddlenlp.utils.env import MAX_BSZ, MAX_DRAFT_TOKENS, SPECULATE_MAX_BSZ
from paddlenlp.utils.log import logger

from .base import BlockInferencePredictorMixin, InferencePredictorMixin
from .utils import PredictorConfig

__all__ = ["DygraphInferencePredictor", "DygraphBlockInferencePredictor"]


class DygraphInferencePredictor(InferencePredictorMixin):
    def __init__(
        self,
        config: PredictorConfig,
        tokenizer: PretrainedTokenizer = None,
        model: PretrainedModel = None,
        **kwargs,
    ):
        if model is None:
            raise ValueError("model should be provided for DygraphInferencePredictor")
        self.cache_kvs_shape = model.get_cache_kvs_shape(model.config, config.batch_size, config.total_max_length)
        InferencePredictorMixin.__init__(self, config, tokenizer, model)
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
        return self.model.generate(
            **inputs,
        )


class DygraphBlockInferencePredictor(BlockInferencePredictorMixin):
    def __init__(
        self, config: PredictorConfig, tokenizer: PretrainedTokenizer = None, model: PretrainedModel = None, **kwargs
    ):
        self.return_full_hidden_states = config.return_full_hidden_states
        self.full_hidden_states = None
        if model is None:
            raise ValueError("model should be provided for DygraphBlockInferencePredictor")
        self.cache_k_shapes, self.cache_v_shapes = model.get_cache_kvs_shape(model.config, config.batch_size)
        BlockInferencePredictorMixin.__init__(self, config, tokenizer, model)

        cachekv_dtype = self.dtype if config.cachekv_int8_type is None else "uint8"

        self.cache_kvs = []
        if self.cache_k_shapes and self.cache_v_shapes:
            for cache_k_shape, cache_v_shape in zip(self.cache_k_shapes, self.cache_v_shapes):
                self.cache_kvs.append(paddle.zeros(cache_k_shape, dtype=cachekv_dtype))
                self.cache_kvs.append(paddle.zeros(cache_v_shape, dtype=cachekv_dtype))
        else:
            # for mla's absorption
            assert self.cache_v_shapes is None
            self.cache_kvs = [paddle.zeros(shape, dtype=cachekv_dtype) for shape in self.cache_k_shapes]

        self.model = model

        self.init_model_inputs(config)
        if config.export_precache:
            self.model_inputs["pre_caches"] = self.pre_caches
        if config.cachekv_int8_type == "dynamic":
            self.model_inputs["k_quant_scales"] = self.k_quant_scales
            self.model_inputs["v_quant_scales"] = self.v_quant_scales
            self.model_inputs["k_dequant_scales"] = self.k_dequant_scales
            self.model_inputs["v_dequant_scales"] = self.v_dequant_scales

        self.model_inputs["cache_kvs"] = self.cache_kvs

        # init speculate components
        if config.speculate_method == "inference_with_reference":
            self.proposer = InferenceWithReferenceProposer(
                config.speculate_max_draft_token_num,
                config.speculate_max_ngram_size,
                config.batch_size,
                config.max_length,
            )
        elif config.speculate_method in ["eagle", "mtp"]:
            speculate_model_args = SpeculateArgument.build_from_predictor(config)
            self.proposer = EagleProposer(args=speculate_model_args)
        else:
            self.proposer = None

    @paddle.no_grad()
    def _infer(self, inputs: dict[str, paddle.Tensor]):
        return self.model.generate(
            **inputs,
        )

    @paddle.no_grad()
    def predict(self, input_texts: list[str], return_tokens=False):
        self._preprocess(input_texts)
        if self.proposer is not None:
            self.proposer.insert_query(
                base_model_inputs=self.model_inputs, real_bs=len(input_texts), seq_lens=self.seq_lens
            )
        result_queue = mp.Queue()
        tensor_queue = mp.Queue()
        done_event = mp.Event()

        # whether speculative decoding
        if self.proposer is None:
            read_res_func = llm_utils.read_res
            output_tensor_shape = [MAX_BSZ + 2, 1]
        else:
            read_res_func = llm_utils.speculate_read_res
            output_tensor_shape = [SPECULATE_MAX_BSZ * MAX_DRAFT_TOKENS + SPECULATE_MAX_BSZ + 2, 1]

        read_res_process = mp.Process(
            target=read_res_func, args=[self.model_name_or_path, tensor_queue, result_queue, done_event]
        )
        if self.tensor_parallel_rank == 0:
            read_res_process.start()

        output_tensor = paddle.full(shape=output_tensor_shape, fill_value=2, dtype="int64").cpu()

        tensor_queue.put(output_tensor)
        if self.tensor_parallel_rank == 0:
            done_event.wait()
        s_time = time.time()
        while self.model_inputs["not_need_stop"]:
            # whether speculative decoding
            if self.proposer is not None:
                self.proposer.run(
                    self.model_inputs,
                    real_batch_size=self.batch_size,
                    seq_lens_this_time=self.model_inputs["seq_lens_this_time"],
                    base_model_full_hidden_states=self.full_hidden_states,
                )
            if self.return_full_hidden_states:
                self.full_hidden_states = self._infer(self.model_inputs)
            else:
                self._infer(self.model_inputs)
        logger.info(f"running spend {time.time() - s_time}")

        if self.tensor_parallel_rank == 0:
            outputs = []
            output_tokens = []
            while len(outputs) < len(input_texts):
                result = result_queue.get(timeout=1)
                outputs.append(result[-1])
                output_tokens.append(result[-2])

            read_res_process.terminate()

            if return_tokens:
                return outputs, output_tokens
            else:
                return outputs
