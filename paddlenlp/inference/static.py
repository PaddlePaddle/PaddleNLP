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

import os
import time

import numpy as np
import paddle
import paddle.incubate.multiprocessing as mp
from paddle.base.framework import in_cinn_mode, in_pir_executor_mode

try:
    from paddlenlp.experimental.transformers import (
        EagleProposer,
        InferenceWithReferenceProposer,
        SpeculateArgument,
    )
except:
    pass
from paddlenlp.taskflow.utils import static_mode_guard
from paddlenlp.transformers import PretrainedModel, PretrainedTokenizer
from paddlenlp.trl import llm_utils
from paddlenlp.utils.env import (
    MAX_BSZ,
    MAX_DRAFT_TOKENS,
    PADDLE_INFERENCE_MODEL_SUFFIX,
    PADDLE_INFERENCE_WEIGHTS_SUFFIX,
    SPECULATE_MAX_BSZ,
)
from paddlenlp.utils.import_utils import is_paddlenlp_ops_available
from paddlenlp.utils.log import logger

from .base import BasePredictor, BlockInferencePredictorMixin, InferencePredictorMixin
from .utils import PredictorArgument

__all__ = ["StaticGraphInferencePredictor", "StaticGraphBlockInferencePredictor", "StaticGraphPredictor"]


class StaticGraphPredictor(BasePredictor):
    def __init__(
        self, config: PredictorArgument, tokenizer: PretrainedTokenizer = None, model: PretrainedModel = None, **kwargs
    ):
        super().__init__(config, tokenizer, model)

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
        # remove `gpu_cpu_map_matmul_v2_to_matmul_pass` to avoid mapping matmul_v2 -> matmul op
        if config.dtype == "bfloat16":
            inference_config.delete_pass("gpu_cpu_map_matmul_v2_to_matmul_pass")
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


class StaticGraphInferencePredictor(InferencePredictorMixin):
    def __init__(
        self,
        config: PredictorArgument,
        tokenizer: PretrainedTokenizer = None,
        model: PretrainedModel = None,
        **kwargs,
    ):
        self.cache_kvs_shape = kwargs.get("cache_kvs_shape", None)
        if self.cache_kvs_shape is None:
            raise ValueError("cache_kvs_shape should be provided for StaticGraphInferencePredictor")
        InferencePredictorMixin.__init__(self, config, tokenizer, model)

        self.predictor = self._create_predictor(config)

    def _create_predictor(self, predictor_args: PredictorArgument):
        if not is_paddlenlp_ops_available():
            raise ValueError(
                "you should install the paddlenlp ops to run inference predictor, "
                "https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md"
            )

        infer_model_path = llm_utils.get_infer_model_path(
            predictor_args.model_name_or_path, predictor_args.model_prefix
        )

        config = paddle.inference.Config(
            infer_model_path + PADDLE_INFERENCE_MODEL_SUFFIX,
            infer_model_path + PADDLE_INFERENCE_WEIGHTS_SUFFIX,
        )

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
            config.enable_new_ir()
            config.disable_mkldnn()
            config.disable_glog_info()
        else:
            device_id = int(os.environ.get("FLAGS_selected_gpus", 0))
            config.enable_use_gpu(100, device_id)
        config.enable_new_executor()

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


class StaticGraphBlockInferencePredictor(BlockInferencePredictorMixin):
    def __init__(
        self,
        config: PredictorArgument,
        tokenizer: PretrainedTokenizer = None,
        model: PretrainedModel = None,
        **kwargs,
    ):
        self.cache_k_shapes = kwargs.get("cache_k_shapes", None)
        self.cache_v_shapes = kwargs.get("cache_v_shapes", None)
        self.model_args = kwargs.get("model_args", None)
        self.return_full_hidden_states = config.return_full_hidden_states
        self.full_hidden_states = None
        if self.cache_k_shapes is None:
            raise ValueError(
                "cache_k_shapes and cache_v_shapes should be provided for StaticGraphBlockInferencePredictor"
            )
        BlockInferencePredictorMixin.__init__(self, config, tokenizer)

        self._create_predictor(config)

        self.init_model_inputs(config)

        if config.export_precache:
            for i in range(self.num_layers):
                self.model_inputs["pre_caches_{}".format(i)] = self.pre_caches[i]

        cachekv_dtype = config.dtype if config.cachekv_int8_type is None else "uint8"

        for i in range(self.num_layers):
            if self.cache_k_shapes is not None:
                self.model_inputs["key_caches_{}".format(i)] = paddle.zeros(
                    self.cache_k_shapes[i], dtype=cachekv_dtype
                )
            if self.cache_v_shapes is not None:
                self.model_inputs["value_caches_{}".format(i)] = paddle.zeros(
                    self.cache_v_shapes[i], dtype=cachekv_dtype
                )

        for i in range(self.num_layers):
            if self.config.cachekv_int8_type == "dynamic":
                self.model_inputs["k_quant_scales_" + str(i)] = self.k_quant_scales[i]
                self.model_inputs["v_quant_scales_" + str(i)] = self.v_quant_scales[i]
                self.model_inputs["k_dequant_scales_" + str(i)] = self.k_dequant_scales[i]
                self.model_inputs["v_dequant_scales_" + str(i)] = self.v_dequant_scales[i]

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

    def _create_predictor(self, predictor_args: PredictorArgument):
        if not is_paddlenlp_ops_available():
            raise ValueError(
                "you should install the paddlenlp ops to run inference predictor, "
                "https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md"
            )

        infer_model_path = llm_utils.get_infer_model_path(
            predictor_args.model_name_or_path, predictor_args.model_prefix
        )

        config = paddle.inference.Config(
            infer_model_path + PADDLE_INFERENCE_MODEL_SUFFIX,
            infer_model_path + PADDLE_INFERENCE_WEIGHTS_SUFFIX,
        )

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

        if predictor_args.device == "npu":
            import paddle_custom_device.npu.passes as passes

            config.switch_ir_optim(True)
            pass_builder = config.pass_builder()
            passes.addPasses(pass_builder, self.model_config.model_type, self.model_config.quant_type)

        self.predictor = paddle.inference.create_predictor(config)

    def predict(self, input_texts: list[str], return_tokens=False):
        s_time = time.time()
        self._preprocess(input_texts)
        if self.proposer is not None:
            self.proposer.insert_query(
                base_model_inputs=self.model_inputs, real_bs=len(input_texts), seq_lens=self.seq_lens
            )
        logger.info(f"preprocess spend {time.time() - s_time}")

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
                self.full_hidden_states = self.predictor.run(list(self.model_inputs.values()))[0]
            else:
                self.predictor.run(list(self.model_inputs.values()))
        logger.info(f"running spend {time.time() - s_time}")

        if self.tensor_parallel_rank == 0:
            outputs = []
            output_tokens = []
            while len(outputs) < self.batch_size:
                result = result_queue.get(timeout=1)
                outputs.append(result[-1])
                output_tokens.append(result[-2])

            read_res_process.terminate()

            if return_tokens:
                return outputs, output_tokens
            else:
                return outputs
