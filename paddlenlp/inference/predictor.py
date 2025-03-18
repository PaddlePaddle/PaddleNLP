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
import time

import paddle
from paddle.distributed import fleet

from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import (
    AutoConfig,
    AutoInferenceModelForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    Llama3Tokenizer,
    LlamaTokenizer,
    PretrainedConfig,
    PretrainedModel,
    PretrainedTokenizer,
)
from paddlenlp.trl import llm_utils
from paddlenlp.utils.log import logger

from .base import DygraphPredictor
from .config import (
    AvxConfig,
    GenerationConfig,
    InferConfig,
    ModelConfig,
    PeftConfig,
    PredictorConfig,
    QuantConfig,
    SpeculateConfig,
)
from .fusemt import DygraphBlockInferencePredictor, DygraphInferencePredictor
from .static import (
    StaticGraphBlockInferencePredictor,
    StaticGraphInferencePredictor,
    StaticGraphPredictor,
)
from .utils import batchfy_text

PredictorMaps = {
    "DygraphPredictor": DygraphPredictor,
    "DygraphInferencePredictor": DygraphInferencePredictor,
    "DygraphBlockInferencePredictor": DygraphBlockInferencePredictor,
    "StaticGraphPredictor": StaticGraphPredictor,
    "StaticGraphInferencePredictor": StaticGraphInferencePredictor,
    "StaticGraphBlockInferencePredictor": StaticGraphBlockInferencePredictor,
}


class AutoPredictor:
    @classmethod
    def from_pretrained(self, model, tokenizer=None, **kwargs):
        self.infer_config = kwargs.pop("InferConfig", InferConfig())
        self.generation_config = kwargs.pop("GenerationConfig", GenerationConfig())
        self.sepeculate_config = kwargs.pop("SpeculateConfig", SpeculateConfig())
        self.peft_config = kwargs.pop("PeftConfig", PeftConfig())
        self.quant_config = kwargs.pop("QuantConfig", QuantConfig())
        self.avx_config = kwargs.pop("AvxConfig", AvxConfig())

        def fuse_all_config(dst_config, other_config):
            for config in other_config:
                print(config, type(config))
                for k, v in config.__dict__.items():
                    setattr(dst_config, k, v)

        fuse_all_config(
            self.infer_config,
            [self.generation_config, self.sepeculate_config, self.peft_config, self.avx_config, self.quant_config],
        )

        paddle.set_device(self.infer_config.device)
        if self.infer_config.dtype:
            paddle.set_default_dtype(self.infer_config.dtype)

        if isinstance(model, str):
            model_name_or_path = model

            from paddlenlp.utils.env import USE_FAST_TOKENIZER

            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, padding_side="left", use_fast=USE_FAST_TOKENIZER
            )

            # init chat_template for tokenizer
            llm_utils.init_chat_template(tokenizer, model_name_or_path, self.infer_config.chat_template)
            config = AutoConfig.from_pretrained(model_name_or_path)
        else:
            config = model.config

        if self.infer_config.dtype is None:
            self.infer_config.dtype = "float32"
            # raise ValueError(config.dtype)

        max_position_embeddings = llm_utils.get_model_max_position_embeddings(config)
        if max_position_embeddings is None:
            max_position_embeddings = self.infer_config.src_length + self.infer_config.max_length
            logger.warning(
                f"Can not retrieval `max_position_embeddings` from config.json, use default value {max_position_embeddings}"
            )
        else:
            if self.infer_config.src_length + self.infer_config.max_length > max_position_embeddings:
                logger.warning(
                    f"The sum of src_length<{self.infer_config.src_length}> and "
                    f"max_length<{self.infer_config.max_length}> should be smaller than or equal to "
                    f"the maximum position embedding size<{max_position_embeddings}>"
                )
                self.infer_config.src_length = max_position_embeddings - self.infer_config.max_length

        # update config parameter for inference predictor
        if self.infer_config.decode_strategy == "greedy_search":
            self.infer_config.top_p = 0.0
            self.infer_config.temperature = 1.0

        tensor_parallel_rank, tensor_parallel_degree = llm_utils.init_dist_env()

        model = None

        # model loading
        if self.infer_config.inference_model:
            model = AutoInferenceModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=config,
                predictor_args=self.infer_config,
                model_args=None,
                dtype=self.infer_config.dtype,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
            )
        else:
            if self.infer_config.mode == "dynamic":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    dtype=self.infer_config.dtype,
                    use_flash_attention=self.infer_config.use_flash_attention,
                    tensor_parallel_degree=tensor_parallel_degree,
                    tensor_parallel_rank=tensor_parallel_rank,
                    tensor_parallel_output=False,
                )

        cache_kvs_shape = None  # used for not block_attn/append_attn
        cache_k_shapes = None  # used for block_attn/append_attn
        cache_v_shapes = None  # used for block_attn/append_attn

        # static or dynamic
        execute_mode = "Dygraph" if self.infer_config.mode == "dynamic" else "StaticGraph"

        # infer/ no infer
        if self.infer_config.inference_model:
            # block/no block
            if self.infer_config.block_attn:
                attn_type = "Block"
                if self.infer_config.mode == "static":
                    cache_k_shapes, cache_v_shapes = model.get_cache_kvs_shape(
                        config, self.infer_config.batch_size, self.infer_config.total_max_length
                    )
            else:
                attn_type = ""
                if self.infer_config.mode == "static":
                    cache_kvs_shape = model.get_cache_kvs_shape(
                        config, self.infer_config.batch_size, self.infer_config.total_max_length
                    )
            inference_mode = f"{attn_type}Inference"
        else:
            inference_mode = ""

        predictor_class_name = execute_mode + inference_mode + "Predictor"

        predictor_class = PredictorMaps[predictor_class_name]
        # instance
        predictor = predictor_class(
            self.infer_config,
            tokenizer=tokenizer,
            model=model,
            cache_k_shapes=cache_k_shapes,
            cache_v_shapes=cache_v_shapes,
            cache_kvs_shape=cache_kvs_shape,
            model_args=None,
        )
        return predictor


class AutoPredictorInner:
    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path).`"
        )

    @classmethod
    def create_predictor(
        cls,
        predictor_args: PredictorConfig,
        config: PretrainedConfig,
        model_args: ModelConfig,
        tokenizer: PretrainedTokenizer = None,
        model: PretrainedModel = None,
        **kwargs,
    ):
        """
        Create a predictor

        Args:
            predictor_args (PredictorConfig): The predictor arguments.
            config (PretrainedConfig): The model configuration.
            model_args (ModelConfig): The model arguments.
            tokenizer (PretrainedTokenizer): The tokenizer.
            **kwargs: Additional keyword arguments.
        Returns:
            Predictor: The predictor.
        """
        cache_kvs_shape = None  # used for not block_attn/append_attn
        cache_k_shapes = None  # used for block_attn/append_attn
        cache_v_shapes = None  # used for block_attn/append_attn

        # static or dynamic
        execute_mode = "Dygraph" if predictor_args.mode == "dynamic" else "StaticGraph"

        # infer/ no infer
        if predictor_args.inference_model:
            # block/no block
            if predictor_args.block_attn:
                attn_type = "Block"
                if predictor_args.mode == "static":
                    cache_k_shapes, cache_v_shapes = model.get_cache_kvs_shape(
                        config, predictor_args.batch_size, predictor_args.total_max_length
                    )
            else:
                attn_type = ""
                if predictor_args.mode == "static":
                    cache_kvs_shape = model.get_cache_kvs_shape(
                        config, predictor_args.batch_size, predictor_args.total_max_length
                    )
            inference_mode = f"{attn_type}Inference"
        else:
            inference_mode = ""

        predictor_class_name = execute_mode + inference_mode + "Predictor"

        predictor_class = PredictorMaps[predictor_class_name]
        # instance
        predictor = predictor_class(
            predictor_args,
            tokenizer=tokenizer,
            model=model,
            cache_k_shapes=cache_k_shapes,
            cache_v_shapes=cache_v_shapes,
            cache_kvs_shape=cache_kvs_shape,
            model_args=model_args,
        )
        return predictor


def create_predictor(
    predictor_args: PredictorConfig,
    model_args: ModelConfig,
):

    paddle.set_device(predictor_args.device)
    paddle.set_default_dtype(predictor_args.dtype)

    from paddlenlp.utils.env import USE_FAST_TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(
        predictor_args.model_name_or_path, padding_side="left", use_fast=USE_FAST_TOKENIZER
    )

    # init chat_template for tokenizer
    llm_utils.init_chat_template(tokenizer, predictor_args.model_name_or_path, predictor_args.chat_template)

    # TODO(wj-Mcat): fix llama tokenzier pad_token bug
    if (isinstance(tokenizer, (LlamaTokenizer, Llama3Tokenizer))) and not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(predictor_args.model_name_or_path)

    max_position_embeddings = llm_utils.get_model_max_position_embeddings(config)
    if max_position_embeddings is None:
        max_position_embeddings = predictor_args.src_length + predictor_args.max_length
        logger.warning(
            f"Can not retrieval `max_position_embeddings` from config.json, use default value {max_position_embeddings}"
        )
    else:
        if predictor_args.src_length + predictor_args.max_length > max_position_embeddings:
            logger.warning(
                f"The sum of src_length<{predictor_args.src_length}> and "
                f"max_length<{predictor_args.max_length}> should be smaller than or equal to "
                f"the maximum position embedding size<{max_position_embeddings}>"
            )
            predictor_args.src_length = max_position_embeddings - predictor_args.max_length

    # update config parameter for inference predictor
    if predictor_args.decode_strategy == "greedy_search":
        predictor_args.top_p = 0.0
        predictor_args.temperature = 1.0

    tensor_parallel_rank, tensor_parallel_degree = llm_utils.init_dist_env()

    model = None

    # model loading
    if predictor_args.inference_model:
        model = AutoInferenceModelForCausalLM.from_pretrained(
            predictor_args.model_name_or_path,
            config=config,
            predictor_args=predictor_args,
            model_args=model_args,
            dtype=predictor_args.dtype,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
        )
    else:
        if predictor_args.mode == "dynamic":
            # model import (gpt-3,ernie) or AutoModel
            # if model_args.model_type == "gpt-3":
            #     sys.path.append("./gpt-3")
            #     from modeling import GPTForCausalLM

            #     model = GPTForCausalLM.from_pretrained(
            #         predictor_args.model_name_or_path,
            #         dtype=predictor_args.dtype,
            #         tensor_parallel_degree=tensor_parallel_degree,
            #         tensor_parallel_rank=tensor_parallel_rank,
            #         tensor_parallel_output=False,
            #     )
            # elif model_args.model_type == "ernie-3.5-se":
            #     sys.path.append("./ernie-3.5-se")
            #     from modeling import Ernie35ForCausalLM

            #     tensor_parallel_degree = paddle.distributed.get_world_size()
            #     tensor_parallel_rank = paddle.distributed.get_rank()
            #     model = Ernie35ForCausalLM.from_pretrained(
            #         predictor_args.model_name_or_path,
            #         dtype=predictor_args.dtype,
            #         tensor_parallel_degree=tensor_parallel_degree,
            #         tensor_parallel_rank=tensor_parallel_rank,
            #         tensor_parallel_output=False,
            #     )
            # else:
            model = AutoModelForCausalLM.from_pretrained(
                predictor_args.model_name_or_path,
                dtype=predictor_args.dtype,
                use_flash_attention=predictor_args.use_flash_attention,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
                tensor_parallel_output=False,
            )

    predictor = AutoPredictorInner.create_predictor(predictor_args, config, model_args, tokenizer, model=model)

    return predictor


def predict():
    parser = PdArgumentParser((PredictorConfig, ModelConfig))
    predictor_args, model_args = parser.parse_args_into_dataclasses()

    llm_utils.set_triton_cache(predictor_args.model_name_or_path, predictor_args.mode)

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
        source_texts = [
            "2014年3月，大范围雾霾天气长时间影响我国东部地区，严重危害人体健康。造成雾霾天气的人为原因有____\r\n①工业生产中使用矿物作为燃料，大量排放污染物     ②汽车尾气的大量排放     \r\n③风力小，空气流动不畅     ④冬季取暖排放粉尘\nA. ①②③\nB. ②③④\nC. ①③④\nD. ①②④"
        ] * predictor_args.batch_size
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

    warmup_time = 5
    test_time = 20

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
