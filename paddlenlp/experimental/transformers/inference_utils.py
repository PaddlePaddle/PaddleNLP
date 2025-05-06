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

from dataclasses import dataclass, field
from typing import List


@dataclass
class PredictorArgument:
    model_name_or_path: str = field(default=None, metadata={"help": "The directory of model."})
    model_prefix: str = field(default="model", metadata={"help": "the prefix name of static model"})
    src_length: int = field(default=None, metadata={"help": "The max length of source text."})
    min_length: int = field(default=1, metadata={"help": "the min length for decoding."})
    max_length: int = field(default=1024, metadata={"help": "the max length for decoding."})
    top_k: int = field(default=0, metadata={"help": "top_k parameter for generation"})
    top_p: float = field(default=0.7, metadata={"help": "top_p parameter for generation"})
    temperature: float = field(default=0.95, metadata={"help": "temperature parameter for generation"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "repetition penalty parameter for generation"})
    device: str = field(default="gpu", metadata={"help": "Device"})
    dtype: str = field(default="bfloat16", metadata={"help": "Model dtype"})
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
    dynamic_insert: bool = field(default=False, metadata={"help": "whether use dynamic insert"})

    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention"},
    )

    mode: str = field(
        default="dynamic", metadata={"help": "the type of predictor, it should be one of [dynamic, static]"}
    )
    inference_model: bool = field(default=True, metadata={"help": "whether use InferenceModel to do generation"})
    quant_type: str = field(
        default="",
        metadata={
            "help": "Quantization type. Supported values: a8w8, a8w8c8, a8w8_fp8, a8w8c8_fp8, weight_only_int4, weight_only_int8"
        },
    )
    avx_model: bool = field(
        default=False, metadata={"help": "whether use AvxModel to do generation when using cpu inference"}
    )
    avx_type: str = field(
        default=None,
        metadata={
            "help": "avx compute type. Supported values: fp16, bf16,fp16_int8\
        fp16: first_token and next_token run in fp16\
        fp16_int8 : first_token run in fp16, next token run in int8"
        },
    )
    avx_cachekv_type: str = field(
        default="fp16",
        metadata={"help": "avx cachekv type. Supported values: fp16,int8"},
    )
    batch_size: int = field(default=1, metadata={"help": "The batch size of data."})
    benchmark: bool = field(
        default=False,
        metadata={
            "help": "If benchmark set as `True`, we will force model decode to max_length, which is helpful to compute throughput. "
        },
    )
    use_fake_parameter: bool = field(default=False, metadata={"help": "use fake parameter, for ptq scales now."})
    block_attn: bool = field(default=False, metadata={"help": "whether use block attention"})
    block_size: int = field(default=64, metadata={"help": "the block size for cache_kvs."})
    cachekv_int8_type: str = field(
        default=None,
        metadata={
            "help": "If cachekv_int8_type set as `dynamic`, cache kv would be quantized to int8 dynamically. If cachekv_int8_type set as `static`, cache kv would be quantized to int8 Statically."
        },
    )

    append_attn: bool = field(default=True, metadata={"help": "whether use append attention"})

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

    total_max_length: int = field(
        default=4096, metadata={"help": "Super parameter. Maximum sequence length(encoder+decoder)."}
    )
    speculate_method: str = field(
        default=None,
        metadata={
            "help": "speculate method, it should be one of ['None', 'inference_with_reference', 'eagle', 'mtp']"
        },
    )
    speculate_max_draft_token_num: int = field(
        default=1,
        metadata={"help": "the max length of draft tokens for speculate method."},
    )
    speculate_max_ngram_size: int = field(default=1, metadata={"help": "the max ngram size of speculate method."})
    speculate_verify_window: int = field(
        default=2, metadata={"help": "the max length of verify window for speculate method."}
    )
    speculate_max_candidate_len: int = field(default=5, metadata={"help": "the max length of candidate tokens."})
    draft_model_name_or_path: str = field(default=None, metadata={"help": "The directory of eagle or draft model"})
    draft_model_quant_type: str = field(
        default="",
        metadata={"help": "Draft model quantization type. Reserved for future"},
    )
    return_full_hidden_states: bool = field(default=False, metadata={"help": "whether return full hidden_states"})

    mla_use_matrix_absorption: bool = field(default=False, metadata={"help": "implement mla with matrix-absorption."})
    weightonly_group_size: int = field(default=-1, metadata={"help": "the max length of candidate tokens."})
    weight_block_size: List[int] = field(
        default_factory=lambda: [128, 128],
        metadata={"help": "Quantitative granularity of weights. Supported values: [128 128]"},
    )
    moe_quant_type: str = field(
        default="",
        metadata={"help": "Quantization type of moe. Supported values: weight_only_int4, weight_only_int8"},
    )
    output_via_mq: bool = field(
        default=True,
        metadata={"help": "Controls whether the message queue is enabled for output"},
    )

    def __post_init__(self):
        if self.speculate_method is not None:
            self.append_attn = True
        if self.append_attn:
            self.block_attn = True
        if self.block_attn:
            self.inference_model = True
        assert self.max_length < self.total_max_length, "max_length should smaller than total_max_length."
        if self.src_length is None:
            self.src_length = self.total_max_length - self.max_length
        # update config parameter for inference predictor
        if self.decode_strategy == "greedy_search":
            self.top_p = 0.0
            self.temperature = 1.0


@dataclass
class ModelArgument:
    model_type: str = field(
        default=None,
        metadata={"help": "the type of the model, which can be one of ['gpt-3', 'ernie-3.5-se', 'llama-img2txt']"},
    )
    data_file: str = field(default=None, metadata={"help": "data file directory"})
    output_file: str = field(default="output.json", metadata={"help": "predict result file directory"})
