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
from dataclasses import dataclass, field

from paddlenlp.trainer import TrainingArguments


@dataclass
class TrainingArguments(TrainingArguments):
    benchmark: bool = field(default=False, metadata={"help": "Whether runs benchmark"})


@dataclass
class DataArgument:
    dataset_name_or_path: str = field(default=None, metadata={"help": "Name or path for dataset"})
    task_name: str = field(default=None, metadata={"help": "Additional name to select a more specific task."})
    intokens: bool = field(default=False, metadata={"help": "Whether to use InTokens data stream"})
    src_length: int = field(default=1024, metadata={"help": "The maximum length of source(context) tokens."})
    max_length: int = field(
        default=2048,
        metadata={
            "help": "The maximum length that model input tokens can have. When intokens is set to True, it's also the maximum length for InTokens data stream"
        },
    )
    eval_with_do_generation: bool = field(default=False, metadata={"help": "Whether to do generation for evaluation"})
    save_generation_output: bool = field(
        default=False,
        metadata={"help": "Whether to save generated text to file when eval_with_do_generation set to True."},
    )
    lazy: bool = field(
        default=False,
        metadata={
            "help": "Weather to return `MapDataset` or an `IterDataset`.True for `IterDataset`. False for `MapDataset`."
        },
    )


@dataclass
class ModelArgument:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})

    # LoRA related parameters
    lora: bool = field(default=False, metadata={"help": "Whether to use LoRA technique"})
    lora_path: str = field(default=None, metadata={"help": "Initialize lora state dict."})
    lora_rank: int = field(default=8, metadata={"help": "Lora attention dimension"})

    # prefix tuning related parameters
    prefix_tuning: bool = field(default=False, metadata={"help": "Whether to use Prefix technique"})
    num_prefix_tokens: int = field(default=128, metadata={"help": "Number of prefix tokens"})


@dataclass
class QuantArgument:
    quant_type: str = field(
        default="A8W8", metadata={"help": "Quantization type. Supported values: A8W8, WINT4,WINT8"}
    )

    # QAT related parameters
    # Not Yet support
    do_qat: bool = field(default=False, metadata={"help": "Whether to use QAT technique"})

    # PTQ related parameters
    do_ptq: bool = field(default=False, metadata={"help": "Whether to use PTQ"})
    ptq_step: int = field(default=32, metadata={"help": "Step for PTQ"})

    shift: bool = field(default=False, metadata={"help": "Whether to use Shift"})
    shift_all_linears: bool = field(default=False, metadata={"help": "Whether to shift all linears"})
    shift_sampler: str = field(
        default="ema", metadata={"help": "The name of shift sampler, choosen from ['ema', 'none']"}
    )
    shift_step: int = field(default=32, metadata={"help": "Sample steps when shift"})

    smooth: bool = field(default=False, metadata={"help": "Whether to use Smooth"})
    smooth_all_linears: bool = field(default=False, metadata={"help": "Whether to smooth all linears"})
    smooth_sampler: str = field(
        default="none", metadata={"help": "The name of smooth sampler, choosen from ['multi_step','none']"}
    )
    smooth_step: int = field(default=32, metadata={"help": "Sample steps when smooth"})
    smooth_piecewise_search: bool = field(
        default=False, metadata={"help": "The number of piece in piecewise search for smooth strategy."}
    )
    smooth_k_piece: int = field(default=3, metadata={"help": "Number of pieces for K-search"})
    smooth_search_piece: bool = field(default=False, metadata={"help": "Whether search k_piece when piecewise search"})

    # GPTQ related parameters
    do_gptq: bool = field(default=False, metadata={"help": "Whether to use GPTQ"})
    gptq_step: int = field(default=8, metadata={"help": "Step for GPTQ"})


@dataclass
class GenerateArgument:
    top_k: int = field(
        default=1,
        metadata={
            "help": "The number of highest probability tokens to keep for top-k-filtering in the sampling strategy"
        },
    )
    top_p: float = field(
        default=1.0, metadata={"help": "The cumulative probability for top-p-filtering in the sampling strategy."}
    )
