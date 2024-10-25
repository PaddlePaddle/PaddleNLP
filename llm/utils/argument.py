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
from typing import List, Optional


@dataclass
class ModelArgument:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_layer_norm: bool = field(
        default=False,
        metadata={"help": "GPT3 model, use fast layernorm"},
    )
    fuse_attention_qkv: bool = field(
        default=None,
        metadata={"help": "whether to fuse attention qkv"},
    )
    fuse_attention_ffn: bool = field(
        default=None,
        metadata={"help": "whether to fuse first up and gate proj in mlp block"},
    )
    hidden_dropout_prob: float = field(default=0.1, metadata={"help": "The hidden dropout prob."})
    attention_probs_dropout_prob: float = field(default=0.1, metadata={"help": "The attention hidden dropout prob."})

    continue_training: bool = field(
        default=True,
        metadata={
            "help": "Whether to train from existing paddlenlp model weights. If set True, the model_name_or_path argument must exist in the paddlenlp models."
        },
    )
    weight_quantize_algo: str = field(
        default=None,
        metadata={
            "help": "Model weight quantization algorithm including 'nf4', 'fp4','weight_only_int4', 'weight_only_int8'."
        },
    )
    weight_blocksize: int = field(
        default=64,
        metadata={"help": "Block size for weight quantization(Only available for nf4 or fp4 quant_scale.)."},
    )
    weight_double_quant: bool = field(
        default=False, metadata={"help": "Whether apply double quant(Only available for nf4 or fp4 quant_scale.)."}
    )
    weight_double_quant_block_size: int = field(
        default=256,
        metadata={
            "help": "Block size for quant_scale of weight quant_scale(Only available for nf4 or fp4 quant_scale.)"
        },
    )

    # LoRA related parameters
    lora: bool = field(default=False, metadata={"help": "Whether to use LoRA technique"})
    lora_path: str = field(default=None, metadata={"help": "Initialize lora state dict."})
    lora_rank: int = field(default=8, metadata={"help": "Lora attention dimension"})
    use_quick_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use quick lora, The use of Quick LoRa will only take effect when lora_dropout is set to 0."
        },
    )
    rslora: bool = field(default=False, metadata={"help": "Whether to use RsLoRA"})
    lora_plus_scale: float = field(default=1.0, metadata={"help": "Lora B scale in LoRA+ technique"})
    pissa: bool = field(default=False, metadata={"help": "Whether to use Pissa: https://arxiv.org/pdf/2404.02948.pdf"})

    # vera related parameters
    vera: bool = field(default=False, metadata={"help": "Whether to use vera technique"})
    vera_rank: int = field(default=8, metadata={"help": "Vera attention dimension"})

    # prefix tuning related parameters
    prefix_tuning: bool = field(default=False, metadata={"help": "Whether to use Prefix technique"})
    prefix_path: str = field(default=None, metadata={"help": "Initialize prefix state dict."})
    num_prefix_tokens: int = field(default=128, metadata={"help": "Number of prefix tokens"})

    from_aistudio: bool = field(default=False, metadata={"help": "Whether to load model from aistudio"})
    save_to_aistudio: bool = field(default=False, metadata={"help": "Whether to save model to aistudio"})
    aistudio_repo_id: str = field(default=None, metadata={"help": "The id of aistudio repo"})
    aistudio_repo_private: bool = field(default=True, metadata={"help": "Whether to create a private repo"})
    aistudio_repo_license: str = field(default="Apache License 2.0", metadata={"help": "The license of aistudio repo"})
    aistudio_token: str = field(default=None, metadata={"help": "The token of aistudio"})
    neftune: bool = field(default=False, metadata={"help": "Whether to apply NEFT"})
    neftune_noise_alpha: float = field(default=5.0, metadata={"help": "NEFT noise alpha"})
    flash_mask: bool = field(default=False, metadata={"help": "Whether to use flash_mask in flash attention."})


@dataclass
class QuantArgument:

    # Quantization method config
    quant_type: str = field(
        default="a8w8",
        metadata={"help": "Quantization type. Supported values: weight_only_int8, weight_only_int4, a8w8, a8w8c8"},
    )

    fp8_type: List[str] = field(
        default_factory=lambda: ["e4m3", "e4m3"],
        metadata={"help": "Quantization type for (activation, weight)", "nargs": "+"},
    )

    skip_list_names: List[str] = field(
        default=lambda: [], metadata={"help": "Skip scales for quantization", "nargs": "+"}
    )

    weight_quant_method: str = field(
        default="abs_max_channel_wise",
        metadata={"help": "Weight quantization method, choosen from ['abs_max_channel_wise', 'groupwise']"},
    )

    act_quant_method: str = field(
        default="avg",
        metadata={"help": "Activation quantization method, choosen from ['abs_max', 'avg']"},
    )

    cachekv_quant_method: str = field(
        default="avg_headwise",
        metadata={"help": "KV quantization method, choosen from ['abs_max_headwise', 'avg_headwise']"},
    )

    # Piecewise Search Smooth related parameters
    search_alpha_min: float = field(
        default=0.2,
        metadata={"help": "The minimum alpha for piece search"},
    )

    search_alpha_max: float = field(
        default=0.8,
        metadata={"help": "The maximum alpha for piece search"},
    )

    search_scale_min: float = field(
        default=1.0,
        metadata={"help": "The minimum scale for piece search"},
    )

    search_scale_max: float = field(
        default=5.0,
        metadata={"help": "The maximum scale for piece search"},
    )

    # QAT related parameters
    # Not Yet support
    do_qat: bool = field(default=False, metadata={"help": "Whether to use QAT technique"})

    # PTQ related parameters
    do_ptq: bool = field(default=False, metadata={"help": "Whether to use PTQ"})
    ptq_step: int = field(default=32, metadata={"help": "Step for PTQ"})

    # Pre-quant method Shift related parameters
    shift: bool = field(default=False, metadata={"help": "Whether to use Shift"})
    shift_all_linears: bool = field(default=False, metadata={"help": "Whether to shift all linears"})
    shift_sampler: str = field(
        default="ema", metadata={"help": "The name of shift sampler, choosen from ['ema', 'none']"}
    )
    shift_step: int = field(default=32, metadata={"help": "Sample steps when shift"})

    # Pre-quant methos Smooth related parameters
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

    # AWQ related parameters, default for WINT4
    do_awq: bool = field(default=False, metadata={"help": "Whether to use AWQ Search"})
    auto_clip: bool = field(default=False, metadata={"help": "Whether to use AutoClip from AWQ"})
    awq_step: int = field(default=8, metadata={"help": "Step for AWQ Search"})
    autoclip_step: int = field(default=8, metadata={"help": "Step for AutoClip"})

    # Other config
    load_quant_model: bool = field(default=False, metadata={"help": "Whether to load quant model"})

    do_quant_debug: bool = field(default=False, metadata={"help": "Whether to use debug"})
    test_sample: Optional[str] = field(default=None, metadata={"help": "Test sample for quantization"})


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
