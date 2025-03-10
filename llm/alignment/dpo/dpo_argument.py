# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Optional

from paddlenlp.trainer import TrainingArguments
from paddlenlp.trainer.trainer_utils import IntervalStrategy
from paddlenlp.trainer.utils.doc import add_start_docstrings
from paddlenlp.transformers.configuration_utils import llmmetaclass


@dataclass
@llmmetaclass
@add_start_docstrings(TrainingArguments.__doc__)
class DPOTrainingArguments(TrainingArguments):
    """DPOTrainingArguments"""

    unified_checkpoint: bool = field(
        default=True,
        metadata={"help": "Enable fused linear grad add strategy."},
    )
    unified_checkpoint_config: Optional[str] = field(
        default="",
        metadata={"help": "Configs to unify hybrid parallel checkpoint.\n"},
    )
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to run benchmark by autotuner. True for from_scratch."},
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to run benchmark by autotuner. True for from_scratch."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.autotuner_benchmark:
            self.num_train_epochs = 1
            self.max_steps = 5
            self.do_train = True
            self.do_export = False
            self.do_predict = False
            self.do_eval = False
            self.overwrite_output_dir = True
            self.load_best_model_at_end = False
            self.report_to = []
            self.save_strategy = IntervalStrategy.NO
            self.evaluation_strategy = IntervalStrategy.NO
            if not self.disable_tqdm:
                self.logging_steps = 1
                self.logging_strategy = IntervalStrategy.STEPS
        if self.benchmark:
            self.do_train = True
            self.do_export = False
            self.do_predict = False
            self.do_eval = False
            self.overwrite_output_dir = True
            self.load_best_model_at_end = False
            self.save_strategy = IntervalStrategy.NO
            self.evaluation_strategy = IntervalStrategy.NO
            if not self.disable_tqdm:
                self.logging_steps = 1
                self.logging_strategy = IntervalStrategy.STEPS
        if self.max_steps > 0:
            self.num_train_epochs = 1


@dataclass
class DPOConfig:
    """DPOConfig"""

    beta: float = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    simpo_gamma: float = field(default=0.5, metadata={"help": "the gamma parameter for SimPO loss"})
    label_smoothing: float = field(default=0.0, metadata={"help": "label_smoothing ratio"})
    loss_type: str = field(default="sigmoid", metadata={"help": "DPO loss type"})
    pref_loss_ratio: float = field(default=1.0, metadata={"help": "DPO loss ratio"})
    sft_loss_ratio: float = field(default=0.0, metadata={"help": "SFT loss ratio"})
    dpop_lambda: float = field(default=50, metadata={"help": "dpop_lambda"})
    ref_model_update_steps: int = field(default=-1, metadata={"help": "Update ref model state dict "})
    reference_free: bool = field(default=False, metadata={"help": "No reference model."})
    lora: bool = field(default=False, metadata={"help": "Use LoRA model."})


@dataclass
class DPODataArgument:
    """DataArgument"""

    train_dataset_path: str = field(default="./data/train.jsonl", metadata={"help": "Path to the train dataset dir."})
    dev_dataset_path: str = field(default="./data/dev.jsonl", metadata={"help": "Path to the dev dataset dir."})
    max_seq_len: int = field(default=4096, metadata={"help": "Maximum sequence length."})
    max_prompt_len: int = field(default=2048, metadata={"help": "Maximum prompt length."})
    greedy_zero_padding: bool = field(
        default=False,
        metadata={"help": "Whether to use Greedy Zero Padding data stream."},
    )
    lazy: bool = field(
        default=False,
        metadata={
            "help": "Weather to return `MapDataset` or an `IterDataset`.True for `IterDataset`. False for `MapDataset`."
        },
    )


@dataclass
class DPOModelArgument:
    """ModelArgument"""

    model_name_or_path: str = field(
        default=None, metadata={"help": "Pretrained model name or path to local directory."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    flash_mask: bool = field(default=False, metadata={"help": "Whether to use flash mask in flash attention."})
    weight_quantize_algo: str = field(
        default=None,
        metadata={"help": "Model weight quantization algorithm including 'nf4'(qlora), 'weight_only_int8'."},
    )
    fuse_attention_qkv: bool = field(
        default=None,
        metadata={"help": "whether to fuse attention qkv"},
    )
    fuse_attention_ffn: bool = field(
        default=None,
        metadata={"help": "whether to fuse first up and gate proj in mlp block"},
    )
    # LoRA
    lora_rank: int = field(default=8, metadata={"help": "Lora rank."})
    lora_path: str = field(default=None, metadata={"help": "Initialize lora state dict."})
    rslora: bool = field(default=False, metadata={"help": "Whether to use RsLoRA"})
    lora_plus_scale: float = field(default=1.0, metadata={"help": "Lora B scale in LoRA+ technique"})
    lora_alpha: int = field(default=-1, metadata={"help": "lora_alpha"})
    rslora_plus: bool = field(default=False, metadata={"help": "Strengthen lora performance"})
    use_quick_lora: bool = field(default=True, metadata={"help": "quick lora"})
