"""DPO utils"""
from dataclasses import dataclass, field
from typing import Optional

from paddlenlp.trainer import TrainingArguments


def add_start_docstrings(*docstr):
    """Adds docstrings for a function."""

    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


@dataclass
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
    dpo_beta: float = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    dpo_normalize_logps: bool = field(
        default=True,
        metadata={"help": "Apply logprobs normalization."},
    )
    zero_padding: bool = field(
        default=True,
        metadata={"help": "Apply zero-padding."},
    )
    offload_strategy: bool = field(
        default=False,
        metadata={"help": "Apply pinned memory offload strategy."},
    )
    num_of_gpus: int = field(default=-1, metadata={"help": "Number of gpus."})
    pipeline_degree: int = field(default=1, metadata={"help": "pipeline_degree for estimate"})
    tensor_degree: int = field(default=1, metadata={"help": "tensor_degree for estimate"})
    sharding_degree: int = field(default=1, metadata={"help": "sharding_degree for estimate"})


@dataclass
class DataArgument:
    """DataArgument"""

    dataset_name_or_path: str = field(
        default="./data/", metadata={"help": "Path to the dataset dir."}
    )
    eval_task_config: str = field(default=None, metadata={"help": "Path to the evaluation task config."})
    max_seq_length: int = field(default=4096, metadata={"help": "Maximum sequence length."})
    max_prompt_len: int = field(default=2048, metadata={"help": "Maximum prompt length."})
    num_samples_each_epoch: int = field(
        default=100000, metadata={"help": "Number of samples per epoch. Used for SFT."}
    )
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to run benchmark by autotuner. True for from_scratch."},
    )
    dpo_benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to run benchmark by autotuner. True for from_scratch."},
    )
    greedy_intokens: bool = field(
        default=True,
        metadata={"help": "Whether apply greedy intokens."},
    )
    buffer_size: int = field(
        default=500, 
        metadata={"help": "Buffer size for greedy_intokens strategy."})


@dataclass
class ModelArgument:
    """ModelArgument"""

    model_name_or_path: str = field(
        default="ernie-bot-10b", metadata={"help": "Pretrained model name or path to local directory."}
    )
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})
    recompute_granularity: str = field(
        default="full",
        metadata={
            "help": "The granularity of recompute training can be selected as `full` or `full_attn` or `core_attn`."
        },
    )
    stage: str = field(default="DPO", metadata={"help": "The type of training."})
    tensor_parallel_output: bool = field(default=True, metadata={"help": "tensor_parallel_output"})