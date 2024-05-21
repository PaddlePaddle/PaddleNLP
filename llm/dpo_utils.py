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
    dpo_beta: float = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})


@dataclass
class DataArgument:
    """DataArgument"""

    dataset_name_or_path: str = field(
        default="./data/", metadata={"help": "Path to the dataset dir."}
    )
    max_seq_length: int = field(default=4096, metadata={"help": "Maximum sequence length."})
    max_prompt_len: int = field(default=2048, metadata={"help": "Maximum prompt length."})
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to run benchmark by autotuner. True for from_scratch."},
    )
    dpo_benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to run benchmark by autotuner. True for from_scratch."},
    )


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