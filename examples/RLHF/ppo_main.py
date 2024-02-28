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

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import paddle
from data import PromptOnlyDataset, SupervisedDataset, parse_dataset
from models import AutoModelForScore
from ppo_trainer import PPOTrainer

from paddlenlp.trainer import PdArgumentParser, TrainingArguments, get_last_checkpoint
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)
from paddlenlp.utils.log import logger

@dataclass
class TrainingArguments(TrainingArguments):
    kl_coeff: float = field(
        default=0.02,
        metadata={"help": "The coefficient for the KL divergence between the reference and actor policy."},
    )
    clip_range_ratio: float = field(
        default=0.2,
        metadata={
            "help": "The clipping range for ratio between the old and new policy. This is the epsilon parameter in the PPO algorithm."
        },
    )
    clip_range_score: float = field(
        default=50.0,
        metadata={
            "help": "The clipping range for the output of the score model. The reward is clipped into [-clip_range_score, clip_range_score]."
        },
    )
    clip_range_value: float = field(
        default=5.0,
        metadata={
            "help": "The clipping range for the value function. The value is clipped into [value_estimate - clip_range_value, value_estimate + clip_range_value] during training."
        },
    )
    ptx_coeff: float = field(
        default=0.0,
        metadata={"help": "The coefficient for the ptx loss."},
    )
    update_iters: float = field(
        default=0.0,
        metadata={"help": "The number of repeated updates on a generated batch."},
    )
    critic_learning_rate: float = field(
        default=None,
        metadata={"help": "Initial learning rate (after the potential warmup period) for the critic model training."},
    )
    critic_weight_decay: float = field(
        default=None,
        metadata={"help": "Weight decay to for the critic model training."},
    )
    critic_lr_scheduler_type: str = field(
        default=None,
        metadata={"help": "The scheduler type for critic model."},
    )
    critic_warmup_ratio: float = field(
        default=None,
        metadata={"help": "Ratio of warm steps over total training steps for the critic lr scheduler."},
    )
    critic_recompute: bool = field(
        default=None,
        metadata={"help": "Enable gradient checkpointing for critic model."},
    )
    normalize_reward: bool = field(
        default=None,
        metadata={"help": "Whether to normalize the reward during RL training."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "The value used to module the next token probabilities."},
    )
    top_k: int = field(
        default=1,
        metadata={"help": "top_k"},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to`top_p` or higher are kept for generation."
        },
    )
    num_return_sequences: int = field(
        default=1,
        metadata={"help": "The number of independently computed returned sequences for each element in the batch."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."},
    )
    per_device_prompt_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size (per device) for the training dataloader."},
    )
    # save_generation_output: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to save generated text to file when eval"},
    # )


@dataclass
class ModelArgument:
    actor_model_name_or_path: str = field(
        default=None, metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    reward_model_name_or_path: str = field(
        default=None, metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    reward_critic_model_name_or_path: str = field(
        default=None, metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})

    # # LoRA related parameters
    # lora: bool = field(default=False, metadata={"help": "Whether to use LoRA technique"})
    # lora_path: str = field(default=None, metadata={"help": "Initialize lora state dict."})
    # lora_rank: int = field(default=8, metadata={"help": "Lora attention dimension"})

    # # prefix tuning related parameters
    # prefix_tuning: bool = field(default=False, metadata={"help": "Whether to use Prefix technique"})
    # num_prefix_tokens: int = field(default=128, metadata={"help": "Number of prefix tokens"})


@dataclass
class DataArgument:
    train_datasets: str = field(default=None, metadata={"help": "Dataset name(s) registered in the raw dataset."})
    eval_datasets: str = field(default=None, metadata={"help": "Dataset name(s) registered in the raw dataset."})
    eval_split_ratio: float = field(default=None, metadata={"help": "Ratio of eval data to train data"})
    ptx_datasets: str = field(default=None, metadata={"help": "Dataset name(s) registered in the raw dataset."})
    max_length: int = field(
        default=2048,
        metadata={
            "help": "The maximum length that model input tokens can have. When intokens is set to True, it's also the maximum length for InTokens data stream"
        },
    )

    @property
    def parsed_train_datasets(self) -> Tuple[str, Dict[str, Any]]:
        """Parse dataset path and its proportion and optionally additional arguments from `train_datasets`."""
        return [parse_dataset(string) for string in self.train_datasets.split(",")]

    @property
    def parsed_eval_datasets(self) -> Tuple[str, Dict[str, Any]]:
        """Parse dataset path and its proportion and optionally additional arguments from `eval_datasets`."""
        if self.eval_datasets is None:
            return None
        return [parse_dataset(string) for string in self.eval_datasets.split(",")]

    @property
    def parsed_ptx_datasets(self) -> Tuple[str, Dict[str, Any]]:
        """Parse dataset path and its proportion and optionally additional arguments from `ptx_datasets`."""
        if self.ptx_datasets is None:
            return None
        return [parse_dataset(string) for string in self.ptx_datasets.split(",")]


def main():
    # Arguments
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 1:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load model
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        elif training_args.bf16:
            dtype = "bfloat16"
        else:
            raise ValueError("Please specific dtype: --fp16 or --bf16")
    else:
        dtype = "float32"
    training_args.max_length = data_args.max_length

    if training_args.pipeline_parallel_degree > 1:
        raise ValueError("Not support pipeline parallel mode.")
    else:
        # actor model
        model_config = AutoConfig.from_pretrained(
            model_args.actor_model_name_or_path,
            tensor_parallel_output=False,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            tensor_parallel_rank=training_args.tensor_parallel_rank,
            dtype=dtype,
        )
        if hasattr(model_config, "use_flash_attention"):
            model_config.use_flash_attention = model_args.use_flash_attention
        actor_model = AutoModelForCausalLM.from_pretrained(
            model_args.actor_model_name_or_path,
            config=model_config,
        )
        # reference model
        actor_reference_model = AutoModelForCausalLM.from_pretrained(
            model_args.actor_model_name_or_path,
            config=model_config,
        )
        actor_tokenizer = AutoTokenizer.from_pretrained(
            model_args.actor_model_name_or_path, model_max_length=data_args.max_length, padding_side="left"
        )

        # reward model
        model_config = AutoConfig.from_pretrained(
            model_args.reward_model_name_or_path,
            tensor_parallel_output=False,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            tensor_parallel_rank=training_args.tensor_parallel_rank,
            dtype=dtype,
        )
        if hasattr(model_config, "use_flash_attention"):
            model_config.use_flash_attention = model_args.use_flash_attention
        reward_model = AutoModelForScore.from_pretrained(
            model_args.reward_model_name_or_path,
            config=model_config,
            score_type="reward",
            do_normalize=training_args.normalize_reward,
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(
            model_args.reward_model_name_or_path, model_max_length=data_args.max_length, padding_side="right"
        )
        # critic model
        if model_args.reward_critic_model_name_or_path is None:
            model_args.reward_critic_model_name_or_path = model_args.reward_model_name_or_path
        reward_critic_model = AutoModelForScore.from_pretrained(
            model_args.reward_critic_model_name_or_path, config=model_config, score_type="critic", do_normalize=False
        )
        reward_critic_tokenizer = AutoTokenizer.from_pretrained(
            model_args.reward_critic_model_name_or_path, model_max_length=data_args.max_length, padding_side="left"
        )
    for tokenizer in [actor_tokenizer, reward_tokenizer, reward_critic_tokenizer]:
        if isinstance(tokenizer, LlamaTokenizer) and tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    train_ds = PromptOnlyDataset(data_args.parsed_train_datasets, tokenizer=actor_tokenizer)
    if data_args.eval_datasets is None and data_args.eval_split_ratio:
        train_ds, dev_ds = train_ds.split_train_test(split_ratio=data_args.eval_split_ratio)
    elif data_args.eval_datasets is not None:
        dev_ds = PromptOnlyDataset(data_args.parsed_eval_datasets, tokenizer=actor_tokenizer)
    else:
        dev_ds = None

    ptx_ds = (
        SupervisedDataset(data_args.parsed_ptx_datasets, tokenizer=actor_tokenizer)
        if data_args.ptx_datasets is not None
        else None
    )

    trainer = PPOTrainer(
        model=(actor_model, actor_reference_model, reward_model, reward_critic_model),
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        ptx_dataset=ptx_ds,
        tokenizer=(actor_tokenizer, actor_tokenizer, reward_tokenizer, reward_critic_tokenizer),
        data_collator=train_ds.get_collator(),
    )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main()
