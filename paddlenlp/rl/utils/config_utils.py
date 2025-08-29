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

from dataclasses import dataclass, field

import paddle

from ...trainer.trainer import ShardingOption, TrainingArguments, logger
from ...trainer.trainer_utils import IntervalStrategy
from ...transformers.configuration_utils import llmmetaclass


@dataclass
@llmmetaclass
class TrainingArguments(TrainingArguments):
    global_batch_size: int = field(
        default=8,
        metadata={"help": "Global batch size for input prompt."},
    )
    global_gen_batch_size: int = field(
        default=-1,
        metadata={"help": "Global generation batch size for dynamic sampling."},
    )
    global_mini_batch_size: int = field(
        default=-1,
        metadata={"help": "Mini-batch size (global) for the training dataloader."},
    )
    per_device_rollout_batch_size: int = field(
        default=-1,
        metadata={"help": "Batch size (per device) for the training dataloader."},
    )
    per_device_logprob_batch_size: int = field(
        default=-1,
        metadata={"help": "Batch size (per device) for the training dataloader."},
    )
    per_device_reward_batch_size: int = field(
        default=-1,
        metadata={"help": "Batch size (per device) for the training dataloader."},
    )
    per_device_value_batch_size: int = field(
        default=-1,
        metadata={"help": "Batch size (per device) for the training dataloader."},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size (per device) for the training dataloader."},
    )
    kl_coeff: float = field(
        default=0.02,
        metadata={"help": "The coefficient for the KL divergence between the reference and actor policy."},
    )
    kl_loss_coeff: float = field(
        default=0.001,
        metadata={"help": "The coefficient for the KL loss for GRPO."},
    )
    pg_loss_coeff: float = field(
        default=1.0,
        metadata={"help": "The coefficient for the PG loss for GRPO."},
    )
    entropy_coeff: float = field(
        default=0.0,
        metadata={"help": "The coefficient for the entropy loss for GRPO."},
    )
    clip_range_ratio: float = field(
        default=0.2,
        metadata={
            "help": "The clipping range for ratio between the old and new policy. "
            "This is the epsilon parameter in the PPO algorithm."
        },
    )
    clip_range_ratio_low: float = field(
        default=None,
        metadata={
            "help": "The clipping range for ratio between the old and new policy. "
            "This is the epsilon parameter in the PPO algorithm."
        },
    )
    clip_range_ratio_high: float = field(
        default=None,
        metadata={
            "help": "The clipping range for ratio between the old and new policy. "
            "This is the epsilon parameter in the PPO algorithm."
        },
    )
    clip_range_score: float = field(
        default=10.0,
        metadata={
            "help": "The clipping range for the output of the score model. "
            "The reward is clipped into [-clip_range_score, clip_range_score]."
        },
    )
    enable_overlong_reward_buffer: bool = field(
        default=False,
        metadata={},
    )
    overlong_reward_buffer: int = field(
        default=256,
        metadata={"help": "The allowed buffer before applying penalty."},
    )
    overlong_penalty_factor: float = field(
        default=1.0,
        metadata={
            "help": "The penalty factor for the overlong reward buffer. "
            "The penalty is deleted to the reward when the buffer is full."
        },
    )
    clip_range_value: float = field(
        default=5.0,
        metadata={
            "help": "The clipping range for the value function. The value is clipped into [value_estimate - "
            "clip_range_value, value_estimate + clip_range_value] during training."
        },
    )
    update_iters: int = field(
        default=1,
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
    normalize_advantage: bool = field(
        default=None,
        metadata={"help": "Whether to normalize the advantage during RL training."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "The value used to module the next token probabilities."},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "If set to float < 1, only the smallest set of most probable tokens "
            "with probabilities that add up to`top_p` or higher are kept for generation."
        },
    )
    rollout_n: int = field(
        default=1,
        metadata={"help": "The number of independently computed returned sequences for each element in the batch."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."},
    )
    rollout_quant_type: str = field(
        default="",
        metadata={"help": "Quantization dtype, optional for: weight_onlt_int8."},
    )
    per_device_prompt_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size (per device) for the training dataloader."},
    )
    dynamic_sampling: bool = field(
        default=False,
        metadata={"help": "whether enable dynamic sample https://arxiv.org/abs/2503.14476"},
    )
    max_gen_batches: int = field(
        default=32,
        metadata={"help": "max gen batches for dynamic sampling"},
    )
    eval_mode: str = field(
        default=None,
        metadata={
            "help": "eval mode for actor model and reward_critic_model, optional for: None, single, tensor_parallel."
        },
    )

    offload_level: str = field(
        default="",
        metadata={"help": "Offload model, optional for: eval, reward, optimizer, train_model"},
    )

    max_dec_len: int = field(default=512, metadata={"help": "Maximum output length."})

    min_dec_len: int = field(default=1, metadata={"help": "Minimum output length."})

    max_src_len: int = field(default=3072, metadata={"help": "Max length of src."})

    eos_token: str = field(
        default="",
        metadata={"help": "Use it as an eos_token if set it to non empty."},
    )

    use_fusemt: bool = field(
        default=True,
        metadata={"help": "use fused inference model to speedup in rollout generation"},
    )

    recompute_use_reentrant: bool = field(
        default=True,
        metadata={"help": "use recompute_use_reentrant to recompute"},
    )

    critic_min_learning_rate: float = field(
        default=None,
        metadata={"help": "Minimum learning rate deacyed to for critic model."},
    )

    critic_decay_steps: int = field(
        default=None,
        metadata={
            "help": "The steps use to control the learing rate for critic model. If the step > decay_steps, "
            "will use the min_learning_rate."
        },
    )

    min_learning_rate: float = field(
        default=None,
        metadata={"help": "Minimum learning rate deacyed to."},
    )

    decay_steps: int = field(
        default=None,
        metadata={
            "help": "The steps use to control the learing rate. If the step > decay_steps, "
            "will use the min_learning_rate."
        },
    )
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to run benchmark by autotuner. True for from_scratch."},
    )
    early_stopping: bool = field(
        default=False,
        metadata={"help": "Whether apply early stopping strategy."},
    )
    early_stopping_patience: int = field(
        default=4,
        metadata={
            "help": "Stop training when the specified metricworsens for early_stopping_patience evaluation calls"
        },
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "how much the specified metric must improve to satisfy early stopping conditions."},
    )
    use_fused_head_and_loss_fn: bool = field(
        default=False,
        metadata={"help": "use fused_head_and_loss_fn."},
    )
    tensor_parallel_output: bool = field(
        default=True,
        metadata={"help": "use tensor_parallel_output."},
    )
    # save_generation_output: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to save generated text to file when eval"},
    # )
    dropout_warmup_steps: int = field(
        default=0,
        metadata={"help": "dropout warmup steps"},
    )
    hidden_dropout_prob: float = field(
        default=0.0,
        metadata={"help": "dropout probability for hidden layers"},
    )
    attention_probs_dropout_prob: float = field(
        default=0.0,
        metadata={"help": "dropout probability for attention layers"},
    )
    rl_algorithm: str = field(
        default="ppo",
        metadata={"help": "RL algorithm (supports PPO, GRPO and Reinforce++)."},
    )
    use_tgt_len_value: bool = field(
        default=False,
        metadata={"help": "Whether to use tgt for KL."},
    )
    use_rm_server: bool = field(default=False, metadata={"help": "Use reward server instead of reward model."})
    use_rule_reward: bool = field(default=False, metadata={"help": "Use rule-based reward only for gsm8k, to date."})
    use_fp32_compute: bool = field(
        default=False, metadata={"help": "Use fp32 to compute xx_log_prob,rewards, advantages and loss."}
    )
    rollout_tensor_parallel_degree: int = field(
        default=-1,
        metadata={"help": ("Tensor parallelism for rollout.")},
    )
    balance_batch: bool = field(
        default=False,
        metadata={"help": "Whether to balance the number of valid tokens on each dp/sharding rank."},
    )
    use_remove_padding: bool = field(
        default=False,
        metadata={"help": "Whether to remove paddings before computing transformer."},
    )
    rollout_max_num_seqs: int = field(
        default=8,
        metadata={
            "help": "The maximum number of sequences that can be processed in a single inference. Default is 8."
        },
    )

    def __post_init__(self):
        """
        Function executed after initialization, used to set some default values and validate parameters.
        If autotuner_benchmark is True, set related parameters to default values and prohibit any other operations.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        # set the unified_checkpoint to True, it will change two cases:
        # 1. use unified_checkpoint
        # 2. data_parallel use hybrid group
        self.unified_checkpoint = True
        # obtain the parallrl degree from the training arguments
        # for auto config the accumulation steps
        self._post_init_parallel_degree()

        if self.global_mini_batch_size < 0:
            self.global_mini_batch_size = self.global_batch_size

        if (
            self.global_batch_size % self.dataset_world_size != 0
            or self.global_mini_batch_size % self.dataset_world_size != 0
        ):
            raise ValueError(
                "global_batch_size(global_mini_batch_size) must be divisible by dataset_world_size! "
                f"Hint: global_batch_size={self.global_batch_size}, global_mini_batch_size={self.global_mini_batch_size}, dataset_world_size={self.dataset_world_size}. "
                f"dataset_world_size({self.dataset_world_size})=data_parallel_degree({self.data_parallel_degree})*sharding_parallel_degree({self.sharding_parallel_degree})."
            )

        if not self.dynamic_sampling or self.global_gen_batch_size <= 0:
            self.global_gen_batch_size = self.global_batch_size

        if self.per_device_rollout_batch_size <= 0:
            self.per_device_rollout_batch_size = self.global_batch_size // self.dataset_world_size
        if self.per_device_logprob_batch_size <= 0:
            self.per_device_logprob_batch_size = self.per_device_train_batch_size
        if self.per_device_reward_batch_size <= 0:
            self.per_device_reward_batch_size = self.per_device_train_batch_size
        if self.per_device_value_batch_size <= 0:
            self.per_device_value_batch_size = self.per_device_train_batch_size

        # conserve kv cache, select the minimum value as the rollout max num seqs for the inference engine
        # self.rollout_max_num_seqs = min(self.per_device_rollout_batch_size * self.rollout_n, self.rollout_max_num_seqs)

        # `gradient_accumulation_steps` specifies the number of mini-batches per gradient update.
        # This value must be set prior to calling `super().__post_init__()`.
        # It is utilized within `super().__post_init__()` for configuring the DistributedStrategy.
        self.gradient_accumulation_steps = (
            self.global_mini_batch_size
            * self.rollout_n
            * self.update_iters
            // self.per_device_train_batch_size
            // self.dataset_world_size
        )

        if self.gradient_accumulation_steps <= 0:
            logger.warning(
                f"gradient_accumulation_steps: {self.gradient_accumulation_steps} must be greater than zero!"
                " Please check your configuration, gradient_accumulation_steps = global_mini_batch_size * rollout_n * update_iters / per_device_train_batch_size / dataset_world_size."
                " dataset_world_size = {self.dataset_world_size} = data_parallel_degree * sharding_parallel_degree."
                " We will set it to 1!"
            )
            self.gradient_accumulation_steps = 1

        train_batch_size_info = {
            "global_batch_size": self.global_batch_size,
            "global_mini_batch_size": self.global_mini_batch_size,
            "rollout_n": self.rollout_n,
            "rollout_max_num_seqs": self.rollout_max_num_seqs,
            "dataset_world_size": self.dataset_world_size,
            "per_device_rollout_batch_size": self.per_device_rollout_batch_size,
            "per_device_logprob_batch_size": self.per_device_logprob_batch_size,
            "per_device_reward_batch_size": self.per_device_reward_batch_size,
            "per_device_value_batch_size": self.per_device_value_batch_size,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }

        logger.info("{:^40}".format("{} Configuration Arguments".format("Train Batch Size")))
        for key, value in train_batch_size_info.items():
            logger.info("{:30}: {}".format(key, value))
        logger.info("===========================================")

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
            self.per_device_prompt_batch_size = self.per_device_train_batch_size
            self.min_dec_len = self.max_dec_len
            # self.skip_profile_timer = False

            if not self.disable_tqdm:
                self.logging_steps = 1
                self.logging_strategy = IntervalStrategy.STEPS

        paddle.set_device(self.device)

        assert self.rl_algorithm in [
            "ppo",
            "grpo",
            "reinforce_plus_plus",
        ], 'self.rl_algorithm should be one of ["ppo", "grpo", "reinforce_plus_plus"]'
        if self.rl_algorithm == "grpo":
            self.normalize_reward = False
            self.normalize_advantage = False

        max_per_device_eval_batch_size = (
            self.global_batch_size * self.rollout_n * self.update_iters // self.dataset_world_size
        )
        if self.per_device_eval_batch_size > max_per_device_eval_batch_size:
            logger.warning(
                f"per_device_eval_batch_size: {self.per_device_eval_batch_size} is larger than "
                f"global_batch_size: {self.global_batch_size} * rollout_n: "
                f"{self.rollout_n} * update_iters: {self.update_iters}, which may cause infer error. "
                f"We will set it to global_batch_size * rollout_n * update_iters // dataset_world_size!"
            )
            self.per_device_eval_batch_size = max_per_device_eval_batch_size

        self.offload_level = self.offload_level.split()

        if self.sequence_parallel:
            if self.tensor_parallel_degree <= 1:
                self.sequence_parallel = False
                logger.info("Tensor_parallel_degree = 1. Set sequence_parallel to False.")

        if self.tensor_parallel_degree <= 1:
            self.tensor_parallel_output = False
            logger.info("Tensor_parallel_degree = 1. Set tensor_parallel_output to False.")

        if self.sharding_parallel_degree > 1:
            if ShardingOption.SHARD_GRAD_OP in self.sharding or ShardingOption.FULL_SHARD in self.sharding:
                if self.release_grads is True:
                    self.release_grads = False

        if self.unified_checkpoint and "async_save" in self.unified_checkpoint_config:
            self.unified_checkpoint_config.remove("async_save")
            logger.warning(
                "PPO training currently does not support asynchronous saving! "
                "Remove `async_save` from unified_checkpoint_config."
            )

        if self.eval_mode is not None and len(self.eval_mode) == 0:
            self.eval_mode = None
        # if self.eval_mode is None and self.offload_level is not None:
        #     self.offload_level = self.offload_level.replace("eval", "")

        if self.decay_steps is None:
            self.decay_steps = self.max_steps

        if self.rollout_tensor_parallel_degree == -1:
            self.rollout_tensor_parallel_degree = self.tensor_parallel_degree
            logger.info(
                f"Set rollout_tensor_parallel_degree to tensor_parallel_degree: {self.tensor_parallel_degree}."
            )

    @property
    def model_dtype(self):
        # Load model
        if self.fp16_opt_level == "O2":
            if self.fp16:
                dtype = "float16"
            elif self.bf16:
                dtype = "bfloat16"
            else:
                raise ValueError("Please specific dtype: --fp16 or --bf16")
        else:
            dtype = "float32"
        return dtype

    @property
    def use_kl_in_reward(self):
        if self.rl_algorithm in ["ppo", "reinforce_plus_plus"]:
            return True
        else:
            return False


@dataclass
class ModelArgument:
    actor_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Built-in pretrained model name or the path to local model."},
    )
    reward_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Built-in pretrained model name or the path to local model."},
    )
    reward_server: str = field(default=None, metadata={"help": "Reward server address."})
    critic_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Built-in pretrained model name or the path to local model."},
    )
    actor_tokenizer_alpha: float = field(default=None, metadata={"help": "Tokenizer will tokenize randomly"})
    reward_tokenizer_alpha: float = field(default=None, metadata={"help": "Tokenizer will tokenize randomly"})
    critic_tokenizer_alpha: float = field(default=None, metadata={"help": "Tokenizer will tokenize randomly"})
    stage: str = field(default="PPO", metadata={"help": "The type of training."})
    critic_recompute_granularity: str = field(
        default="full",
        metadata={
            "help": "The granularity of recompute in critic model, "
            "can be selected as `full` or `full_attn` or `core_attn`. "
        },
    )
    chat_template: str = field(
        default="none",
        metadata={
            "help": "the path of `chat_template.json` file to handle multi-rounds conversation. "
            "If is None(do not set --chat_template argument), it will use the default `chat_template.json`;"
            "If is equal with `model_name_or_path`, it will use the default loading; "
            "If is directory, it will find the `chat_template.json` under the directory; If is file, it will load it."
            "If is none string, it will not use chat_template.json."
        },
    )


@dataclass
class DataArgument:
    train_datasets: str = field(default=None, metadata={"help": "Dataset name(s) registered in the raw dataset."})
    eval_datasets: str = field(default=None, metadata={"help": "Dataset name(s) registered in the raw dataset."})
    max_length: int = field(
        default=2048,
        metadata={
            "help": "The maximum length that model input tokens can have. When intokens is set to True, it's also the maximum length for InTokens data stream"
        },
    )
    max_prompt_len: int = field(default=4096, metadata={"help": "Maximum prompt length."})
    prompt_key: str = field(default="src", metadata={"help": "The key of prompt(question) in the dataset."})
    response_key: str = field(default="tgt", metadata={"help": "The key of response(answer) in the dataset."})
