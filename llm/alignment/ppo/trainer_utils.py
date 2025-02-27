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

import inspect
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import paddle
import tqdm
from data import parse_dataset
from models.ppo_model_utils import make_attention_mask, make_position_ids
from paddle.distributed import fleet
from paddle.io import DataLoader

from paddlenlp.generation.utils import GenerationMixin
from paddlenlp.trainer import IntervalStrategy
from paddlenlp.trainer.trainer import (
    TRAINER_STATE_NAME,
    HybridParallelOptimizer,
    NlpDistributedBatchSampler,
    ShardingOption,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    _obtain_optimizer_parameters_list,
    distributed_file,
    distributed_isfile,
    fused_allreduce_gradients,
    logger,
    reshard_util,
    split_inputs_sequence_dim,
)
from paddlenlp.transformers import BatchEncoding, PretrainedModel, PretrainedTokenizer
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_outputs import ModelOutput
from paddlenlp.transformers.tokenizer_utils_base import PaddingStrategy


@dataclass
class TrainingArguments(TrainingArguments):
    kl_coeff: float = field(
        default=0.02,
        metadata={"help": "The coefficient for the KL divergence between the reference and actor policy."},
    )
    kl_loss_coeff: float = field(
        default=0.001,
        metadata={"help": "The coefficient for the KL loss for GRPO."},
    )
    clip_range_ratio: float = field(
        default=0.2,
        metadata={
            "help": "The clipping range for ratio between the old and new policy. "
            "This is the epsilon parameter in the PPO algorithm."
        },
    )
    clip_range_score: float = field(
        default=50.0,
        metadata={
            "help": "The clipping range for the output of the score model. "
            "The reward is clipped into [-clip_range_score, clip_range_score]."
        },
    )
    clip_range_value: float = field(
        default=5.0,
        metadata={
            "help": "The clipping range for the value function. The value is clipped into [value_estimate - "
            "clip_range_value, value_estimate + clip_range_value] during training."
        },
    )
    ptx_coeff: float = field(
        default=0.0,
        metadata={"help": "The coefficient for the ptx loss."},
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
    unified_checkpoint: bool = field(
        default=True,
        metadata={
            "help": "Enable fused linear grad add strategy, which will reduce elementwise "
            "add for grad accumulation in the backward of nn.Linear ."
        },
    )
    unified_checkpoint_config: Optional[str] = field(
        default="",
        metadata={
            "help": (
                "Configs to unify hybrid parallel checkpoint.\n"
                "Following options are supports:\n"
                "- skip_save_model_weight: do not save model weights when the masters weight exist\n"
                "- master_weight_compatible: 1. if the master weights exist, only load when needed\n"
                "                            2. if master weights does not exist, convert model weights"
                " to master weights when needed\n"
                "- async_save: enable asynchronous saving checkpoints to disk\n"
                "- enable_all_options: enable all optimization configurations\n"
            )
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
            "help": "Stop training when the specified metric" "worsens for early_stopping_patience evaluation calls"
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
    per_device_rollout_batch_size: int = field(
        default=-1,
        metadata={"help": "Batch size per GPU core/CPU for rollout."},
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
        metadata={"help": "RL algorithm (supports PPO and GRPO)."},
    )
    use_tgt_len_value: bool = field(
        default=False,
        metadata={"help": "Whether to use tgt for KL."},
    )
    use_rm_server: bool = field(default=False, metadata={"help": "Use reward server instead of reward model."})

    def __post_init__(self):
        """
            在初始化后执行的函数，用于设置一些默认值和验证参数。
        如果 autotuner_benchmark 为 True，则将相关参数设置为默认值，并禁止其他任何操作。

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
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
        if self.per_device_rollout_batch_size < 0:
            self.per_device_rollout_batch_size = self.per_device_train_batch_size
        assert self.rl_algorithm in ["ppo", "grpo"], 'self.rl_algorithm should be one of ["ppo", "grpo"]'
        if self.rl_algorithm == "grpo":
            self.normalize_reward = False
            self.normalize_advantage = False


@dataclass
class ModelArgument:
    actor_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Build-in pretrained model name or the path to local model."},
    )
    reward_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Build-in pretrained model name or the path to local model."},
    )
    reward_server: str = field(
        default=None,
        metadata={"help": "Reward server address."},
    )
    reward_critic_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Build-in pretrained model name or the path to local model."},
    )
    actor_tokenizer_alpha: float = field(default=None, metadata={"help": "Tokenizer will tokenize randomly"})
    reward_tokenizer_alpha: float = field(default=None, metadata={"help": "Tokenizer will tokenize randomly"})
    reward_critic_tokenizer_alpha: float = field(default=None, metadata={"help": "Tokenizer will tokenize randomly"})
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})
    use_attn_mask_start_row_indices: bool = field(default=False, metadata={"help": "Should in data args"})
    stage: str = field(default="PPO", metadata={"help": "The type of training."})
    fused_linear: bool = field(default=True, metadata={"help": "Whether to use fused_gemm_epilogue"})
    recompute_granularity: str = field(
        default="full",
        metadata={
            "help": "The granularity of recompute in policy model, "
            "can be selected as `full` or `full_attn` or `core_attn`. "
        },
    )
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
    eval_split_ratio: float = field(default=None, metadata={"help": "Ratio of eval data to train data"})
    ptx_datasets: str = field(default=None, metadata={"help": "Dataset name(s) registered in the raw dataset."})
    max_length: int = field(
        default=2048,
        metadata={
            "help": "The maximum length that model input tokens can have. When intokens is set to True, it's also the maximum length for InTokens data stream"
        },
    )
    max_prompt_len: int = field(default=4096, metadata={"help": "Maximum prompt length."})

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


# ########## patches for Trianer ##########
def init_train_model_opt(
    self: Trainer,
    max_steps: int,
    resume_from_checkpoint: bool = False,
    clear_master_weight: bool = False,
) -> PretrainedModel:
    """
    初始化训练模型和优化器，并返回已包装的模型。

    Args:
        self (Trainer): Trainer实例对象。
        max_steps (int): 最大训练步数。
        resume_from_checkpoint (bool, optional, default=False): 是否从保存点中恢复训练，默认为False。
            Defaults to False.
        clear_master_weight (bool, optional, default=False): 在使用Trainer的分布式硬件加速时，清除主参数权重，默认为False。
            Defaults to False.

    Returns:
        PretrainedModel: 已经包装好的模型。
    """
    # Copy of model/optimizer init and resuming related code in `Trainer.train`.
    # NOTE: this `_load_from_checkpoint` is indeed to load model states in the
    # following elif-else branches, though they are apart away in `Trainer.train`.
    if not self.args.should_load_sharding_stage1_model:
        self._load_from_checkpoint(resume_from_checkpoint)

    # delay_optimizer_creation = (
    #     self.sharding is not None
    #     and ShardingOption.SHARD_OP in self.args.sharding
    # )
    delay_optimizer_creation = False

    if not delay_optimizer_creation:
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    if self.args.should_load_sharding_stage1_model:
        model = self._wrap_model_and_load_sharded_checkpoint(resume_from_checkpoint)
    elif self.args.should_save_sharding_stage1_model:
        # In the non-sharded mode, should invoke _load_from_checkpoint before _wrap_model.
        # In this mode, the rank0 load all params and the _wrap_model implicitly broadcast
        # params from rank0 to the other ranks.
        model = self._wrap_model(self.model_wrapped)
        if self.sharding_io is not None:
            assert delay_optimizer_creation is False, "delay_optimizer_creation should be False"
            # the self.optimizer should be wrapped and it is done in _wrap_model
            self.sharding_io.set_optimizer(self.optimizer)
        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model
        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
    else:
        model = self._wrap_model(self.model_wrapped)
        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model
        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

    if ShardingOption.FULL_SHARD in self.args.sharding and clear_master_weight:
        # for inference model to use Trainer sharding stage3, clear master_weight
        # which is created in GroupShardedStage3.__init__
        self.optimizer._master_weights = None

    if self.args.device == "npu" and self.args.flatten_param_grads:
        from .plugins.npu_plugin import npu_accelerate_plugin

        npu_accelerate_plugin(self.optimizer)

    return model


def init_train_state(
    self: Trainer,
    resume_from_checkpoint: bool,
    train_dataloader: DataLoader,
    max_steps: int,
    num_train_epochs: int,
    num_update_steps_per_epoch: int,
):
    """
    初始化训练状态。

    Args:
        self (Trainer): Trainer实例，用于记录训练状态。
        resume_from_checkpoint (bool, optional): 是否从检查点继续训练，默认为None。
        train_dataloader (DataLoader, optional): 训练数据加载器，默认为None。
        max_steps (int, optional): 最大训练步数，默认为-1。
        num_train_epochs (int, optional): 训练的最大轮数，默认为3。
        num_update_steps_per_epoch (int, optional): 每个轮次更新模型的步数，默认为1。

    Returns:
        Tuple[int, int, Optional[tqdm]]:
            - epochs_trained (int): 已经训练了多少个epoch。
            - steps_trained_in_current_epoch (int): 如果不忽略数据跳过，则为当前epoch中已经训练了多少个批次；否则为0。
            - steps_trained_progress_bar (Optional[tqdm]): 如果不忽略数据跳过，则为一个tqdm进度条，用于显示正在跳过第一个批次；否则为None。
    """
    args = self.args

    self.state = TrainerState()
    self.state.epoch = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    steps_trained_progress_bar = None

    # Check if continuing training from a checkpoint
    if resume_from_checkpoint is not None and distributed_isfile(
        os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    ):
        self.state = TrainerState.load_from_json(
            distributed_file(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        )
        epochs_trained = self.state.global_step // num_update_steps_per_epoch
        if not args.ignore_data_skip:
            steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
            steps_trained_in_current_epoch *= args.gradient_accumulation_steps
        else:
            steps_trained_in_current_epoch = 0

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {self.state.global_step}")
        if not args.ignore_data_skip:
            logger.info(
                f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                "flag to your launch command, but you will resume the training on data already seen by your model."
            )
            if self.is_local_process_zero() and not args.disable_tqdm:
                steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                steps_trained_progress_bar.set_description("Skipping the first batches")
        if not args.ignore_data_skip:
            if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                train_dataloader.batch_sampler, NlpDistributedBatchSampler
            ):
                consumed_samples = (
                    self.state.global_step
                    * args.train_batch_size
                    * args.gradient_accumulation_steps
                    * args.dataset_world_size
                )
                train_dataloader.batch_sampler.set_epoch(consumed_samples=consumed_samples)
                logger.info(f"Set DistributedBatchSampler consumed_samples to {consumed_samples}")

    self.state.max_steps = int(max_steps)
    self.state.num_train_epochs = num_train_epochs
    self.state.is_local_process_zero = self.is_local_process_zero()
    self.state.is_world_process_zero = self.is_world_process_zero()

    return (
        epochs_trained,
        steps_trained_in_current_epoch,
        steps_trained_progress_bar,
    )


def init_train_log(
    self: Trainer,
    num_examples: int,
    num_train_epochs: int,
    total_train_batch_size: int,
    max_steps: int,
    num_train_samples: int,
    model: PretrainedModel,
):
    """
    初始化训练日志。

    Args:
        self (Trainer): Trainer实例，包含了训练所需的参数和信息。
        num_examples (int): 训练集中样本的总数。
        num_train_epochs (int): 训练的 epoch 数量。
        total_train_batch_size (int): 单个设备上的训练 batch 大小之和。
        max_steps (int): 最大训练步数。
        num_train_samples (int): 训练集中样本的总数。
        model (PretrainedModel): 被训练的模型。

    Returns:
        None, 该函数不返回任何值。
    """
    args = self.args

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples:,}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps:,}")
    logger.info(f"  Total num train samples = {num_train_samples:,}")
    # per_device_trainable_numel = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
    # TODO: Temporary fix since Tensor.numel() not supported in distributed mode
    per_device_trainable_numel = sum(np.prod(p.shape) for p in model.parameters() if not p.stop_gradient)
    logger.debug(f"  Number of trainable parameters = {per_device_trainable_numel:,} (per device)")
    if self.args.use_hybrid_parallel:
        # todo fix for pipeline_parallel_degree
        parts_num = max(self.args.tensor_parallel_degree, 1) * max(self.args.pipeline_parallel_degree, 1)
        if parts_num > 1:
            all_reduce_dtype = "int64"
            if paddle.get_device().split(":")[0] in ["npu", "xpu"]:
                # TODO(duanyanhui): fix when NPU all_reduce supports int64
                all_reduce_dtype = "float32"
            trainable_numel_tensor = paddle.to_tensor(per_device_trainable_numel, dtype=all_reduce_dtype)
            paddle.distributed.all_reduce(trainable_numel_tensor)
            trainable_numel = int(trainable_numel_tensor.item()) // self.args.dataset_world_size
            # the numel is roughly, because the tensor parallel still hold own bias or layer_norm weight without splited
            # so, the trainable numel is a little bigger than real.
            logger.debug(f"  Number of trainable parameters = {trainable_numel:,} (all devices, roughly)")


def full_training_step(self: Trainer, inputs: Dict[str, paddle.Tensor], **kwargs):
    """
    Just a copy of single training step complete code in Trainer.train while loop
    which including forward+backward+step, while wraps the inputs and outputs to
    make the complicated copied code no need to change. Maybe a better way is to
    add fine-grained methods including these steps to Trainer which is similar to
    DeepSpeed engine.
    """

    # TODO(guosheng): step, steps_trained_in_current_epoch and steps_trained_progress_bar
    # should use reference since they would be overwrite.
    # for state update
    epoch = kwargs.get("epoch", 0)
    step = kwargs.get("step", 0)
    steps_in_epoch = kwargs.get("steps_in_epoch", 0)
    step_control = kwargs.get("step_control", 0)
    # for step and progress update when resuming data
    train_dataloader = kwargs.get("train_dataloader", None)
    resume_from_checkpoint = kwargs.get("resume_from_checkpoint", None)
    steps_trained_in_current_epoch = kwargs.get("steps_trained_in_current_epoch", 0)
    steps_trained_progress_bar = kwargs.get("steps_trained_progress_bar", None)
    # for eval output ignore to gather
    ignore_keys_for_eval = kwargs.get("ignore_keys_for_eval", None)
    # timer_name = kwargs.get("timer_name", "")
    tr_loss = kwargs.get("tr_loss", 0.0)
    model = kwargs.get("model", self.model_wrapped)
    # needed in _maybe_log_save_evaluate
    self._globalstep_last_logged = getattr(self, "_globalstep_last_logged", 0)
    self._globalstep_last_start_time = getattr(self, "_globalstep_last_start_time", time.time())

    args = self.args

    if self.args.use_hybrid_parallel and self.args.sep_parallel_degree > 1:
        inputs = split_inputs_sequence_dim(inputs)
    # self.timers and self.timers("read-data").stop()
    os.environ["TRAINER_GLOBAL_STEP"] = str(self.state.global_step)
    self.callback_handler.on_load_data_end(args, self.state, self.control, inputs=inputs)

    # Skip past any already trained steps if resuming training
    # for paddlenlp.utils.batch_sampler.DistributedBatchSampler
    # We use consumed_samples to reset the status
    if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
        train_dataloader.batch_sampler, NlpDistributedBatchSampler
    ):
        if step == 0:
            if steps_trained_progress_bar is not None:
                steps_trained_progress_bar.update(steps_trained_in_current_epoch)
                steps_trained_progress_bar.close()
                steps_trained_progress_bar = None
            self._load_rng_state(resume_from_checkpoint)
        step += steps_trained_in_current_epoch
    elif steps_trained_in_current_epoch > 0:
        steps_trained_in_current_epoch -= 1
        if steps_trained_progress_bar is not None:
            steps_trained_progress_bar.update(1)
        if steps_trained_in_current_epoch == 0:
            self._load_rng_state(resume_from_checkpoint)
        # continue
        final_local_vars = locals()
        for k in kwargs.keys():
            if k in final_local_vars:
                kwargs[k] = final_local_vars[k]
        return kwargs
    elif steps_trained_progress_bar is not None:
        steps_trained_progress_bar.close()
        steps_trained_progress_bar = None

    if step_control % args.gradient_accumulation_steps == 0:
        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
        # self.timers and self.timers(f"{timer_name}: forward-backward").start()

    dp_enabled = self.args.data_parallel_degree > 1 if self.args.use_hybrid_parallel else args.local_rank != -1
    forbidden_no_sync = False
    # stage2 and stage3 should not no_sync, because the is no DDP wrapper and no_sync API
    # hybrid_parallel (tp or pp or sharding stage 1) should not no_sync
    if self.args.use_hybrid_parallel:
        forbidden_no_sync = True

    availiable_no_sync = dp_enabled and not forbidden_no_sync

    is_no_sync = (
        ((step_control + 1) % args.gradient_accumulation_steps != 0)
        and availiable_no_sync
        and args._no_sync_in_gradient_accumulation
    ) or (args.recompute and availiable_no_sync)
    # sharding
    # stage1. the same as ddp
    # stage2. manualy collect gradient on dp group

    dp_master_grad = self.args.world_size > 1 and self.args.amp_master_grad and not self.args.use_hybrid_parallel
    if dp_master_grad:
        is_no_sync = True

    if is_no_sync:
        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
        with model.no_sync():
            tr_loss_step = self.training_step(model, inputs)
    else:
        tr_loss_step = self.training_step(model, inputs)

    tr_loss += tr_loss_step

    if (step_control + 1) % args.gradient_accumulation_steps == 0 or (
        # last step in epoch but step is always smaller than gradient_accumulation_steps
        steps_in_epoch <= args.gradient_accumulation_steps
        and (step + 1) == steps_in_epoch
    ):
        if self.args.pipeline_parallel_degree <= 1 and self._enable_delay_scale_loss():
            tr_loss /= self.args.gradient_accumulation_steps

        # self.timers and self.timers(f"{timer_name}: forward-backward").stop()

        # Maunally collect gradients
        # Case 1: Use recompute and dp
        # Case 2: Hack dp with master_grad
        # Case 3: Pipeline or sharding overlap
        # local_rank != -1 don't means dp in networks.
        # self.timers and self.timers(f"{timer_name}: all-reduce").start()

        # Case 1: Use recompute and dp / sharding stage1,
        # manualy collect gradient for dp.
        if args.recompute and availiable_no_sync:
            fused_allreduce_gradients(list(model.parameters()), None)

        # Case 2: hack dp with master_grad
        if dp_master_grad and not (args.recompute and availiable_no_sync):
            fused_allreduce_gradients(list(model.parameters()), None)

        # Pipeline parallel mode,  handle gradient reduce here to overlap
        pipeline_parallel_config = (
            set(args.pipeline_parallel_config.split(" ")) if args.pipeline_parallel_degree > 1 else set()
        )
        enable_dp_comm_overlap = "enable_dp_comm_overlap" in pipeline_parallel_config
        enable_release_grads = "enable_release_grads" in pipeline_parallel_config

        # Case 3: Pipeline parallel mode, overlap with dp
        if isinstance(self.optimizer, HybridParallelOptimizer) and not self.do_grad_scaling:
            parameters_list = _obtain_optimizer_parameters_list(self.optimizer._inner_opt)

            if not enable_dp_comm_overlap:
                if self.optimizer._sharding_enable:
                    assert reshard_util.is_sharding_opt(self.optimizer)
                    self.optimizer._inner_opt.reduce_gradients(list(parameters_list), self.optimizer._hcg)

                if self.optimizer._dp_enable or getattr(self.optimizer, "_sep_enable", False):
                    fused_allreduce_gradients(list(parameters_list), self.optimizer._hcg)

        # self.timers and self.timers(f"{timer_name}: all-reduce").stop()
        # self.timers and self.timers(f"{timer_name}: optimizer-step").start()

        if self.args.gradient_accumulation_steps > 1 and self._enable_delay_scale_loss():
            for p in model._layers.parameters():
                with paddle.no_grad():
                    if hasattr(p, "main_grad") and p.main_grad is not None:
                        assert p.grad is None
                        p.main_grad.scale_(1.0 / self.args.gradient_accumulation_steps)
                    elif p.grad is not None:
                        p.grad.scale_(1.0 / self.args.gradient_accumulation_steps)

        # Optimizer step
        self.callback_handler.on_optimizer_begin(
            args,
            self.state,
            self.control,
            scaler=self.scaler if self.do_grad_scaling else None,
        )
        optimizer_was_run = True
        if self.do_grad_scaling:
            scale_before = paddle.assign(self.scaler._scale)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler._scale
            # Compatible with paddlepaddle 2.6.0 using typo word.
            if hasattr(self.scaler, "_cache_founf_inf"):
                optimizer_was_run = not self.scaler._cache_founf_inf
            else:
                optimizer_was_run = not self.scaler._cache_found_inf
            if not optimizer_was_run:
                scale_before_value = scale_before.cpu().numpy()
                scale_after_value = scale_after.cpu().numpy()
                logger.warning(
                    f"optimizer not run, scale_before: {scale_before_value[0]}, scale_after: {scale_after_value[0]}"
                )
        elif isinstance(self.optimizer, HybridParallelOptimizer):
            self.optimizer._step(parameters_list)
        else:
            self.optimizer.step()

        # self.timers and self.timers(f"{timer_name}: optimizer-step").stop()

        if optimizer_was_run:
            self.lr_scheduler.step()

        if enable_release_grads and args.pipeline_parallel_degree > 1:
            self.optimizer.clear_grad(set_to_zero=False)
            for _, buffers in model._chunk_2_comm_buffers.items():
                for buffer in buffers:
                    buffer._clear_grad_storage()
        else:
            self.optimizer.clear_grad(set_to_zero=False)

        self.callback_handler.on_optimizer_end(
            args,
            self.state,
            self.control,
            scaler=self.scaler if self.do_grad_scaling else None,
        )

        self.state.global_step += 1
        self.state.epoch = epoch + (step + 1) / steps_in_epoch
        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
        self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
        self._print_timer()
        step_control = 0
    else:
        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
        step_control += 1

    if self.control.should_epoch_stop or self.control.should_training_stop:
        # break
        final_local_vars = locals()
        for k in kwargs.keys():
            if k in final_local_vars:
                kwargs[k] = final_local_vars[k]
        return kwargs
    # self.timers and self.timers("read-data").start()

    final_local_vars = locals()
    for k in kwargs.keys():
        if k in final_local_vars:
            kwargs[k] = final_local_vars[k]
    return kwargs


Trainer.init_train_model_opt = init_train_model_opt
Trainer.init_train_log = init_train_log
Trainer.init_train_state = init_train_state
Trainer.full_training_step = full_training_step
# ########## patches for Trianer ##########


class MuteDefaultFlowCallback(TrainerCallback):
    """
    Add this callback can cencel logging/evaluation/saving by DefaultFlowCallback.
    Use this when having multi trainer.
    """

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        在一个步骤结束时调用，可以用来更新控制流程。

        Args:
            args (TrainingArguments): 训练参数对象。
            state (TrainerState): 训练器状态对象。
            control (TrainerControl): 训练控制对象，包含了训练过程中的控制信息，如是否保存模型、是否进行评估和是否记录日志等。
            kwargs (dict, optional): 其他关键字参数，默认为None，没有使用。

        Returns:
            TrainerControl: 返回一个TrainerControl对象，包含了训练过程中的控制信息，如是否保存模型、是否进行评估和是否记录日志等。

        Raises:
            None
        """
        control.should_save = False
        control.should_evaluate = False
        control.should_log = False
        return control


@contextmanager
def guard_set_args(args, arg_name_values):
    """
    在一个上下文中，设置给定的参数名称和值，并在上下文结束后将其还原。

    Args:
        args (object): 需要修改参数的对象，通常是命令行解析器的实例。
        arg_name_values (dict[str, Any]): 包含参数名称和新值的字典，该函数会在上下文中修改这些参数。
            key (str): 参数名称。
            value (Any): 参数的新值。

    Yields:
        None: 无返回值，只是用于上下文管理。

    Returns:
        None: 无返回值，只是用于上下文管理。

    Raises:
        None: 不会引发任何异常。
    """
    for k, v in arg_name_values.items():
        old_value = getattr(args, k, None)
        setattr(args, k, v)
        arg_name_values[k] = old_value
    yield
    for k, v in arg_name_values.items():
        old_value = getattr(args, k)
        setattr(args, k, v)
        arg_name_values[k] = old_value


class PipeEvalModel(GenerationMixin):
    """
    Wrapper for PipelineParallel to do evaluate and generate. Currently only
    support .
    """

    def __init__(self, trainer: Trainer):
        """
        Args:
        trainer (Trainer): Trainer object.
            The trainer should have a attribute named `_inner_eval_model` which is the model used for evaluation.
            If it does not exist, then the model in `trainer.model_wrapped` will be used.
        """
        eval_model = getattr(trainer, "_inner_eval_model", None)
        self.model: fleet.model.PipelineParallel = trainer.model_wrapped if eval_model is None else eval_model
        self.config: PretrainedConfig = trainer.model.config
        self._is_gen = False
        self.update_model_kwargs_for_generation = (
            self.model._layers._non_pipe_model_class.update_model_kwargs_for_generation
        )

    @property
    def pp_group(self):
        """
        获取当前模型的属性分组，返回值为str类型。
        如果模型没有设置属性分组，则返回None。

        Returns:
            str, optional: 当前模型的属性分组，默认为None。
        """
        return self.model.pp_group

    def eval(self):
        """
        将模型置于评估模式，禁用梯度计算和 dropout。
        返回：None
        """
        self.model.eval()

    def train(self):
        """
        将模型设置为训练模式。
        在调用任何前向传播函数之前，必须先调用此函数。

        Returns:
            None, 无返回值。
        """
        self.model.train()

    def __getattr__(self, name):
        """
        如果在当前类中没有找到对应的属性，则尝试从模型中获取。
        如果在模型中也没有找到对应的属性，则会引发AttributeError异常。

        Args:
            name (str): 要查询的属性名称。

        Returns:
            Any: 返回属性值，如果在当前类和模型中都没有找到该属性，则会引发AttributeError异常。

        Raises:
            AttributeError: 如果在当前类和模型中都没有找到对应的属性。
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def _broadcast_outputs(self, outputs):
        """
        将输出广播到所有进程中，如果不是最后一个阶段则返回元组，否则返回ModelOutput或者paddle.Tensor。
        如果不是最后一个阶段，会对输入的每个张量创建一个与其形状、类型相同但内容为空的新张量，并广播这些张量。

        Args:
            outputs (Union[paddle.Tensor, Tuple[paddle.Tensor], ModelOutput]): 模型的输出，可以是单个张量或张量元组，也可以是ModelOutput。

        Returns:
            Union[paddle.Tensor, Tuple[paddle.Tensor], ModelOutput]: 如果不是最后一个阶段，返回元组；否则返回ModelOutput或者paddle.Tensor。
        """
        # outputs is PipelineParallel.eval_batch which is a list of batches.
        out = []
        outputs = (outputs,) if isinstance(outputs, paddle.Tensor) else outputs
        for tensors in outputs:
            if not self.model.is_pipeline_last_stage():
                tensor = tensors if isinstance(tensors, paddle.Tensor) else tensors[0]
                head_out_meta = (
                    (self.model._layers.head_out_meta,)
                    if isinstance(
                        self.model._layers.head_out_meta,
                        paddle.static.InputSpec,
                    )
                    else self.model._layers.head_out_meta
                )
                tensors = tuple(
                    paddle.empty(
                        shape=[
                            (tensor.shape[i] if (meta.shape[i] is None or meta.shape[i] < 0) else meta.shape[i])
                            for i in range(len(meta.shape))
                        ],
                        dtype=(tensor.dtype if meta.dtype is None else meta.dtype),
                    )
                    for meta in head_out_meta
                )
            else:
                # Currently use tuple instead of ModelOutput and require the
                # caller use the return result as tuple.
                tensors = (
                    (tensors,)
                    if isinstance(tensors, paddle.Tensor)
                    else (tensors.to_tuple() if isinstance(tensors, ModelOutput) else tensors)
                )

            # use map_structure seems hung
            for tensor in tensors:
                paddle.distributed.broadcast(
                    tensor,
                    src=self.model.pp_group.ranks[-1],
                    group=self.model.pp_group,
                )
            out.append(tensors[0] if len(tensors) == 1 else tensors)
        return out[0] if len(out) == 1 else out

    def __call__(self, *args, **kwargs):
        """
        Call the method to generate output from given input.

        Args:
            *args (tuple, optional): Input arguments to the method. Defaults to ().
            **kwargs (dict, optional): Keyword arguments to the method. Defaults to {}.

        Returns:
            Union[List[Any], Tuple[Any]]: Output generated from the input. If the method is
                called multiple times, each call returns one output. The type of the output
                depends on the implementation of the method.
        """
        model = self.model
        assert self.model.training is False
        if self._is_gen:
            # inputs by `prepare_inputs_for_generation` is a dict with following keys:
            # "input_ids", "position_ids", "past_key_values", "use_cache", "attention_mask"
            # NOTE: 1. cache/past_key_values should be passed across decoding steps
            # by using as model attr rather than input args to reduce comm overhead.
            # Also, pipe model defined for training not support this cache input.
            # 2. ignore use_cache since _check_data_vaild requires tensor if not None.
            # 3. attention_mask can reuse _prepare_decoder_attention_mask in LlamaEmbeddingPipe.
            # 4. position_ids pass through _prepare_pipeline_inputs_func and PipeLayer.
            inputs, labels = model._prepare_pipeline_inputs_func(*args, **kwargs)
            # currently, set accumulate_steps to 1 to avoid multi-batch eval/gen
            with guard_set_args(model, {"_compute_loss": False, "accumulate_steps": 1}):
                outputs = model.eval_batch([inputs, labels], compute_loss=False)
            # TODO(guosheng): Broadcasted logits are used to get next_scores, remove
            # it to reduce comm overhead. Also note that we still need broadcast
            # next_tokens though logits are broadcasted since pp ranks' seeds differs.
            # Currently, just slice the last token to reduce comm overhead.
            outputs = [
                (
                    micro_batch_output[:, -1, :].unsqueeze(1).contiguous()
                    if isinstance(micro_batch_output, paddle.Tensor)
                    else micro_batch_output[0][:, -1, :].unsqueeze(1).contiguous()
                )
                for micro_batch_output in outputs
            ]
            outputs = self._broadcast_outputs(outputs)
        else:
            # use _prepare_pipeline_inputs_func to convert pipeline inputs
            inputs, labels = model._prepare_pipeline_inputs_func(*args, **kwargs)
            # NOTE(guosheng): bug seems exist. pp.eval_batch(compute_loss=False)
            # will set pp._compute_loss to False and would not set it back. Thus
            # hack here to set it back.
            with guard_set_args(model, {"_compute_loss": False, "accumulate_steps": 1}):
                outputs = model.eval_batch([inputs, labels], compute_loss=False)
            outputs = self._broadcast_outputs(outputs)
        return outputs

    def generate(self, *args, **kwargs):
        """
            重写父类的方法，在生成文本时使用缓存。
        首先将self._is_gen设置为True，然后修改DecoderLayerPipe以使用缓存。
        接下来，调用super().generate(*args, **kwargs)进行文本生成。
        最后，清除所有层中的缓存（包括子层），并将self._has_cache设置为False。

        Args:
            args (Tuple[Any], optional): 可变参数列表，默认为空元组。
            kwargs (Dict[str, Any], optional): 关键字参数字典，默认为空字典。

        Returns:
            Tuple[Any]: 返回一个元组，其中包含了生成的文本和相应的概率分布。

        Raises:
            无。
        """
        self._is_gen = True
        # patch DecoderLayerPipe to use cache, DecoderLayerPipe is subclass of
        # DecoderLayer, and would call super().forward
        ori_decoder_layer_forward = self.model._layers._non_pipe_decoder_layer_class.forward

        def decoder_layer_forward(layer_self, *args, **kwargs):
            kwargs.update(
                {
                    "use_cache": True,
                    "cache": getattr(layer_self, "_cache", None),
                }
            )
            outputs = ori_decoder_layer_forward(layer_self, *args, **kwargs)
            output = outputs[0]
            layer_self._cache = outputs[1]
            self._has_cache = True
            return output

        with guard_set_args(
            self.model._layers._non_pipe_decoder_layer_class,
            {"forward": decoder_layer_forward},
        ):
            outputs = super().generate(*args, **kwargs)
        self._is_gen = False
        # clear cache of decoder layers, sublayers is incursive thus suitable
        # to both 1F1B and interleave
        for layer in self.model._layers.sublayers():
            if isinstance(layer, self.model._layers._non_pipe_decoder_layer_class):
                layer._cache = None
        self._has_cache = False
        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
            Prepare the input for generation. This method is used by
        :meth:`~transformers.Pipeline.__call__` to generate text from prompts.

        Args:
            *args (tuple, optional): Arguments passed to :meth:`~transformers.Pipeline.__call__`.
            **kwargs (dict, optional): Keyword arguments passed to :meth:`~transformers.Pipeline.__call__`.

        Returns:
            dict: A dictionary containing the prepared inputs for generation. The keys are:

                - "prompt" (:obj:`str`, `optional`, defaults to :obj:`None`):
                  Text to be decoded. If not provided, the pipeline will try to use the cached prompts.
                - "cache" (:obj:`bool`, `optional`, defaults to :obj:`False`):
                  Whether to use the cached past key values. If not provided, it will be set to :obj:`True` when
                  the pipeline has cache.
                - Other keyword arguments are passed to :meth:`~transformers.Pipeline.__call__`.

        Raises:
            ValueError: If both ``prompt`` and ``cache`` are not provided.
        """
        arg_bind = inspect.signature(self.model._layers._non_pipe_model_class.prepare_inputs_for_generation).bind(
            *((self,) + args), **kwargs
        )
        arg_bind.apply_defaults()
        arg_dict = arg_bind.arguments
        last_arg_name, last_arg_value = arg_dict.popitem()
        if arg_bind.signature.parameters[last_arg_name].kind == inspect.Parameter.VAR_KEYWORD:
            arg_dict.update(last_arg_value)
        else:
            arg_dict[last_arg_name] = last_arg_value
        arg_dict.pop("self")
        cache = arg_dict.get("cache", None)
        # prepare_inputs_for_generation use cache to discrimate prefill
        # or decode and slice inputs accordingly.
        if getattr(self, "_has_cache", False):
            arg_dict.update({"cache": True})
        model_inputs = self.model._layers._non_pipe_model_class.prepare_inputs_for_generation(self, **arg_dict)
        model_inputs.update({"cache": cache})
        return model_inputs


def is_same_tokenizer(
    tokenizer: PretrainedTokenizer,
    other_tokenizer: PretrainedTokenizer,
) -> bool:
    """Check if two tokenizers are the same."""
    return tokenizer is other_tokenizer or (
        tokenizer.__class__ == other_tokenizer.__class__ and tokenizer.get_vocab() == other_tokenizer.get_vocab()
    )


def retokenize(src_tokenizer, dest_tokenizer, token_ids, skip_special_tokens):
    """Retokenize a sequence of token ids from one tokenizer to another."""
    tokens = src_tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
    part_tokens = []
    result_ids = []
    for token in tokens:
        if token in src_tokenizer.all_special_tokens:
            if part_tokens:
                decoded_text = src_tokenizer.decode(
                    src_tokenizer.convert_tokens_to_ids(part_tokens),
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=False,
                )
                tmp_tokens = dest_tokenizer.tokenize(decoded_text)
                result_ids.extend(dest_tokenizer.convert_tokens_to_ids(tmp_tokens))
                part_tokens = []  # 清空
            # 转换当前特殊 token
            special_token = dest_tokenizer.convert_tokens_to_ids(token)
            result_ids.append(special_token)
        else:
            part_tokens.append(token)
    # 如果有，处理最后一段(一般不应该走到, 应该以special token结尾)
    if part_tokens:
        decoded_text = src_tokenizer.decode(
            src_tokenizer.convert_tokens_to_ids(part_tokens),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
        tmp_tokens = dest_tokenizer.tokenize(decoded_text)
        result_ids.extend(dest_tokenizer.convert_tokens_to_ids(tmp_tokens))
    return result_ids


def batch_retokenize(
    input_ids: paddle.Tensor,
    src_tokenizer: PretrainedTokenizer,
    dest_tokenizer: PretrainedTokenizer,
    *,
    padding: bool | str | PaddingStrategy = PaddingStrategy.LONGEST,
    skip_special_tokens: bool = False,
) -> BatchEncoding:
    """Re-tokenize a batch of input ids from one tokenizer to another."""
    all_ids = []
    for token_ids in input_ids:
        tmp_ids = retokenize(src_tokenizer, dest_tokenizer, token_ids, skip_special_tokens)
        all_ids.append(tmp_ids)
    output = {}

    output["input_ids"] = dest_tokenizer.pad(
        {"input_ids": all_ids},
        padding=padding,
        return_attention_mask=False,
        return_tensors="pd",
    )["input_ids"]
    output["attention_mask"] = make_attention_mask(
        output["input_ids"],
        pad_id=dest_tokenizer.pad_token_id,
        eos_id=dest_tokenizer.eos_token_id,
        unk_id=dest_tokenizer.unk_token_id,
        causal_mask=True,
    ).cast(paddle.bfloat16)
    output["position_ids"] = make_position_ids(output["attention_mask"])
    return output


def process_row(row, remove_value=0, remove_side="both"):
    """
    从张量中去除前导/尾随的特定值。

    Args:
        row (paddle.Tensor): 待处理的张量，一维。
        remove_value (int, optional): 要去除的值，默认为0。
        remove_side (str, optional): 去除的位置，可选"left"（只去除前导）、"right"（只去除尾随）、"both"（去除前导和尾随），默认为"both"。

    Returns:
        paddle.Tensor: 处理后的张量，一维。

    """
    non_zero_indices = paddle.nonzero(row != remove_value).flatten()
    if non_zero_indices.shape[0] == 0:
        # 行全为0，警告，不处理
        logger.warning("Row is all zeros, no trimming will be performed.")
        return row
    start_index = non_zero_indices[0]
    end_index = non_zero_indices[-1]
    # 切取中间的非零部分
    if remove_side == "left":
        trimmed_row = row[start_index:]
    elif remove_side == "right":
        trimmed_row = row[: end_index + 1]
    elif remove_side == "both":
        trimmed_row = row[start_index : end_index + 1]
    else:
        logger.warning("unknown remove_side, using both remove_side.")
        trimmed_row = row[start_index : end_index + 1]

    return trimmed_row
