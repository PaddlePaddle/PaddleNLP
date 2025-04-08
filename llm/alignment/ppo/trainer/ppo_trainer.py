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
import copy
import json
import math
import os
import sys
import time
import types
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed as dist
from algos.advantage import (
    compute_grpo_advantages,
    compute_reinforce_plus_plus_advantages_and_returns,
)
from models.ppo_model_utils import make_position_ids_from_input_ids
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import PipelineLayer
from paddle.io import DataLoader, Dataset, DistributedBatchSampler
from paddle.utils import map_structure
from rich.console import Console
from rich.table import Table
from utils.comm_utils import (
    ActorStages,
    RolloutStages,
    data_group_merge,
    data_group_split,
    gather_and_pad,
    get_timer_label,
    new_timer_log,
)
from utils.infer_utils import infer_guard
from utils.offload_utils import reload_and_offload_scope, reload_tensor_to_gpu
from utils.timer_utils import TimerScope, TimerScopeManualLabel

from paddlenlp.data import DataCollator
from paddlenlp.trainer.trainer import (
    EvalLoopOutput,
    EvalPrediction,
    ProgressCallback,
    ShardingOption,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainOutput,
    logger,
    speed_metrics,
)
from paddlenlp.trainer.utils.helper import (
    broadcast_dataset_rank0_model,
    distributed_concat,
)
from paddlenlp.transformers import (
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
    PretrainedModel,
    PretrainedTokenizer,
)
from paddlenlp.transformers.model_utils import _add_variant
from paddlenlp.utils.env import PADDLE_WEIGHTS_NAME

from .actor_trainer import ActorReferenceTrainer
from .critic_trainer import CriticTrainer
from .reward_trainer import RewardTrainer
from .trainer_utils import (
    MuteDefaultFlowCallback,
    batch_retokenize,
    guard_set_args,
    is_same_tokenizer,
    process_row,
)


class PPOMetric:
    def set_metric_meta(self):
        """
        设置指标的元信息，包括指标名称和运算方式。

        Args:

        Returns:
            None: 无返回值，直接修改了类属性。
        """
        self.metric_names = [
            "train_" + name
            for name in (
                [
                    "policy_loss",
                    "ptx_loss",
                    *(["value_loss"] if self.args.rl_algorithm == "ppo" else []),
                    "reward",
                    "norm_reward",
                    "kl_reward",
                    "norm_reward_with_kl",
                    "pure_policy_loss",
                    "entropy_loss",
                    *(["values"] if self.args.rl_algorithm == "ppo" else []),
                    "returns",
                    "kl_divergence",
                    "mean_generated_length",
                    "max_generated_length",
                    "min_generated_length",
                ]
                if self.args.rl_algorithm in ["ppo", "reinforce_plus_plus"]
                else [
                    "policy_loss",
                    "pure_policy_loss",
                    "kl_loss",
                    "entropy_loss",
                    "reward",
                    "kl_divergence",
                    "mean_generated_length",
                    "max_generated_length",
                    "min_generated_length",
                ]
            )
        ]

        if self.args.rl_algorithm == "ppo":
            self.metric_ops = ["mean"] * 13 + ["max", "min"]
        elif self.args.rl_algorithm == "reinforce_plus_plus":
            self.metric_ops = ["mean"] * 11 + ["max", "min"]
        else:
            self.metric_ops = ["mean"] * 8 + ["max", "min"]

    def __init__(self, freq, args, use_stack=True):
        """
        Args:
        freq (int): frequency of metrics collection.
        use_stack (bool, optional): whether to stack the metrics into a single tensor. Defaults to True.
        use_ptx (bool, optional): whether to use ptx or not. Defaults to True.

        Raises:
            ValueError: when freq is less than 1.
        """
        self.args = args
        self.set_metric_meta()
        self.freq = freq
        self.counter = 0
        self.use_stack = use_stack
        if use_stack:
            self.metrics = paddle.zeros([freq, len(self.metric_names)], dtype=paddle.float32)
        else:
            self.metrics = [None] * len(self.metric_names)
            for i in range(len(self.metrics)):
                self.metrics[i] = paddle.zeros([freq], dtype=paddle.float32)

    @paddle.no_grad()
    def update(self, metrics: Dict[str, paddle.Tensor]) -> Union[None, Dict[str, float]]:
        """
        If has updated for`freq` times then return metrics (results reduced from
        all worker) and reset metric states, otherwise return `None`.
        """
        for name in self.metric_names:
            # PipelineParallel broadcast loss with shape [1]
            if len(metrics[name].shape) != 0:
                metrics[name] = metrics[name].squeeze()
            if metrics[name].dtype != paddle.float32:
                metrics[name] = metrics[name].cast(paddle.float32)
        if self.use_stack:
            self.metrics[self.counter] = paddle.stack([metrics[name] for name in self.metric_names])
        else:
            for i, name in enumerate(self.metric_names):
                self.metrics[i][self.counter] = metrics[name]

        self.counter += 1
        if self.counter == self.freq:
            metrics = distributed_concat(self.metrics) if paddle.distributed.get_world_size() > 1 else self.metrics

            out_metrics = {}
            if self.use_stack:
                mean_metric = metrics.mean(0)
                max_metric = metrics.max(0)
                min_metric = metrics.min(0)
            for i, (name, op) in enumerate(zip(self.metric_names, self.metric_ops)):
                if op == "max":
                    out_metrics[name] = max_metric[i].item() if self.use_stack else metrics[i].max().item()
                elif op == "min":
                    out_metrics[name] = min_metric[i].item() if self.use_stack else metrics[i].min().item()
                else:
                    out_metrics[name] = mean_metric[i].item() if self.use_stack else metrics[i].mean().item()

            # reset
            self.counter = 0
            if self.use_stack:
                self.metrics.fill_(0.0)
            else:
                for i, name in enumerate(self.metric_names):
                    self.metrics[i].fill_(0.0)
            return out_metrics


class PPOTrainer(Trainer):
    def __init__(
        self,
        actor_model: Union[PretrainedModel, nn.Layer],
        reference_model: Union[PretrainedModel, nn.Layer] = None,
        reward_model: Union[PretrainedModel, nn.Layer] = None,
        critic_model: Union[PretrainedModel, nn.Layer] = None,
        actor_model_eval: Union[PretrainedModel, nn.Layer] = None,
        critic_model_eval: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        actor_tokenizer: Optional[PretrainedTokenizer] = None,
        reference_tokenizer: Optional[PretrainedTokenizer] = None,
        reward_tokenizer: Optional[PretrainedTokenizer] = None,
        critic_tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]] = None,
    ):
        """
        Args:
        model (Union[PretrainedModel, nn.Layer], optional): The model to be trained. If not provided, it will be
            initialized based on the values of ``args``. Defaults to None.
        criterion (nn.Layer, optional): The loss function used for training. Defaults to None.
        args (TrainingArguments, optional): Training arguments. If not provided, it will be initialized with
            default values. Defaults to None.
        data_collator (Optional[DataCollator], optional): The function to batch data samples together into
            mini-batches. If not provided, a simple batching function that drops remaining samples will be used.
            Defaults to None.
        train_dataset (Optional[Dataset], optional): The dataset to be used for training. Defaults to None.
        eval_dataset (Union[Dataset, Dict[str, Dataset]], optional): The dataset to be used for evaluation.
            Defaults to None.
        tokenizer (Optional[PretrainedTokenizer], optional): The tokenizer used for encoding. Defaults to None.
            actor_tokenizer and critic_tokenizer should be same
        compute_metrics (Optional[Callable[[EvalPrediction], Dict]], optional): The function to compute metrics
            during evaluation. Defaults to None.
        callbacks (Optional[List[TrainerCallback]], optional): A list of callbacks to customize the training
            process. Defaults to None.
        optimizers (Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler], optional): The tuple of
            optimizer and learning rate scheduler. Defaults to (None, None).
        preprocess_logits_for_metrics (Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor], optional): The
            function to preprocess logits before computing metrics. Defaults to None.
        """
        with guard_set_args(
            args,
            {
                "recompute": False,
                "fp16_opt_level": "O1",
                "pipeline_parallel_degree": 1,  # workaround for pipeline parallel model check
            },
        ):
            # just used to create trival attrs might be used in the training
            # process of trainer, while changing some args to avoid model usage
            # in __init__ such as recompute and AMP-O2
            super().__init__(
                (actor_model, reference_model, reward_model, critic_model, actor_model_eval, critic_model_eval),
                criterion,
                args,
                data_collator,
                train_dataset,
                eval_dataset,
                (actor_tokenizer, reference_tokenizer, reward_tokenizer, critic_tokenizer),
                compute_metrics,
                callbacks,
                optimizers,
                preprocess_logits_for_metrics,
            )

        trainer_agrs = {
            # "model": None,
            "criterion": criterion,
            "args": args,
            "data_collator": data_collator,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            # "tokenizer": None,
            "compute_metrics": compute_metrics,
            "callbacks": callbacks,
            "optimizers": optimizers,
            "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
        }

        self.actor_trainer = self.create_actor_trainer(
            model=actor_model,
            model_eval=actor_model_eval,
            tokenizer=actor_tokenizer,
            **trainer_agrs,
        )

        if args.rl_algorithm == "ppo":
            self.critic_trainer = self.create_critic_trainer(
                model=critic_model,
                tokenizer=critic_tokenizer,
                **trainer_agrs,
            )

        # use trainer for reference_model/reward_model to enable sharding stage-3
        # and PipelineParallel. allow reference_model/reward_model to use different
        # dist strategy
        self.reference_trainer = self.create_reference_trainer(
            model=reference_model,
            tokenizer=reference_tokenizer,
            **trainer_agrs,
        )
        self.reward_trainer = self.create_reward_trainer(
            model=reward_model,
            tokenizer=reward_tokenizer,
            **trainer_agrs,
        )

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._model_config = actor_model.config
        self._actor_model_eval = actor_model_eval
        self._critic_model_eval = critic_model_eval
        self.reference_model.eval()
        if isinstance(reward_model, PretrainedModel):
            self.reward_model.eval()

        self.tokenizer = actor_tokenizer
        if is_same_tokenizer(actor_tokenizer, reward_tokenizer):
            self.reward_tokenizer = actor_tokenizer
        else:
            self.reward_tokenizer = reward_tokenizer

        # Those value can be changed
        self.kl_coeff = self.args.kl_coeff
        self.clip_range_score = self.args.clip_range_score
        self.gamma = 1.0
        self.gae_lambda = 0.95

        # for reward norm
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.sample_batch_num = 0

        # dummy class and object for model to be compatible with methods of
        # Trainer, such as evaluation_loop
        self.DummyPPOModel = type(
            "DummyPPOModel",
            (object,),
            {
                "eval": lambda _: self.set_eval(),
                "train": lambda _: self.set_train(),
            },
        )
        self.model = self.model_wrapped = self.DummyPPOModel()
        if self.timers:
            self.timers.log = types.MethodType(new_timer_log, self.timers)

    def create_actor_trainer(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        model_eval: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]] = None,
    ):
        policy_training_args = copy.deepcopy(args)
        lr_scheduler = self.get_scheduler(policy_training_args)
        actor_trainer = ActorReferenceTrainer(
            model,
            criterion,
            policy_training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
            callbacks,
            [None, lr_scheduler],
            preprocess_logits_for_metrics,
        )
        actor_trainer.set_eval_model(model_eval)
        actor_trainer.timers = self.timers

        actor_trainer.add_callback(MuteDefaultFlowCallback)
        if not args.disable_tqdm:
            actor_trainer.pop_callback(ProgressCallback)
        return actor_trainer

    def create_critic_trainer(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        model_eval: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]] = None,
    ):
        value_training_args = copy.deepcopy(args)
        for attr_name in [
            "critic_learning_rate",
            "critic_weight_decay",
            "critic_lr_scheduler_type",
            "critic_warmup_ratio",
            "critic_recompute",
        ]:
            if getattr(value_training_args, attr_name, None) is not None:
                setattr(
                    value_training_args,
                    attr_name[len("critic_") :],
                    getattr(value_training_args, attr_name),
                )
        lr_scheduler = self.get_scheduler(value_training_args)
        critic_trainer = CriticTrainer(
            model,
            criterion,
            value_training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
            callbacks,
            [None, lr_scheduler],
            preprocess_logits_for_metrics,
        )

        critic_trainer.set_eval_model(model_eval)
        critic_trainer.timers = self.timers

        critic_trainer.add_callback(MuteDefaultFlowCallback)
        if not args.disable_tqdm:
            critic_trainer.pop_callback(ProgressCallback)
        return critic_trainer

    def create_reference_trainer(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]] = None,
    ):
        with guard_set_args(
            args,
            {
                "recompute": False,
                # "fp16_opt_level": "O1",
                "pipeline_parallel_degree": (
                    args.pipeline_parallel_degree if isinstance(model, PipelineLayer) else 1
                ),  # workaround for pipeline parallel model check
            },
        ):
            reference_trainer = ActorReferenceTrainer(
                model,
                criterion,
                copy.deepcopy(args),
                data_collator,
                train_dataset,
                eval_dataset,
                tokenizer,
                compute_metrics,
                callbacks,
                optimizers,
                preprocess_logits_for_metrics,
            )
            if args.pipeline_parallel_degree > 1 or ShardingOption.FULL_SHARD in args.sharding:
                reference_trainer.init_train_model_opt(100, None, clear_master_weight=True)  # dummy max_steps

        reference_trainer.timers = self.timers

        return reference_trainer

    def create_reward_trainer(
        self,
        model: Union[PretrainedModel, nn.Layer, str] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]] = None,
    ):
        with guard_set_args(
            args,
            {
                "recompute": False,
                # "fp16_opt_level": "O1",
                "pipeline_parallel_degree": (
                    args.pipeline_parallel_degree if isinstance(model, PipelineLayer) else 1
                ),  # workaround for pipeline parallel model check
            },
        ):
            reward_trainer = RewardTrainer(
                model,
                criterion,
                copy.deepcopy(args),
                data_collator,
                train_dataset,
                eval_dataset,
                tokenizer,
                compute_metrics,
                callbacks,
                optimizers,
                preprocess_logits_for_metrics,
                reward_server=model,
            )

            if not self.args.use_rm_server:
                if args.pipeline_parallel_degree > 1 or ShardingOption.FULL_SHARD in args.sharding:
                    reward_trainer.init_train_model_opt(100, None, clear_master_weight=True)  # dummy max_steps

        reward_trainer.timers = self.timers

        return reward_trainer

    @property
    def reference_model(self):
        """
        获取参考模型，如果没有则返回None。
        该方法只能在初始化后使用，否则会引发异常。

        Returns:
            torch.nn.Module, optional - 参考模型，如果没有则返回None。

        Raises:
            Exception - 当调用此方法前未初始化reference_trainer时，将引发异常。
        """
        return self.reference_trainer.get_model(train=False)

    @property
    def reward_model(self):
        """
        获取奖励模型，如果没有则创建一个。
        返回值：tf.keras.models.Model，奖励模型。
        """
        if self.args.use_rm_server:
            return self.reward_server
        else:
            return self.reward_trainer.get_model(train=False)

    @property
    def actor_model(self):
        """
        获取当前的actor模型，如果在训练中则返回训练后的模型，否则返回eval时使用的模型。

        Returns:
            torch.nn.Module, torch.jit.ScriptModule: Actor模型，可以是torch.nn.Module或者torch.jit.ScriptModule类型。
        """
        return self.actor_trainer.get_model(train=self.training)

    @property
    def critic_model(self):
        """
        获取 critic model，仅在使用 value-based 策略时有效。

        Returns:
            tf.keras.Model, optional: critic model，如果没有设置则返回 None。
        """
        return self.critic_trainer.get_model(train=self.training)

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for all models."""
        if mode:
            self.training = True
            self.actor_model.train()
            if self.args.rl_algorithm == "ppo":
                self.critic_model.train()
        else:
            self.training = False
            self.actor_model.eval()
            if self.args.rl_algorithm == "ppo":
                self.critic_model.eval()

    def set_eval(self) -> None:
        """Set model to evaluation mode."""
        self.set_train(mode=False)

    def get_scheduler(self, args):
        """
        获取学习率调度器，如果没有设置最小学习率则返回None。
        支持两种类型的学习率调度器："cosine"和"linear"。

        Args:
            args (argparse.Namespace): 命令行参数，包含了学习率相关的参数。

        Returns:
            torch.optim.lr_scheduler._LRScheduler or None, optional: 学习率调度器或者None，默认为None。
        """
        if args.decay_steps is None:
            args.decay_steps = args.max_steps
        if args.warmup_steps > 0:
            warmup_steps = args.warmup_steps
        else:
            warmup_steps = args.warmup_ratio * args.max_steps
        lr_scheduler = None
        if args.min_learning_rate is not None:
            if args.lr_scheduler_type == "cosine":
                lr_scheduler = CosineAnnealingWithWarmupDecay(
                    max_lr=args.learning_rate,
                    min_lr=args.min_learning_rate,
                    warmup_step=warmup_steps,
                    decay_step=args.decay_steps,
                    last_epoch=0,
                )
            elif args.lr_scheduler_type == "linear":
                lr_scheduler = LinearAnnealingWithWarmupDecay(
                    max_lr=args.learning_rate,
                    min_lr=args.min_learning_rate,
                    warmup_step=warmup_steps,
                    decay_step=args.decay_steps,
                    last_epoch=0,
                )
        return lr_scheduler

    @paddle.no_grad()
    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        """
        预测步骤，用于生成下一个输入序列。

        Args:
            model (nn.Layer): 模型实例，需要是 `paddle.nn.Layer` 的子类。
            inputs (Dict[str, Union[paddle.Tensor, Any]]): 包含输入数据的字典，其中包含以下键：
                - "input_ids" (paddle.Tensor, optional): 输入序列的编号 ID，默认为None。
                - "attention_mask" (paddle.Tensor, optional): 输入序列的注意力掩码，默认为None。
                - "position_ids" (paddle.Tensor, optional): 输入序列的位置ID，默认为None。
            prediction_loss_only (bool): 仅返回预测损失，不返回其他任何值。
            ignore_keys (Optional[List[str]], optional): 忽略的键列表，默认为None。

        Returns:
            Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
            三元组，包含以下元素：
                - Optional[paddle.Tensor]: 如果 `prediction_loss_only` 为False，则为预测得分，否则为None。
                - Optional[paddle.Tensor]: 当前未定义，始终为None。
                - Optional[paddle.Tensor]: 当前未定义，始终为None。

        Raises:
            ValueError: 如果 `ignore_keys` 不是可选参数或者不是一个列表。
        """
        inputs = self._prepare_inputs(inputs)
        with reload_and_offload_scope(self, self.actor_model, self.reference_model, self.actor_trainer):
            with infer_guard(self.actor_trainer):
                prompt_only_batch = {
                    "input_ids": inputs["input_ids"],
                    **({"label_ids": inputs["label_ids"]} if self.args.use_rm_server else {}),
                }
                generated_seq = self.actor_trainer.generate_sequences(prompt_only_batch, do_eval=True)[0]["input_ids"]

            if not self.args.use_rm_server:
                if self._model_config.sequence_parallel:
                    # pad to max_sequence_length
                    seq = self.tokenizer.pad(
                        {"input_ids": [s for s in generated_seq]},
                        padding="longest",
                        max_length=None,
                        return_attention_mask=False,
                        pad_to_multiple_of=self.args.tensor_parallel_degree,
                    )["input_ids"]
                else:
                    seq = generated_seq

                if self.reward_tokenizer is not self.tokenizer:
                    reward_tokenize_output = batch_retokenize(
                        input_ids=seq,
                        src_tokenizer=self.tokenizer,
                        dest_tokenizer=self.reward_tokenizer,
                    )
                    reward_input_ids = reward_tokenize_output["input_ids"]
                    # reward_attention_mask = reward_tokenize_output["attention_mask"]
                    reward_position_ids = reward_tokenize_output["position_ids"]
                else:
                    reward_input_ids = seq
                    reward_attention_mask = None
                    reward_position_ids = make_position_ids_from_input_ids(
                        reward_attention_mask, self.reward_tokenizer.pad_token_id
                    )

                # .end_scores
                reward_score = self.reward_model(
                    reward_input_ids,
                    attention_mask=reward_attention_mask,
                    position_ids=reward_position_ids,
                    # return_dict=True,
                )[1]
            else:
                prompt_len = inputs["input_ids"].shape[-1]
                if "label_ids" not in inputs:
                    raise ValueError("Rule-based reward needs labels.")
                src = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=False)
                tgt = self.tokenizer.batch_decode(inputs["label_ids"], skip_special_tokens=False)
                response = self.tokenizer.batch_decode(generated_seq[:, prompt_len:], skip_special_tokens=False)
                reward_score = self.reward_trainer.request_reward_server(
                    [i.replace(self.tokenizer.pad_token, "") for i in src],
                    [i.replace(self.tokenizer.pad_token, "") for i in tgt],
                    [i.replace(self.tokenizer.pad_token, "") for i in response],
                )

            reward_score = reward_score.squeeze(axis=-1).cast(paddle.float32)
        # keep the first batch of eval output sequence to print and check
        prompt = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        generated = self.tokenizer.batch_decode(generated_seq, skip_special_tokens=True)  # no padding
        reward_score_list = reward_score.tolist()
        for i, text in enumerate(generated):
            item = {
                "Prompt": text[: len(prompt[i]) - 1],
                "Generated": text[len(prompt[i]) :],
                "Reward": reward_score_list[i],
            }
            self._eval_out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

        if getattr(self, "_eval_seq", None) is None:
            generated = [text[len(prompt[i]) :] for i, text in enumerate(generated)]
            # prompts.extend(prompt)
            # generateds.extend(generated)
            self._eval_seq = (prompt, generated, reward_score_list)
        return reward_score.mean(), reward_score, reward_score

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_eval_iters: Optional[int] = -1,
    ) -> EvalLoopOutput:
        """
        循环访问数据集，并对模型进行评估。

        Args:
            dataloader (DataLoader, optional): 用于评估的数据加载器。默认为None。
            description (str, optional): 描述评估过程的字符串。默认为''.
            prediction_loss_only (Optional[bool], optional): 是否只计算预测损失。默认为None。
            ignore_keys (Optional[List[str]], optional): 要忽略的键列表。默认为None。
            metric_key_prefix (str, optional): 指标键前缀。默认为'eval'.
            max_eval_iters (Optional[int], optional): 最大评估次数。默认为-1，表示无限制。

        Returns:
            EvalLoopOutput: 包含评估结果和指标的类实例。

        Raises:
            ValueError: 如果`prediction_loss_only`不是布尔值，则引发ValueError异常。
        """
        # to save eval generated sequence
        eval_out_file = os.path.join(
            self.args.output_dir,
            f"eval_out-step{self.state.global_step}-rank{self.args.local_rank}.jsonl",
        )
        self._eval_out_file = open(eval_out_file, "w", encoding="utf-8")

        # TODO(guosheng): use _inner_eval_model (if trainer has one) instead of
        # original trainer model to eval, especially when using sharded EMA
        # NOTE: use here rather than in prediction_step since actor_model would
        # be set to eval out of prediction_step
        # with guard_set_args(
        #     self.actor_trainer,  # disable _inner_eval_model
        #     {
        #         "_eval_model": None,  # otherwise would use cached _eval_model
        #         "_inner_eval_model": None,  # otherwise would use _inner_eval_model to create _eval_model
        #     },
        # ):
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
            max_eval_iters,
        )
        output.metrics[f"{metric_key_prefix}_reward"] = output.metrics.pop(f"{metric_key_prefix}_loss")

        columns = ["Prompt", "Generated", "Reward"]
        rows = list(zip(*self._eval_seq))
        rows = [[str(item) for item in row] for row in rows]
        max_num_rows = 5
        table = Table(title="Evaluating...", show_lines=True, title_justify="left")
        for column in columns:
            table.add_column(column)
        for row in rows[:max_num_rows]:
            table.add_row(*row)
        Console(soft_wrap=True, markup=False, emoji=False).print(table)
        self._eval_seq = None

        self._eval_out_file.close()

        return output

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        获取用于评估模型的数据加载器。如果未提供`eval_dataset`，则使用`self.eval_dataset`。
            该函数会设置一个名为"data_collator"的参数，并将其传递给`super().get_eval_dataloader()`。

            Args:
                eval_dataset (Optional[Dataset], optional): 用于评估的数据集. Defaults to None.

            Returns:
                DataLoader: 包含用于评估的数据的DataLoader实例。
        """
        with guard_set_args(self, {"data_collator": self.data_collator}):
            return super().get_eval_dataloader(eval_dataset)

    def _save_checkpoint(self, model, metrics=None):
        """
        保存模型和指标到两个不同的 checkpoint，一个是 policy 模型，另一个是 value 模型。
        这里使用了 `guard_set_args` 来防止在调用 `_save_checkpoint` 时修改了原始参数。

        Args:
            model (nn.Module): 需要保存的模型。
            metrics (Optional[Dict], optional): 可选的指标字典，默认为 None。
                key 是指标名称，value 是对应的指标值。

        Returns:
            None.
        """
        # maybe change args.output_dir of actor_trainer/critic_trainer directly
        self.runtime_timer.start("checkpoint saving time")
        with guard_set_args(
            self.actor_trainer.args,
            {"output_dir": os.path.join(self.args.output_dir, "policy")},
        ):
            if self.actor_trainer.args.unified_checkpoint:
                if "train_model" in self.actor_trainer.args.offload_level:
                    reload_tensor_to_gpu((self.actor_trainer.model, "train_model"))
                if (
                    "optimizer" in self.actor_trainer.args.offload_level
                    and not self.actor_trainer.args.ignore_save_lr_and_optim
                ):
                    reload_tensor_to_gpu((self.actor_trainer.optimizer, "optimizer"))
            self.actor_trainer._save_checkpoint(model, metrics)
        if self.args.rl_algorithm == "ppo":
            with guard_set_args(
                self.critic_trainer.args,
                {"output_dir": os.path.join(self.args.output_dir, "value")},
            ):
                if self.critic_trainer.args.unified_checkpoint:
                    if "train_model" in self.critic_trainer.args.offload_level:
                        reload_tensor_to_gpu((self.critic_trainer.model, "train_model"))
                    if (
                        "optimizer" in self.critic_trainer.args.offload_level
                        and not self.critic_trainer.args.ignore_save_lr_and_optim
                    ):
                        reload_tensor_to_gpu((self.critic_trainer.optimizer, "optimizer"))
                self.critic_trainer._save_checkpoint(model, metrics)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                metrics = {
                    "policy": self.actor_trainer.state.best_model_checkpoint,
                    **(
                        {"value": self.critic_trainer.state.best_model_checkpoint}
                        if self.args.rl_algorithm == "ppo"
                        else {}
                    ),
                }
                self.state.best_model_checkpoint = json.dumps(metrics)

    def save_model(
        self,
        output_dir: Optional[str] = None,
        merge_tensor_parallel: Optional[bool] = False,
    ):
        """
            保存模型。

        Args:
            output_dir (Optional[str], optional): 输出目录，默认为None，使用命令行参数--output-dir。 Defaults to None.
            merge_tensor_parallel (Optional[bool], optional): 是否合并tensor parallel，默认为False。 Defaults to False.

        Raises:
            ValueError: 如果output_dir不在当前工作目录下，则会引发ValueError异常。
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if "train_model" in self.args.offload_level:
            reload_tensor_to_gpu((self.actor_trainer.model, "model"))
            if self.args.rl_algorithm == "ppo":
                reload_tensor_to_gpu((self.critic_trainer.model, "model"))
        self.actor_trainer.save_model(os.path.join(output_dir, "policy"), merge_tensor_parallel)
        if self.args.rl_algorithm == "ppo":
            self.critic_trainer.save_model(os.path.join(output_dir, "value"), merge_tensor_parallel)

    def init_train_model_opt(
        self: Trainer,
        max_steps: int,
        resume_from_checkpoint: bool = False,
        clear_master_weight: bool = False,
    ) -> PretrainedModel:
        """
            初始化训练模型和优化器。
        如果`resume_from_checkpoint`为字符串，则将其作为路径，并在该路径下恢复模型和优化器状态；否则，将其视为布尔值，表示是否从最后一个保存的检查点中恢复。
        如果`clear_master_weight`为True，则清除主要权重。

        Args:
            max_steps (int): 最大训练步数。
            resume_from_checkpoint (bool, optional): 是否从检查点中恢复模型和优化器状态（默认为False）。
                如果为字符串，则将其作为路径，并在该路径下恢复模型和优化器状态。
            clear_master_weight (bool, optional): 是否清除主要权重（默认为False）。

        Returns:
            Tuple[PretrainedModel, PretrainedModel]: 返回两个元组，分别包含策略模型和价值函数模型。
        """
        # resume should be triggered here
        # maybe change args.output_dir of actor_trainer/critic_trainer directly
        with guard_set_args(
            self.actor_trainer.args,
            {"output_dir": os.path.join(self.args.output_dir, "policy")},
        ):
            actor_model = self.actor_trainer.init_train_model_opt(
                max_steps,
                (
                    os.path.join(resume_from_checkpoint, "policy")
                    if isinstance(resume_from_checkpoint, str)
                    else resume_from_checkpoint
                ),
            )
        if self.args.rl_algorithm == "ppo":
            with guard_set_args(
                self.critic_trainer.args,
                {"output_dir": os.path.join(self.args.output_dir, "value")},
            ):
                critic_model = self.critic_trainer.init_train_model_opt(
                    max_steps,
                    (
                        os.path.join(resume_from_checkpoint, "value")
                        if isinstance(resume_from_checkpoint, str)
                        else resume_from_checkpoint
                    ),
                )
        else:
            critic_model = None
        return actor_model, critic_model

    def init_train_num(self: Trainer, train_dataloader: DataLoader):
        """
            初始化训练数据的批次大小，以及相关参数。

        Args:
            self (Trainer): Trainer实例。
            train_dataloader (DataLoader): 用于训练的DataLoader对象。

        Returns:
            tuple (int, Optional[int], int, int, int, int, int):
                返回一个元组，包含：
                1. total_train_batch_size (int) - 总训练批次大小。
                2. len_dataloader (Optional[int]) - 如果不是可迭代的数据集，则为DataLoader长度；否则为None。
                3. max_steps (int) - 最大训练步数。
                4. num_train_epochs (int) - 训练的最大轮数。
                5. num_update_steps_per_epoch (int) - 每个epoch中更新模型的次数。
                6. num_examples (int) - 训练数据的样本数量。
                7. num_train_samples (int) - 训练数据的样本总数。
        """
        args = self.args

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.dataset_world_size
        len_dataloader = None
        if not self._is_iterable_dataset(self.train_dataset):
            len_dataloader = len(train_dataloader)
            num_train_sub_steps = (
                len_dataloader
                * self.args.update_iters
                * self.args.per_device_prompt_batch_size
                * self.args.num_return_sequences
                // self.args.per_device_train_batch_size
            )
            num_update_steps_per_epoch = num_train_sub_steps // args.gradient_accumulation_steps
            num_examples = len(self.train_dataset)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
                num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = total_train_batch_size * max_steps
        else:
            assert args.max_steps > 0
            max_steps = args.max_steps
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = args.max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        return (
            total_train_batch_size,
            len_dataloader,
            max_steps,
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
        )

    def is_step_end(self):
        """
            判断是否到达了步数结尾，当累加步数等于args.gradient_accumulation_steps时返回True。
        返回值：bool，如果到达了步数结尾则返回True，否则返回False。
        """
        # reach accumulation_steps, value trainer has the same step_control and
        # gradient_accumulation_steps as PPO trainer.
        # if (step_control + 1) % args.gradient_accumulation_steps == 0
        if self.args.rl_algorithm == "ppo":
            return self.critic_trainer.is_accumulation_step
        return self.actor_trainer.is_accumulation_step

    def get_step_loss(self, loss_prefix: str = "") -> Dict:
        """
            获取当前步骤的损失，包括策略训练和价值函数训练的损失。
        如果提供了loss_prefix参数，则将损失名称加上该前缀。

        Args:
            loss_prefix (str, optional): 损失名称的前缀字符串，默认为"".

        Returns:
            Dict[str, float]: 返回一个字典，包含两个损失项：rl_loss（策略训练的损失）和value_loss（价值函数训练的损失）。
        """
        rl_loss = self.actor_trainer.get_step_loss(loss_prefix)
        if self.args.rl_algorithm == "ppo":
            value_loss = self.critic_trainer.get_step_loss(loss_prefix)
            rl_loss.update(value_loss)
        return rl_loss

    def remove_pad_tokens_after_generate(self, generated_batches):
        cleanup_batches, indices, label_ids_batches = [], [], []

        for batch in generated_batches:
            cleanup_batches.extend(
                [
                    process_row(row, remove_value=self.tokenizer.pad_token_id, remove_side="right")
                    for row in batch["input_ids"]
                ]
            )
            if self.args.use_rm_server:
                label_ids_batches.extend(
                    [
                        process_row(row, remove_value=self.tokenizer.pad_token_id, remove_side="right")
                        for row in batch["label_ids"]
                    ]
                )
            indices.append(batch["index"])

        return cleanup_batches, indices, label_ids_batches

    def truncate_batch_data(self, batch, truncate_max_len):
        if len(batch) > truncate_max_len:
            batch = self.tokenizer.truncate_sequences(
                batch,
                num_tokens_to_remove=len(batch) - truncate_max_len,
                truncation_strategy="longest_first",
            )[0]
        return batch

    def pad_batch_data(self, batches, padding_strategy="longest", padding_max_len=None, pad_to_multiple_of=None):
        input_ids = self.tokenizer.pad(
            {"input_ids": batches},
            padding=padding_strategy,
            padding_side="right",
            max_length=padding_max_len,
            return_attention_mask=False,
            pad_to_multiple_of=pad_to_multiple_of,
        )["input_ids"]

        position_ids = make_position_ids_from_input_ids(input_ids)
        return input_ids, position_ids

    def distribute_gather_and_pad_data(self, micro_batches):
        old_log_probs = [micro_batch["log_probs"] for micro_batch in micro_batches]
        ref_log_probs = [micro_batch["ref_log_probs"] for micro_batch in micro_batches]
        rewards = [micro_batch["rewards"] for micro_batch in micro_batches]
        eos_mask = [
            (micro_batch["input_ids"] != self.tokenizer.pad_token_id)[:, micro_batch["prompt"].shape[-1] :].to(
                old_log_probs[0].dtype
            )
            for micro_batch in micro_batches
        ]
        try:
            hcg = fleet.get_hybrid_communicate_group()
            sd_group = hcg.get_sharding_parallel_group()
            dp_group = hcg.get_data_parallel_group()
        except AttributeError:
            pass
        new_batch = {
            "rewards": gather_and_pad(rewards, dp_group, sd_group, pad=False),
            "log_probs": gather_and_pad(old_log_probs, dp_group, sd_group),
            "ref_log_probs": gather_and_pad(ref_log_probs, dp_group, sd_group),
            "eos_mask": gather_and_pad(eos_mask, dp_group, sd_group),
        }

        return new_batch

    def get_rank_data(self, tensor):
        return tensor.split(self.args.dataset_world_size)[self.args.dataset_rank]

    def distribute_get_rank_data(self, micro_batches, new_batches):
        shapes = [micro_batch["log_probs"].shape for micro_batch in micro_batches]
        local_data = {
            "reward_advantages": self.get_rank_data(new_batches[0]["reward_advantages"]),
            "rewards": self.get_rank_data(new_batches[0]["rewards"]),
            "ori_rewards": self.get_rank_data(new_batches[0]["ori_rewards"]),
            "reward_returns": self.get_rank_data(new_batches[0]["reward_returns"]),
            "kl_rewards": self.get_rank_data(new_batches[0]["kl_rewards"]),
            "rewards_with_kl": self.get_rank_data(new_batches[0]["rewards_with_kl"]),
            "eos_mask": self.get_rank_data(new_batches[0]["eos_mask"]),
        }
        offset = 0
        for idx, batch in enumerate(micro_batches):
            for k, v in local_data.items():
                if local_data[k][offset].ndim < 1:
                    micro_batches[idx].update(
                        {k: local_data[k][offset : offset + len(batch["log_probs"])][: shapes[idx][-1]]}
                    )
                else:
                    micro_batches[idx].update(
                        {k: local_data[k][offset : offset + len(batch["log_probs"])][:, : shapes[idx][-1]]}
                    )
            offset += len(batch["log_probs"])

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
    ) -> None:
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (Optional[Union[str, bool]], optional):
                Checkpoint path from which training should be resumed. If a
                path is given, training will restart from this checkpoint. If
                set to ``True``, the last checkpoint in ``output_dir`` will be
                loaded. If ``False`` or ``None`` (default), training will
                start from scratch. Defaults to ``None``.

            ignore_keys_for_eval (Optional[List[str]], optional):
                List of keys to ignore when computing the metrics during
                evaluation. Defaults to ``None``.

        Returns:
            None:
            Training process is finished, no return value.
        """
        # ##### The following code try to keep same as the Trainer.train #####
        args = self.args
        self.is_in_train = True

        # ##### trainging data and related num setting #####
        # TODO(guosheng): remove the binding method get_collator of dataset
        with (
            guard_set_args(
                args,
                {"per_device_train_batch_size": self.args.per_device_prompt_batch_size},
            ),
            guard_set_args(
                self,
                {"train_dataset": self.train_dataset, "data_collator": self.data_collator},
            ),
        ):
            train_dataloader = self.prompt_only_dataloader = self.get_train_dataloader()

        (
            total_train_batch_size,
            len_dataloader,
            max_steps,
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
        ) = self.init_train_num(train_dataloader)

        # ##### model and optimizer related setting #####
        actor_model, critic_model = self.init_train_model_opt(max_steps, resume_from_checkpoint)
        paddle.device.cuda.empty_cache()

        # ##### traing statistic logging #####
        # Number of trainable parameters only account for actor_model
        self.init_train_log(
            num_examples,
            num_train_epochs,
            total_train_batch_size,
            max_steps,
            num_train_samples,
            actor_model,
        )

        # ##### set training state and resume #####
        # consumed_samples used to set train_dataloader.batch_sampler may not be
        # correct. Thus, data cannot be resumed perfectly when not breaking at epoch end.
        epochs_trained, steps_trained_in_current_epoch, steps_trained_progress_bar = self.init_train_state(
            resume_from_checkpoint,
            train_dataloader,
            max_steps,
            num_train_epochs,
            num_update_steps_per_epoch,
        )

        steps_in_epoch = num_update_steps_per_epoch * args.gradient_accumulation_steps

        # self.callback_handler.model = self.model
        # self.callback_handler.optimizer = self.optimizer
        # self.callback_handler.lr_scheduler = self.lr_scheduler
        # self.callback_handler.train_dataloader = train_dataloader
        self.state.max_steps = int(max_steps)
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        self._globalstep_last_logged = self.state.global_step
        metric = PPOMetric(freq=self.args.logging_steps, args=self.args)

        start_time = time.time()
        self._globalstep_last_start_time = start_time

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                train_dataloader.batch_sampler, DistributedBatchSampler
            ):
                train_dataloader.batch_sampler.set_epoch(epoch)

            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            for prompt_only_batch in self.prompt_only_dataloader:
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                # step 1-1: rollout data with actor model (eval) and reward model
                self.set_eval()

                data_trans_group = getattr(self.actor_trainer, "_data_trans_group", None)
                prompt_only_batch = data_group_split(prompt_only_batch, group=data_trans_group)

                cleanup_batches, indices, label_ids_batches = [], [], []
                total_batch_size = prompt_only_batch["input_ids"].shape[0]
                per_device_rollout_batch_size = self.args.per_device_rollout_batch_size

                timer_scope_actor_model = TimerScope(
                    self.timers,
                    RolloutStages.ACTOR_MODEL_ENABLE_DISABLE,
                    minus_names=[RolloutStages.GENERATE, RolloutStages.ROLLOUT_LOGPROB],
                )
                timer_scope_actor_model.start()
                with reload_and_offload_scope(self, self.actor_model, self.reference_model):
                    timer_scope_rollout = TimerScope(self.timers, RolloutStages.GENERATE)
                    timer_scope_rollout.start()
                    with infer_guard(self.actor_trainer):
                        for i in range(0, total_batch_size, per_device_rollout_batch_size):
                            micro_batch = map_structure(
                                lambda tensor: tensor[i : i + per_device_rollout_batch_size],
                                prompt_only_batch,
                            )

                            # generate for multi batches and then disable FuseMT model
                            generated_batches = self.actor_trainer.generate_sequences(micro_batch)
                            # NOTE(drownfish19): do process for each micro_batch, prepare for split mode
                            micro_ret = self.remove_pad_tokens_after_generate(generated_batches)
                            micro_cleanup_batches, micro_indices, micro_label_ids_batches = micro_ret
                            cleanup_batches.extend(micro_cleanup_batches)
                            indices.extend(micro_indices)
                            label_ids_batches.extend(micro_label_ids_batches)
                        indices = np.concatenate(indices)
                    timer_scope_rollout.stop()

                    # step 2-1: compute logprob for rollout data
                    with TimerScope(self.timers, RolloutStages.ROLLOUT_LOGPROB):
                        per_device_train_batch_size = self.args.per_device_train_batch_size
                        micro_batches = []

                        for i in range(0, len(cleanup_batches), per_device_train_batch_size):
                            cur_batch = [
                                self.truncate_batch_data(
                                    batch,
                                    truncate_max_len=self._model_config.max_position_embeddings,
                                )
                                for batch in cleanup_batches[i : i + per_device_train_batch_size]
                            ]

                            pad_to_multiple_of = (
                                self.args.tensor_parallel_degree if self._model_config.sequence_parallel else None
                            )
                            input_ids, position_ids = self.pad_batch_data(
                                cur_batch, pad_to_multiple_of=pad_to_multiple_of
                            )
                            prompt = prompt_only_batch["input_ids"][i : i + per_device_train_batch_size]

                            micro_batch = {
                                "prompt": prompt,
                                "input_ids": input_ids,
                                "position_ids": position_ids,
                                "index": indices[i : i + per_device_train_batch_size],
                                **(
                                    {"label_ids": label_ids_batches[i : i + per_device_train_batch_size]}
                                    if self.args.use_rm_server
                                    else {}
                                ),
                            }

                            with TimerScope(self.timers, RolloutStages.ROLLOUT_OLD_LOGPROB):
                                micro_batch["log_probs"] = self.actor_trainer.compute_logprob(**micro_batch)
                            with TimerScope(self.timers, RolloutStages.ROLLOUT_REF_LOGPROB):
                                micro_batch["ref_log_probs"] = self.reference_trainer.compute_logprob(**micro_batch)
                            micro_batches.append(micro_batch)

                timer_scope_actor_model.stop()

                # step 2-2: compute reward for rollout data
                with TimerScope(
                    self.timers,
                    RolloutStages.REWARD_MODEL_ENABLE_DISABLE,
                    minus_names=[RolloutStages.ROLLOUT_REWARD_VALUE],
                ):
                    with reload_and_offload_scope(
                        self,
                        self.reward_critic_model if self.args.rl_algorithm == "ppo" else None,
                        self.reward_model if not self.args.use_rm_server else None,
                    ):
                        with TimerScope(self.timers, RolloutStages.ROLLOUT_REWARD_VALUE):
                            for micro_batch in micro_batches:
                                micro_batch["rewards"] = self.reward_trainer.compute_reward(
                                    input_ids_tokenizer=self.tokenizer,
                                    **micro_batch,
                                )
                                if self.args.rl_algorithm == "ppo":
                                    micro_batch["reward_values"] = self.critic_trainer.compute_reward(**micro_batch)

                # prepare data for reinforce_plus_plus
                if self.args.rl_algorithm == "reinforce_plus_plus":
                    rl_batches = self.distribute_gather_and_pad_data(micro_batches)
                else:
                    rl_batches = micro_batches

                # step 2-3: compute reward normalization
                for rl_batch in rl_batches:
                    rl_batch["ori_rewards"] = rl_batch["rewards"].clone()

                if self.args.normalize_reward:
                    rl_batches = self.compute_reward_normalization(rl_batches)

                with TimerScope(self.timers, RolloutStages.ROLLOUT_ADVANTAGE):
                    # step 2-4: compute advantage
                    rl_batches = self.compute_advantage(rl_batches, use_tgt_len_value=args.use_tgt_len_value)

                    # step 2-5: compute advantage normalization
                    if self.args.normalize_advantage:
                        rl_batches = self.compute_advantage_normalization(rl_batches)

                # prepare data for reinforce_plus_plus
                if self.args.rl_algorithm == "reinforce_plus_plus":
                    train_batch = self.distribute_get_rank_data(micro_batches, rl_batches)
                else:
                    train_batch = rl_batches

                train_batch = data_group_merge(train_batch, group=data_trans_group)

                # step 3: train actor model and critic model with rollout data
                self.set_train()
                with TimerScope(self.timers, ActorStages.MODEL_ENABLE_DISABLE, minus_names=[ActorStages.RL_STEP]):
                    with reload_and_offload_scope(self, self.actor_model, self.actor_trainer.optimizer):
                        with TimerScope(self.timers, ActorStages.RL_STEP):
                            # timer_info = {} # prepare for each micro_step

                            for micro_step, rl_batch in enumerate(train_batch):
                                step = 0 if step == -1 else step
                                with TimerScopeManualLabel(
                                    self.timers,
                                    get_timer_label(ActorStages.MICRO_STEPS) + f"_{micro_step}",
                                    minus_names=[get_timer_label(ActorStages.OPTIMIZE_STEP)],
                                ):
                                    rl_info = self.actor_trainer.update_actor(rl_batch)

                                paddle.device.cuda.empty_cache()

                                if self.args.rl_algorithm == "ppo":
                                    rl_info["train_value_loss"] = self.critic_trainer.update_critc(rl_batch)
                                if self.is_step_end():
                                    self.state.global_step += 1
                                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                                    rl_info.update(self.get_step_loss(loss_prefix="train_"))
                                    rl_info = metric.update(rl_info)
                                    self.timers and rl_info.update(
                                        self.timers.info(self.timers.timers.keys(), reset=False)
                                    )
                                    # on_step_end
                                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                                else:
                                    # on_sub_step_end
                                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                                step += 1

                self._print_timer()
                self._maybe_log_save_evaluate(rl_info, None, epoch, ignore_keys_for_eval, inputs=rl_batch)
                paddle.device.cuda.empty_cache()
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({self.state.max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            # argument model is not used in _maybe_log_save_evaluate, thus use None
            self._maybe_log_save_evaluate(rl_info, None, epoch, ignore_keys_for_eval, inputs=rl_batch)

            if self.control.should_training_stop:
                break
        # TODO(guosheng): add epilogue of training
        logger.info("\nTraining completed. \n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            if args.local_rank != -1:
                dist.barrier()

            best_model_checkpoint = json.loads(self.state.best_model_checkpoint)

            logger.info(f"Loading best model from {best_model_checkpoint['value']}(score: {self.state.best_metric}).")
            self.load_best_ckpt(best_model_checkpoint["value"], self.critic_trainer)

            logger.info(f"Loading best model from {best_model_checkpoint['policy']}(score: {self.state.best_metric}).")
            self.load_best_ckpt(best_model_checkpoint["policy"], self.actor_trainer)

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )

        self.is_in_train = False
        self.log(metrics)
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        tr_loss = 0.0
        for history in self.state.log_history:
            if "train_policy_loss" in history:
                tr_loss += history["train_policy_loss"]
        tr_loss = tr_loss / self.state.global_step
        return TrainOutput(self.state.global_step, tr_loss, metrics)

    def load_best_ckpt(self, model_path, trainer, **kwargs):
        """
        Load the best checkpoint from the given path into the specified trainer.

        Args:
            args (TrainingArguments): The arguments object containing the configuration settings.
            model_path (str): The path to the directory where the best checkpoint is located.
            trainer (Trainer): The trainer instance that will receive the loaded weights.
            kwargs (Any, optional): Additional keyword arguments passed to the `load_unified_checkpoint` function.
        """

        if trainer.args.unified_checkpoint:
            trainer.unified_checkpoint_handler.load_unified_checkpoint(
                trainer.model,
                model_path,
            )
            if trainer.args.sharding_parallel_degree > 1 or trainer.args.data_parallel_degree > 1:
                broadcast_dataset_rank0_model(trainer.model)
        else:
            weight_name = PADDLE_WEIGHTS_NAME
            best_model_path = os.path.join(
                model_path,
                _add_variant(weight_name, trainer.args.weight_name_suffix),
            )
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = paddle.load(best_model_path, return_numpy=True)
                # If the model is on the GPU, it still works!
                trainer._set_state_dict_in_model(state_dict)
            else:
                logger.warning(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        """
        记录、保存和评估，如果需要。
            如果控制变量指示应该记录，则记录损失，并将模型保存到磁盘上。
            如果控制变量指示应该评估，则评估模型并将结果保存到磁盘上。

            Args:
                tr_loss (Optional[Dict[str, float]]): 字典形式的训练损失，包含键'train_policy_loss'和'train_ptx_loss'。
                    如果为None，则不记录任何内容。默认为None。
                model (Model): 用于评估的模型。
                epoch (int): 当前迭代次数。
                ignore_keys_for_eval (List[str]): 在评估时要忽略的键列表。默认为空列表。
                kwargs (Any, optional): 其他可选参数，将被传递给`log()`和`save()`方法。默认为空字典。

            Returns:
                None.

            Raises:
                None.
        """
        if self.control.should_log and tr_loss is not None:
            logs: Dict[str, float] = {}
            # use_ptx would double the gradient_accumulation_steps which causes
            # policy_loss and ptx_loss reduced by half. Moreover, ptx_loss should
            # be divided by ptx_coeff for logging.
            logs.update(tr_loss)
            logs["global_step"] = int(self.state.global_step)
            logs["train_actor_lr"] = float(f"{self.actor_trainer._get_learning_rate():.3e}")
            if self.args.rl_algorithm == "ppo":
                logs["train_reward_critic_lr"] = float(f"{self.critic_trainer._get_learning_rate():.3e}")

            total_train_batch_size = (
                self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.dataset_world_size
            )
            num_steps = self.state.global_step - self._globalstep_last_logged
            logs.update(
                speed_metrics(
                    "interval",
                    self._globalstep_last_start_time,
                    num_samples=total_train_batch_size * num_steps,
                    num_steps=num_steps,
                )
            )

            self._globalstep_last_logged = self.state.global_step
            self._globalstep_last_start_time = time.time()

            self.log(logs, **kwargs)

        # To trigger evaluation and save but avoid log again
        with guard_set_args(self.control, {"should_log": False}):
            super()._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval)

    def add_kl_divergence_regularization(
        self,
        prompt: paddle.Tensor,  # size = (B, S) # pylint: disable=unused-argument
        log_probs: paddle.Tensor,  # size = (B, L)
        ref_log_probs: paddle.Tensor,  # size = (B, L)
        reward_score: paddle.Tensor,  # size = (B,)
        sequence_mask: paddle.Tensor,  # size = (B, L)
    ) -> paddle.Tensor:
        """
            计算KL散度迭代增益，并将其添加到回报中。
        参数：
            prompt (paddle.Tensor, shape=(B, S)): 输入序列的prompt，未使用。
            log_probs (paddle.Tensor, shape=(B, L)): 当前预测的log概率分布。
            ref_log_probs (paddle.Tensor, shape=(B, L)): 基线预测的log概率分布。
            reward_score (paddle.Tensor, shape=(B,)): 基于prompt和输出序列的基本奖励得分。
            sequence_mask (paddle.Tensor, shape=(B, L)): 序列的mask，用于确定序列的长度。
        返回值（paddle.Tensor, shape=(B, L)}：
            包含KL散度迭代增益的向量。
        """

        kl_divergence_estimate = -self.kl_coeff * (log_probs - ref_log_probs)  # size = (B, L)
        rewards = kl_divergence_estimate  # size = (B, L)
        reward_clip = paddle.clip(  # size = (B,)
            reward_score,
            min=-self.clip_range_score,
            max=self.clip_range_score,
        )
        # TODO(guosheng): use scatter_add/put_along_axis
        index = paddle.cumsum(sequence_mask.cast(paddle.int64), axis=-1).argmax(-1, keepdim=True)

        rewards = paddle.put_along_axis(
            rewards,
            index,
            reward_clip.unsqueeze(axis=-1),
            axis=-1,
            reduce="add",
        )
        return rewards, kl_divergence_estimate

    def get_advantages_and_returns(
        self,
        values: paddle.Tensor,
        rewards: paddle.Tensor,
        sequence_mask: paddle.Tensor,
        start: int,
        use_tgt_len_return: bool = True,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute advantages and returns using Generalized Advantage Estimation (GAE)."""
        # Modified from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        last_gae_lambda = 0.0
        advantages_reversed = []
        values = values * sequence_mask
        rewards = rewards * sequence_mask
        length = rewards.shape[-1]
        if use_tgt_len_return and start > 0:
            # consistent with Beaver
            # values length is src+tgt-1, start is src-1, return length is tgt
            pass
        elif use_tgt_len_return:
            # values length is tgt, start is 0, return length is tgt
            assert start == 0
        else:
            # values length is src+tgt-1, start is src-1, return length is src+tgt-1
            pass
        for t in reversed(range(start, length)):  # pylint: disable=invalid-name
            next_values = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * next_values - values[:, t]
            last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda
            advantages_reversed.append(last_gae_lambda)
        advantages = paddle.stack(advantages_reversed[::-1], axis=1)
        returns = advantages + values[:, start:].contiguous()

        if not use_tgt_len_return:
            advantages = paddle.concat(
                [
                    paddle.zeros([advantages.shape[0], start], dtype=advantages.dtype),
                    advantages,
                ],
                axis=-1,
            )
            returns = paddle.concat(
                [
                    paddle.zeros([returns.shape[0], start], dtype=returns.dtype),
                    returns,
                ],
                axis=-1,
            )

        return advantages.detach(), returns

    @paddle.no_grad()
    def compute_reward_normalization(self, rl_batches):
        batch_rewards_list = [rl_batch["rewards"] for rl_batch in rl_batches]
        batch_rewards = paddle.concat(batch_rewards_list, axis=0)
        batch_rewards = batch_rewards.cast(paddle.float32)

        try:
            hcg = fleet.get_hybrid_communicate_group()
            sd_group = hcg.get_sharding_parallel_group()
            dp_group = hcg.get_data_parallel_group()

            if sd_group.nranks > 1:
                all_gather_batch_rewards = []
                dist.all_gather(all_gather_batch_rewards, batch_rewards, group=sd_group)
                batch_rewards = paddle.flatten(paddle.stack(all_gather_batch_rewards))
            if dp_group.nranks > 1:
                all_gather_batch_rewards = []
                dist.all_gather(all_gather_batch_rewards, batch_rewards, group=dp_group)
                batch_rewards = paddle.flatten(paddle.stack(all_gather_batch_rewards))
        except AttributeError:
            pass

        batch_rewards_mean = batch_rewards.mean()
        # batch_rewards_std = batch_rewards.std()
        batch_rewards_var = batch_rewards.var()

        current_batch_num = batch_rewards.shape[0]
        delta = batch_rewards_mean - self.reward_mean
        total_batch_num = self.sample_batch_num + current_batch_num

        new_mean = self.reward_mean + delta * current_batch_num / total_batch_num
        m_a = self.reward_var * self.sample_batch_num
        m_b = batch_rewards_var * current_batch_num
        m2 = m_a + m_b + paddle.square(delta) * (self.sample_batch_num * current_batch_num / total_batch_num)
        new_var = m2 / total_batch_num

        self.reward_mean = new_mean
        self.reward_var = new_var
        self.sample_batch_num = total_batch_num

        for rl_batch in rl_batches:
            reward_mean = self.reward_mean.cast(paddle.bfloat16)
            reward_std = self.reward_var.sqrt().cast(paddle.bfloat16)
            rl_batch["rewards"] = (rl_batch["rewards"] - reward_mean) / (reward_std + 1e-8)

        return rl_batches

    @paddle.no_grad()
    def compute_advantage(self, rl_batches, use_tgt_len_value):
        for rl_batch in rl_batches:
            old_log_probs = rl_batch["log_probs"]  # length: src + tgt -1
            ref_log_probs = rl_batch["ref_log_probs"]  # length: src + tgt -1
            rewards = rl_batch["rewards"]  # length: 1
            if self.args.rl_algorithm == "ppo":
                old_reward_values = rl_batch["reward_values"]  # length: src + tgt -1

            if self.args.rl_algorithm == "grpo":
                eos_mask = (rl_batch["input_ids"] != self.tokenizer.pad_token_id)[:, 1:].to(old_log_probs.dtype)
                start = rl_batch["prompt"].shape[-1] - 1
                reward_advantages = compute_grpo_advantages(
                    rewards, rl_batch["index"], eos_mask[:, start:], old_log_probs.shape[-1]
                )
            elif self.args.rl_algorithm == "ppo":
                start = rl_batch["prompt"].shape[-1] - 1
                eos_mask = (rl_batch["input_ids"] != self.tokenizer.pad_token_id)[:, 1:].to(old_log_probs.dtype)
                rewards_with_kl, kl_rewards = self.add_kl_divergence_regularization(
                    None,  # prompt,
                    old_log_probs,
                    ref_log_probs,
                    rewards,
                    eos_mask[:, start:],
                )  # length: tgt if use_tgt_len_value src + tgt -1
                reward_advantages, reward_returns = self.get_advantages_and_returns(
                    old_reward_values,
                    rewards_with_kl,
                    eos_mask[:, start:],
                    start=0 if use_tgt_len_value else start,
                    use_tgt_len_return=use_tgt_len_value,
                )  # length: tgt if use_tgt_len_value src + tgt -1
            elif self.args.rl_algorithm == "reinforce_plus_plus":
                start = 0
                eos_mask = rl_batch["eos_mask"]
                rewards_with_kl, kl_rewards = self.add_kl_divergence_regularization(
                    None,  # prompt,
                    old_log_probs,
                    ref_log_probs,
                    rewards,
                    eos_mask[:, start:],
                )  # length: tgt if use_tgt_len_value src + tgt -1
                reward_advantages, reward_returns = compute_reinforce_plus_plus_advantages_and_returns(
                    rewards_with_kl,
                    eos_mask[:, start:],
                    self.gamma,
                )  # length: tgt if use_tgt_len_value src + tgt -1
            else:
                raise ValueError(f"Unknown rl_algorithm: {self.args.rl_algorithm}")

            rl_batch.update(
                {
                    "log_probs": old_log_probs,
                    "reward_advantages": reward_advantages,
                    "reward_advantages_clean": reward_advantages[eos_mask[:, start:] != 0],
                    "ref_log_probs": ref_log_probs,
                    "rewards": rewards,
                    "eos_mask": eos_mask[:, start:],
                }
            )
            if self.args.rl_algorithm in ["reinforce_plus_plus", "ppo"]:
                if self.args.rl_algorithm == "ppo":
                    rl_batch.update({"reward_values": old_reward_values})

                rl_batch.update(
                    {
                        "reward_returns": reward_returns,
                        "kl_rewards": kl_rewards,
                        "rewards_with_kl": rewards_with_kl,
                    }
                )

            # pop out to reduce data dispatch comm overhead
            # rl_batch.pop("prompt")

        return rl_batches

    @paddle.no_grad()
    def compute_advantage_normalization(rl_batches):
        all_advantages_list = []
        for rl_batch in rl_batches:
            all_advantages_list.append(rl_batch["reward_advantages_clean"])
        all_advantages = paddle.concat(all_advantages_list, axis=0)
        all_advantages = all_advantages.cast(paddle.float32)

        try:
            hcg = fleet.get_hybrid_communicate_group()
            sd_group = hcg.get_sharding_parallel_group()
            dp_group = hcg.get_data_parallel_group()

            if sd_group.nranks > 1:
                object_list = []
                dist.all_gather_object(object_list, all_advantages.tolist(), group=sd_group)
                flattened_data = [item for sublist in object_list for item in sublist]
                all_advantages = paddle.to_tensor(flattened_data, dtype="float32")
            if dp_group.nranks > 1:
                object_list = []
                dist.all_gather_object(object_list, all_advantages.tolist(), group=dp_group)
                flattened_data = [item for sublist in object_list for item in sublist]
                all_advantages = paddle.to_tensor(flattened_data, dtype="float32")
        except AttributeError:
            pass
        all_advantages_mean = all_advantages.mean()
        all_advantages_std = all_advantages.std()
        for rl_batch in rl_batches:
            all_advantages_mean = all_advantages_mean.cast(paddle.bfloat16)
            all_advantages_std = all_advantages_std.cast(paddle.bfloat16)
            rl_batch["reward_advantages"] = (rl_batch["reward_advantages"] - all_advantages_mean) / (
                all_advantages_std + 1e-8
            )
            rl_batch["reward_advantages"] = rl_batch["reward_advantages"] * rl_batch["eos_mask"]
