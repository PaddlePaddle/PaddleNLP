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
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import PipelineLayer
from paddle.io import DataLoader, Dataset, DistributedBatchSampler
from rich.console import Console
from rich.table import Table

from ...data import DataCollator
from ...datasets.rlhf_datasets.protocol import DataProto, TensorDict
from ...generation import GenerationConfig
from ...trainer.trainer import (
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
from ...trainer.trainer_utils import SchedulerType
from ...trainer.utils.helper import broadcast_dataset_rank0_model, distributed_concat
from ...transformers import (
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
    PretrainedModel,
    PretrainedTokenizer,
)
from ...transformers.model_utils import _add_variant
from ...trl import llm_utils
from ...utils.env import PADDLE_WEIGHTS_NAME
from ..algos.advantage import (
    add_kl_divergence_regularization,
    compute_gae_advantage_return,
    compute_grpo_advantages,
    compute_reinforce_plus_plus_advantages_and_returns,
)
from ..algos.penalty import apply_overlong_penalty
from ..models.ppo_model_utils import make_position_ids_from_input_ids
from ..utils.comm_utils import (
    ActorStages,
    RolloutStages,
    data_group_merge,
    data_group_split,
    filter_valid_reward_groups,
    gather_and_pad_dataproto,
    gather_tensor_list,
    get_timer_label,
    new_timer_log,
    split_batch_by_rank,
)
from ..utils.infer_utils import infer_guard
from ..utils.offload_utils import reload_and_offload_scope, reload_tensor_to_gpu
from ..utils.reshard_utils import ReshardController
from ..utils.timer_utils import TimerScope, TimerScopeManualLabel
from .actor_trainer import ActorReferenceTrainer
from .critic_trainer import CriticTrainer
from .reward_trainer import RewardTrainer
from .rl_trainer import RLTrainerBase
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
        Set the meta-information of metrics, including metric names and operations.

        Args:
            None.

        Returns:
            None: Directly modifies class attributes.
        """
        self.metric_names = [
            "train_" + name
            for name in (
                [
                    "policy_loss",
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

        self.metric_ops = ["mean"] * (len(self.metric_names) - 2) + ["max", "min"]

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
    def update(self, metrics: DataProto) -> Union[None, DataProto]:
        """
        If has updated for`freq` times then return metrics (results reduced from
        all worker) and reset metric states, otherwise return `None`.
        """
        # PipelineParallel broadcast loss with shape [1]
        metrics = TensorDict(
            {
                k: (v.squeeze() if isinstance(v, paddle.Tensor) and v.shape == [1] else v)
                for k, v in metrics.meta_info["metrics"].items()
            }
        )
        for name in self.metric_names:
            # if len(metrics[name].shape) != 0:
            #     metrics[name] = metrics[name].squeeze()
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
            return DataProto(meta_info={"metrics": out_metrics})


class PPOTrainer(RLTrainerBase):
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
        generation_config: Optional[GenerationConfig] = None,
        reshard_controller: Optional[ReshardController] = None,
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
            # just used to create trivial attrs might be used in the training
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

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._model_config = actor_model.config
        self._actor_model_eval = actor_model_eval
        self._critic_model_eval = critic_model_eval

        # ##### trainging data and related num setting #####
        with (
            guard_set_args(
                args,
                {"per_device_train_batch_size": self.args.global_gen_batch_size // self.args.dataset_world_size},
            ),
            guard_set_args(
                self,
                {"train_dataset": self.train_dataset, "data_collator": self.data_collator},
            ),
        ):
            self.train_dataloader = self.prompt_only_dataloader = self.get_train_dataloader()  # 64

        (
            self.total_train_batch_size,
            self.len_dataloader,
            self.max_steps,
            self.num_train_epochs,
            self.num_update_steps_per_epoch,
            self.num_examples_,  # There is a problem with duplicate names
            self.num_train_samples,
        ) = self.init_train_num(self.train_dataloader)

        args.max_steps = self.max_steps

        self.reshard_controller = reshard_controller
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
            reshard_controller=reshard_controller,
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
        # [gae_lambda] value needs to be set manually!
        # On the gsm8k benchmark, this value is 1.0.
        self.gae_lambda = 1.0

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
        self.generation_config = generation_config

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
        reshard_controller: Optional[ReshardController] = None,
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
            reshard_controller,
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
            "critic_min_learning_rate",
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

            if not self.args.use_rm_server and not self.args.use_rule_reward:
                if args.pipeline_parallel_degree > 1 or ShardingOption.FULL_SHARD in args.sharding:
                    reward_trainer.init_train_model_opt(100, None, clear_master_weight=True)  # dummy max_steps

        reward_trainer.timers = self.timers

        return reward_trainer

    @property
    def reference_model(self):
        """
        Get the reference model, return None if it doesn't exist.
        This method can only be used after initialization, otherwise an exception will be raised.

        Returns:
            paddle.nn.Layer, optional - The reference model, return None if it doesn't exist.

        Raises:
            Exception - An exception will be raised if the reference_trainer is not initialized before calling this method.
        """
        return self.reference_trainer.get_model(train=False)

    @property
    def reward_model(self):
        """
        Get the reward model, create one if it doesn't exist.

        Returns:
            paddle.nn.Layer: The reward model.
        """
        if self.args.use_rm_server:
            return self.reward_server
        else:
            return self.reward_trainer.get_model(train=False)

    @property
    def actor_model(self):
        """
        Get the current actor model. If in training mode, return the trained model; otherwise, return the model for evaluation.

        Returns:
            paddle.nn.Layer: The actor model.
        """
        return self.actor_trainer.get_model(train=self.training)

    @property
    def critic_model(self):
        """
        Get the critic model, which is only valid when using value-based strategies.

        Returns:
            paddle.nn.Layer, optional: The critic model, return None if not set.
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
        Get the learning rate scheduler, return None if the minimum learning rate is not set.
        Supports two types of learning rate schedulers: "cosine" and "linear".

        Args:
            args (argparse.Namespace): Command-line arguments containing parameters related to the learning rate.

        Returns:
            paddle.optimizer.lr.LRScheduler or None, optional: The learning rate scheduler or None, default is None.
        """
        if args.decay_steps is None:
            args.decay_steps = args.max_steps
        if args.warmup_steps > 0:
            warmup_steps = args.warmup_steps
        else:
            warmup_steps = args.warmup_ratio * args.max_steps
        lr_scheduler = None
        if args.min_learning_rate is not None:
            if args.lr_scheduler_type == SchedulerType.COSINE or args.lr_scheduler_type == "cosine":
                lr_scheduler = CosineAnnealingWithWarmupDecay(
                    max_lr=args.learning_rate,
                    min_lr=args.min_learning_rate,
                    warmup_step=warmup_steps,
                    decay_step=args.decay_steps,
                    last_epoch=0,
                )
            elif args.lr_scheduler_type == SchedulerType.LINEAR or args.lr_scheduler_type == "linear":
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
        Prediction step to generate the next input sequence.

        Args:
            model (nn.Layer): The model instance, which should be a subclass of `paddle.nn.Layer`.
            inputs (Dict[str, Union[paddle.Tensor, Any]]): A dictionary containing input data, with the following keys:
                - "input_ids" (paddle.Tensor, optional): IDs of the input sequences, default is None.
                - "attention_mask" (paddle.Tensor, optional): Attention mask for the input sequences, default is None.
                - "position_ids" (paddle.Tensor, optional): Position IDs of the input sequences, default is None.
            prediction_loss_only (bool): Only return the prediction loss and not any other values.
            ignore_keys (Optional[List[str]], optional): A list of keys to ignore, default is None.

        Returns:
            Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
            A tuple containing the following elements:
                - Optional[paddle.Tensor]: Prediction scores if `prediction_loss_only` is False, otherwise None.
                - Optional[paddle.Tensor]: Currently undefined, always None.
                - Optional[paddle.Tensor]: Currently undefined, always None.

        Raises:
            ValueError: If `ignore_keys` is not an optional parameter or is not a list.
        """
        inputs = self._prepare_inputs(inputs)
        data_trans_group = getattr(self.actor_trainer, "_data_trans_group", None)
        inputs = data_group_split(inputs, group=data_trans_group)
        with reload_and_offload_scope(self, self.actor_model, self.reference_model, self.actor_trainer):
            with infer_guard(self.actor_trainer):
                prompt_only_batch = DataProto.from_single_dict(
                    {
                        "input_ids": inputs["input_ids"],
                        **(
                            {"label_ids": inputs["label_ids"]}
                            if self.args.use_rm_server or self.args.use_rule_reward
                            else {}
                        ),
                    }
                )
                generated_seq = self.actor_trainer.generate_sequences(prompt_only_batch, do_eval=True)[0].batch[
                    "input_ids"
                ]

            if self.reshard_controller is not None:
                self.reshard_controller.set_train_env("[after prediction_step]")
            if self.args.use_rule_reward:
                prompt_len = inputs["input_ids"].shape[-1]
                if "label_ids" not in inputs:
                    raise ValueError("Rule-based reward needs labels.")
                tgt = self.tokenizer.batch_decode(inputs["label_ids"], skip_special_tokens=False)
                response = self.tokenizer.batch_decode(generated_seq[:, prompt_len:], skip_special_tokens=False)
                ground_truth = [i.replace(self.tokenizer.pad_token, "") for i in tgt]
                response_str = [i.replace(self.tokenizer.pad_token, "") for i in response]
                reward_score = self.reward_trainer.compute_score(response_str, ground_truth)
            elif not self.args.use_rm_server:
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
                if "label_ids" not in inputs.keys():
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
        Iterate over the dataset and evaluate the model.

        Args:
            dataloader (DataLoader): The data loader used for evaluation.
            description (str): A string describing the evaluation process.
            prediction_loss_only (Optional[bool]): Whether to only compute the prediction loss. Default is None.
            ignore_keys (Optional[List[str]]): A list of keys to ignore. Default is None.
            metric_key_prefix (str): The prefix for metric keys. Default is 'eval'.
            max_eval_iters (Optional[int]): The maximum number of evaluation iterations. Default is -1, which means no limit.

        Returns:
            EvalLoopOutput: An instance of the class containing evaluation results and metrics.

        Raises:
            ValueError: If `prediction_loss_only` is not a boolean value, a ValueError exception will be raised.
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
        Get the DataLoader for evaluating the model. If `eval_dataset` is not provided, `self.eval_dataset` will be used.
        This function sets a parameter named "data_collator" and passes it to `super().get_eval_dataloader()`.

        Args:
            eval_dataset (Optional[Dataset], optional): The dataset used for evaluation. Defaults to None.

        Returns:
            DataLoader: An instance of DataLoader containing the data for evaluation.
        """
        with guard_set_args(self, {"data_collator": self.data_collator}):
            return super().get_eval_dataloader(eval_dataset)

    def _save_checkpoint(self, model, metrics=None):
        """
        Save the model and metrics to two separate checkpoints, one for the policy model and one for the value model.
        This method uses `guard_set_args` to prevent modifying the original parameters when `_save_checkpoint` is called.

        Args:
            model (nn.Module): The model to be saved.
            metrics (Optional[Dict], optional): An optional dictionary of metrics, default is None.
                The key is the metric name, and the value is the corresponding metric value.

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
        Save the model.

        Args:
            output_dir (Optional[str], optional): The output directory to save the model. Defaults to None,
                which uses the command-line argument '--output-dir'.
            merge_tensor_parallel (Optional[bool], optional): Whether to merge tensor parallel parameters.
                Defaults to False.

        Raises:
            ValueError: If `output_dir` is not within the current working directory, a ValueError exception will be raised.
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
        resume_from_checkpoint: Union[bool, str] = False,
        clear_master_weight: bool = False,
    ) -> Tuple[PretrainedModel, PretrainedModel]:
        """
        Initialize the training model and optimizer.

        If `resume_from_checkpoint` is a string, it will be treated as a path to resume the model and optimizer states
        from that location; otherwise, it will be treated as a boolean indicating whether to resume from the last saved
        checkpoint.

        If `clear_master_weight` is True, the master weights will be cleared.

        Args:
            max_steps (int): The maximum number of training steps.
            resume_from_checkpoint (Union[bool, str], optional): Whether to resume the model and optimizer states from
                a checkpoint (default is False). If it is a string, it will be treated as the path to resume from.
            clear_master_weight (bool, optional): Whether to clear the master weights (default is False).

        Returns:
            Tuple[PretrainedModel, PretrainedModel]: A tuple containing the policy model and the value function model.
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

    def init_train_num(
        self: Trainer, train_dataloader: DataLoader
    ) -> Tuple[int, Optional[int], int, int, int, int, int]:
        """
        Initialize the batch size for training data and related parameters.

        Args:
            self (Trainer): The instance of the Trainer class.
            train_dataloader (DataLoader): The DataLoader object used for training.

        Returns:
            tuple (int, Optional[int], int, int, int, int, int):
                A tuple containing:
                1. total_train_batch_size (int) - The total batch size for training.
                2. len_dataloader (Optional[int]) - The length of the DataLoader if it is not an iterable dataset; otherwise, None.
                3. max_steps (int) - The maximum number of training steps.
                4. num_train_epochs (int) - The maximum number of training epochs.
                5. num_update_steps_per_epoch (int) - The number of model updates per epoch.
                6. num_examples (int) - The number of samples in the training data.
                7. num_train_samples (int) - The total number of samples in the training data.
        """
        args = self.args
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.dataset_world_size
        len_dataloader = None
        if not self._is_iterable_dataset(self.train_dataset):
            len_dataloader = len(train_dataloader)
            num_train_sub_steps = (
                args.global_mini_batch_size * args.rollout_n * args.update_iters
            ) // args.per_device_train_batch_size
            num_update_steps_per_epoch = (num_train_sub_steps // args.gradient_accumulation_steps) * len_dataloader
            num_examples = len(self.train_dataset)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = (
                self.args.global_batch_size * self.args.update_iters * self.args.rollout_n
            ) * len_dataloader
        else:
            assert args.max_steps > 0
            max_steps = args.max_steps
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
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
        Determine if the end of the step has been reached.
        Return True when the accumulated steps equal to args.gradient_accumulation_steps.

        Returns:
            bool: Return True if the end of the step is reached, otherwise False.
        """
        # reach accumulation_steps, value trainer has the same step_control and
        # gradient_accumulation_steps as PPO trainer.
        # if (step_control + 1) % args.gradient_accumulation_steps == 0
        if self.args.rl_algorithm == "ppo":
            return self.critic_trainer.is_accumulation_step
        return self.actor_trainer.is_accumulation_step

    def get_step_loss(self, loss_prefix: str = "") -> Dict:
        """
        Get the current step's losses, including the policy training loss and the value function training loss.
        If the `loss_prefix` parameter is provided, it will be added to the loss names.

        Args:
            loss_prefix (str, optional): A prefix string for the loss names, defaults to "".

        Returns:
            Dict[str, float]: A dictionary containing two loss items: `rl_loss` (the policy training loss)
                and `value_loss` (the value function training loss).
        """
        rl_loss = self.actor_trainer.get_step_loss(loss_prefix)
        if self.args.rl_algorithm == "ppo":
            value_loss = self.critic_trainer.get_step_loss(loss_prefix)
            rl_loss.update(value_loss)
        return rl_loss

    def remove_pad_tokens_after_generate(self, generated_batches: List[DataProto]):
        cleanup_batches, indices, label_ids_batches = [], [], []

        for batch in generated_batches:
            cleanup_batches.extend(
                [
                    process_row(
                        row,
                        remove_value=self.tokenizer.pad_token_id,
                        remove_side="right",
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    for row in batch.batch["input_ids"]
                ]
            )
            if self.args.use_rm_server or self.args.use_rule_reward:
                label_ids_batches.extend(
                    [
                        process_row(
                            row,
                            remove_value=self.tokenizer.pad_token_id,
                            remove_side="left",
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                        for row in batch.batch["label_ids"]
                    ]
                )
            indices.append(batch.non_tensor_batch["index"])

        return cleanup_batches, indices, label_ids_batches

    def truncate_batch_data(self, batch, truncate_max_len):
        if len(batch) > truncate_max_len:
            batch = self.tokenizer.truncate_sequences(
                batch,
                num_tokens_to_remove=len(batch) - truncate_max_len,
                truncation_strategy="longest_first",
            )[0]
        return batch

    def get_rank_data(self, tensor):
        return tensor.split(self.args.dataset_world_size)[self.args.dataset_rank]

    def distribute_get_rank_data(self, local_batch: DataProto, global_batch: DataProto):
        local_data = {
            "reward_advantages": self.get_rank_data(global_batch.batch["reward_advantages"]),
            "rewards": self.get_rank_data(global_batch.batch["rewards"]),
            "ori_rewards": self.get_rank_data(global_batch.batch["ori_rewards"]),
            "eos_mask": self.get_rank_data(global_batch.batch["eos_mask"]),
        }
        if self.args.rl_algorithm == "reinforce_plus_plus":
            local_data["reward_returns"] = self.get_rank_data(global_batch.batch["reward_returns"])
            local_data["kl_rewards"] = self.get_rank_data(global_batch.batch["kl_rewards"])
            local_data["rewards_with_kl"] = self.get_rank_data(global_batch.batch["rewards_with_kl"])

        # shape = local_batch.batch["log_probs"].shape
        # for k, v in local_data.items():
        #     if local_data[k].ndim <= 1:
        #         local_batch.batch.update({k: local_data[k][: shape[-1]]})
        #     else:
        #         local_batch.batch.update({k: local_data[k][:, : shape[-1]]})

        # TODO(downfish19): test following code instead of above without any error
        for k, v in local_data.items():
            local_batch.batch.update({k: local_data[k]})
        return local_batch

    def _balance_batch(self, local_batch: DataProto):
        """Reorder the data such that each dp/sharding rank gets similar total tokens"""
        dp_degree, sharding_degree = max(self.args.data_parallel_degree, 1), max(self.args.sharding_parallel_degree, 1)
        # dp or sharding degree = 1, no need to balance batch
        if dp_degree * sharding_degree == 1:
            return local_batch

        # otherwise, need to balance batch across DP and Sharding groups
        try:
            hcg = fleet.get_hybrid_communicate_group()
            sharding_parallel_group = hcg.get_sharding_parallel_group()
            data_parallel_group = hcg.get_data_parallel_group()
        except:
            sharding_parallel_group = None
            data_parallel_group = None

        tensors_to_gather_per_key = defaultdict(list)
        for key in local_batch.batch.keys():
            tensors_to_gather_per_key[key].append(local_batch.batch[key])
        for key in local_batch.non_tensor_batch.keys():
            tensors_to_gather_per_key[key].append(local_batch.non_tensor_batch[key])

        global_balanced_batch_dict = {}
        # Collect and pad tensors from all workers (across DP and Sharding groups)
        for key in tensors_to_gather_per_key.keys():
            tensor_list_from_local_batch = tensors_to_gather_per_key[key]

            global_balanced_batch_dict[key] = gather_tensor_list(data_parallel_group, sharding_parallel_group)(
                DataProto.pad_or_concat_tensor_list
            )(tensor_list_from_local_batch, self.tokenizer.pad_token_id, key)

        # Truncate total_batch to match expected total batch size
        # Split total_batch evenly across all DP × Sharding ranks
        balanced_slice_for_current_rank = split_batch_by_rank(
            total_batch=global_balanced_batch_dict,
            dp_rank=hcg.get_data_parallel_rank(),
            sharding_rank=hcg.get_sharding_parallel_rank(),
            dp_degree=dp_degree,
            sharding_degree=sharding_degree,
            balance_batch_across_dp_group=True,
        )

        return DataProto.from_single_dict(balanced_slice_for_current_rank)

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

        # ##### model and optimizer related setting #####
        actor_model, critic_model = self.init_train_model_opt(self.max_steps, resume_from_checkpoint)
        paddle.device.cuda.empty_cache()

        # ##### traing statistic logging #####
        # Number of trainable parameters only account for actor_model
        self.init_train_log(
            self.num_examples_,
            self.num_train_epochs,
            self.total_train_batch_size,
            self.max_steps,
            self.num_train_samples,
            actor_model,
        )

        # ##### set training state and resume #####
        # consumed_samples used to set train_dataloader.batch_sampler may not be
        # correct. Thus, data cannot be resumed perfectly when not breaking at epoch end.
        epochs_trained, steps_trained_in_current_epoch, steps_trained_progress_bar = self.init_train_state(
            resume_from_checkpoint,
            self.train_dataloader,
            self.max_steps,
            self.num_train_epochs,
            self.num_update_steps_per_epoch,
        )

        steps_in_epoch = self.num_update_steps_per_epoch * args.gradient_accumulation_steps
        # self.callback_handler.model = self.model
        # self.callback_handler.optimizer = self.optimizer
        # self.callback_handler.lr_scheduler = self.lr_scheduler
        # self.callback_handler.train_dataloader = train_dataloader
        self.state.max_steps = int(self.max_steps)
        self.state.num_train_epochs = self.num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        self._globalstep_last_logged = self.state.global_step
        metric = PPOMetric(freq=self.args.logging_steps, args=self.args)

        start_time = time.time()
        self._globalstep_last_start_time = start_time

        num_gen_batches = 0
        if self.args.dynamic_sampling:
            total_valid_prompt = 0
            total_batch = defaultdict(list)

        is_fleet_init = True
        try:
            hcg = fleet.get_hybrid_communicate_group()
            sharding_parallel_group = hcg.get_sharding_parallel_group()
            data_parallel_group = hcg.get_data_parallel_group()
        except:
            is_fleet_init = False
            sharding_parallel_group = None
            data_parallel_group = None

        for epoch in range(epochs_trained, int(self.num_train_epochs)):
            if isinstance(self.train_dataloader, paddle.io.DataLoader) and isinstance(
                self.train_dataloader.batch_sampler, DistributedBatchSampler
            ):
                self.train_dataloader.batch_sampler.set_epoch(epoch)

            num_gen_batches += 1
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            for prompt_only_batch_dict in self.prompt_only_dataloader:
                prompt_only_batch: DataProto = DataProto.from_single_dict(prompt_only_batch_dict)
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                # step 1-1: rollout data with actor model (eval) and reward model
                self.set_eval()
                data_trans_group = getattr(self.actor_trainer, "_data_trans_group", None)
                prompt_only_batch = data_group_split(prompt_only_batch, group=data_trans_group)
                eos_token_ids = llm_utils.get_eos_token_id(self.tokenizer, self.generation_config)
                pad_token_id = self.tokenizer.pad_token_id
                prompt_only_batch.meta_info = {
                    "eos_token_ids": eos_token_ids,
                    "pad_token_id": pad_token_id,
                }

                # expand input_ids and raw_prompt_len for all sequences
                batch_keys_to_pop = ["raw_label_ids_len", "raw_prompt_len", "input_ids"]
                prompt_only_batch_expand = prompt_only_batch.select(
                    batch_keys=batch_keys_to_pop,
                )
                prompt_only_batch_expand = prompt_only_batch_expand.repeat(
                    repeat_times=self.args.rollout_n, interleave=True
                )
                prompt_only_batch_expand.rename("raw_prompt_len", "raw_prompt_len_expand")

                expand_prompt = prompt_only_batch_expand.batch["input_ids"]
                per_device_rollout_batch_size = self.args.per_device_rollout_batch_size
                cleanup_batches, indices, label_ids_batches = [], [], []

                timer_scope_actor_model = TimerScope(
                    self.timers,
                    RolloutStages.ACTOR_MODEL_ENABLE_DISABLE,
                    minus_names=[RolloutStages.GENERATE],
                )

                timer_scope_actor_model.start()
                with reload_and_offload_scope(self, self.actor_model):
                    timer_scope_rollout = TimerScope(self.timers, RolloutStages.GENERATE)
                    timer_scope_rollout.start()
                    with infer_guard(self.actor_trainer):
                        total_batch_size = prompt_only_batch.batch["input_ids"].shape[0]

                        for i in range(0, total_batch_size, per_device_rollout_batch_size):
                            micro_prompt_batch = prompt_only_batch[i : i + per_device_rollout_batch_size]
                            # generate for multi batches and then disable FuseMT model
                            generated_batches: List[DataProto] = self.actor_trainer.generate_sequences(
                                micro_prompt_batch
                            )
                            # NOTE(drownfish19): do process for each micro_batch, prepare for split mode
                            micro_ret = self.remove_pad_tokens_after_generate(generated_batches)
                            micro_cleanup_batches, micro_indices, micro_label_ids_batches = micro_ret
                            cleanup_batches.extend(micro_cleanup_batches)
                            indices.extend(micro_indices)
                            label_ids_batches.extend(micro_label_ids_batches)
                        indices = np.concatenate(indices)
                    self.timers and (dist.get_world_size() > 1) and dist.barrier()
                    timer_scope_rollout.stop()
                timer_scope_actor_model.stop()
                if self.reshard_controller is not None:
                    self.reshard_controller.set_train_env("[after rollout]")

                # step 2-1: truncate data
                truncate_input_ids = [
                    self.truncate_batch_data(batch, truncate_max_len=self._model_config.max_position_embeddings)
                    for batch in cleanup_batches
                ]
                input_ids_len = paddle.to_tensor([len(item) for item in truncate_input_ids])

                # padding data
                pad_to_multiple_of = self.args.tensor_parallel_degree if self._model_config.sequence_parallel else None
                input_ids = self.tokenizer.pad(
                    {"input_ids": truncate_input_ids},
                    padding="longest",
                    padding_side="right",
                    max_length=None,
                    return_attention_mask=False,
                    pad_to_multiple_of=pad_to_multiple_of,
                )["input_ids"]
                if len(label_ids_batches) > 0:
                    label_ids = DataProto.pad_batch_data(label_ids_batches, pad_token_id=pad_token_id)
                position_ids = make_position_ids_from_input_ids(input_ids, pad_token_id=pad_token_id)

                prompt_len = paddle.full(shape=[expand_prompt.shape[0]], fill_value=expand_prompt.shape[1], dtype=expand_prompt.dtype)  # fmt: skip
                prompt_len_without_pad = prompt_only_batch_expand.batch["raw_prompt_len_expand"]
                response_len_without_pad = input_ids_len - prompt_len

                batch = DataProto.from_single_dict(
                    {
                        "prompt": expand_prompt,
                        "input_ids": input_ids,
                        "position_ids": position_ids,
                        "prompt_len": prompt_len,
                        "prompt_len_without_pad": prompt_len_without_pad,
                        "response_len_without_pad": response_len_without_pad,
                        "index": indices,
                        **({"label_ids": label_ids} if self.args.use_rm_server or self.args.use_rule_reward else {}),
                        **(
                            {"raw_label_ids_len": prompt_only_batch_expand.batch["raw_label_ids_len"]}
                            if self.args.use_rm_server or self.args.use_rule_reward
                            else {}
                        ),
                    }
                )

                batch = data_group_merge(batch, group=data_trans_group)
                # step 2-2: balance batches based on batch tokens
                if self.args.balance_batch:
                    batch = self._balance_batch(batch)

                # step 2-3: compute logprob for rollout data
                with self.autocast_smart_context_manager():
                    with TimerScope(self.timers, RolloutStages.ROLLOUT_LOGPROB):
                        with reload_and_offload_scope(self, self.reference_model):
                            with TimerScope(self.timers, RolloutStages.ROLLOUT_REF_LOGPROB):
                                ref_log_probs = self.reference_trainer.compute_logprob(batch, key="ref_log_probs")
                                batch = batch.union(ref_log_probs)

                        with reload_and_offload_scope(self, self.actor_model):
                            with TimerScope(self.timers, RolloutStages.ROLLOUT_OLD_LOGPROB):
                                log_probs = self.actor_trainer.compute_logprob(batch, key="log_probs")
                                batch = batch.union(log_probs)

                # step 2-2: compute reward for rollout data
                with TimerScope(
                    self.timers,
                    RolloutStages.REWARD_MODEL_ENABLE_DISABLE,
                    minus_names=[RolloutStages.ROLLOUT_REWARD_VALUE],
                ):
                    with reload_and_offload_scope(
                        self,
                        self.critic_model if self.args.rl_algorithm == "ppo" else None,
                        self.reward_model if not self.args.use_rm_server and not self.args.use_rule_reward else None,
                    ):
                        with TimerScope(self.timers, RolloutStages.ROLLOUT_REWARD_VALUE):
                            reward_tensor = self.reward_trainer.compute_reward(
                                batch,
                                input_ids_tokenizer=self.tokenizer,
                            )
                            batch.batch["rewards"] = reward_tensor
                            if self.args.enable_overlong_reward_buffer:
                                overlong_penalty = apply_overlong_penalty(
                                    response_length=batch.batch["response_len_without_pad"],
                                    max_dec_len=self.args.max_dec_len,
                                    overlong_buffer_len=self.args.overlong_reward_buffer,
                                    penalty_factor=self.args.overlong_penalty_factor,
                                )
                                batch.batch["rewards_before_length_penalty"] = reward_tensor.clone()
                                batch.batch["rewards"] = reward_tensor + overlong_penalty

                            if self.args.rl_algorithm == "ppo":
                                reward_values = self.critic_trainer.compute_value(batch)
                                batch.union(reward_values)

                # dynamic sampling: filter generated samples by rewards, keep generating until valid samples are enough
                if self.args.dynamic_sampling:
                    local_valid_prompt = 0
                    combined_batch = batch
                    total_batch, local_valid_prompt = filter_valid_reward_groups(
                        combined_batch=combined_batch,
                        total_batch=total_batch,
                        rollout_n=self.args.rollout_n,
                        variance_threshold=1e-6,
                    )

                    dp_degree, sharding_degree = (
                        max(self.args.data_parallel_degree, 1),
                        max(self.args.sharding_parallel_degree, 1),
                    )
                    local_valid_prompt = paddle.to_tensor(local_valid_prompt, dtype="int32")
                    if sharding_degree > 1:
                        dist.all_reduce(local_valid_prompt, op=dist.ReduceOp.SUM, group=sharding_parallel_group)
                    if dp_degree > 1:
                        if is_fleet_init:
                            dist.all_reduce(local_valid_prompt, op=dist.ReduceOp.SUM, group=data_parallel_group)
                        else:
                            dist.all_reduce(local_valid_prompt, op=dist.ReduceOp.SUM)

                    total_valid_prompt += int(local_valid_prompt)

                    if total_valid_prompt >= self.args.global_batch_size:
                        # Collect and pad tensors from all workers (across DP and Sharding groups)
                        for key in total_batch.keys():
                            tensor_list = total_batch[key]

                            gathered_and_padded_tensor = gather_tensor_list(
                                data_parallel_group, sharding_parallel_group
                            )(DataProto.pad_or_concat_tensor_list)(tensor_list, pad_token_id, key)

                            total_batch[key] = gathered_and_padded_tensor

                        # Truncate total_batch to match expected total batch size
                        for key in total_batch.keys():
                            total_batch[key] = total_batch[key][: self.args.global_batch_size * self.args.rollout_n]

                        # Split total_batch evenly across all DP × Sharding ranks
                        if is_fleet_init and dp_degree * sharding_degree > 1:
                            total_batch = split_batch_by_rank(
                                total_batch=total_batch,
                                dp_rank=hcg.get_data_parallel_rank(),
                                sharding_rank=hcg.get_sharding_parallel_rank(),
                                dp_degree=dp_degree,
                                sharding_degree=sharding_degree,
                                balance_batch_across_dp_group=False,
                            )

                        batch = DataProto.from_single_dict(dict(total_batch))

                        # Reset for next accumulation
                        total_batch = defaultdict(list)
                        total_valid_prompt = 0
                        num_gen_batches = 0
                        logger.info("Dynamic sampling completed. \n")

                    else:
                        if self.args.max_gen_batches > 0 and num_gen_batches > self.args.max_gen_batches:
                            raise ValueError("Generated batches exceeds `max_gen_batches`. Please check your data.")
                        else:
                            logger.info(
                                f"Collected {total_valid_prompt} valid prompts, "
                                f"need {self.args.global_batch_size}. Continue Dynamic Sampling..."
                            )
                            continue

                # prepare data for reinforce_plus_plus & grpo
                if self.args.rl_algorithm in ["reinforce_plus_plus", "grpo"]:
                    local_batch = copy.deepcopy(batch)
                    select_keys = ["index", "rewards", "eos_mask", "log_probs", "ref_log_probs"]
                    batch = gather_and_pad_dataproto(
                        batch, data_parallel_group, sharding_parallel_group, eos_token_ids, pad_token_id, select_keys
                    )
                else:
                    local_batch = batch
                    batch = batch
                # step 2-3: compute reward normalization
                batch.batch["ori_rewards"] = batch.batch["rewards"].clone()

                if self.args.normalize_reward:
                    batch = self.compute_reward_normalization(batch)

                with TimerScope(self.timers, RolloutStages.ROLLOUT_ADVANTAGE):
                    # step 2-4: compute advantage
                    batch = self.compute_advantage(batch)

                    # step 2-5: compute advantage normalization
                    if self.args.normalize_advantage:
                        batch = self.compute_advantage_normalization(batch)

                # prepare data for reinforce_plus_plus & grpo
                if self.args.rl_algorithm in ["reinforce_plus_plus", "grpo"]:
                    batch = self.distribute_get_rank_data(local_batch, batch)
                else:
                    batch = batch

                # step 3: train actor model and critic model with rollout data
                self.set_train()
                with TimerScope(self.timers, ActorStages.MODEL_ENABLE_DISABLE, minus_names=[ActorStages.RL_STEP]):
                    with reload_and_offload_scope(self, self.actor_model, self.actor_trainer.optimizer):
                        with TimerScope(self.timers, ActorStages.RL_STEP):
                            # timer_info = {} # prepare for each micro_step
                            micro_batches = batch.split_batch_into_micro_batches(
                                batch_size=self.args.per_device_train_batch_size,
                                pad_token_id=self.tokenizer.pad_token_id,
                            )

                            for micro_step, micro_batch in enumerate(micro_batches * self.args.update_iters):
                                step = 0 if step == -1 else step

                                with TimerScopeManualLabel(
                                    self.timers, get_timer_label(ActorStages.MICRO_STEPS) + f"_{micro_step}"
                                ):
                                    rl_info = self.actor_trainer.update_actor(micro_batch)

                                paddle.device.cuda.empty_cache()

                                if self.args.rl_algorithm == "ppo":
                                    train_value_loss = self.critic_trainer.update_critic(micro_batch)
                                    rl_info.meta_info["metrics"].update(train_value_loss)

                                    paddle.device.cuda.empty_cache()

                                if self.is_step_end():
                                    self.state.global_step += 1
                                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                                    rl_info.meta_info["metrics"].update(self.get_step_loss(loss_prefix="train_"))
                                    rl_info = metric.update(rl_info)
                                    self.timers and rl_info.meta_info["metrics"].update(
                                        self.timers.info(self.timers.timers.keys(), reset=False)
                                    )
                                    # on_step_end
                                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                                else:
                                    # on_sub_step_end
                                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                                step += 1

                self._print_timer()
                self._maybe_log_save_evaluate(rl_info, None, epoch, ignore_keys_for_eval, inputs=micro_batch)
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
            self._maybe_log_save_evaluate(rl_info, None, epoch, ignore_keys_for_eval, inputs=micro_batch)

            if self.control.should_training_stop:
                break
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
            num_samples=self.num_train_samples,
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

    def _maybe_log_save_evaluate(self, tr_loss: DataProto, model, epoch, ignore_keys_for_eval, **kwargs):
        """
        Log, save, and evaluate if needed.

        If the control variables indicate logging is required, log the losses and save the model to disk.
        If the control variables indicate evaluation is required, evaluate the model and save the results to disk.

        Args:
            tr_loss (Optional[Dict[str, float]], optional): Training losses in dictionary form, with keys 'train_policy_loss' and 'train_ptx_loss'.
                If None, nothing will be logged. Defaults to None.
            model (Model): The model to be evaluated.
            epoch (int): The current epoch number.
            ignore_keys_for_eval (List[str]): A list of keys to ignore during evaluation. Defaults to an empty list.
            kwargs (Any, optional): Additional optional parameters that will be passed to the `log()` and `save()` methods. Defaults to an empty dictionary.

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
            logs.update(tr_loss.meta_info["metrics"])
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
            super()._maybe_log_save_evaluate(tr_loss.meta_info["metrics"], model, epoch, ignore_keys_for_eval)

    @paddle.no_grad()
    def compute_reward_normalization(self, batch):
        batch_rewards = batch["rewards"].cast(paddle.float32)

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

        reward_mean = self.reward_mean.cast(paddle.bfloat16)
        reward_std = self.reward_var.sqrt().cast(paddle.bfloat16)
        batch["rewards"] = (batch["rewards"] - reward_mean) / (reward_std + 1e-8)
        return batch

    @paddle.no_grad()
    def compute_advantage(self, batch: DataProto) -> DataProto:
        if "log_probs" in batch.batch:
            old_log_probs = batch.batch["log_probs"]  # length: src + tgt -1
        if "ref_log_probs" in batch.batch:
            ref_log_probs = batch.batch["ref_log_probs"]  # length: src + tgt -1
        rewards = batch.batch["rewards"]  # length: 1
        if self.args.rl_algorithm == "ppo":
            old_reward_values = batch.batch["reward_values"]  # length: src + tgt -1

        if self.args.rl_algorithm == "grpo":
            eos_mask = batch.batch["eos_mask"]
            reward_advantages = compute_grpo_advantages(
                rewards, batch.non_tensor_batch["index"], eos_mask, eos_mask.shape[-1]
            )
        elif self.args.rl_algorithm == "ppo":
            eos_mask = (batch.batch["input_ids"] != self.tokenizer.pad_token_id)[
                :, batch.batch["prompt"].shape[-1] :
            ].to(old_log_probs.dtype)
            rewards_with_kl, kl_rewards = add_kl_divergence_regularization(
                None,  # prompt,
                old_log_probs,
                ref_log_probs,
                rewards,
                eos_mask,
                self.kl_coeff,
                self.clip_range_score,
            )
            reward_advantages, reward_returns = compute_gae_advantage_return(
                rewards_with_kl,
                old_reward_values,
                eos_mask,
                gamma=self.gamma,
                lam=self.gae_lambda,
            )
        elif self.args.rl_algorithm == "reinforce_plus_plus":
            eos_mask = batch.batch["eos_mask"]
            rewards_with_kl, kl_rewards = add_kl_divergence_regularization(
                None,  # prompt,
                old_log_probs,
                ref_log_probs,
                rewards,
                eos_mask,
                self.kl_coeff,
                self.clip_range_score,
            )
            reward_advantages, reward_returns = compute_reinforce_plus_plus_advantages_and_returns(
                rewards_with_kl,
                eos_mask,
                self.gamma,
            )
        else:
            raise ValueError(f"Unknown rl_algorithm: {self.args.rl_algorithm}")

        batch.batch.update(
            {
                # "log_probs": old_log_probs,
                "reward_advantages": reward_advantages,
                "reward_advantages_clean": reward_advantages[eos_mask != 0],
                # "ref_log_probs": ref_log_probs,
                "rewards": rewards,
                "eos_mask": eos_mask,
            }
        )
        if self.args.rl_algorithm in ["reinforce_plus_plus", "ppo"]:
            if self.args.rl_algorithm == "ppo":
                batch.batch.update({"reward_values": old_reward_values})

            batch.batch.update(
                {
                    "reward_returns": reward_returns,
                    "kl_rewards": kl_rewards,
                    "rewards_with_kl": rewards_with_kl,
                }
            )

        # pop out to reduce data dispatch comm overhead
        # rl_batch.pop("prompt")

        return batch

    @paddle.no_grad()
    def compute_advantage_normalization(self, batch: DataProto):
        all_advantages = batch.batch["reward_advantages_clean"].cast(paddle.float32)

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
        all_advantages_mean = all_advantages.mean().cast(paddle.bfloat16)
        all_advantages_std = all_advantages.std().cast(paddle.bfloat16)
        batch.batch["reward_advantages"] = (batch.batch["reward_advantages"] - all_advantages_mean) / (
            all_advantages_std + 1e-8
        )
        batch.batch["reward_advantages"] = batch.batch["reward_advantages"] * batch.batch["eos_mask"]

        return batch
